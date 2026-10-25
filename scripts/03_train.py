"""
03_train.py
===========
Trains CNN / Separable-CNN / Xception on the leak-free dataset produced by
02_rebuild_dataset.py.

    python 03_train.py --data ./dataset_rebuilt --model xception
    python 03_train.py --cv ./dataset_cv --model xception     # all folds

What is different from the original pipeline
--------------------------------------------
1. Preprocessing is BAKED INTO the model as a Rescaling layer, so the Flask app
   can never disagree with training. (Original bug: Xception was fed x/255 at
   inference, but ImageNet Xception weights expect x/127.5 - 1. That mismatch
   alone costs several accuracy points.)
2. Every model exposes a layer literally named `gradcam_target`, so the app
   stops guessing auto-generated names like "conv2d_25".
3. Xception uses proper two-phase transfer learning: frozen head warm-up, then
   low-LR fine-tuning of the top blocks with BatchNorm kept frozen.
4. Class weights + label smoothing + macro-F1 early stopping (accuracy is a bad
   monitor on a 16-class imbalanced problem).
5. CNN / S-CNN get BatchNorm, GlobalAveragePooling and a sane input size
   instead of 64x64 grayscale, so the baselines are honest baselines.

"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASS_NAMES = [
    "anold-chiari-malformation", "arachnoid-cyst", "cerebellah-hypoplasia",
    "colphocephaly", "encephalocele", "holoprosencephaly", "hydracenphaly",
    "intracranial-hemorrdge", "intracranial-tumor", "m-magna",
    "mild-ventriculomegaly", "moderate-ventriculomegaly", "normal",
    "polencephaly", "severe-ventriculomegaly", "vein-of-galen",
]
NUM_CLASSES = len(CLASS_NAMES)
AUTOTUNE = tf.data.AUTOTUNE


def load_class_names(data_dir):
    """Use the class list written by 02_rebuild_dataset.py when present, so a
    merged/grouped dataset trains and evaluates without editing any script."""
    for d in (Path(data_dir), Path(data_dir).parent):
        p = d / "class_names.json"
        if p.exists():
            names = json.loads(p.read_text())
            print(f"class names from {p}: {len(names)} classes")
            return names
    return None



# ==========================================================================
# data
# ==========================================================================
def clahe_batch(images):
    """CLAHE on the luminance channel. Ultrasound is low-contrast; this helps."""
    import cv2
    out = np.empty_like(images)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i, img in enumerate(images):
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(images.dtype)
    return out


def apply_clahe(x, y):
    x = tf.py_function(clahe_batch, [x], tf.float32)
    x.set_shape([None, None, None, 3])
    return x, y


def build_augmenter(hflip: bool):
    aug = [
        layers.RandomRotation(0.06, fill_mode="constant"),
        layers.RandomZoom(0.10, fill_mode="constant"),
        layers.RandomTranslation(0.06, 0.06, fill_mode="constant"),
        layers.RandomContrast(0.15),
    ]
    if hflip:
        aug.insert(0, layers.RandomFlip("horizontal"))
    if hasattr(layers, "RandomBrightness"):
        aug.append(layers.RandomBrightness(0.10, value_range=(0, 255)))
    return keras.Sequential(aug, name="augment")


def cutout_batch(max_frac=0.25):
    """Erase one random rectangle per image (filled with the image mean). Stops the
    network leaning on a single corner overlay / caliper / text label."""
    def fn(x, y):
        x = tf.cast(x, tf.float32)
        shp = tf.shape(x)
        b, h, w = shp[0], shp[1], shp[2]
        fh = tf.cast(tf.cast(h, tf.float32) * tf.random.uniform([b], 0.08, max_frac), tf.int32)
        fw = tf.cast(tf.cast(w, tf.float32) * tf.random.uniform([b], 0.08, max_frac), tf.int32)
        top = tf.cast(tf.random.uniform([b]) * tf.cast(h - fh, tf.float32), tf.int32)
        left = tf.cast(tf.random.uniform([b]) * tf.cast(w - fw, tf.float32), tf.int32)
        rows = tf.range(h)[None, :, None]
        cols = tf.range(w)[None, None, :]
        inside = ((rows >= top[:, None, None]) & (rows < (top + fh)[:, None, None]) &
                  (cols >= left[:, None, None]) & (cols < (left + fw)[:, None, None]))
        keep = tf.cast(~inside, x.dtype)[..., None]
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        return x * keep + mean * (1 - keep), y
    return fn


def grayscale_layer(name="to_gray"):
    """Fixed (non-trainable) luminance projection built from a 1x1 conv, so it is
    saved inside the .keras file (no Lambda) and the app needs no extra step.
    Removes colour overlays/annotations as a source cue."""
    luma = np.array([0.299, 0.587, 0.114], np.float32)
    kernel = np.tile(luma[:, None], (1, 3)).reshape(1, 1, 3, 3)
    return layers.Conv2D(3, 1, use_bias=False, trainable=False, name=name,
                         kernel_initializer=keras.initializers.Constant(kernel))


def mixup_batch(alpha):
    """Batch-level mixup: blends each image with a shuffled partner. Strong
    regulariser on ~1k-image problems; labels become soft."""
    def fn(x, y):
        # mixed precision makes the dataset emit float16 labels; mix in float32
        x, y = tf.cast(x, tf.float32), tf.cast(y, tf.float32)
        g1 = tf.random.gamma([], alpha)
        g2 = tf.random.gamma([], alpha)
        lam = g1 / (g1 + g2)
        idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
        return (lam * x + (1 - lam) * tf.gather(x, idx),
                lam * y + (1 - lam) * tf.gather(y, idx))
    return fn


def make_datasets(data_dir: Path, img_size: int, batch: int, clahe: bool, hflip: bool,
                  mixup: float = 0.0, cutout: float = 0.0):
    for split in ("train", "valid", "test"):
        for c in CLASS_NAMES:
            (data_dir / split / c).mkdir(parents=True, exist_ok=True)

    def load(split, shuffle):
        return keras.utils.image_dataset_from_directory(
            data_dir / split,
            labels="inferred",
            label_mode="categorical",
            class_names=CLASS_NAMES,
            image_size=(img_size, img_size),
            batch_size=batch,
            shuffle=shuffle,
            seed=42,
            interpolation="bilinear",
        )

    train_ds, val_ds, test_ds = load("train", True), load("valid", False), load("test", False)

    if clahe:
        train_ds = train_ds.map(apply_clahe, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(apply_clahe, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(apply_clahe, num_parallel_calls=AUTOTUNE)

    augmenter = build_augmenter(hflip)
    train_ds = (train_ds
                .cache()
                .shuffle(512, seed=42)
                .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE))
    if cutout > 0:
        train_ds = train_ds.map(cutout_batch(cutout), num_parallel_calls=AUTOTUNE)
    if mixup > 0:
        train_ds = train_ds.map(mixup_batch(mixup), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds


def class_weights_from_dir(train_dir: Path, power: float = 0.5):
    """Inverse-frequency weights raised to `power`. 1.0 = fully balanced (which
    on tiny classes makes a handful of images dominate the loss and drives
    memorisation); 0.5 keeps minority classes audible without that."""
    counts = Counter()
    for i, c in enumerate(CLASS_NAMES):
        counts[i] = sum(1 for _ in (train_dir / c).glob("*") if _.is_file())
    total = sum(counts.values())
    present = [i for i in counts if counts[i] > 0]
    raw = {i: (total / (len(present) * counts[i])) ** power if counts[i] else 0.0
           for i in counts}
    # renormalise so the mean sample weight stays ~1 (keeps the LR meaningful)
    mean = sum(raw[i] * counts[i] for i in counts) / max(total, 1)
    return {i: (w / mean if mean else w) for i, w in raw.items()}, counts


# ==========================================================================
# models  (all expose a `gradcam_target` layer and take raw 0-255 input)
# ==========================================================================
def build_cnn(img_size, separable=False, dropout=0.4, gray=True):
    Conv = layers.SeparableConv2D if separable else layers.Conv2D
    inp = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    x = grayscale_layer()(inp) if gray else inp
    x = layers.Rescaling(1.0 / 255.0, name="preprocess")(x)

    # first block always standard conv (separable on 3 channels learns nothing useful)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    for filters in (64, 128, 256):
        x = Conv(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = Conv(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)

    x = Conv(512, 3, padding="same", use_bias=False, name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu", name="gradcam_target")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)
    return keras.Model(inp, out, name="scnn" if separable else "cnn")


def build_xception(img_size, dropout=0.5, gray=True):
    inp = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    # Xception's preprocess_input == x/127.5 - 1, expressed as a layer so the
    # saved .keras file needs no Lambda and no safe_mode=False on load.
    x = grayscale_layer()(inp) if gray else inp
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)
    try:
        base = keras.applications.Xception(include_top=False, weights="imagenet", input_tensor=x)
        feat = base.output
    except Exception:  # fall back to a nested backbone
        base = keras.applications.Xception(include_top=False, weights="imagenet",
                                           input_shape=(img_size, img_size, 3))
        feat = base(x, training=False)
    feat = layers.Activation("linear", name="gradcam_target")(feat)
    y = layers.GlobalAveragePooling2D()(feat)
    y = layers.Dropout(dropout)(y)
    out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(y)
    return keras.Model(inp, out, name="xception"), base


def set_backbone_trainable(base, unfreeze_from, train_bn=False):
    """Freeze everything, then unfreeze from the named block onward."""
    for l in base.layers:
        l.trainable = False
    if unfreeze_from is None:
        return 0
    names = [l.name for l in base.layers]
    start = next((i for i, n in enumerate(names) if n.startswith(unfreeze_from)), None)
    if start is None:
        start = int(len(names) * 0.7)
    n = 0
    for l in base.layers[start:]:
        if isinstance(l, layers.BatchNormalization) and not train_bn:
            continue
        l.trainable = True
        n += 1
    return n


# ==========================================================================
# training
# ==========================================================================
def metrics_list():
    m = [keras.metrics.CategoricalAccuracy(name="acc")]
    try:
        m.append(keras.metrics.F1Score(average="macro", name="macro_f1"))
    except Exception:
        pass
    return m


def make_optimizer(lr, weight_decay):
    if weight_decay > 0 and hasattr(keras.optimizers, "AdamW"):
        return keras.optimizers.AdamW(lr, weight_decay=weight_decay)
    return keras.optimizers.Adam(lr)


def fit_phase(model, train_ds, val_ds, epochs, lr, ckpt, cw, tag, args, patience=8,
              best_so_far=None):
    model.compile(
        optimizer=make_optimizer(lr, args.weight_decay),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=metrics_list(),
    )
    # val_loss is far less noisy than val macro-F1 on a validation set of a few
    # dozen images (one image can swing a rare class's F1 from 0 to 1)
    if args.monitor == "macro_f1" and any(getattr(m, "name", "") == "macro_f1"
                                          for m in model.metrics):
        mon, mode = "val_macro_f1", "max"
    else:
        mon, mode = "val_loss", "min"
    cbs = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor=mon, mode=mode, save_best_only=True, verbose=1,
                                        initial_value_threshold=best_so_far),
        keras.callbacks.EarlyStopping(monitor=mon, mode=mode, patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=mon, mode=mode, factor=0.3,
                                          patience=max(3, patience // 2), min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(str(Path(ckpt).with_suffix("")) + f"_{tag}_history.csv"),
    ]
    print(f"\n--- {tag}: {epochs} epochs @ lr={lr} (monitor {mon}) ---")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     class_weight=cw, callbacks=cbs, verbose=1)
    vals = hist.history.get(mon)
    hist.best_monitor = (min(vals) if mode == "min" else max(vals)) if vals else None
    return hist


def train_one(model_key, args, tag=""):
    img_size = args.img_size or (299 if model_key == "xception" else 224)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(out_dir / f"best_model_{model_key}{tag}.keras")
    if args.skip_existing and Path(ckpt).exists():
        print(f"skip {ckpt} (already trained)")
        return ckpt

    train_ds, val_ds, _ = make_datasets(Path(args.data), img_size, args.batch_size,
                                        args.clahe, not args.no_hflip, args.mixup, args.cutout)
    cw, counts = class_weights_from_dir(Path(args.data) / "train", args.cw_power)
    print("\nTrain images per class:", {CLASS_NAMES[i]: c for i, c in sorted(counts.items()) if c})

    if model_key == "xception":
        model, base = build_xception(img_size, dropout=args.dropout if args.dropout is not None else 0.5,
                                     gray=not args.no_grayscale)
        if args.init_backbone:
            base.load_weights(args.init_backbone)
            print(f"Backbone initialised from {args.init_backbone} (anatomy pretraining)")
        set_backbone_trainable(base, None)
        h1 = fit_phase(model, train_ds, val_ds, args.warmup_epochs, 1e-3, ckpt, cw,
                       "phase1-head", args, patience=6)
        n = set_backbone_trainable(base, args.unfreeze_from, args.train_bn)
        print(f"Unfroze {n} backbone layers from '{args.unfreeze_from}' onward (BN frozen={not args.train_bn})")
        hist = fit_phase(model, train_ds, val_ds, args.epochs, args.ft_lr, ckpt, cw, "phase2-finetune",
                         args, patience=args.patience,
                         best_so_far=h1.best_monitor)
    else:
        model = build_cnn(img_size, separable=(model_key == "scnn"),
                          dropout=args.dropout if args.dropout is not None else 0.5,
                          gray=not args.no_grayscale)
        hist = fit_phase(model, train_ds, val_ds, args.epochs, args.lr, ckpt, cw, "scratch",
                         args, patience=args.patience)

    with open(out_dir / f"best_model_{model_key}{tag}.json", "w") as f:
        json.dump({"model": model_key, "img_size": img_size, "class_names": CLASS_NAMES,
                   "clahe": args.clahe, "hflip": not args.no_hflip, "mixup": args.mixup, "cutout": args.cutout, "grayscale": not args.no_grayscale,
                   "label_smoothing": args.label_smoothing, "weight_decay": args.weight_decay,
                   "cw_power": args.cw_power, "monitor": args.monitor,
                   "data": str(args.data), "init_backbone": args.init_backbone,
                   "input_range": "0-255 (preprocessing is inside the model)",
                   "gradcam_layer": "gradcam_target"}, f, indent=2)
    print(f"\nSaved {ckpt}")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="output of 02_rebuild_dataset.py")
    ap.add_argument("--cv", default=None, help="a dataset_cv/ dir: trains every fold")
    ap.add_argument("--model", choices=["cnn", "scnn", "xception", "all"], default="xception")
    ap.add_argument("--out", default="./models")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup-epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3, help="scratch models")
    ap.add_argument("--ft-lr", type=float, default=1e-5, help="fine-tuning LR for Xception")
    ap.add_argument("--unfreeze-from", default="block13",
                    help="Xception block to unfreeze from (later = fewer trainable params = less overfit)")
    ap.add_argument("--init-backbone", default=None,
                    help="Xception backbone weights (.weights.h5) from 15_pretrain_anatomy.py; "
                         "replaces the ImageNet start (xception only)")
    ap.add_argument("--dropout", type=float, default=None,
                    help="head dropout (default 0.5; try 0.6-0.7 if still overfitting)")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW decoupled weight decay")
    ap.add_argument("--mixup", type=float, default=0.2, help="mixup alpha; 0 disables")
    ap.add_argument("--cw-power", type=float, default=0.5,
                    help="class-weight exponent: 0 none, 0.5 dampened (default), 1 fully balanced")
    ap.add_argument("--cutout", type=float, default=0.25,
                    help="max side fraction of the random erased box; 0 disables")
    ap.add_argument("--no-grayscale", action="store_true",
                    help="keep colour (default: fixed luminance layer inside the model)")
    ap.add_argument("--monitor", choices=["loss", "macro_f1"], default="loss")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not retrain a fold whose checkpoint already exists (resume)")
    ap.add_argument("--only-folds", type=int, nargs="*", default=None,
                    help="with --cv: train just these fold indices (e.g. 0 3)")
    ap.add_argument("--train-bn", action="store_true", help="also fine-tune BatchNorm (usually worse on small data)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--clahe", action="store_true", help="CLAHE contrast enhancement")
    ap.add_argument("--no-hflip", action="store_true", help="disable horizontal flip augmentation")
    ap.add_argument("--mixed-precision", action="store_true")
    args = ap.parse_args()

    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.utils.set_random_seed(42)

    global CLASS_NAMES, NUM_CLASSES
    names = load_class_names(args.cv or args.data)
    if names:
        CLASS_NAMES, NUM_CLASSES = names, len(names)

    keys = ["cnn", "scnn", "xception"] if args.model == "all" else [args.model]
    if args.cv:
        folds = sorted(p for p in Path(args.cv).glob("fold_*") if p.is_dir())
        if not folds:
            raise SystemExit(f"no fold_* directories under {args.cv}")
        if args.only_folds is not None:
            folds = [f for f in folds if int(f.name.split("_")[1]) in args.only_folds]
        for f in folds:
            args.data = str(f)
            for k in keys:
                print(f"\n########## {k} — {f.name} ##########")
                train_one(k, args, tag=f"_{f.name}")
        print("\nNext:  python 04_evaluate.py --cv ./dataset_cv --models ./models")
    else:
        if not args.data:
            raise SystemExit("pass --data ./dataset_rebuilt or --cv ./dataset_cv")
        for k in keys:
            train_one(k, args)
        print("\nNext:  python 04_evaluate.py --data ./dataset_rebuilt --models ./models")


if __name__ == "__main__":
    main()

