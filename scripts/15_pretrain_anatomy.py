"""
15_pretrain_anatomy.py
======================
Pretrain the Xception backbone on the Fetal-Head anatomy sets (Trans-cerebellum,
Trans-ventricular, Diverse Fetal Head Images) so the diagnosis model starts from
ultrasound features instead of ImageNet only.

What is used, and what is NOT:
  * The only labels are the ones the archives actually carry: which of Brain /
    CSP / LV is present in the segmentation mask (multi-label, sigmoid head).
  * NO diagnosis is assigned to these images and none enters the diagnosis
    folds. They carry no normal/abnormal information.
  * Leakage guard: Fetal-Head images that are pixel-duplicates of a Roboflow image
    or share a Patient##### token with a Roboflow file are excluded, so the
    backbone never sees a scan that appears in a diagnosis test fold.

    python scripts/15_pretrain_anatomy.py --build-dir datasets/final_build_20261022_010449 \
        --out /content/anatomy_pretrain
    # then
    python scripts/03_train.py --cv ./dataset_cv --model xception \
        --init-backbone /content/anatomy_pretrain/xception_anatomy_backbone.weights.h5 ...

`--prepare-only` writes the filtered manifest without importing TensorFlow.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ANATOMY = ["Brain", "CSP", "LV"]
PATIENT = re.compile(r"(Patient\d+)", re.I)


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prepare(build: Path, repo: Path):
    """Return (kept_rows, excluded_rows[(row, reason)]) from the anatomy manifest."""
    anat = read_csv(build / "auxiliary_anatomy" / "anatomy_manifest.csv")
    dups = read_csv(build / "manifests" / "duplicate_groups.csv")
    img = read_csv(build / "manifests" / "image_manifest.csv")

    by_dup = defaultdict(set)
    rid_dup = {}
    for r in dups:
        by_dup[r["duplicate_group_id"]].add(r["source_dataset"])
        rid_dup[r["record_id"]] = r["duplicate_group_id"]
    cross_dup = {g for g, s in by_dup.items() if "roboflow_fetal_brain" in s and len(s) > 1}
    rob_tokens = {m.group(1).lower() for r in img if r.get("source_dataset") == "roboflow_fetal_brain"
                  for m in [PATIENT.search(r.get("original_filename", "") or "")] if m}

    kept, excluded = [], []
    for r in anat:
        path = repo / r["image_path"]
        if not path.exists():
            excluded.append((r, "image file missing (re-run scripts/14_build_unified_dataset.py)"))
        elif rid_dup.get(r["record_id"]) in cross_dup:
            excluded.append((r, "pixel-duplicate of a Roboflow image"))
        elif (m := PATIENT.search(Path(r["image_path"]).name)) and m.group(1).lower() in rob_tokens:
            excluded.append((r, "Patient token also present in Roboflow"))
        elif not (set((r["anatomy_classes_present"] or "").split(";")) & set(ANATOMY)):
            excluded.append((r, "no anatomy class present in mask"))
        else:
            kept.append(r)
    return kept, excluded


def split_by_group(rows, val_frac=0.15, seed=42):
    """Deterministic group split. Group = case_group_id (patient token for Trans-*;
    singleton for Diverse, whose subjects are unknown -> val is optimistic there)."""
    def h(g):
        return int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), 16) % 10_000 / 10_000
    tr = [r for r in rows if h(r["case_group_id"] or r["record_id"]) >= val_frac]
    va = [r for r in rows if h(r["case_group_id"] or r["record_id"]) < val_frac]
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True, help="datasets/final_build_<timestamp>")
    ap.add_argument("--repo-root", default=str(REPO), help="base for the relative image_path column")
    ap.add_argument("--out", default="./anatomy_pretrain")
    ap.add_argument("--img-size", type=int, default=299)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze-from", default="block5", help="Xception block to unfreeze from")
    ap.add_argument("--no-grayscale", action="store_true")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="use only N images (smoke test)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept, excluded = prepare(Path(a.build_dir), Path(a.repo_root))
    if a.limit:
        kept = kept[::max(1, len(kept) // a.limit)][:a.limit]
    tr, va = split_by_group(kept)
    with open(out / "pretrain_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "image_path", "source_dataset", "anatomy_classes_present", "split"])
        for split, rows in (("train", tr), ("val", va)):
            for r in rows:
                w.writerow([r["record_id"], r["image_path"], r["source_dataset"],
                            r["anatomy_classes_present"], split])
    ex = Counter(reason for _, reason in excluded)
    summary = {"kept": len(kept), "train": len(tr), "val": len(va), "excluded": dict(ex),
               "label_prevalence": {c: sum(c in r["anatomy_classes_present"].split(";") for r in kept)
                                    for c in ANATOMY}}
    (out / "pretrain_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if a.prepare_only:
        return

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    spec = importlib.util.spec_from_file_location("train_mod", HERE / "03_train.py")
    train_mod = importlib.util.module_from_spec(spec)
    sys.modules["train_mod"] = train_mod
    spec.loader.exec_module(train_mod)

    if a.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    keras.utils.set_random_seed(42)

    repo = Path(a.repo_root)
    size = a.img_size

    def make_ds(rows, train):
        paths = [str(repo / r["image_path"]) for r in rows]
        y = np.array([[float(c in r["anatomy_classes_present"].split(";")) for c in ANATOMY]
                      for r in rows], np.float32)

        def load(p, t):
            img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
            img = tf.image.resize(img, (size, size))
            if train:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, 20.0)
                img = tf.clip_by_value(img, 0.0, 255.0)
            return img, t
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
        if train:
            ds = ds.shuffle(len(paths), seed=42)
        return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch_size).prefetch(2)

    train_ds, val_ds = make_ds(tr, True), make_ds(va, False)

    # reuse the exact diagnosis-model backbone/preprocessing so weights transfer 1:1
    train_mod.CLASS_NAMES, train_mod.NUM_CLASSES = ANATOMY, len(ANATOMY)
    model, base = train_mod.build_xception(size, dropout=0.3, gray=not a.no_grayscale)
    feat = model.get_layer("gradcam_target").output
    y_ = layers.GlobalAveragePooling2D()(feat)
    y_ = layers.Dropout(0.3)(y_)
    y_ = layers.Dense(len(ANATOMY), activation="sigmoid", dtype="float32", name="anatomy")(y_)
    net = keras.Model(model.input, y_, name="xception_anatomy")

    def compile_(lr):
        net.compile(keras.optimizers.Adam(lr), loss="binary_crossentropy",
                    metrics=[keras.metrics.AUC(name="auc", multi_label=True),
                             keras.metrics.BinaryAccuracy(name="acc")])

    cbs = lambda: [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                 restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2)]
    train_mod.set_backbone_trainable(base, None)
    compile_(1e-3)
    h1 = net.fit(train_ds, validation_data=val_ds, epochs=a.head_epochs, callbacks=cbs(), verbose=2)
    n = train_mod.set_backbone_trainable(base, a.unfreeze_from, False)
    print(f"Unfroze {n} backbone layers from '{a.unfreeze_from}' (BN frozen)")
    compile_(a.ft_lr)
    h2 = net.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs(), verbose=2)

    res = net.evaluate(val_ds, return_dict=True, verbose=0)
    wpath = out / "xception_anatomy_backbone.weights.h5"
    base.save_weights(wpath)
    (out / "pretrain_result.json").write_text(json.dumps({
        "val_metrics": {k: float(v) for k, v in res.items()}, "backbone_weights": wpath.name,
        "epochs_run": [len(h1.history["loss"]), len(h2.history["loss"])],
        "note": "pretext task = anatomy presence (Brain/CSP/LV) from Fetal-Head masks; "
                "no diagnosis labels were used"}, indent=2))
    print("val:", res, "\nsaved", wpath)


if __name__ == "__main__":
    main()
