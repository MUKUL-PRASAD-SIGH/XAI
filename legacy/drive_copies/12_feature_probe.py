"""
12_feature_probe.py
===================
Trains a sparse linear classifier on frozen deep features instead of
fine-tuning the whole network. No GPU training: features are extracted once,
then scikit-learn does the rest in seconds.

Why this can beat fine-tuning here
----------------------------------
Fine-tuning Xception adapts ~20 million parameters to 63 training patients,
which is why the training accuracy reaches 0.95 while held-out accuracy sits
near 0.55. An L1-penalised logistic regression on 2048 frozen features fits a
few thousand parameters at most, and the L1 penalty drives most of them to
exactly zero. Fewer parameters, less to overfit.

It also answers the feature-selection requirement literally: L1 selects a
subset of the 2048 features and this script reports exactly how many survived,
so "which features were selected and why" has a concrete answer.

Pipeline: preprocess -> extract frozen features -> standardise ->
          L1 feature selection -> train -> validate (choose C) -> test

Usage
-----
    # features from your fine-tuned checkpoint
    python 12_feature_probe.py --data ./dataset_rebuilt --models ./models

    # features from plain ImageNet Xception (the cleaner baseline)
    python 12_feature_probe.py --data ./dataset_rebuilt --models ./models \
        --backbone imagenet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*penalty.*")
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

CLASS_NAMES = [
    "anold-chiari-malformation", "arachnoid-cyst", "cerebellah-hypoplasia",
    "colphocephaly", "encephalocele", "holoprosencephaly", "hydracenphaly",
    "intracranial-hemorrdge", "intracranial-tumor", "m-magna",
    "mild-ventriculomegaly", "moderate-ventriculomegaly", "normal",
    "polencephaly", "severe-ventriculomegaly", "vein-of-galen",
]
AUTOTUNE = tf.data.AUTOTUNE
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp"}


def l1_logreg(C, max_iter=4000):
    """L1 multinomial logistic. 'saga' is the solver that supports L1 together
    with multiclass across every recent scikit-learn version; 'liblinear' was
    dropped for multiclass in 1.8."""
    return LogisticRegression(penalty="l1", C=C, solver="saga",
                              class_weight="balanced", max_iter=max_iter,
                              tol=1e-3, n_jobs=-1)


def load_class_names(d):
    for p in (Path(d) / "class_names.json", Path(d).parent / "class_names.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


def make_ds(data_dir: Path, split: str, size: int, batch: int):
    d = data_dir / split
    if not d.is_dir() or not any(p.suffix.lower() in IMG_SUFFIX for p in d.rglob("*")):
        return None
    present = [c for c in CLASS_NAMES if (d / c).is_dir()]
    for c in CLASS_NAMES:
        (d / c).mkdir(parents=True, exist_ok=True)
    return keras.utils.image_dataset_from_directory(
        d, labels="inferred", label_mode="categorical", class_names=CLASS_NAMES,
        image_size=(size, size), batch_size=batch, shuffle=False).prefetch(AUTOTUNE)


def build_extractor(models_dir: Path, backbone: str, model_key: str):
    """Returns (feature_model, img_size). Output is the pooled feature vector."""
    if backbone == "imagenet":
        size = 299
        inp = keras.Input((size, size, 3))
        x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
        base = keras.applications.Xception(include_top=False, weights="imagenet",
                                           input_tensor=x, pooling="avg")
        return keras.Model(inp, base.output), size

    ck = models_dir / f"best_model_{model_key}.keras"
    if not ck.exists():
        raise SystemExit(f"no {ck}")
    cfg_p = ck.with_suffix(".json")
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    model = keras.models.load_model(ck)
    size = cfg.get("img_size") or model.input_shape[1]
    gap = [l for l in model.layers
           if l.__class__.__name__ == "GlobalAveragePooling2D"]
    if gap:
        feat = gap[-1].output
    else:
        feat = model.layers[-2].output
        print(f"  (no GAP layer; using {model.layers[-2].name})")
    return keras.Model(model.inputs, feat), size


def extract(fmodel, ds):
    X, y = [], []
    for xb, yb in ds:
        X.append(fmodel.predict(xb, verbose=0))
        y.append(yb.numpy().argmax(1))
    return np.concatenate(X), np.concatenate(y)


def metrics(y, yp, proba, keep, proba_classes=None):
    """proba_classes gives the label each probability COLUMN corresponds to.

    scikit-learn's predict_proba has one column per entry of clf.classes_, which
    is not the same as the original label values when some classes never appear
    in training. Keras output has one column per class index. Getting this wrong
    silently scrambles the ROC-AUC, so the mapping is explicit.
    """
    m = np.isin(y, keep)
    yk, ypk = y[m], yp[m]
    p, r, f, _ = precision_recall_fscore_support(yk, ypk, labels=keep,
                                                 average="macro", zero_division=0)
    auc = None
    if proba is not None and len(keep) > 1:
        try:
            if proba_classes is None:
                proba_classes = np.arange(proba.shape[1])
            pos = {int(c): i for i, c in enumerate(proba_classes)}
            if all(c in pos for c in keep):
                sub = proba[m][:, [pos[c] for c in keep]]
                sub = sub / np.maximum(sub.sum(1, keepdims=True), 1e-9)
                remap = {c: i for i, c in enumerate(keep)}
                auc = float(roc_auc_score(np.array([remap[v] for v in yk]), sub,
                                          multi_class="ovr", average="macro"))
        except Exception:
            auc = None
    return dict(accuracy=float(accuracy_score(yk, ypk)),
                balanced_accuracy=float(balanced_accuracy_score(yk, ypk)),
                precision=float(p), recall=float(r), macro_f1=float(f),
                roc_auc_ovr=auc, n=int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--models", default="./models", type=Path)
    ap.add_argument("--model", default="xception")
    ap.add_argument("--backbone", choices=["finetuned", "imagenet"],
                    default="finetuned")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="./results_probe", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    global CLASS_NAMES
    names = load_class_names(a.data)
    if names:
        CLASS_NAMES = names
        print(f"class names from dataset: {len(names)} classes")

    L = []
    W = lambda s="": (print(s), L.append(s))

    fmodel, size = build_extractor(a.models, a.backbone, a.model)
    W("=" * 72)
    W(f"L1 FEATURE PROBE -- backbone: {a.backbone}, input {size}x{size}")
    W("=" * 72)

    data = {}
    for sp in ("train", "valid", "test"):
        ds = make_ds(a.data, sp, size, a.batch_size)
        if ds is None:
            W(f"  {sp}: absent"); continue
        X, y = extract(fmodel, ds)
        data[sp] = (X, y)
        W(f"  {sp}: {X.shape[0]} images -> {X.shape[1]} features")
    if "train" not in data or "test" not in data:
        raise SystemExit("need at least train and test splits")

    Xtr, ytr = data["train"]
    Xte, yte = data["test"]
    Xva, yva = data.get("valid", (None, None))

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    Xva_s = scaler.transform(Xva) if Xva is not None else None

    keep = sorted(set(yte.tolist()))
    W(f"\n  {len(keep)} evaluable classes in the test split")

    # ---- choose the L1 strength on validation ----------------------------
    W("\n" + "-" * 72)
    W("FEATURE SELECTION -- sweeping the L1 strength on the validation split")
    W("-" * 72)
    W(f"  {'C':>10}{'features kept':>16}{'valid macro-F1':>18}")
    grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    best, best_score = None, -1.0
    for C in grid:
        clf = l1_logreg(C)
        clf.fit(Xtr_s, ytr)
        nz = int((np.abs(clf.coef_).sum(0) > 0).sum())
        if Xva_s is not None and len(set(yva.tolist())) > 1:
            sc = f1_score(yva, clf.predict(Xva_s), average="macro", zero_division=0)
        else:
            sc = f1_score(ytr, clf.predict(Xtr_s), average="macro", zero_division=0)
        W(f"  {C:>10}{nz:>16}{sc:>18.3f}")
        if sc > best_score:
            best, best_score, best_nz = C, sc, nz
    W(f"\n  chosen C = {best}  ({best_nz} of {Xtr.shape[1]} features retained, "
      f"{100*best_nz/Xtr.shape[1]:.1f}%)")

    # ---- fit the candidates ----------------------------------------------
    W("\n" + "-" * 72)
    W("TEST RESULTS")
    W("-" * 72)
    results = {}
    models = {
        f"L1 logistic (C={best})": l1_logreg(best),
        "L2 logistic (C=1)": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=3000),
        "linear SVM": LinearSVC(class_weight="balanced", max_iter=5000),
    }
    for name, clf in models.items():
        clf.fit(Xtr_s, ytr)
        yp = clf.predict(Xte_s)
        proba = clf.predict_proba(Xte_s) if hasattr(clf, "predict_proba") else None
        results[name] = metrics(yte, yp, proba, keep,
                                proba_classes=getattr(clf, "classes_", None))
        if name.startswith("L1"):
            best_clf, best_pred = clf, yp

    # ---- the fine-tuned CNN on the same test set -------------------------
    if a.backbone == "finetuned":
        ck = a.models / f"best_model_{a.model}.keras"
        cnn = keras.models.load_model(ck)
        ds = make_ds(a.data, "test", size, a.batch_size)
        p = cnn.predict(ds, verbose=0)
        results["fine-tuned CNN (for comparison)"] = metrics(
            yte, p.argmax(1), p, keep)

    hdr = f"  {'model':<34}{'acc':>7}{'bal acc':>9}{'macro-F1':>10}{'ROC-AUC':>9}"
    W(hdr); W("  " + "-" * (len(hdr) - 2))
    for name, s in results.items():
        auc = f"{s['roc_auc_ovr']:.3f}" if s["roc_auc_ovr"] is not None else "  -  "
        W(f"  {name:<34}{s['accuracy']:>7.3f}{s['balanced_accuracy']:>9.3f}"
          f"{s['macro_f1']:>10.3f}{auc:>9}")

    counts = np.bincount(yte[np.isin(yte, keep)], minlength=len(CLASS_NAMES))
    W(f"\n  majority-class baseline: {counts.max()/counts.sum():.3f} "
      f"({CLASS_NAMES[int(counts.argmax())]})")

    # ---- per-class for the L1 model --------------------------------------
    m = np.isin(yte, keep)
    rep = classification_report(yte[m], best_pred[m], labels=keep,
                                target_names=[CLASS_NAMES[i] for i in keep],
                                zero_division=0, digits=3)
    W("\n" + "-" * 72)
    W("PER CLASS -- L1 logistic")
    W("-" * 72)
    W(rep)

    # ---- which features were selected ------------------------------------
    coef = np.abs(best_clf.coef_)
    used = np.where(coef.sum(0) > 0)[0]
    strength = coef.sum(0)[used]
    order = used[np.argsort(-strength)]
    W("-" * 72)
    W("SELECTED FEATURES")
    W("-" * 72)
    W(f"  {len(used)} of {Xtr.shape[1]} deep features have a non-zero coefficient")
    W(f"  top 15 by total absolute weight: "
      f"{', '.join(str(int(i)) for i in order[:15])}")
    W("  These are channel indices of the backbone's pooled feature map, not")
    W("  hand-crafted features -- describe them as automatically selected deep")
    W("  descriptors, and cite the count and the L1 strength that produced it.")

    (a.out / f"probe_{a.backbone}.txt").write_text("\n".join(L))
    (a.out / f"probe_{a.backbone}.json").write_text(json.dumps({
        "backbone": a.backbone, "n_features": int(Xtr.shape[1]),
        "chosen_C": best, "features_retained": int(len(used)),
        "selected_feature_indices": [int(i) for i in sorted(used)],
        "results": results,
    }, indent=2))
    np.save(a.out / f"confusion_{a.backbone}.npy",
            confusion_matrix(yte[m], best_pred[m], labels=keep))
    print(f"\nWrote {a.out}/probe_{a.backbone}.txt and .json")


if __name__ == "__main__":
    main()
