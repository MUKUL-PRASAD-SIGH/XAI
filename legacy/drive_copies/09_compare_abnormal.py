"""
09_compare_abnormal.py
======================
EXPERIMENT 3 analysis. Answers one question with a controlled comparison:

    Does removing the normal class from training improve discrimination
    between abnormalities, or is the ceiling in the data?

Both models are scored on the IDENTICAL 33 abnormal test images:

  A) the 16-class model, with its argmax restricted to the abnormal classes
     (it was trained with normal, so its normal logit is discarded here)
  B) the abnormality-only model, trained without normal at all

Same images, same cases, same protocol. The only difference is whether normal
was present during training, so any gap is attributable to that.

Because both models score the same items, the difference is tested with
McNemar's exact test rather than compared by eye. On 33 images the detection
threshold is coarse -- read the per-class recall table alongside the headline.

Usage
-----
    python 09_compare_abnormal.py \
        --data-abnormal ./dataset_abnormal \
        --model-16 ./models/best_model_xception.keras \
        --model-abnormal ./models_abnormal/best_model_xception.keras \
        --out ./results_abnormal
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

CLASS_NAMES_16 = [
    "anold-chiari-malformation", "arachnoid-cyst", "cerebellah-hypoplasia",
    "colphocephaly", "encephalocele", "holoprosencephaly", "hydracenphaly",
    "intracranial-hemorrdge", "intracranial-tumor", "m-magna",
    "mild-ventriculomegaly", "moderate-ventriculomegaly", "normal",
    "polencephaly", "severe-ventriculomegaly", "vein-of-galen",
]
AUTOTUNE = tf.data.AUTOTUNE
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp"}


def load_test(data_dir: Path, names, size, batch):
    d = data_dir / "test"
    if not d.is_dir() or not any(p.suffix.lower() in IMG_SUFFIX for p in d.rglob("*")):
        raise SystemExit(f"{d} has no images. Build it with 07_build_grouped.py "
                         f"--like ./dataset_rebuilt --drop-classes normal")
    for c in names:
        (d / c).mkdir(parents=True, exist_ok=True)
    return keras.utils.image_dataset_from_directory(
        d, labels="inferred", label_mode="categorical", class_names=names,
        image_size=(size, size), batch_size=batch, shuffle=False).prefetch(AUTOTUNE)


def img_size_of(model, ckpt):
    p = ckpt.with_suffix(".json")
    if p.exists():
        cfg = json.loads(p.read_text())
        if cfg.get("img_size"):
            return cfg["img_size"]
    return model.input_shape[1]


def mcnemar(a_ok, b_ok):
    b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok)); n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), b, c


def chance_macro_f1(y, k, seed=0, reps=4000):
    rng = np.random.default_rng(seed)
    prior = np.bincount(y, minlength=k) / len(y)
    return float(np.mean([f1_score(y, rng.choice(k, len(y), p=prior),
                                   average="macro", zero_division=0)
                          for _ in range(reps)]))


def plot_cm(y, yp, names, title, path):
    cm = confusion_matrix(y, yp, labels=range(len(names)))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-abnormal", required=True, type=Path)
    ap.add_argument("--model-16", required=True, type=Path)
    ap.add_argument("--model-abnormal", required=True, type=Path)
    ap.add_argument("--out", default="./results_abnormal", type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    names = json.loads((a.data_abnormal / "class_names.json").read_text())
    K = len(names)
    print(f"Abnormal label space: {K} classes\n")

    # ---- B: the abnormality-only model ------------------------------------
    m_ab = keras.models.load_model(a.model_abnormal)
    ds_ab = load_test(a.data_abnormal, names, img_size_of(m_ab, a.model_abnormal),
                      a.batch_size)
    y = np.concatenate([b.numpy() for _, b in ds_ab]).argmax(1)
    yp_ab = m_ab.predict(ds_ab, verbose=0).argmax(1)

    # ---- A: the 16-class model, argmax restricted to abnormal classes ------
    m16 = keras.models.load_model(a.model_16)
    ds16 = load_test(a.data_abnormal, names, img_size_of(m16, a.model_16),
                     a.batch_size)
    p16 = m16.predict(ds16, verbose=0)
    keep16 = [CLASS_NAMES_16.index(n) for n in names if n in CLASS_NAMES_16]
    sub = p16[:, keep16]
    yp_16 = sub.argmax(1)  # already indexed in the abnormal label space

    present = sorted(set(y.tolist()))
    maj = max(np.bincount(y, minlength=K)) / len(y)
    chance = chance_macro_f1(y, K)

    def score(yp):
        p, r, f, _ = precision_recall_fscore_support(y, yp, labels=present,
                                                     average="macro", zero_division=0)
        return dict(accuracy=accuracy_score(y, yp),
                    balanced_accuracy=balanced_accuracy_score(y, yp),
                    precision=p, recall=r, macro_f1=f,
                    n_correct=int((y == yp).sum()), n=len(y))

    s16, sab = score(yp_16), score(yp_ab)
    p, b, c = mcnemar(y == yp_16, y == yp_ab)

    L = []
    W = lambda t="": (print(t), L.append(t))
    W("=" * 72)
    W("EXPERIMENT 3: does removing 'normal' improve abnormality discrimination?")
    W("=" * 72)
    W(f"\nBoth models scored on the same {len(y)} abnormal test images, "
      f"{len(present)} classes present.")
    W(f"Majority baseline (largest class): {maj:.3f}")
    W(f"Chance macro-F1 (prior-weighted) : {chance:.3f}\n")
    W(f"{'Metric':<22}{'16-class (restricted)':>24}{'Abnormal-only':>16}{'Delta':>10}")
    W("-" * 72)
    for k, lab in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                   ("precision", "Macro precision"), ("recall", "Macro recall"),
                   ("macro_f1", "Macro-F1")]:
        W(f"{lab:<22}{s16[k]:>24.3f}{sab[k]:>16.3f}{sab[k]-s16[k]:>+10.3f}")
    W(f"{'Correct':<22}{str(s16['n_correct'])+'/'+str(s16['n']):>24}"
      f"{str(sab['n_correct'])+'/'+str(sab['n']):>16}"
      f"{sab['n_correct']-s16['n_correct']:>+10d}")
    W(f"\nMcNemar exact: {c} images fixed, {b} broken, net "
      f"{c-b:+d}, p = {p:.3f}"
      f"{'  -> NOT significant' if p >= 0.05 else '  -> significant'}")

    W("\nPer-class recall (the metric that shows which conditions are found at all)")
    W(f"{'class':<28}{'support':>8}{'16-class':>10}{'abn-only':>10}{'delta':>8}")
    W("-" * 64)
    for i in present:
        m = y == i
        r16 = float((yp_16[m] == i).mean())
        rab = float((yp_ab[m] == i).mean())
        W(f"{names[i]:<28}{int(m.sum()):>8}{r16:>10.3f}{rab:>10.3f}{rab-r16:>+8.3f}")

    never16 = [names[i] for i in present if (yp_16 == i).sum() == 0]
    neverab = [names[i] for i in present if (yp_ab == i).sum() == 0]
    W(f"\nNever predicted, 16-class    : {', '.join(never16) or 'none'}")
    W(f"Never predicted, abnormal-only: {', '.join(neverab) or 'none'}")

    W("\nCONCLUSIONS")
    if p < 0.05 and sab["macro_f1"] > s16["macro_f1"]:
        W("  Removing normal measurably improved abnormality discrimination.")
        W("  The normal class was absorbing capacity during training.")
    elif sab["macro_f1"] > chance * 1.5:
        W("  The abnormal-only model clears chance but the paired difference is")
        W("  not significant on 33 images. Cross-validation is needed to")
        W("  distinguish a real gain from noise.")
    else:
        W("  Removing normal did not lift abnormality discrimination above chance.")
        W("  The difficulty is fine-grained abnormal diagnosis, not the presence")
        W("  of the normal class. The ceiling is in the data: roughly 60 abnormal")
        W("  cases spread over 15 conditions.")

    plot_cm(y, yp_16, names, "16-class model, restricted to abnormal classes",
            a.out / "confusion_16class_restricted.png")
    plot_cm(y, yp_ab, names, "Abnormality-only model",
            a.out / "confusion_abnormal_only.png")
    (a.out / "per_class_abnormal_only.txt").write_text(
        classification_report(y, yp_ab, labels=present,
                              target_names=[names[i] for i in present],
                              zero_division=0, digits=3))
    (a.out / "comparison_vs_16class.txt").write_text("\n".join(L))
    (a.out / "comparison_vs_16class.json").write_text(json.dumps({
        "n": len(y), "classes_present": len(present),
        "majority_baseline": maj, "chance_macro_f1": chance,
        "sixteen_class_restricted": s16, "abnormal_only": sab,
        "mcnemar": {"p": p, "fixed": c, "broken": b, "net": c - b},
    }, indent=2))
    print(f"\nWrote {a.out}/ (comparison, per-class report, 2 confusion matrices)")


if __name__ == "__main__":
    main()
