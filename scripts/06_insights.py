"""
06_insights.py
==============
Runs on checkpoints you ALREADY have. No retraining. Answers the questions that
should be settled BEFORE changing the pipeline:

  1. Is the reported accuracy distinguishable from a trivial baseline?
  2. Is the model overfitting? (train vs valid vs test gap)
  3. Does preprocessing at inference match training? (quantifies the /255 bug)
  4. How do accuracy, balanced accuracy and macro-F1 compare, with CIs?
  5. Which class pairs dominate the errors, and which classes are never predicted?
  6. How much of the score is the normal-vs-abnormal (source-confounded) boundary?
  7. Can trivial pixel statistics separate normal from abnormal without any CNN?
     If yes, the normal class is a source artefact, not a diagnostic finding.

Usage
-----
    python 06_insights.py --data ./dataset_rebuilt --models ./models
    python 06_insights.py --data ./dataset_rebuilt --models ./models --model xception
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, precision_recall_fscore_support)

CLASS_NAMES = [
    "anold-chiari-malformation", "arachnoid-cyst", "cerebellah-hypoplasia",
    "colphocephaly", "encephalocele", "holoprosencephaly", "hydracenphaly",
    "intracranial-hemorrdge", "intracranial-tumor", "m-magna",
    "mild-ventriculomegaly", "moderate-ventriculomegaly", "normal",
    "polencephaly", "severe-ventriculomegaly", "vein-of-galen",
]
PRETTY = {"cnn": "CNN", "scnn": "S-CNN", "xception": "Xception"}
AUTOTUNE = tf.data.AUTOTUNE
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp"}


def load_class_names(data_dir):
    for d in (Path(data_dir), Path(data_dir).parent):
        p = d / "class_names.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def make_ds(data_dir: Path, split: str, size: int, batch: int):
    d = data_dir / split
    if not d.is_dir() or not any(p.suffix.lower() in IMG_SUFFIX for p in d.rglob("*")):
        return None
    for c in CLASS_NAMES:
        (d / c).mkdir(parents=True, exist_ok=True)
    return keras.utils.image_dataset_from_directory(
        d, labels="inferred", label_mode="categorical", class_names=CLASS_NAMES,
        image_size=(size, size), batch_size=batch, shuffle=False).prefetch(AUTOTUNE)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - m), min(1.0, c + m)


def binom_tail(k, n, p0):
    return sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))


def bootstrap_ci(y, yp, fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(fn(y[idx], yp[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (0, 0)


def pixel_features(path):
    g = np.asarray(Image.open(path).convert("L").resize((128, 128)), np.float32)
    return [g.mean(), g.std(), (g < 8).mean(), (g > 200).mean(),
            np.abs(np.diff(g, axis=0)).mean()]


def section(t, L):
    for s in ("", "=" * 70, t, "=" * 70):
        print(s); L.append(s)


def line(s, L):
    print(s); L.append(s)


def analyse(key, ckpt, data_dir, batch, L):
    cfg_p = ckpt.with_suffix(".json")
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    model = keras.models.load_model(ckpt)
    size = cfg.get("img_size") or model.input_shape[1]

    test = make_ds(data_dir, "test", size, batch)
    if test is None:
        raise SystemExit(f"{data_dir}/test has no images. Rebuild with 02_rebuild_dataset.py")
    y = np.concatenate([b.numpy() for _, b in test]).argmax(1)
    probs = model.predict(test, verbose=0)
    yp = probs.argmax(1)
    n = len(y)
    present = sorted(set(y.tolist()))

    section(f"{PRETTY.get(key, key)}  —  n = {n} test images", L)

    # -- 1. baseline and significance ---------------------------------------
    line("\n1. IS THE ACCURACY DISTINGUISHABLE FROM A TRIVIAL BASELINE?", L)
    acc = accuracy_score(y, yp)
    counts = Counter(y.tolist())
    maj_cls, maj_n = counts.most_common(1)[0]
    maj = maj_n / n
    lo, hi = wilson(int(round(acc * n)), n)
    p = binom_tail(int(round(acc * n)), n, maj)
    line(f"   accuracy                {acc:.3f}", L)
    line(f"   majority baseline       {maj:.3f}  (always predict "
         f"'{CLASS_NAMES[maj_cls]}', {maj_n}/{n})", L)
    line(f"   95% CI on accuracy      [{lo:.3f}, {hi:.3f}]  "
         f"(width {100*(hi-lo):.1f} points)", L)
    line(f"   exact binomial vs baseline  p = {p:.3f}"
         f"{'   -> NOT better than baseline' if p > 0.05 else ''}", L)

    # -- 2. metric comparison ------------------------------------------------
    line("\n2. ACCURACY vs BALANCED ACCURACY vs MACRO-F1", L)
    bal = balanced_accuracy_score(y, yp)
    mf1 = f1_score(y, yp, labels=present, average="macro", zero_division=0)
    wf1 = f1_score(y, yp, labels=present, average="weighted", zero_division=0)
    f_lo, f_hi = bootstrap_ci(y, yp,
                              lambda a, b: f1_score(a, b, average="macro", zero_division=0))
    line(f"   accuracy           {acc:.3f}", L)
    line(f"   balanced accuracy  {bal:.3f}   (mean per-class recall)", L)
    line(f"   macro F1           {mf1:.3f}   95% bootstrap CI [{f_lo:.3f}, {f_hi:.3f}]", L)
    line(f"   weighted F1        {wf1:.3f}", L)
    rng = np.random.default_rng(0)
    prior = np.array([counts.get(i, 0) for i in range(len(CLASS_NAMES))]) / n
    null = np.array([f1_score(y, rng.choice(len(CLASS_NAMES), n, p=prior),
                              average="macro", zero_division=0) for _ in range(2000)])
    line(f"   chance macro-F1    {null.mean():.3f}   P(chance >= observed) = "
         f"{(null >= mf1).mean():.3f}", L)

    # -- 3. overfitting ------------------------------------------------------
    line("\n3. OVERFITTING CHECK", L)
    for sp in ("train", "valid", "test"):
        ds = make_ds(data_dir, sp, size, batch)
        if ds is None:
            line(f"   {sp:<6} (absent)", L); continue
        yy = np.concatenate([b.numpy() for _, b in ds]).argmax(1)
        pp = model.predict(ds, verbose=0).argmax(1)
        line(f"   {sp:<6} accuracy {accuracy_score(yy, pp):.3f}   "
             f"macro-F1 {f1_score(yy, pp, average='macro', zero_division=0):.3f}   "
             f"n={len(yy)}", L)
    hist = sorted(ckpt.parent.glob(f"{ckpt.stem}*history.csv"))
    if hist:
        line(f"   training history: {', '.join(h.name for h in hist)}", L)

    # -- 4. preprocessing consistency ---------------------------------------
    line("\n4. PREPROCESSING CONSISTENCY (cost of the /255 mismatch)", L)
    has_rescale = any("rescal" in l.__class__.__name__.lower() for l in model.layers)
    line(f"   Rescaling layer inside the model: {has_rescale}", L)
    wrong = test.map(lambda x, t: (x / 255.0, t), num_parallel_calls=AUTOTUNE)
    acc_wrong = accuracy_score(y, model.predict(wrong, verbose=0).argmax(1))
    line(f"   accuracy with correct 0-255 input : {acc:.3f}", L)
    line(f"   accuracy if the app divides by 255: {acc_wrong:.3f}"
         f"   -> costs {100*(acc-acc_wrong):+.1f} points", L)

    # -- 5. where the errors are --------------------------------------------
    line("\n5. ERROR STRUCTURE", L)
    cm = confusion_matrix(y, yp, labels=range(len(CLASS_NAMES)))
    pairs = [(cm[i, j], i, j) for i in present for j in range(len(CLASS_NAMES)) if i != j and cm[i, j]]
    for c, i, j in sorted(pairs, reverse=True)[:8]:
        line(f"   {c:>3}x  {CLASS_NAMES[i]:<28} -> {CLASS_NAMES[j]}", L)
    never = [CLASS_NAMES[i] for i in present if (yp == i).sum() == 0]
    if never:
        line(f"   never predicted at all: {', '.join(never)}", L)
    top = Counter(yp.tolist()).most_common(1)[0]
    line(f"   most frequent prediction: {CLASS_NAMES[top[0]]} "
         f"({top[1]}/{n} = {top[1]/n:.0%} of all predictions)", L)

    # -- 6. how much is just normal vs abnormal -----------------------------
    if "normal" in CLASS_NAMES:
        NORMAL = CLASS_NAMES.index("normal")
        line("\n6. HOW MUCH OF THE SCORE IS normal-vs-abnormal?", L)
        yb, ypb = (y != NORMAL).astype(int), (yp != NORMAL).astype(int)
        line(f"   binary normal/abnormal accuracy {accuracy_score(yb, ypb):.3f}", L)
        m = y != NORMAL
        if m.sum():
            ab = [i for i in range(len(CLASS_NAMES)) if i != NORMAL]
            yps = np.array(ab)[probs[m][:, ab].argmax(1)]
            line(f"   abnormality-only accuracy       {accuracy_score(y[m], yps):.3f}   "
                 f"macro-F1 {f1_score(y[m], yps, average='macro', zero_division=0):.3f}  "
                 f"(n={int(m.sum())})", L)
            line("   -> the gap between these two lines is the confound.", L)
    return model


def shortcut_test(data_dir: Path, L):
    """Can 5 trivial pixel statistics separate normal from abnormal?"""
    if "normal" not in CLASS_NAMES:
        return
    section("SOURCE-SHORTCUT TEST (no CNN, 5 pixel statistics only)", L)
    data = {}
    for sp in ("train", "test"):
        X, y = [], []
        for c in CLASS_NAMES:
            d = data_dir / sp / c
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.suffix.lower() in IMG_SUFFIX:
                    X.append(pixel_features(p)); y.append(int(c == "normal"))
        data[sp] = (np.array(X), np.array(y))
    (Xtr, ytr), (Xte, yte) = data["train"], data["test"]
    if len(set(ytr.tolist())) < 2 or len(Xte) == 0:
        line("   not enough data for the test", L); return
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    base = max(np.bincount(yte, minlength=2)) / len(yte)
    line(f"   features: mean, std, dark fraction, bright fraction, row gradient", L)
    line(f"   trained on {len(Xtr)} train images, tested on {len(Xte)} "
         f"case-disjoint test images", L)
    line(f"   normal-vs-abnormal from pixel statistics alone : {acc:.3f}", L)
    line(f"   majority baseline for the same binary task     : {base:.3f}", L)
    line(f"   lift over baseline                             : {acc-base:+.3f}", L)
    if acc >= 0.90:
        line("   -> 5 crude numbers nearly solve it. A high normal F1 is not", L)
        line("      evidence of anatomical learning.", L)
    elif acc - base > 0.10:
        line("   -> A source signal is measurably present in raw pixel statistics,", L)
        line("      so the normal/abnormal boundary is partly an acquisition", L)
        line("      artefact. This does NOT by itself prove the CNN relies on it --", L)
        line("      state it as a confound risk, not a proven mechanism.", L)
    else:
        line("   -> No meaningful source signal in crude pixel statistics.", L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--models", default="./models", type=Path)
    ap.add_argument("--model", choices=["cnn", "scnn", "xception", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="./results", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    global CLASS_NAMES
    names = load_class_names(a.data)
    if names:
        CLASS_NAMES = names
        print(f"class names from dataset: {len(names)} classes")

    L, ran = [], 0
    for k in (["cnn", "scnn", "xception"] if a.model == "all" else [a.model]):
        ck = a.models / f"best_model_{k}.keras"
        if not ck.exists():
            print(f"skip {k} (no {ck})")
            continue
        analyse(k, ck, a.data, a.batch_size, L)
        ran += 1
    if not ran:
        raise SystemExit(f"No checkpoints in {a.models}")

    shortcut_test(a.data, L)
    (a.out / "insights.txt").write_text("\n".join(L))
    print(f"\nWrote {a.out}/insights.txt")


if __name__ == "__main__":
    main()

