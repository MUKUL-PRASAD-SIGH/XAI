"""
05_diagnostics.py
=================
Runs on the checkpoints you ALREADY have. No retraining. Answers the three
questions the per-class report raises:

  1. How much of the accuracy is just "normal vs abnormal"?
     Every normal image comes from FETAL_PLANES_DB and every abnormal image
     does not, so this boundary is confounded with source dataset. If binary
     performance is near-perfect while multi-class is near-chance, the model is
     detecting acquisition, not anatomy.

  2. Given that an image IS abnormal, can the model say which abnormality?
     Restricts the argmax to abnormal classes and scores only abnormal test
     images. This subtask is confound-free: all abnormal images share a source.
     THIS is the number that reflects real diagnostic ability.

  3. Does merging the ventriculomegaly severities help, and is it justified?
     Reports metrics with mild/moderate/severe collapsed into one class, plus
     the 3x3 sub-confusion matrix so the merge is evidence-driven, not assumed.

Usage
-----
    python 05_diagnostics.py --data ./dataset_rebuilt --models ./models
    python 05_diagnostics.py --data ./dataset_rebuilt --models ./models --model xception
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

CLASS_NAMES = [
    "anold-chiari-malformation", "arachnoid-cyst", "cerebellah-hypoplasia",
    "colphocephaly", "encephalocele", "holoprosencephaly", "hydracenphaly",
    "intracranial-hemorrdge", "intracranial-tumor", "m-magna",
    "mild-ventriculomegaly", "moderate-ventriculomegaly", "normal",
    "polencephaly", "severe-ventriculomegaly", "vein-of-galen",
]
NORMAL = CLASS_NAMES.index("normal")
VENT = [CLASS_NAMES.index(c) for c in
        ("mild-ventriculomegaly", "moderate-ventriculomegaly", "severe-ventriculomegaly")]
PRETTY = {"cnn": "CNN", "scnn": "S-CNN", "xception": "Xception"}
AUTOTUNE = tf.data.AUTOTUNE


def get_predictions(ckpt: Path, data_dir: Path, batch: int):
    cfg_p = ckpt.with_suffix(".json")
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    model = keras.models.load_model(ckpt)
    size = cfg.get("img_size") or model.input_shape[1]
    for c in CLASS_NAMES:
        (data_dir / "test" / c).mkdir(parents=True, exist_ok=True)
    ds = keras.utils.image_dataset_from_directory(
        data_dir / "test", labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=(size, size),
        batch_size=batch, shuffle=False).prefetch(AUTOTUNE)
    y = np.concatenate([b.numpy() for _, b in ds]).argmax(1)
    return y, model.predict(ds, verbose=0)


def macro(y, yp, labels):
    p, r, f, _ = precision_recall_fscore_support(y, yp, labels=labels,
                                                 average="macro", zero_division=0)
    return accuracy_score(y, yp), p, r, f


def report(name, y, probs, lines):
    P = lambda s="": (print(s), lines.append(s))
    abnormal_idx = [i for i in range(len(CLASS_NAMES)) if i != NORMAL]
    yp = probs.argmax(1)

    P(f"\n{'='*66}\n{name}\n{'='*66}")

    # ---- 1. binary: is this the source shortcut? -------------------------
    yb = (y != NORMAL).astype(int)
    ypb = (yp != NORMAL).astype(int)
    acc = accuracy_score(yb, ypb)
    p, r, f, _ = precision_recall_fscore_support(yb, ypb, average="binary",
                                                 zero_division=0)
    P("\n1. NORMAL vs ABNORMAL (confounded with source dataset)")
    P(f"   accuracy {acc:.3f} | abnormal precision {p:.3f} recall {r:.3f} F1 {f:.3f}")
    P(f"   n = {len(yb)}  ({int((yb==0).sum())} normal, {int(yb.sum())} abnormal)")
    if acc > 0.90:
        P("   -> near-perfect on a boundary that is also a source boundary.")
        P("      Treat this as evidence of a shortcut, not of clinical skill.")

    # ---- 2. abnormal-only: the confound-free subtask ----------------------
    m = y != NORMAL
    P("\n2. WHICH ABNORMALITY (abnormal images only, argmax over abnormal classes)")
    if m.sum() == 0:
        P("   no abnormal images in the test split")
    else:
        sub = probs[m][:, abnormal_idx]
        yp_sub = np.array(abnormal_idx)[sub.argmax(1)]
        present = sorted(set(y[m].tolist()))
        a2, p2, r2, f2 = macro(y[m], yp_sub, present)
        chance = 1.0 / max(len(present), 1)
        P(f"   accuracy {a2:.3f}  (chance ~{chance:.3f} over {len(present)} classes)")
        P(f"   macro precision {p2:.3f} recall {r2:.3f} F1 {f2:.3f}   n = {int(m.sum())}")
        P("   -> this is the number that reflects real diagnostic ability.")

    # ---- 3. ventriculomegaly merge ---------------------------------------
    P("\n3. VENTRICULOMEGALY SEVERITIES")
    vent_present = [i for i in VENT if (y == i).any()]
    if len(vent_present) >= 2:
        names = [CLASS_NAMES[i].replace("-ventriculomegaly", "") for i in VENT]
        cm = confusion_matrix(y, yp, labels=VENT)
        # images predicted OUTSIDE the family would otherwise vanish from the table
        outside = np.array([int(((y == v) & ~np.isin(yp, VENT)).sum()) for v in VENT])
        P("   confusion among the three severities (rows = true, cols = predicted)")
        P("            " + "".join(f"{n:>10}" for n in names) + f"{'other':>10}")
        for i, n in enumerate(names):
            P(f"   {n:>8} " + "".join(f"{v:>10}" for v in cm[i]) + f"{outside[i]:>10}")
        within = cm.sum()
        total_vent = int(np.isin(y, VENT).sum())
        P(f"   {within}/{total_vent} ventriculomegaly images were predicted as SOME "
          f"ventriculomegaly class")
        if total_vent and within / total_vent >= 0.5:
            P("   -> most errors stay inside the severity family: merging is")
            P("      empirically justified, not just convenient.")
        else:
            P("   -> errors leave the severity family: merging will not fix this.")

        merged = np.where(np.isin(y, VENT), VENT[0], y)
        pm = probs.copy()
        pm[:, VENT[0]] = probs[:, VENT].sum(1)
        pm[:, VENT[1:]] = 0
        ypm = pm.argmax(1)
        present = sorted(set(merged.tolist()))
        am, pmm, rm, fm = macro(merged, ypm, present)
        P(f"   merged ({len(present)} classes): accuracy {am:.3f} macro-F1 {fm:.3f}")
    else:
        P("   fewer than two severities present in the test split")

    # ---- summary ----------------------------------------------------------
    present_all = sorted(set(y.tolist()))
    a0, p0, r0, f0 = macro(y, yp, present_all)
    P(f"\n   for reference, full multi-class: accuracy {a0:.3f} macro-F1 {f0:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--models", default="./models", type=Path)
    ap.add_argument("--model", choices=["cnn", "scnn", "xception", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="./results", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    keys = ["cnn", "scnn", "xception"] if a.model == "all" else [a.model]
    lines, ran = [], 0
    for k in keys:
        ck = a.models / f"best_model_{k}.keras"
        if not ck.exists():
            print(f"skip {k} (no {ck})")
            continue
        y, probs = get_predictions(ck, a.data, a.batch_size)
        report(PRETTY[k], y, probs, lines)
        ran += 1

    if not ran:
        raise SystemExit(f"No checkpoints found in {a.models}")
    (a.out / "diagnostics.txt").write_text("\n".join(lines))
    print(f"\nWrote {a.out}/diagnostics.txt")


if __name__ == "__main__":
    main()

