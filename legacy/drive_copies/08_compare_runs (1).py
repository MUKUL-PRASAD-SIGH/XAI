"""
08_compare_runs.py
==================
Harvests every model variant you have trained (models/, models_reg/,
models_clahe/, models_pat4/, ...), evaluates them all on the SAME test set, and
reports whether any difference between them is statistically real.

Why the significance test matters: with 75 test images, two models must differ
by a net 8-10 correctly classified images -- about 11-13 accuracy points --
before the difference clears McNemar's exact test at alpha = 0.05. Picking the
best of several variants on a test set this small is fitting the test set, not
model selection.

Usage
-----
    python 08_compare_runs.py --data ./dataset_rebuilt --glob "models*" \
        --model xception --reference models

    # include every architecture
    python 08_compare_runs.py --data ./dataset_rebuilt --glob "models*" --model all
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)

CLASS_NAMES = [
    "anold-chiari-malformation", "arachnoid-cyst", "cerebellah-hypoplasia",
    "colphocephaly", "encephalocele", "holoprosencephaly", "hydracenphaly",
    "intracranial-hemorrdge", "intracranial-tumor", "m-magna",
    "mild-ventriculomegaly", "moderate-ventriculomegaly", "normal",
    "polencephaly", "severe-ventriculomegaly", "vein-of-galen",
]
AUTOTUNE = tf.data.AUTOTUNE
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp"}


def load_class_names(d):
    for p in (Path(d) / "class_names.json", Path(d).parent / "class_names.json"):
        if p.exists():
            return json.loads(p.read_text())
    return None


def make_test(data_dir: Path, size: int, batch: int):
    d = data_dir / "test"
    if not d.is_dir() or not any(p.suffix.lower() in IMG_SUFFIX for p in d.rglob("*")):
        raise SystemExit(f"{d} has no images. Rebuild with 02_rebuild_dataset.py")
    for c in CLASS_NAMES:
        (d / c).mkdir(parents=True, exist_ok=True)
    return keras.utils.image_dataset_from_directory(
        d, labels="inferred", label_mode="categorical", class_names=CLASS_NAMES,
        image_size=(size, size), batch_size=batch, shuffle=False).prefetch(AUTOTUNE)


def split_acc(model, data_dir, split, size, batch):
    d = data_dir / split
    if not d.is_dir() or not any(p.suffix.lower() in IMG_SUFFIX for p in d.rglob("*")):
        return None
    for c in CLASS_NAMES:
        (d / c).mkdir(parents=True, exist_ok=True)
    ds = keras.utils.image_dataset_from_directory(
        d, labels="inferred", label_mode="categorical", class_names=CLASS_NAMES,
        image_size=(size, size), batch_size=batch, shuffle=False).prefetch(AUTOTUNE)
    y = np.concatenate([b.numpy() for _, b in ds]).argmax(1)
    return accuracy_score(y, model.predict(ds, verbose=0).argmax(1))


def mcnemar(a_ok, b_ok):
    """Exact two-sided McNemar. a_ok/b_ok are boolean correctness vectors."""
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), b, c


def describe(cfg):
    if not cfg:
        return "-"
    bits = []
    for k, short in (("img_size", "img"), ("clahe", "clahe"), ("hflip", "hflip")):
        if k in cfg:
            v = cfg[k]
            if isinstance(v, bool):
                if v:
                    bits.append(short)
            else:
                bits.append(f"{short}={v}")
    return ", ".join(bits) or "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--glob", default="models*", help="directory glob for variants")
    ap.add_argument("--model", choices=["cnn", "scnn", "xception", "all"],
                    default="xception")
    ap.add_argument("--reference", default="models",
                    help="directory name treated as the baseline for the paired test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="./results", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    global CLASS_NAMES
    names = load_class_names(a.data)
    if names:
        CLASS_NAMES = names
        print(f"class names from dataset: {len(names)} classes")

    keys = ["cnn", "scnn", "xception"] if a.model == "all" else [a.model]
    dirs = sorted(p for p in Path(".").glob(a.glob) if p.is_dir())
    if not dirs:
        raise SystemExit(f"No directories match {a.glob}")
    print(f"Scanning: {', '.join(d.name for d in dirs)}\n")

    rows, correct = [], {}
    y_ref = None
    for d in dirs:
        for k in keys:
            ck = d / f"best_model_{k}.keras"
            if not ck.exists():
                continue
            cfg_p = ck.with_suffix(".json")
            cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
            model = keras.models.load_model(ck)
            size = cfg.get("img_size") or model.input_shape[1]
            ds = make_test(a.data, size, a.batch_size)
            y = np.concatenate([b.numpy() for _, b in ds]).argmax(1)
            y_ref = y if y_ref is None else y_ref
            yp = model.predict(ds, verbose=0).argmax(1)
            present = sorted(set(y.tolist()))
            tag = f"{d.name}/{k}"
            correct[tag] = (y == yp)
            tr = split_acc(model, a.data, "train", size, a.batch_size)
            rows.append({
                "run": tag, "config": describe(cfg),
                "accuracy": accuracy_score(y, yp),
                "balanced_accuracy": balanced_accuracy_score(y, yp),
                "macro_f1": f1_score(y, yp, labels=present, average="macro",
                                     zero_division=0),
                "train_accuracy": tr,
                "train_test_gap": (tr - accuracy_score(y, yp)) if tr is not None else None,
                "n_correct": int((y == yp).sum()), "n": len(y),
            })
            print(f"  {tag:<28} acc={rows[-1]['accuracy']:.3f} "
                  f"macroF1={rows[-1]['macro_f1']:.3f} "
                  f"({rows[-1]['n_correct']}/{rows[-1]['n']})")

    if not rows:
        raise SystemExit("No checkpoints found")

    # paired significance against the reference run
    ref = next((r["run"] for r in rows if r["run"].split("/")[0] == a.reference), None)
    if ref:
        print(f"\nPaired comparison against {ref} (McNemar exact):")
        for r in rows:
            if r["run"] == ref:
                r["vs_ref"] = "reference"
                continue
            p, b, c = mcnemar(correct[ref], correct[r["run"]])
            net = c - b
            r["vs_ref"] = (f"{net:+d} images, p={p:.3f}"
                           + ("" if p < 0.05 else "  (not significant)"))
            print(f"  {r['run']:<28} {r['vs_ref']}")

    md = ["| Run | Config | Correct | Accuracy | Balanced acc | Macro-F1 | "
          "Train acc | Train-test gap | vs reference |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: -x["macro_f1"]):
        g = f"{r['train_test_gap']:+.3f}" if r["train_test_gap"] is not None else "-"
        t = f"{r['train_accuracy']:.3f}" if r["train_accuracy"] is not None else "-"
        md.append(f"| {r['run']} | {r['config']} | {r['n_correct']}/{r['n']} | "
                  f"{r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | "
                  f"{r['macro_f1']:.3f} | {t} | {g} | {r.get('vs_ref','-')} |")

    n = rows[0]["n"]
    note = (f"\n\nNet gain needed for significance on n={n} (McNemar exact, "
            f"alpha=0.05): about 8-10 images, i.e. 11-13 accuracy points. "
            f"Differences smaller than that are not distinguishable, and "
            f"selecting the best variant on this test set is fitting the test set.\n")
    (a.out / "run_comparison.md").write_text("\n".join(md) + note)
    (a.out / "run_comparison.json").write_text(json.dumps(rows, indent=2))
    print("\n" + "\n".join(md) + note)
    print(f"Wrote {a.out}/run_comparison.md and .json")


if __name__ == "__main__":
    main()
