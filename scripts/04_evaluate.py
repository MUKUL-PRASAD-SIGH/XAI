"""
04_evaluate.py
==============
Evaluates trained models and writes the tables the paper needs.

Two modes:
    python 04_evaluate.py --data ./dataset_rebuilt --models ./models
    python 04_evaluate.py --cv ./dataset_cv --models ./models     # pools all folds

In CV mode every case is tested exactly once across the folds, and predictions
are pooled before computing metrics. That is the number to report: a single
75-image test split cannot support per-class claims on this dataset.

Classes listed in the rebuild's `train_only_classes` are excluded from the
metrics automatically, because they have no held-out case and reporting a score
for them would be meaningless. They are named in the output so the exclusion is
explicit rather than hidden.

Reported numbers (one definition, so they cannot disagree between sections):
  * macro-averages run over the classes that have at least one test image
    (minus train-only classes), pooled over all folds;
  * the majority-class baseline gets REAL precision/recall/F1 computed from its
    constant prediction, not zeros;
  * 95 % CIs come from a CASE-level bootstrap (images of one case are resampled
    together), because images of one patient are not independent;
  * an exact one-sided binomial test says whether accuracy beats the baseline.
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

CLASS_NAMES = [
    "anold-chiari-malformation", "arachnoid-cyst", "cerebellah-hypoplasia",
    "colphocephaly", "encephalocele", "holoprosencephaly", "hydracenphaly",
    "intracranial-hemorrdge", "intracranial-tumor", "m-magna",
    "mild-ventriculomegaly", "moderate-ventriculomegaly", "normal",
    "polencephaly", "severe-ventriculomegaly", "vein-of-galen",
]
_spec = importlib.util.spec_from_file_location(
    "audit_mod", Path(__file__).with_name("01_dataset_audit.py"))
_audit = importlib.util.module_from_spec(_spec)
sys.modules["audit_mod"] = _audit
_spec.loader.exec_module(_audit)
parse_name = _audit.parse_name

PRETTY = {"cnn": "CNN", "scnn": "S-CNN", "xception": "Xception"}
AUTOTUNE = tf.data.AUTOTUNE


def load_class_names(root: Path):
    """class_names.json written by 02b / 07 wins over the built-in 16 classes."""
    for d in (root, root.parent):
        p = d / "class_names.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def load_test(data_dir: Path, img_size: int, batch: int):
    for c in CLASS_NAMES:
        (data_dir / "test" / c).mkdir(parents=True, exist_ok=True)
    ds = keras.utils.image_dataset_from_directory(
        data_dir / "test", labels="inferred", label_mode="categorical",
        class_names=CLASS_NAMES, image_size=(img_size, img_size),
        batch_size=batch, shuffle=False)
    files = list(ds.file_paths)
    return ds.prefetch(AUTOTUNE), files


def predict(model, ds, tta: bool):
    p = model.predict(ds, verbose=0)
    if tta:
        flip = ds.map(lambda x, y: (tf.image.flip_left_right(x), y), num_parallel_calls=AUTOTUNE)
        zoom = ds.map(lambda x, y: (tf.image.resize(tf.image.central_crop(x, 0.9),
                                                    tf.shape(x)[1:3]), y),
                      num_parallel_calls=AUTOTUNE)
        p = (p + model.predict(flip, verbose=0) + model.predict(zoom, verbose=0)) / 3.0
    return p


def scores(y, yp, keep):
    m = np.isin(y, keep)
    y, yp = y[m], yp[m]
    p, r, f, _ = precision_recall_fscore_support(y, yp, labels=keep, average="macro",
                                                 zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, yp, labels=keep, average="weighted",
                                                    zero_division=0)
    return dict(accuracy=accuracy_score(y, yp), precision=p, recall=r, f1=f,
                precision_w=pw, recall_w=rw, f1_w=fw, n=int(m.sum()))


def plot_cm(y, yp, keep, title, path):
    names = [CLASS_NAMES[i] for i in keep]
    m = np.isin(y, keep)
    cm = confusion_matrix(y[m], yp[m], labels=keep)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def eval_one(ckpt: Path, data_dir: Path, batch, tta):
    cfg_p = ckpt.with_suffix(".json")
    cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
    model = keras.models.load_model(ckpt)
    size = cfg.get("img_size") or model.input_shape[1]
    ds, files = load_test(data_dir, size, batch)
    n_out = model.output_shape[-1]
    if n_out != len(CLASS_NAMES):
        raise SystemExit(f"{ckpt.name} has {n_out} outputs but the dataset has "
                         f"{len(CLASS_NAMES)} classes; wrong --models/--data pairing?")
    y = np.concatenate([b.numpy() for _, b in ds]).argmax(1)
    return y, predict(model, ds, tta), files


def case_ids(files):
    """Cluster key for the bootstrap: the case a file belongs to."""
    return np.array([f"{Path(f).parent.name}/{parse_name(Path(f))[1]}" for f in files])


def bootstrap_ci(y, yp, keep, cases, n_boot=1000, seed=0):
    """95 % CI of accuracy and macro-F1, resampling whole cases."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(cases, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    acc, f1 = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        sc = scores(y[idx], yp[idx], [k for k in keep if (y[idx] == k).any()])
        acc.append(sc["accuracy"]); f1.append(sc["f1"])
    return (np.percentile(acc, [2.5, 97.5]).tolist(), np.percentile(f1, [2.5, 97.5]).tolist())


def binom_p_greater(k, n, p0):
    """Exact one-sided P(X >= k) for X ~ Binomial(n, p0)."""
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--cv", default=None)
    ap.add_argument("--models", default="./models")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    a = ap.parse_args()
    if not (a.data or a.cv):
        raise SystemExit("pass --data ./dataset_rebuilt or --cv ./dataset_cv")

    global CLASS_NAMES
    names = load_class_names(Path(a.cv or a.data))
    if names:
        CLASS_NAMES = names
        print(f"class names from class_names.json: {len(names)} classes")

    mdir, out = Path(a.models), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(a.cv or a.data)
    cfg_file = next((p for p in (root / "rebuild_config.json", root / "cv_config.json")
                     if p.exists()), None)
    train_only = []
    if cfg_file:
        train_only = json.loads(cfg_file.read_text()).get("train_only_classes", [])

    results, pooled, y_all, per_fold = {}, {}, {}, {}

    for key in ("cnn", "scnn", "xception"):
        ys, ps, fold_names, fl, fold_id = [], [], [], [], []
        if a.cv:
            for fold in sorted(p for p in Path(a.cv).glob("fold_*") if p.is_dir()):
                ck = mdir / f"best_model_{key}_{fold.name}.keras"
                if not ck.exists():
                    continue
                y, p, files = eval_one(ck, fold, a.batch_size, a.tta)
                ys.append(y); ps.append(p); fold_names.append(fold.name)
                fl += files; fold_id += [fold.name] * len(y)
        else:
            ck = mdir / f"best_model_{key}.keras"
            if ck.exists():
                y, p, files = eval_one(ck, Path(a.data), a.batch_size, a.tta)
                ys.append(y); ps.append(p)
                fl += files; fold_id += ["single"] * len(y)
        if not ys:
            print(f"skip {key} (no checkpoints in {mdir})")
            continue
        y = np.concatenate(ys); p = np.concatenate(ps)
        y_all[key], pooled[key] = y, p

        present = sorted(set(y.tolist()))
        keep = [i for i in present if CLASS_NAMES[i] not in train_only]
        yp = p.argmax(1)
        results[PRETTY[key]] = scores(y, yp, keep)
        m = np.isin(y, keep)
        cases = case_ids(fl)
        acc_ci, f1_ci = bootstrap_ci(y[m], yp[m], keep, cases[m])
        results[PRETTY[key]].update(acc_ci=acc_ci, f1_ci=f1_ci, n_cases=int(len(set(cases[m]))))
        with open(out / f"predictions_{key}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "file", "true", "pred"] + [f"p_{c}" for c in CLASS_NAMES])
            for fo, fi, t, pr, pp in zip(fold_id, fl, y, yp, p):
                w.writerow([fo, Path(fi).name, CLASS_NAMES[t], CLASS_NAMES[pr]]
                           + [f"{v:.4f}" for v in pp])
        plot_cm(y, yp, keep, f"{PRETTY[key]} — pooled test", out / f"confusion_{key}.png")
        m = np.isin(y, keep)
        (out / f"per_class_{key}.txt").write_text(
            classification_report(y[m], yp[m], labels=keep,
                                  target_names=[CLASS_NAMES[i] for i in keep],
                                  zero_division=0, digits=3))
        s = results[PRETTY[key]]
        print(f"{PRETTY[key]:<10} n={s['n']:<5} acc={s['accuracy']:.3f} "
              f"[{s['acc_ci'][0]:.3f}-{s['acc_ci'][1]:.3f}]  macro-F1={s['f1']:.3f} "
              f"[{s['f1_ci'][0]:.3f}-{s['f1_ci'][1]:.3f}]  over {len(keep)} classes, "
              f"{s['n_cases']} cases")

        # per-fold spread: with this few cases, one number hides everything
        if a.cv and len(ys) > 1:
            fs = []
            for fy, fp, fn in zip(ys, ps, fold_names):
                k = [i for i in sorted(set(fy.tolist())) if CLASS_NAMES[i] not in train_only]
                if k:
                    fs.append((fn, scores(fy, fp.argmax(1), k)))
            per_fold[PRETTY[key]] = fs
            for fn, fsc in fs:
                print(f"    {fn:<8} n={fsc['n']:<4} acc={fsc['accuracy']:.3f} "
                      f"macro-F1={fsc['f1']:.3f}")

        # majority-class collapse check
        counts = np.bincount(y[np.isin(y, keep)], minlength=len(CLASS_NAMES))
        maj = int(counts.argmax())
        maj_acc = counts[maj] / counts.sum()
        pred_share = np.bincount(yp[np.isin(y, keep)], minlength=len(CLASS_NAMES))
        top_pred = int(pred_share.argmax())
        print(f"    majority-class baseline ({CLASS_NAMES[maj]}) = {maj_acc:.3f}"
              f" | model predicts '{CLASS_NAMES[top_pred]}' for "
              f"{pred_share[top_pred]/pred_share.sum():.0%} of test images")
        if abs(s["accuracy"] - maj_acc) < 0.02 or pred_share[top_pred] / pred_share.sum() > 0.8:
            print(f"    *** WARNING: looks like majority-class collapse. "
                  f"Accuracy here is not evidence of learning; read macro-F1. ***")
        results[PRETTY[key]]["majority_baseline"] = float(maj_acc)
        # the baseline predicts `maj` for everything: score it with the same function
        results[PRETTY[key]]["_baseline_scores"] = scores(
            y, np.full_like(y, maj), keep)
        k_ok = int(round(s["accuracy"] * s["n"]))
        pv = binom_p_greater(k_ok, s["n"], maj_acc)
        results[PRETTY[key]]["p_vs_majority"] = pv
        print(f"    accuracy vs majority baseline: exact binomial p = {pv:.3f} "
              f"({'beats' if pv < 0.05 else 'NOT distinguishable from'} baseline)")

    if a.ensemble and len(pooled) > 1 and len({v.shape for v in pooled.values()}) == 1:
        key0 = next(iter(y_all))
        y = y_all[key0]
        ens = np.mean(list(pooled.values()), axis=0)
        keep = [i for i in sorted(set(y.tolist())) if CLASS_NAMES[i] not in train_only]
        results["Ensemble (soft vote)"] = scores(y, ens.argmax(1), keep)
        plot_cm(y, ens.argmax(1), keep, "Ensemble — pooled test", out / "confusion_ensemble.png")

    if results:
        first = next(iter(results.values()))
        if "_baseline_scores" in first:
            results["Majority-class baseline"] = first["_baseline_scores"]
        for v in results.values():
            v.pop("_baseline_scores", None)

    md = ["| Model | Accuracy (95% CI) | Precision | Recall | Macro-F1 (95% CI) |",
          "|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{|l|c|c|c|c|}", r"\hline",
           r"Model & Accuracy & Precision & Recall & F1-score \\ \hline"]
    for name, s in results.items():
        ci_a = (f" ({s['acc_ci'][0]*100:.0f}-{s['acc_ci'][1]*100:.0f})" if "acc_ci" in s else "")
        ci_f = (f" ({s['f1_ci'][0]:.2f}-{s['f1_ci'][1]:.2f})" if "f1_ci" in s else "")
        md.append(f"| {name} | {s['accuracy']*100:.1f}%{ci_a} | {s['precision']:.2f} | "
                  f"{s['recall']:.2f} | {s['f1']:.2f}{ci_f} |")
        tex.append(f"{name} & {s['accuracy']*100:.1f}\\% & {s['precision']:.2f} & "
                   f"{s['recall']:.2f} & {s['f1']:.2f} \\\\ \\hline")
    tex.append(r"\end{tabular}")

    note = (f"\n\nMacro-averaged over evaluable classes on a case-level, leak-free "
            f"{'pooled ' + str(len(list(Path(a.cv).glob('fold_*')))) + '-fold CV' if a.cv else 'held-out'} "
            f"test set.")
    if train_only:
        note += (f" Excluded (no held-out case available): {', '.join(train_only)}.")
    if per_fold:
        md += ["", "", "| Model | Accuracy (mean ± sd) | Macro-F1 (mean ± sd) | folds |",
               "|---|---|---|---|"]
        for name, fs in per_fold.items():
            acc = np.array([x[1]["accuracy"] for x in fs])
            f1 = np.array([x[1]["f1"] for x in fs])
            md.append(f"| {name} | {acc.mean()*100:.1f}% ± {acc.std(ddof=1)*100:.1f} | "
                      f"{f1.mean():.2f} ± {f1.std(ddof=1):.2f} | {len(fs)} |")
        (out / "per_fold.json").write_text(json.dumps(
            {k: {fn: v for fn, v in fs} for k, fs in per_fold.items()}, indent=2))

    (out / "results_table.md").write_text("\n".join(md) + note + "\n")
    (out / "results_table.tex").write_text("\n".join(tex) + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("\n" + "\n".join(md) + note)
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()

