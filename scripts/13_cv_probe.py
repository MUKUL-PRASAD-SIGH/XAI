"""
13_cv_probe.py
==============
Low-variance comparator for the fine-tuned network: a logistic-regression probe
on FROZEN ImageNet Xception features, evaluated on exactly the same stratified
grouped folds as 03/04.

On ~1k images with a handful of independent cases per diagnosis, a probe has a
few thousand parameters instead of millions, so it cannot memorise the training
set the way fine-tuning does. If it matches or beats the fine-tuned model, the
fine-tuned model is only memorising; if fine-tuning clearly wins, it is learning.

Per fold: features are standardised on train, C is chosen on the fold's VALID
split (macro-F1), and the TEST split is scored once. Output mirrors 04:
predictions_probe.csv, results_probe.json, per-fold and pooled metrics with
case-level bootstrap CIs.

    python 13_cv_probe.py --cv ./dataset_cv --out ./results_probe
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def extract_features(paths, size, batch, gray):
    """GAP features of frozen Xception, averaged with the horizontal flip."""
    import tensorflow as tf
    from tensorflow import keras
    base = keras.applications.Xception(include_top=False, weights="imagenet",
                                       pooling="avg", input_shape=(size, size, 3))
    luma = tf.constant([0.299, 0.587, 0.114])
    out = []
    for i in range(0, len(paths), batch):
        imgs = []
        for p in paths[i:i + batch]:
            im = tf.io.decode_image(tf.io.read_file(str(p)), channels=3, expand_animations=False)
            imgs.append(tf.image.resize(im, (size, size)))
        x = tf.stack(imgs)
        if gray:
            x = tf.tile(tf.reduce_sum(x * luma, axis=-1, keepdims=True), [1, 1, 1, 3])
        x = x / 127.5 - 1.0
        f = (base(x, training=False) + base(tf.image.flip_left_right(x), training=False)) / 2
        out.append(f.numpy())
        print(f"\r  features {min(i + batch, len(paths))}/{len(paths)}", end="")
    print()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cv", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("./results_probe"))
    ap.add_argument("--img-size", type=int, default=299)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-grayscale", action="store_true")
    ap.add_argument("--Cs", type=float, nargs="*", default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    ev = _load("eval_mod", "04_evaluate.py")

    classes = json.loads((a.cv / "class_names.json").read_text())
    cidx = {c: i for i, c in enumerate(classes)}
    folds = sorted(p for p in a.cv.glob("fold_*") if p.is_dir())

    # one feature extraction per unique file (folds share files via hardlinks)
    files = {}
    for f in folds:
        for sp in ("train", "valid", "test"):
            for p in sorted((f / sp).glob("*/*")):
                if p.suffix.lower() in IMG_EXT:
                    files.setdefault(p.name, p)
    names = sorted(files)
    cache = a.out / "features_cache.npz"
    if cache.exists() and list(np.load(cache, allow_pickle=True)["names"]) == names:
        feats = np.load(cache, allow_pickle=True)["feats"]
        print(f"loaded cached features for {len(names)} files")
    else:
        print(f"extracting features for {len(names)} unique files")
        feats = extract_features([files[n] for n in names], a.img_size, a.batch_size,
                                 not a.no_grayscale)
        np.savez(cache, names=np.array(names, dtype=object), feats=feats)
    F = dict(zip(names, feats))

    def split(fold, sp):
        ps = [p for p in sorted((fold / sp).glob("*/*")) if p.suffix.lower() in IMG_EXT]
        return ps, np.array([cidx[p.parent.name] for p in ps]), np.stack([F[p.name] for p in ps])

    ys, ps_all, fl, fold_id, per_fold = [], [], [], [], []
    for fold in folds:
        tr_p, ytr, Xtr = split(fold, "train")
        va_p, yva, Xva = split(fold, "valid")
        te_p, yte, Xte = split(fold, "test")
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        best = None
        for C in a.Cs:
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
            clf.fit(Xtr, ytr)
            f1 = f1_score(yva, clf.predict(Xva), average="macro", zero_division=0)
            if best is None or f1 > best[0]:
                best = (f1, C, clf)
        _, C, clf = best
        prob = np.zeros((len(yte), len(classes)))
        prob[:, clf.classes_] = clf.predict_proba(Xte)
        yp = prob.argmax(1)
        keep = sorted(set(yte.tolist()))
        s = ev.scores(yte, yp, keep)
        per_fold.append((fold.name, s))
        print(f"{fold.name}: C={C:<6} valid-F1={best[0]:.3f}  test acc={s['accuracy']:.3f} "
              f"macro-F1={s['f1']:.3f} (n={s['n']})")
        ys.append(yte); ps_all.append(prob)
        fl += [str(p) for p in te_p]; fold_id += [fold.name] * len(yte)

    y, p = np.concatenate(ys), np.concatenate(ps_all)
    yp = p.argmax(1)
    keep = sorted(set(y.tolist()))
    s = ev.scores(y, yp, keep)
    cases = ev.case_ids(fl)
    s["acc_ci"], s["f1_ci"] = ev.bootstrap_ci(y, yp, keep, cases)
    counts = np.bincount(y, minlength=len(classes))
    maj = int(counts.argmax())
    base = ev.scores(y, np.full_like(y, maj), keep)
    pv = ev.binom_p_greater(int(round(s["accuracy"] * s["n"])), s["n"], counts[maj] / counts.sum())
    print(f"\nPOOLED probe: acc={s['accuracy']:.3f} [{s['acc_ci'][0]:.3f}-{s['acc_ci'][1]:.3f}] "
          f"macro-F1={s['f1']:.3f} [{s['f1_ci'][0]:.3f}-{s['f1_ci'][1]:.3f}]  "
          f"classes={len(keep)} n={s['n']}")
    print(f"majority baseline ({classes[maj]}): acc={base['accuracy']:.3f} "
          f"macro-F1={base['f1']:.3f};  binomial p vs baseline = {pv:.3f}")

    with open(a.out / "predictions_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "file", "true", "pred"] + [f"p_{c}" for c in classes])
        for fo, fi, t, pr, pp in zip(fold_id, fl, y, yp, p):
            w.writerow([fo, Path(fi).name, classes[t], classes[pr]] + [f"{v:.4f}" for v in pp])
    (a.out / "results_probe.json").write_text(json.dumps(
        {"pooled": s, "baseline": base, "p_vs_majority": pv,
         "per_fold": {n: v for n, v in per_fold}}, indent=2))


if __name__ == "__main__":
    main()
