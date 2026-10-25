"""
16_audit_cv_results.py
======================
Read-only audit of an EXISTING cross-validation run. Trains nothing, loads no
model, modifies no input. Writes a new report directory (refuses to reuse a
non-empty one).

    python scripts/16_audit_cv_results.py --cv /content/dataset_cv \
        --results /content/results_cv --models /content/models_cv \
        --probe /content/results_probe --out /content/audit_cv_<tag>

Answers, from the saved files only:
  * which artifacts exist (CNN / S-CNN / Xception checkpoints, predictions);
  * image / case-group counts and why they differ (e.g. 430 images vs ~91 cases);
  * per-class precision / recall / F1 / support, confusion matrix, top confusions;
  * group separation across folds and exact-duplicate overlap (SHA-256);
  * classes absent from a test fold and the macro-F1 class set;
  * how the CI / baseline test behave when images are (not) resampled by case.
Verified facts and untested hypotheses are kept in separate sections.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

MODEL_KEYS = ("cnn", "scnn", "xception")


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def macro_f1(y, yp, labels):
    return float(precision_recall_fscore_support(y, yp, labels=labels, average="macro",
                                                 zero_division=0)[2])


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def inventory(cv, results, models, probe):
    lines, found = [], {}
    n_folds = len(list(cv.glob("fold_*"))) if cv else 0
    for k in MODEL_KEYS:
        ck = sorted(models.glob(f"best_model_{k}_fold_*.keras")) if models and models.is_dir() else []
        pred = (results / f"predictions_{k}.csv") if results else None
        found[k] = dict(checkpoints=len(ck), predictions=bool(pred and pred.exists()))
        lines.append([k, f"{len(ck)}/{n_folds}", "yes" if found[k]["predictions"] else "no"])
    extra = [str(p) for p in (probe.glob("*") if probe and probe.is_dir() else [])][:20]
    return found, lines, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", required=True, help="dataset_cv dir (cv_manifest.csv, class_names.json, fold_*/)")
    ap.add_argument("--results", required=True, help="dir with predictions_<model>.csv from 04_evaluate.py")
    ap.add_argument("--models", default=None, help="dir with best_model_*_fold_*.keras (existence check only)")
    ap.add_argument("--probe", default=None, help="results_probe dir (optional)")
    ap.add_argument("--model", default="xception", choices=MODEL_KEYS, help="which predictions to audit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 duplicate scan")
    a = ap.parse_args()

    cv, results = Path(a.cv), Path(a.results)
    models = Path(a.models) if a.models else None
    probe = Path(a.probe) if a.probe else None
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new --out (existing outputs are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)

    classes = json.loads((cv / "class_names.json").read_text())
    cfg = json.loads((cv / "cv_config.json").read_text()) if (cv / "cv_config.json").exists() else {}
    man = read_csv(cv / "cv_manifest.csv")
    pred_path = results / f"predictions_{a.model}.csv"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found; run 04_evaluate.py first (this script does not predict)")
    pred = read_csv(pred_path)

    R = []   # report lines
    P = R.append

    # ---------------- 1. artifacts ----------------
    found, inv_rows, extra = inventory(cv, results, models, probe)
    P("# CV results audit\n")
    P(f"Audited: `{pred_path}` ({a.model}); dataset `{cv}`; {len(classes)} classes: {', '.join(classes)}\n")
    P("## 1. Artifacts that actually exist\n")
    P(md_table(["model", "fold checkpoints", "predictions csv"], inv_rows))
    missing = [k for k in MODEL_KEYS if not found[k]["checkpoints"] and not found[k]["predictions"]]
    P(f"\nNo checkpoints or predictions for: **{', '.join(missing) or 'none'}**. "
      "Nothing is reported for those models. Train them with the Colab notebook cell "
      "'CNN / S-CNN baselines' (`03_train.py --model cnn|scnn`) and re-run this audit with `--model`.\n")
    if extra:
        P("Probe dir files: " + ", ".join(Path(e).name for e in extra) + "\n")

    # ---------------- 2. counts ----------------
    test_rows = [r for r in man if r["split"] == "test"]
    by_fold_test = defaultdict(list)
    for r in test_rows:
        by_fold_test[r["fold"]].append(r)
    folds = sorted(by_fold_test, key=int)
    test_groups = {r["group"] for r in test_rows}
    test_cases = {r["case_id"] for r in test_rows}
    test_scans = {r["scan_id"] for r in test_rows}
    aug_in_test = sum(r["aug"] not in ("", "0", "False", "false") for r in test_rows)
    names = [r["filename"] for r in test_rows]
    P("## 2. What is the evaluation unit? (images vs cases)\n")
    P(md_table(["quantity", "value"], [
        ["test rows in cv_manifest (all folds)", len(test_rows)],
        ["unique test filenames", len(set(names))],
        ["prediction rows", len(pred)],
        ["unique predicted filenames", len({r["file"] for r in pred})],
        ["distinct `scan_id` (original scan, augmented copies share it)", len(test_scans)],
        ["distinct `case_id` (filename case token)", len(test_cases)],
        ["distinct linked `group` (case tokens + duplicate links; the CV unit)", len(test_groups)],
        ["test rows flagged augmented", aug_in_test],
    ]))
    ratio = len(test_rows) / max(len(test_groups), 1)
    P(f"\n**Verified:** {len(test_rows)} test images belong to {len(test_groups)} linked groups "
      f"(~{ratio:.1f} images per group). The 'cases' figure in the log counts groups, not images, so "
      "the two numbers measure different things; they do not conflict. Within a group the images are "
      "near-copies (augmentations or the same scan), so the effective sample size for confidence "
      "intervals is closer to the group count than to the image count.\n")

    # ---------------- 3. per-class ----------------
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[r["true"]] for r in pred])
    yp = np.array([cidx[r["pred"]] for r in pred])
    folds_p = np.array([r["fold"] for r in pred])
    fname2group = {(r["fold"], r["filename"]): r["group"] for r in test_rows}
    fold_key = lambda f: re.sub(r"\D", "", f)
    group = np.array([fname2group.get((fold_key(r["fold"]), r["file"]), f"?{r['file']}") for r in pred])
    unmatched = int(sum(g.startswith("?") for g in group))
    labels_all = list(range(len(classes)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=labels_all, zero_division=0)
    n_pred = np.bincount(yp, minlength=len(classes))
    P("## 3. Per-class metrics (pooled over folds)\n")
    P(md_table(["class", "support", "predicted", "precision", "recall", "F1", "test groups"],
               [[classes[i], int(sup[i]), int(n_pred[i]), f"{pr[i]:.3f}", f"{rc[i]:.3f}", f"{f1[i]:.3f}",
                 len({g for g, t in zip(group, y) if t == i})] for i in labels_all]))
    with open(out / "per_class_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "predicted", "precision", "recall", "f1"])
        for i in labels_all:
            w.writerow([classes[i], sup[i], n_pred[i], f"{pr[i]:.4f}", f"{rc[i]:.4f}", f"{f1[i]:.4f}"])
    top = int(np.argmax(n_pred))
    P(f"\nThe model predicts `{classes[top]}` for {n_pred[top] / len(yp):.0%} of images; "
      f"true share of `{classes[top]}` is {sup[top] / len(y):.0%}. "
      f"Classes never predicted: {', '.join(classes[i] for i in labels_all if n_pred[i] == 0) or 'none'}.\n")
    if unmatched:
        P(f"(Note: {unmatched} prediction rows could not be matched to a manifest group; "
          "they are treated as singleton groups in the bootstrap.)\n")

    # ---------------- 4. confusion ----------------
    cm = confusion_matrix(y, yp, labels=labels_all)
    np.savetxt(out / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(classes), comments="")
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{a.model}: pooled test (row-normalised)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=170)
    plt.close(fig)
    pairs = sorted(((cm[i, j], i, j) for i in labels_all for j in labels_all if i != j and cm[i, j]), reverse=True)[:8]
    P("## 4. Most frequent confusions\n")
    P(md_table(["true", "predicted as", "images", "share of true class"],
               [[classes[i], classes[j], int(n), f"{n / max(sup[i], 1):.0%}"] for n, i, j in pairs]))
    P("\nPlots: `confusion_matrix.png`, `per_class_f1.png`, `per_fold.png`.\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(classes)), f1, color="#4c78a8")
    ax.set_xticks(range(len(classes)), [f"{c} (n={int(s)})" for c, s in zip(classes, sup)], rotation=70,
                  ha="right", fontsize=7)
    ax.set_ylabel("F1"); ax.set_ylim(0, 1); ax.set_title(f"{a.model}: per-class F1"); fig.tight_layout()
    fig.savefig(out / "per_class_f1.png", dpi=170); plt.close(fig)

    # ---------------- 5. group separation / duplicates ----------------
    P("## 5. Group separation and duplicate overlap\n")
    train_by_fold = defaultdict(set)
    for r in man:
        if r["split"] in ("train", "valid"):
            train_by_fold[r["fold"]].add(r["group"])
    leaks = {f: sorted(set(g["group"] for g in by_fold_test[f]) & train_by_fold[f]) for f in folds}
    P(f"- Group overlap between a fold's test set and its own train/valid sets: "
      f"{ {f: len(v) for f, v in leaks.items()} } (verified from `cv_manifest.csv`).")
    seen = Counter()
    for f in folds:
        for g in {r['group'] for r in by_fold_test[f]}:
            seen[g] += 1
    P(f"- Groups tested in more than one fold: {sum(1 for v in seen.values() if v > 1)}; "
      f"groups tested exactly once: {sum(1 for v in seen.values() if v == 1)} of {len(seen)}.")
    if not a.no_hash:
        dup_rows, hashed = [], 0
        for f in folds:
            fd = cv / f"fold_{f}"
            if not fd.is_dir():
                continue
            tr = {}
            for sp in ("train", "valid"):
                for p in (fd / sp).rglob("*"):
                    if p.is_file():
                        tr.setdefault(sha(p), p.name); hashed += 1
            for p in (fd / "test").rglob("*"):
                if p.is_file():
                    hashed += 1
                    h = sha(p)
                    if h in tr:
                        dup_rows.append((f, p.name, tr[h]))
        if hashed:
            P(f"- Exact (SHA-256) duplicates between a fold's test images and its train/valid images: "
              f"**{len(dup_rows)}** (hashed {hashed} files).")
            with open(out / "exact_duplicate_overlap.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["fold", "test_file", "train_or_valid_file"]); w.writerows(dup_rows)
        else:
            P("- SHA-256 scan skipped: `fold_*/` image directories are not present on this machine.")
    P("- Near-duplicate (perceptual-hash) overlap is NOT re-computed here; `01_dataset_audit.py` "
      "defined the groups and `02b` links near-duplicates into them before splitting.\n")

    # ---------------- 6. folds / class sets ----------------
    P("## 6. Classes per test fold and the macro-F1 class set\n")
    rows = []
    for f in folds:
        m = np.array([fold_key(x) == f for x in folds_p])
        present = sorted(set(y[m].tolist()))
        acc = float((y[m] == yp[m]).mean()) if m.any() else float("nan")
        rows.append([f, int(m.sum()), len(present), ", ".join(classes[i] for i in labels_all if i not in present) or "-",
                     f"{acc:.3f}", f"{macro_f1(y[m], yp[m], present):.3f}" if m.any() else "-"])
    P(md_table(["fold", "images", "classes present", "classes absent from test", "accuracy",
                "macro-F1 (present classes)"], rows))
    keep = sorted(set(y.tolist()))
    P(f"\nPooled macro-F1 over classes with support>0 ({len(keep)}): **{macro_f1(y, yp, keep):.3f}**; "
      f"over all {len(classes)} classes (absent ones score 0): {macro_f1(y, yp, labels_all):.3f}. "
      "`04_evaluate.py` uses the first definition; per-fold macro-F1 above uses each fold's own class set, "
      "so fold values are not averaged on a common basis.\n")
    fa = [float(r[4]) for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([f"fold {r[0]}" for r in rows], fa, color="#4c78a8", label="accuracy")
    ax.bar([f"fold {r[0]}" for r in rows], [float(r[5]) for r in rows], color="#f58518", width=0.4, label="macro-F1")
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("per fold"); fig.tight_layout()
    fig.savefig(out / "per_fold.png", dpi=170); plt.close(fig)

    # ---------------- 7. CI / baseline test ----------------
    P("## 7. Do the CI and baseline test respect the design?\n")
    maj = int(np.bincount(y, minlength=len(classes)).argmax())
    base_acc = float((y == maj).mean())
    acc = float((y == yp).mean())
    rng = np.random.default_rng(0)
    ug, inv = np.unique(group, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]

    def boot(cluster):
        accs, f1s, dif = [], [], []
        for _ in range(a.n_boot):
            if cluster:
                idx = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
            else:
                idx = rng.integers(0, len(y), len(y))
            yy, pp = y[idx], yp[idx]
            accs.append((yy == pp).mean())
            f1s.append(macro_f1(yy, pp, sorted(set(yy.tolist()))))
            dif.append((yy == pp).mean() - (yy == maj).mean())
        return (np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5]), float(np.mean(np.array(dif) <= 0)))
    ci_g, ci_i = boot(True), boot(False)
    from scipy.stats import binom
    p_binom = float(binom.sf(round(acc * len(y)) - 1, len(y), base_acc))
    P(md_table(["method", "accuracy 95% CI", "macro-F1 95% CI", "P(model acc <= majority acc) in resamples"], [
        [f"bootstrap by group ({len(ug)} clusters)", f"{ci_g[0][0]:.3f}-{ci_g[0][1]:.3f}",
         f"{ci_g[1][0]:.3f}-{ci_g[1][1]:.3f}", f"{ci_g[2]:.3f}"],
        [f"bootstrap by image ({len(y)} units, ignores dependence)", f"{ci_i[0][0]:.3f}-{ci_i[0][1]:.3f}",
         f"{ci_i[1][0]:.3f}-{ci_i[1][1]:.3f}", f"{ci_i[2]:.3f}"],
    ]))
    P(f"\nPoint estimates: accuracy {acc:.3f}, majority baseline (`{classes[maj]}`) {base_acc:.3f}. "
      f"Exact binomial p treating all {len(y)} images as independent: {p_binom:.4f}.\n")
    P(f"**Verified:** the binomial test assumes {len(y)} independent Bernoulli trials; the data have only "
      f"{len(ug)} groups, so that p-value is anti-conservative. The group-resampled column is the one to quote. "
      "The case-level CI in `04_evaluate.py` clusters by `fold/case-token`, which is coarser than the "
      "linked `group` used for the split, so its width can differ from the group bootstrap here.\n")

    # ---------------- 8. interpretation ----------------
    P("## 8. Why performance is modest\n")
    P("### Verified from the files above")
    n_ok = sum(1 for c in labels_all if sup[c] >= 1 and f1[c] >= 0.5)
    P(f"- Only {n_ok} of {len(keep)} classes with test images reach F1 >= 0.5; "
      f"{sum(1 for c in labels_all if n_pred[c] == 0)} classes are never predicted.")
    P(f"- Majority-class dominance: `{classes[top]}` is {sup[top] / len(y):.0%} of the test set and "
      f"{n_pred[top] / len(y):.0%} of predictions.")
    P(f"- Effective sample size is {len(ug)} groups for {len(y)} images (section 2); per-class group counts are in "
      "section 3 (last column).")
    spread = (min(fa), max(fa))
    P(f"- Fold accuracy ranges {spread[0]:.3f}-{spread[1]:.3f}; with this few independent groups per class a fold "
      "is decided by 1-2 groups.")
    P("\n### Hypotheses (NOT tested by this audit)")
    P("- Labels are uploader-asserted and unverified; label noise would cap achievable F1.")
    P("- Normal vs abnormal is confounded with file family/source (see `data_quality_report.md` of the dataset "
      "build), so some accuracy may come from non-anatomical cues instead of pathology.")
    P("- Rare diagnoses may be visually near-identical to neighbouring classes (e.g. ventriculomegaly grades); "
      "the confusion pairs in section 4 are consistent with this but do not prove it.")
    P("- Anatomy pretraining (`15_pretrain_anatomy.py`) might help feature quality; untested until a run completes.\n")

    P("## 9. Recommended next experiment\n")
    P("Run the **same 5 folds** with `03_train.py --init-backbone` (anatomy-pretrained) versus the ImageNet start, "
      "everything else fixed, then compare group-bootstrap CIs and per-class F1 from this script. "
      "Only keep the pretraining if the macro-F1 CI moves clearly above the ImageNet run; do not tune on test folds.\n")

    (out / "audit_report.md").write_text("\n".join(R), encoding="utf-8")
    (out / "audit_summary.json").write_text(json.dumps({
        "model": a.model, "n_images": int(len(y)), "n_groups": int(len(ug)), "accuracy": acc,
        "majority_baseline": base_acc, "macro_f1_present": macro_f1(y, yp, keep),
        "per_class": {classes[i]: dict(support=int(sup[i]), precision=float(pr[i]), recall=float(rc[i]),
                                        f1=float(f1[i])) for i in labels_all},
        "artifacts": found, "config": {k: cfg.get(k) for k in ("folds", "seed", "group_level")},
    }, indent=2))
    print("\n".join(R))
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
