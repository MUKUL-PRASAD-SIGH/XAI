"""
02b_make_cv_folds.py
====================
Builds a STRATIFIED, GROUPED K-fold cross-validation dataset (default K=5).

    dataset_cv/
        class_names.json
        cv_config.json
        cv_manifest.csv          one row per file: fold, split-in-each-fold, group...
        cv_report.txt            per-class / per-fold audit incl. leakage checks
        fold_0/{train,valid,test}/<class>/*.jpg
        ...
        fold_4/...

Why this replaces `02_rebuild_dataset.py --folds`
-------------------------------------------------
* STRATIFIED: every class is spread over the K test folds as evenly as its
  number of independent groups allows (sklearn StratifiedGroupKFold, then the
  best of N random seeds is kept, scored on empty-class-in-fold and per-class
  imbalance). The old code dealt groups round-robin per class, which was not
  stratified overall and ignored group size.
* GROUPED: a group is a case / near-duplicate cluster / exact-duplicate set,
  built AFTER the class map is applied, so a patient whose planes carry
  different fine labels that merge into one family can never straddle folds.
* Conflicting labels inside one group are resolved (majority wins, otherwise
  the whole group is dropped) and logged, instead of silently landing in two
  folds.
* Every fold has its own validation set (carved out of the training groups,
  also stratified + grouped) so early stopping / LR scheduling never touch the
  test fold. The old code used the neighbouring test fold, which cost 20 % of
  the training data and made folds share validation/test roles.
* valid / test contain TRUE ORIGINALS only (one image per scan); stored
  `_aug_N` copies live in train only.

Usage
-----
    python 02b_make_cv_folds.py --src ./ds --out ./dataset_cv \
        --class-map ../configs/class_map_cv.json --link

    python 02b_make_cv_folds.py --src ./ds --out ./dataset_cv --manifest-only
"""

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_audit = _load("audit_mod", "01_dataset_audit.py")
_rebuild = _load("rebuild_mod", "02_rebuild_dataset.py")
load_records, sha256 = _audit.load_records, _audit.sha256
build_groups = _rebuild.build_groups


# ==========================================================================
# pure logic (no image I/O) -- unit-tested in tests/test_cv_folds.py
# ==========================================================================
def apply_class_map(recs, cmap, drop=()):
    """Relabel in place; keeps the fine label in r['orig_label']."""
    out = []
    unmapped = set()
    for r in recs:
        r["orig_label"] = r["label"]
        if r["label"] in drop:
            continue
        if cmap:
            if r["label"] not in cmap:
                unmapped.add(r["label"])
                continue
            r["label"] = cmap[r["label"]]
        out.append(r)
    if unmapped:
        raise SystemExit(f"class map has no entry for: {', '.join(sorted(unmapped))}")
    return out


def resolve_conflicts(recs, log):
    """Each group must carry one label. Majority wins; a tie drops the group."""
    by_g = defaultdict(list)
    for r in recs:
        by_g[r["group"]].append(r)
    keep, dropped_imgs, dropped_groups = [], 0, 0
    for g, rs in by_g.items():
        c = Counter(r["label"] for r in rs)
        if len(c) == 1:
            keep += rs
            continue
        (l1, n1), (_, n2) = c.most_common(2)
        if n1 == n2:
            dropped_groups += 1
            dropped_imgs += len(rs)
            log(f"  conflict (tie) -> group dropped: {dict(c)}")
            continue
        kept = [r for r in rs if r["label"] == l1]
        dropped_imgs += len(rs) - len(kept)
        log(f"  conflict -> kept '{l1}': {dict(c)}")
        keep += kept
    return keep, dropped_imgs, dropped_groups


def originals_only(subset):
    """One entry per source scan: the true original, else one stored aug."""
    by_scan = defaultdict(list)
    for r in subset:
        by_scan[(r["label"], r["scan"])].append(r)
    keep = []
    for v in by_scan.values():
        orig = [r for r in v if r["aug"] is None]
        keep += orig if orig else [v[0]]
    return keep


def group_table(recs):
    """group -> (label, n_original_images)."""
    lab, n = {}, Counter()
    for r in originals_only(recs):
        lab[r["group"]] = r["label"]
        n[r["group"]] += 1
    return lab, n


def _fold_score(fold_of, glab, gn, k, classes):
    """Lower is better: empty (class, fold) cells dominate, then imbalance."""
    per = {c: np.zeros(k) for c in classes}
    ng = Counter()
    for g, f in fold_of.items():
        per[glab[g]][f] += gn[g]
        ng[glab[g]] += 1
    empty = 0
    imb = 0.0
    for c in classes:
        achievable = min(ng[c], k)
        nonempty = int((per[c] > 0).sum())
        empty += max(0, achievable - nonempty)
        imb += float(np.std(per[c] / max(per[c].sum(), 1)))
    sizes = sum(per.values())
    return 1000 * empty + imb + float(np.std(sizes) / max(np.mean(sizes), 1))


def assign_test_folds(recs, k, seed, n_trials):
    """group -> fold index, stratified by class, best of `n_trials` seeds."""
    glab, gn = group_table(recs)
    groups = sorted(glab)
    classes = sorted(set(glab.values()))
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[glab[g]] for g in groups])
    w = np.array([gn[g] for g in groups])
    # StratifiedGroupKFold balances by sample; feed it one row per image
    rows_g = np.repeat(np.arange(len(groups)), w)
    rows_y = y[rows_g]

    best, best_s = None, None
    for t in range(n_trials):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + t)
        fold_of = {}
        try:
            for f, (_, te) in enumerate(sgkf.split(rows_g, rows_y, rows_g)):
                for gi in np.unique(rows_g[te]):
                    fold_of[groups[gi]] = f
        except ValueError:
            continue
        s = _fold_score(fold_of, glab, gn, k, classes)
        if best_s is None or s < best_s:
            best, best_s = fold_of, s
    if best is None:
        raise SystemExit("could not build stratified folds (too few groups per class?)")
    return best, best_s


def assign_valid_groups(recs, fold_of, k_test, val_frac, seed):
    """From the groups NOT in the test fold, pick a stratified validation set."""
    glab, gn = group_table(recs)
    pool = sorted(g for g in glab if fold_of[g] != k_test)
    nv = max(2, int(round(1 / val_frac)))
    cidx = {c: i for i, c in enumerate(sorted(set(glab[g] for g in pool)))}
    rows_g = np.repeat(np.arange(len(pool)), [gn[g] for g in pool])
    rows_y = np.array([cidx[glab[pool[i]]] for i in rows_g])
    sgkf = StratifiedGroupKFold(n_splits=nv, shuffle=True, random_state=seed + 1000 + k_test)
    _, va = next(iter(sgkf.split(rows_g, rows_y, rows_g)))
    valid = {pool[i] for i in np.unique(rows_g[va])}
    # a class must keep at least one training group
    by_c = defaultdict(list)
    for g in pool:
        by_c[glab[g]].append(g)
    for c, gs in by_c.items():
        if all(g in valid for g in gs):
            valid.discard(max(gs, key=lambda g: gn[g]))
    return valid


def split_of(recs, fold_of, valid_groups, k_test):
    """Returns the records of fold k with r['split'] set, aug policy applied."""
    for r in recs:
        g = r["group"]
        r["split"] = ("test" if fold_of[g] == k_test
                      else "valid" if g in valid_groups else "train")
    train = [r for r in recs if r["split"] == "train"]
    held = originals_only([r for r in recs if r["split"] != "train"])
    return train + held


def check_no_leakage(fold_records):
    """Group and byte-identical-content overlap between splits of one fold."""
    issues = []
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        ga = {r["group"] for r in fold_records if r["split"] == a}
        gb = {r["group"] for r in fold_records if r["split"] == b}
        if ga & gb:
            issues.append(f"{len(ga & gb)} groups shared by {a}/{b}")
        sa = {r["sha"] for r in fold_records if r["split"] == a and r.get("sha")}
        sb = {r["sha"] for r in fold_records if r["split"] == b and r.get("sha")}
        if sa & sb:
            issues.append(f"{len(sa & sb)} identical files shared by {a}/{b}")
    return issues


# ==========================================================================
# I/O
# ==========================================================================
def materialise(recs, out_dir, link):
    for r in recs:
        d = out_dir / r["split"] / r["label"]
        d.mkdir(parents=True, exist_ok=True)
        dst = d / r["path"].name
        if dst.exists():
            continue
        if link:
            try:
                dst.hardlink_to(r["path"])
                continue
            except Exception:
                pass
        shutil.copy2(r["path"], dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="original export (train/valid/test)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--class-map", type=Path, default=None,
                    help="JSON fine-label -> family. Omit to keep all classes.")
    ap.add_argument("--drop-classes", nargs="*", default=[])
    ap.add_argument("--group-level", choices=["scan", "case"], default="case")
    ap.add_argument("--phash-dist", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.12,
                    help="share of the non-test groups used for validation")
    ap.add_argument("--min-groups", type=int, default=3,
                    help="classes with fewer independent groups are excluded")
    ap.add_argument("--trials", type=int, default=300,
                    help="random seeds tried; the best-balanced fold layout is kept")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="hardlink instead of copy")
    ap.add_argument("--manifest-only", action="store_true", help="skip copying images")
    a = ap.parse_args()

    if a.out.exists() and any(a.out.iterdir()) and not a.manifest_only:
        raise SystemExit(f"{a.out} is not empty; refusing to overwrite a CV dataset.")
    a.out.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def P(s=""):
        print(s)
        log_lines.append(s)

    cmap = {}
    if a.class_map:
        raw = json.loads(a.class_map.read_text())
        cmap = {k: v for k, v in raw.items() if not k.startswith("_")}
        P(f"Class map: {a.class_map} ({len(cmap)} labels -> {len(set(cmap.values()))} classes)")

    recs = load_records(a.src)          # pools the original train/valid/test
    P(f"Loaded {len(recs)} files from {a.src}")
    for r in recs:
        r["sha"] = sha256(r["path"])
    recs = apply_class_map(recs, cmap, set(a.drop_classes))
    groups = build_groups(recs, a.group_level, a.phash_dist)
    for r, g in zip(recs, groups):
        r["group"] = g

    P("\nLabel-conflict resolution:")
    recs, di, dg = resolve_conflicts(recs, P)
    P(f"  dropped {di} images ({dg} tied groups)")

    # ---- classes that cannot be cross-validated ----------------------------
    glab, gn = group_table(recs)
    ng = Counter(glab.values())
    small = sorted(c for c, n in ng.items() if n < a.min_groups)
    if small:
        P(f"\nExcluded (< {a.min_groups} independent groups): "
          + ", ".join(f"{c} ({ng[c]})" for c in small))
        recs = [r for r in recs if r["label"] not in small]
        glab, gn = group_table(recs)
        ng = Counter(glab.values())
    classes = sorted(ng)
    if len(classes) < 2:
        raise SystemExit("fewer than two classes left after filtering")
    P(f"\n{len(classes)} classes, {len(glab)} independent groups, "
      f"{sum(1 for _ in originals_only(recs))} original scans")

    # ---- stratified grouped folds ------------------------------------------
    fold_of, score = assign_test_folds(recs, a.folds, a.seed, a.trials)
    P(f"Best of {a.trials} fold layouts: balance score {score:.3f} (lower is better)")

    fam = {r["group"]: r.get("family", "?") for r in recs}
    manifest = []
    P(f"\n{'class':<30}{'groups':>7}" + "".join(f"{'f'+str(k):>7}" for k in range(a.folds)))
    P("-" * (37 + 7 * a.folds))
    test_cnt = Counter()
    for r in originals_only(recs):
        test_cnt[(r["label"], fold_of[r["group"]])] += 1
    warn = []
    for c in classes:
        row = [test_cnt[(c, k)] for k in range(a.folds)]
        P(f"{c:<30}{ng[c]:>7}" + "".join(f"{v:>7}" for v in row))
        if min(row) == 0 and ng[c] >= a.folds:
            warn.append(c)
    P("-" * (37 + 7 * a.folds))
    P("(test images per class per fold; each group is tested in exactly one fold)")
    if warn:
        P(f"WARNING: stratification left empty test folds for: {', '.join(warn)}")

    P("\nSource family of test images per fold (patient vs copy-of = source proxy):")
    for k in range(a.folds):
        c = Counter(fam[r["group"]] for r in originals_only(recs) if fold_of[r["group"]] == k)
        P(f"  fold_{k}: {dict(c)}")

    # ---- per fold ----------------------------------------------------------
    all_ok = True
    for k in range(a.folds):
        valid = assign_valid_groups(recs, fold_of, k, a.val_frac, a.seed)
        sub = split_of(recs, fold_of, valid, k)
        issues = check_no_leakage(sub)
        all_ok &= not issues
        tot = Counter(r["split"] for r in sub)
        P(f"\nfold_{k}: train={tot['train']} valid={tot['valid']} test={tot['test']}  "
          f"leakage check: {'PASS' if not issues else 'FAIL ' + '; '.join(issues)}")
        # classes absent from train would be unlearnable in this fold
        tr_c = {r["label"] for r in sub if r["split"] == "train"}
        missing = [c for c in classes if c not in tr_c]
        if missing:
            P(f"  WARNING: no training images for {missing}")
        if not a.manifest_only:
            materialise(sub, a.out / f"fold_{k}", a.link)
        for r in sub:
            manifest.append([k, r["split"], r["path"].name, r["label"], r["orig_label"],
                             r["group"], r["case"], r["scan"], r["aug"], r.get("family")])
    if not all_ok:
        raise SystemExit("LEAKAGE DETECTED; refusing to write a usable dataset")

    # every group is tested exactly once
    tested = Counter()
    for k in range(a.folds):
        for row in manifest:
            if row[0] == k and row[1] == "test":
                tested[row[5]] += 1
    P(f"\nEvery one of the {len(glab)} groups is in exactly one test fold: "
      f"{'PASS' if set(tested) == set(glab) else 'FAIL'}")

    (a.out / "class_names.json").write_text(json.dumps(classes, indent=2))
    with open(a.out / "cv_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "split", "filename", "label", "orig_label", "group",
                    "case_id", "scan_id", "aug", "family"])
        w.writerows(manifest)
    (a.out / "cv_config.json").write_text(json.dumps({
        "folds": a.folds, "stratified": True, "grouped": True,
        "group_level": a.group_level, "aug_policy": "train-only",
        "val_frac": a.val_frac, "seed": a.seed, "trials": a.trials,
        "class_map": cmap, "dropped": list(a.drop_classes),
        "train_only_classes": [], "excluded_small_classes": small,
        "class_names": classes,
    }, indent=2))
    (a.out / "cv_report.txt").write_text("\n".join(log_lines))
    P(f"\nWrote {a.out}/  (+ class_names.json, cv_manifest.csv, cv_config.json, cv_report.txt)")
    P(f"\nNext:\n  python 03_train.py --cv {a.out} --model xception --out ./models_cv\n"
      f"  python 04_evaluate.py --cv {a.out} --models ./models_cv --out ./results_cv --tta")


if __name__ == "__main__":
    main()
