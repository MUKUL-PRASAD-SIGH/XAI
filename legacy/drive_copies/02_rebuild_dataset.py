"""
02_rebuild_dataset.py
=====================
Rebuilds the dataset ONCE into a leak-free folder structure. Run after
01_dataset_audit.py. Reads the original export; never modifies it.

    dataset_rebuilt/
        train/<class>/*.jpg
        valid/<class>/*.jpg
        test/<class>/*.jpg

Key policies (all switchable, all recorded in rebuild_report.txt so the paper
can state exactly what was done):

  --protocol group     Split by CASE, so no scan or patient spans two splits.
  --protocol random    Reproduce the original image-level random split. Use this
                       to generate the "as published" comparison arm.

  --aug-policy train-only  (default) Keep the stored _aug_N copies in TRAIN, but
                           use only true originals in valid/test, so one scan
                           cannot be counted as six test samples. Best of both.
  --aug-policy drop-if-original    Discard stored _aug_N copies everywhere when
                           the original exists (it does for 100% of v1 scans).
  --aug-policy keep        Keep every stored file in every split.

  --min-cases N        Classes with fewer than N cases cannot be honestly
                       evaluated. --small-class controls what happens to them:
                         train-only  keep for training, exclude from val/test
                         exclude     drop the class entirely
                         keep        force a split anyway (not defensible)

Usage
-----
    python 02_rebuild_dataset.py --src ./ds --out ./dataset_rebuilt
    python 02_rebuild_dataset.py --src ./ds --out ./dataset_random --protocol random
"""

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "audit_mod", Path(__file__).with_name("01_dataset_audit.py"))
_audit = importlib.util.module_from_spec(_spec)
sys.modules["audit_mod"] = _audit
_spec.loader.exec_module(_audit)

load_records, parse_name, sha256 = _audit.load_records, _audit.parse_name, _audit.sha256
hashes, near_pairs, UF = _audit.hashes, _audit.near_pairs, _audit.UF
CLASS_NAMES = _audit.CLASS_NAMES

import numpy as np


def build_groups(recs, level, phash_dist):
    n = len(recs)
    uf = UF(n)
    key = "scan" if level == "scan" else "case"
    buckets = defaultdict(list)
    by_sha = defaultdict(list)
    for i, r in enumerate(recs):
        buckets[(r["label"], r[key])].append(i)
        by_sha[r["sha"]].append(i)
    for idx in list(buckets.values()) + list(by_sha.values()):
        for j in idx[1:]:
            uf.union(idx[0], j)
    if phash_dist > 0:
        AH = np.stack([hashes(r["path"])[0] for r in recs])
        DH = np.stack([hashes(r["path"])[1] for r in recs])
        a = set((i, j) for i, j, _ in near_pairs(AH, phash_dist))
        d = set((i, j) for i, j, _ in near_pairs(DH, phash_dist))
        for i, j in a & d:
            uf.union(i, j)
    return [uf.find(i) for i in range(n)]


def balanced_group_split(items_by_group, ratios, seed, preassigned=None):
    """Assign whole groups to splits, targeting the image-count ratios.

    Greedy largest-first: place each group in whichever split is currently
    furthest below its target share. Much less lopsided than shuffling when
    groups differ wildly in size (they do here).

    `preassigned` pins groups that were already placed while processing another
    class. Near-duplicate images can carry conflicting labels, which merges two
    classes into one group; without this the same group would land in two
    splits and reintroduce leakage.
    """
    rng = random.Random(seed)
    groups = sorted(items_by_group, key=lambda g: (-len(items_by_group[g]), str(g)))
    total = sum(len(v) for v in items_by_group.values())
    names = ["train", "valid", "test"]
    target = {s: total * r for s, r in zip(names, ratios)}
    have = {s: 0 for s in names}
    out = {}

    preassigned = preassigned or {}
    for g in groups:
        if g in preassigned:
            out[g] = preassigned[g]
            have[out[g]] += len(items_by_group[g])

    free = [g for g in groups if g not in out]
    # guarantee one group each for valid/test when there are enough groups
    if len(free) >= 3:
        mid = free[len(free) // 2:]
        rng.shuffle(mid)
        for s in ("test", "valid"):
            if not any(v == s for v in out.values()) and mid:
                g = mid.pop()
                out[g] = s
                have[s] += len(items_by_group[g])

    for g in groups:
        if g in out:
            continue
        s = max(names, key=lambda k: target[k] - have[k])
        out[g] = s
        have[s] += len(items_by_group[g])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--protocol", choices=["group", "random"], default="group")
    ap.add_argument("--group-level", choices=["scan", "case"], default="case")
    ap.add_argument("--aug-policy", choices=["train-only", "drop-if-original", "keep"],
                    default="train-only")
    ap.add_argument("--phash-dist", type=int, default=4)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--min-cases", type=int, default=3)
    ap.add_argument("--small-class", choices=["train-only", "exclude", "keep"],
                    default="train-only")
    ap.add_argument("--folds", type=int, default=0,
                    help="emit N grouped CV folds (fold_0..fold_N-1) instead of one split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="hardlink instead of copy")
    a = ap.parse_args()

    recs = load_records(a.src)
    for r in recs:
        r["sha"] = sha256(r["path"])
    print(f"Loaded {len(recs)} images")

    groups = build_groups(recs, a.group_level, a.phash_dist)
    for i, r in enumerate(recs):
        r["group"] = groups[i]

    log = []
    P = lambda s="": (print(s), log.append(s))

    # ---- augmentation policy (part 1: applied before splitting) -------------
    def originals_only(subset):
        """One entry per source scan: the true originals, or one stored aug."""
        by_scan = defaultdict(list)
        for r in subset:
            by_scan[(r["label"], r["scan"])].append(r)
        keep = []
        for v in by_scan.values():
            originals = [r for r in v if r["aug"] is None]
            keep += originals if originals else [v[0]]
        return keep

    if a.aug_policy == "drop-if-original":
        before = len(recs)
        recs = originals_only(recs)
        P(f"Augmentation policy 'drop-if-original': {before} -> {len(recs)} images")
    else:
        P(f"Augmentation policy '{a.aug_policy}': carrying all {len(recs)} files "
          f"into the split")

    # ---- small classes ------------------------------------------------------
    cases_per_class = defaultdict(set)
    for r in recs:
        cases_per_class[r["label"]].add(r["group"])
    small = {c for c, s in cases_per_class.items() if len(s) < a.min_cases}
    if small:
        P(f"\nClasses with < {a.min_cases} independent cases: "
          f"{', '.join(f'{c} ({len(cases_per_class[c])})' for c in sorted(small))}")
        P(f"  policy = {a.small_class}")
        if a.small_class == "exclude":
            recs = [r for r in recs if r["label"] not in small]

    # ---- grouped k-fold -----------------------------------------------------
    if a.folds and a.folds > 1:
        rng = random.Random(a.seed)
        fold_of = {}
        for cls in sorted(cases_per_class):
            gs = sorted({r["group"] for r in recs if r["label"] == cls},
                        key=lambda g: -sum(1 for r in recs if r["group"] == g))
            gs = [g for g in gs if g not in fold_of]
            for i, g in enumerate(gs):
                fold_of[g] = i % a.folds
        P(f"\nGrouped {a.folds}-fold CV: {len(fold_of)} groups distributed across folds")
        for k in range(a.folds):
            for r in recs:
                f = fold_of[r["group"]]
                r["split"] = ("test" if f == k
                              else "valid" if f == (k + 1) % a.folds else "train")
            if a.aug_policy == "train-only":
                sub = [r for r in recs if r["split"] == "train"] + \
                      originals_only([r for r in recs if r["split"] != "train"])
            else:
                sub = recs
            fold_dir = a.out / f"fold_{k}"
            for r in sub:
                d = fold_dir / r["split"] / r["label"]
                d.mkdir(parents=True, exist_ok=True)
                dst = d / r["path"].name
                if dst.exists():
                    continue
                if a.link:
                    try:
                        dst.hardlink_to(r["path"]); continue
                    except Exception:
                        pass
                shutil.copy2(r["path"], dst)
            c = Counter(r["split"] for r in sub)
            P(f"  fold_{k}: train={c['train']} valid={c['valid']} test={c['test']}")
        (a.out / "cv_config.json").write_text(json.dumps(
            {"folds": a.folds, "group_level": a.group_level,
             "aug_policy": a.aug_policy, "seed": a.seed}, indent=2))
        (a.out / "rebuild_report.txt").write_text("\n".join(log))
        P(f"\nWrote {a.out}/fold_0..fold_{a.folds-1}")
        return

    # ---- split --------------------------------------------------------------
    if a.protocol == "random":
        rng = random.Random(a.seed)
        rng.shuffle(recs)
        n = len(recs)
        n_tr = int(n * a.ratios[0]); n_va = int(n * a.ratios[1])
        for i, r in enumerate(recs):
            r["split"] = "train" if i < n_tr else ("valid" if i < n_tr + n_va else "test")
        P("\nProtocol: RANDOM image-level split (reproduces the published, leaky protocol)")
    else:
        P(f"\nProtocol: GROUP-AWARE split at {a.group_level} level")
        placed = {}
        for cls in sorted(cases_per_class):
            idx = [r for r in recs if r["label"] == cls]
            if not idx:
                continue
            if cls in small and a.small_class == "train-only":
                for r in idx:
                    r["split"] = "train"
                    placed[r["group"]] = "train"
                continue
            by_g = defaultdict(list)
            for r in idx:
                by_g[r["group"]].append(r)
            assign = balanced_group_split(by_g, a.ratios, a.seed, preassigned=placed)
            placed.update(assign)
            for g, rs in by_g.items():
                for r in rs:
                    r["split"] = assign[g]

    # ---- augmentation policy (part 2: applied after splitting) --------------
    if a.aug_policy == "train-only":
        before = len(recs)
        train = [r for r in recs if r["split"] == "train"]
        evalr = originals_only([r for r in recs if r["split"] != "train"])
        recs = train + evalr
        P(f"Augmentation policy 'train-only': stored augs kept in train; "
          f"valid/test reduced to originals ({before} -> {len(recs)} images)")

    # ---- verify no leakage --------------------------------------------------
    spl = defaultdict(set)
    for r in recs:
        spl[r["group"]].add(r["split"])
    leaks = [g for g, s in spl.items() if len(s) > 1]
    P(f"\nVerification: {len(leaks)} groups span more than one split "
      f"({'PASS' if not leaks or a.protocol == 'random' else 'FAIL'})")

    # ---- write --------------------------------------------------------------
    cnt = Counter((r["split"], r["label"]) for r in recs)
    tot = Counter(r["split"] for r in recs)
    P(f"\n{'class':<28}{'train':>7}{'valid':>7}{'test':>6}")
    P("-" * 48)
    for c in CLASS_NAMES:
        if any(cnt[(s, c)] for s in ("train", "valid", "test")):
            flag = "  (train-only)" if c in small and a.small_class == "train-only" else ""
            P(f"{c:<28}{cnt[('train',c)]:>7}{cnt[('valid',c)]:>7}{cnt[('test',c)]:>6}{flag}")
    P("-" * 48)
    P(f"{'TOTAL':<28}{tot['train']:>7}{tot['valid']:>7}{tot['test']:>6}")

    a.out.mkdir(parents=True, exist_ok=True)
    for r in recs:
        d = a.out / r["split"] / r["label"]
        d.mkdir(parents=True, exist_ok=True)
        dst = d / r["path"].name
        if dst.exists():
            continue
        if a.link:
            try:
                dst.hardlink_to(r["path"]); continue
            except Exception:
                pass
        shutil.copy2(r["path"], dst)

    with open(a.out / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "split", "group", "scan_id", "case_id", "aug"])
        for r in recs:
            w.writerow([r["path"].name, r["label"], r["split"], r["group"],
                        r["scan"], r["case"], r["aug"]])
    (a.out / "rebuild_report.txt").write_text("\n".join(log))
    (a.out / "rebuild_config.json").write_text(json.dumps({
        "protocol": a.protocol, "group_level": a.group_level,
        "aug_policy": a.aug_policy, "ratios": a.ratios, "min_cases": a.min_cases,
        "small_class_policy": a.small_class,
        "train_only_classes": sorted(small) if a.small_class == "train-only" else [],
        "excluded_classes": sorted(small) if a.small_class == "exclude" else [],
        "seed": a.seed, "images": len(recs),
    }, indent=2))
    P(f"\nWrote {a.out}/  (+ split_manifest.csv, rebuild_report.txt, rebuild_config.json)")


if __name__ == "__main__":
    main()
