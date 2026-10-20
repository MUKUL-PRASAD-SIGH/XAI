"""
02c_design_classes.py
=====================
Data-driven class design. Decides, from the number of INDEPENDENT cases each
diagnosis really has, which fine classes must be merged so that every final class
can be cross-validated, and writes the class map that 02b_make_cv_folds.py uses.

Rules (all inside the anatomical families of configs/class_hierarchy.json):
  1. Fine classes inside a family whose mutual confusion is high in a previous
     run (--predictions predictions_xception.csv from 04_evaluate) are merged:
     the model has already shown it cannot tell them apart.
  2. Any class with fewer than --min-groups independent cases (default 5 = the
     number of folds, so it can appear in every test fold) is merged with the
     smallest sibling in its family.
  3. A family that is still too small is pooled into the catch-all class.
  4. Classes in `always_separate` (normal) are never merged.

Nothing is merged across families unless it ends up in the catch-all, and every
merge is logged with the case counts that forced it.

Usage
-----
    python 02c_design_classes.py --src ./ds --out ./class_map_auto.json
    python 02c_design_classes.py --src ./ds --out ./class_map_auto.json \
        --predictions ./results_cv/predictions_xception.csv
"""

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# pure logic -- unit-tested
# ==========================================================================
EXCLUDE = "__exclude__"   # 02b drops images mapped to this


def confusion_clusters(pred_rows, members, thr):
    """Union fine classes of one family whose symmetric confusion rate >= thr."""
    parent = {m: m for m in members}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sup = Counter(t for t, _ in pred_rows)
    pair = Counter((t, p) for t, p in pred_rows if t != p)
    for i in members:
        for j in members:
            if i < j and sup[i] and sup[j]:
                rate = (pair[(i, j)] / sup[i] + pair[(j, i)] / sup[j]) / 2
                if rate >= thr:
                    parent[find(i)] = find(j)
    out = defaultdict(list)
    for m in members:
        out[find(m)].append(m)
    return [sorted(v) for v in out.values()]


def design(groups_by_class, hierarchy, min_groups, pred_rows=None, conf_thr=0.25, log=print):
    """groups_by_class: {fine: set(group ids)}. Returns {fine: final_class}."""
    sep = set(hierarchy.get("always_separate", []))
    catch = hierarchy.get("catch_all", "other_abnormal")
    fams = hierarchy["families"]
    fam_of = {m: f for f, ms in fams.items() for m in ms}
    cmap, leftovers = {}, []

    for c in sorted(groups_by_class):
        if c in sep:
            cmap[c] = c
            if len(groups_by_class[c]) < min_groups:
                log(f"  WARNING: '{c}' has only {len(groups_by_class[c])} independent cases")

    for fam, members in fams.items():
        present = [m for m in members if m in groups_by_class and m not in sep]
        if not present:
            continue
        clusters = ([list(c) for c in confusion_clusters(pred_rows, present, conf_thr)]
                    if pred_rows else [[m] for m in present])
        for c in clusters:
            if len(c) > 1:
                log(f"  [{fam}] merged by model confusion: {', '.join(c)}")

        def n_groups(cl):
            return len(set().union(*(groups_by_class[m] for m in cl)))

        while True:
            small = [c for c in clusters if n_groups(c) < min_groups]
            if not small:
                break
            s = min(small, key=n_groups)
            others = [c for c in clusters if c is not s]
            if not others:
                log(f"  [{fam}] still only {n_groups(s)} cases after merging -> catch-all")
                leftovers += s
                clusters = []
                break
            t = min(others, key=n_groups)
            log(f"  [{fam}] {', '.join(s)} ({n_groups(s)} cases) < {min_groups} -> merged "
                f"with {', '.join(t)} ({n_groups(t)} cases)")
            clusters = [c for c in clusters if c is not s and c is not t] + [s + t]

        for cl in clusters:
            if len(cl) == 1:
                name = cl[0]
            elif set(cl) == set(members):
                name = fam
            else:
                name = fam + "_" + "+".join(x.split("-")[0] for x in sorted(cl))
            for m in cl:
                cmap[m] = name

    # classes not described by the hierarchy stay as they are, flagged
    for c in groups_by_class:
        if c not in cmap and c not in leftovers:
            if c in fam_of:
                continue
            log(f"  '{c}' is not in the hierarchy; kept as its own class")
            cmap[c] = c
    if leftovers:
        pooled = set().union(*(groups_by_class[m] for m in leftovers))
        if len(pooled) < min_groups:
            log(f"  catch-all '{catch}' would have only {len(pooled)} cases; these classes "
                f"cannot be cross-validated and are EXCLUDED: {', '.join(sorted(leftovers))}")
            for m in leftovers:
                cmap[m] = EXCLUDE
        else:
            for m in leftovers:
                cmap[m] = catch
    return cmap


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hierarchy", type=Path,
                    default=Path(__file__).resolve().parent.parent / "configs" / "class_hierarchy.json")
    ap.add_argument("--min-groups", type=int, default=5)
    ap.add_argument("--predictions", type=Path, default=None,
                    help="predictions_<model>.csv from 04_evaluate (enables confusion merging)")
    ap.add_argument("--confusion-thr", type=float, default=0.25)
    ap.add_argument("--phash-dist", type=int, default=4)
    a = ap.parse_args()

    cv = _load("cv_mod", "02b_make_cv_folds.py")
    log_lines = []

    def P(s=""):
        print(s)
        log_lines.append(s)

    recs = cv.load_records(a.src)
    for r in recs:
        r["sha"] = cv.sha256(r["path"])
    groups = cv.build_groups(recs, "case", a.phash_dist)
    for r, g in zip(recs, groups):
        r["group"] = g
    gbc = defaultdict(set)
    for r in recs:
        gbc[r["label"]].add(r["group"])
    P("Independent cases per fine class:")
    for c, g in sorted(gbc.items(), key=lambda kv: len(kv[1])):
        P(f"  {c:<30}{len(g):>5}")

    hierarchy = json.loads(a.hierarchy.read_text())
    pred_rows = None
    if a.predictions:
        with open(a.predictions, newline="") as f:
            pred_rows = [(r["true"], r["pred"]) for r in csv.DictReader(f)]
        P(f"\nUsing {len(pred_rows)} predictions from {a.predictions} for confusion merging")

    P(f"\nDesign (min {a.min_groups} independent cases per class):")
    cmap = design(gbc, hierarchy, a.min_groups, pred_rows, a.confusion_thr, P)

    final = defaultdict(set)
    members = defaultdict(list)
    for fine, fin in cmap.items():
        final[fin] |= gbc[fine]
        members[fin].append(fine)
    P("\nFinal classes:")
    for fin in sorted(final, key=lambda k: len(final[k])):
        P(f"  {fin:<46}{len(final[fin]):>5} cases  <- {', '.join(sorted(members[fin]))}")

    out = {"_review": {k: "merged: " + ", ".join(sorted(v))
                       for k, v in members.items() if len(v) > 1}}
    out.update(cmap)
    a.out.write_text(json.dumps(out, indent=2))
    a.out.with_suffix(".report.txt").write_text("\n".join(log_lines))
    P(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
