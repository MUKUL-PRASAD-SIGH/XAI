"""
07_build_grouped.py
===================
Builds a DERIVED experiment dataset from an existing leak-free dataset. Touches
nothing that already exists. Two independent uses:

  (a) grouped families  --class-map class_map_grouped.json
  (b) class subset      --drop-classes normal        (no merging at all)

Either may be used alone or together.

The split is not recomputed. Every image keeps the exact train/valid/test
assignment it already has in the source dataset, so the grouped experiment and
the 16-class main experiment are evaluated on the identical case-level
partition and their numbers are directly comparable. Only the class label
changes.

Works with a single split (train/valid/test) or a CV root containing fold_*
directories, detected automatically.

Usage
-----
    python 07_build_grouped.py --like ./dataset_rebuilt \
        --class-map class_map_grouped.json --out ./dataset_grouped --link

    python 07_build_grouped.py --like ./dataset_rebuilt \
        --drop-classes normal --out ./dataset_abnormal --link

Then, using the SAME pipeline as the main experiment (03/04 read
class_names.json automatically, so no script needs editing):

    python 03_train.py    --data ./dataset_grouped --model xception \
                          --out ./models_grouped
    python 04_evaluate.py --data ./dataset_grouped --models ./models_grouped \
                          --out ./results_grouped --tta
"""

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "audit_mod", Path(__file__).with_name("01_dataset_audit.py"))
_audit = importlib.util.module_from_spec(_spec)
sys.modules["audit_mod"] = _audit
_spec.loader.exec_module(_audit)
parse_name = _audit.parse_name

SPLITS = ("train", "valid", "test")
IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp"}


def read_split(root: Path):
    """[(path, original_label, split)] read straight off the directory tree."""
    out = []
    for sp in SPLITS:
        d = root / sp
        if not d.is_dir():
            continue
        for cls_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in IMG_SUFFIX:
                    out.append((p, cls_dir.name, sp))
    return out


def build_one(src_root: Path, out_root: Path, cmap: dict, link: bool, log: list,
              drop=()):
    P = lambda s="": (print(s), log.append(s))
    recs = read_split(src_root)
    if not recs:
        raise SystemExit(f"No images found under {src_root}/train|valid|test")

    if drop:
        before = len(recs)
        recs = [r for r in recs if r[1] not in set(drop)]
        P(f"  dropped {', '.join(drop)}: {before} -> {len(recs)} images")
        if not recs:
            raise SystemExit("every image was dropped")

    if cmap:
        unmapped = sorted({lab for _, lab, _ in recs} - set(cmap))
        if unmapped:
            raise SystemExit(f"--class-map has no entry for: {', '.join(unmapped)}")
    else:  # identity: every class stays independent
        cmap = {lab: lab for _, lab, _ in recs}

    # ---- verify the grouping cannot introduce leakage ----------------------
    case_splits = defaultdict(set)
    for p, lab, sp in recs:
        _, case, _, _ = parse_name(p)
        case_splits[(cmap[lab], case)].add(sp)
    bad = [k for k, v in case_splits.items() if len(v) > 1]
    P(f"  leak check after grouping: {len(bad)} cases span >1 split "
      f"({'PASS' if not bad else 'FAIL'})")
    if bad:
        for k in bad[:5]:
            P(f"    {k[0]} / {k[1]} -> {sorted(case_splits[k])}")

    # ---- materialise -------------------------------------------------------
    for p, lab, sp in recs:
        d = out_root / sp / cmap[lab]
        d.mkdir(parents=True, exist_ok=True)
        dst = d / p.name
        if dst.exists():
            continue
        if link:
            try:
                dst.hardlink_to(p)
                continue
            except Exception:
                pass
        shutil.copy2(p, dst)

    classes = sorted({cmap[lab] for _, lab, _ in recs})
    (out_root / "class_names.json").write_text(json.dumps(classes, indent=2))

    with open(out_root / "grouped_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "original_label", "grouped_label", "split", "case_id"])
        for p, lab, sp in recs:
            _, case, _, _ = parse_name(p)
            w.writerow([p.name, lab, cmap[lab], sp, case])

    # ---- report ------------------------------------------------------------
    cnt = Counter((sp, cmap[lab]) for _, lab, sp in recs)
    cases = defaultdict(set)
    for p, lab, sp in recs:
        _, case, _, _ = parse_name(p)
        cases[cmap[lab]].add(case)
    P(f"\n  {'group':<26}{'train':>7}{'valid':>7}{'test':>6}{'cases':>7}")
    P("  " + "-" * 53)
    for c in classes:
        n_case = len(cases[c])
        flag = "  (train-only: <3 cases)" if n_case < 3 else ""
        P(f"  {c:<26}{cnt[('train',c)]:>7}{cnt[('valid',c)]:>7}"
          f"{cnt[('test',c)]:>6}{n_case:>7}{flag}")
    tot = Counter(sp for _, _, sp in recs)
    P("  " + "-" * 53)
    P(f"  {'TOTAL':<26}{tot['train']:>7}{tot['valid']:>7}{tot['test']:>6}"
      f"{sum(len(v) for v in cases.values()):>7}")
    return classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True, type=Path,
                    help="existing leak-free dataset whose split is inherited")
    ap.add_argument("--class-map", type=Path, default=None,
                    help="optional label -> group mapping. Omit to keep every "
                         "class independent (no merging).")
    ap.add_argument("--drop-classes", nargs="*", default=[],
                    help="labels removed entirely, e.g. normal")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--link", action="store_true", help="hardlink instead of copy")
    a = ap.parse_args()

    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} already exists and is not empty. "
                         f"Refusing to overwrite an existing experiment.")

    if a.class_map:
        raw = json.loads(a.class_map.read_text())
        cmap = {k: v for k, v in raw.items() if not k.startswith("_")}
        review = raw.get("_review")
    else:
        cmap, review = {}, None
    if not a.class_map and not a.drop_classes:
        raise SystemExit("nothing to do: pass --class-map, --drop-classes, or both")

    log = []
    P = lambda s="": (print(s), log.append(s))
    P(f"Source split : {a.like}")
    if a.class_map:
        P(f"Class map    : {a.class_map}  ({len(cmap)} labels -> "
          f"{len(set(cmap.values()))} groups)")
    else:
        P("Class map    : none (every class kept independent, no merging)")
    if a.drop_classes:
        P(f"Dropped      : {', '.join(a.drop_classes)}")
    P(f"Output       : {a.out}")
    if review:
        P("\nGroupings flagged for clinical review before publication:")
        for k, v in review.items():
            P(f"  * {k}\n      {v}")

    folds = sorted(p for p in a.like.glob("fold_*") if p.is_dir())
    if folds:
        P(f"\nCV source detected: {len(folds)} folds")
        for f in folds:
            P(f"\n--- {f.name} ---")
            classes = build_one(f, a.out / f.name, cmap, a.link, log,
                                a.drop_classes)
        (a.out / "class_names.json").write_text(json.dumps(classes, indent=2))
    else:
        P("")
        classes = build_one(a.like, a.out, cmap, a.link, log, a.drop_classes)

    (a.out / "grouping_config.json").write_text(json.dumps({
        "experiment": ("class subset" if not a.class_map else
                       "grouped anomaly families"),
        "dropped_classes": list(a.drop_classes),
        "inherited_split_from": str(a.like),
        "class_map_file": str(a.class_map),
        "class_map": cmap,
        "class_names": classes,
        "note": "Split inherited case-for-case from the source dataset; not recomputed.",
    }, indent=2))
    (a.out / "grouping_report.txt").write_text("\n".join(log))
    P(f"\nWrote {a.out}/ (+ class_names.json, grouped_manifest.csv, "
      f"grouping_config.json, grouping_report.txt)")
    tag = "grouped" if a.class_map else "abnormal"
    P(f"\nNext:\n  python 03_train.py    --data {a.out} --model xception "
      f"--out ./models_{tag}\n  python 04_evaluate.py --data {a.out} "
      f"--models ./models_{tag} --out ./results_{tag} --tta")


if __name__ == "__main__":
    main()
