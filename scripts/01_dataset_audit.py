"""
01_dataset_audit.py
===================
READ-ONLY audit of the Roboflow "Fetal Brain Abnormalities Ultrasound v1"
multiclass export. Changes nothing on disk except the report files it writes.

Answers, with evidence:
  1. How many UNIQUE source scans are behind the 1,768 images?
  2. Do augmented copies of the same scan appear in more than one split?
  3. Are there exact-duplicate and near-duplicate images?
  4. Do duplicate images carry conflicting labels?
  5. Do the classes come from more than one source dataset? (shortcut risk)

Usage
-----
    python 01_dataset_audit.py --src ./ds
    python 01_dataset_audit.py --src ./ds --out ./audit --phash-dist 4
"""

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = [
    "anold-chiari-malformation", "arachnoid-cyst", "cerebellah-hypoplasia",
    "colphocephaly", "encephalocele", "holoprosencephaly", "hydracenphaly",
    "intracranial-hemorrdge", "intracranial-tumor", "m-magna",
    "mild-ventriculomegaly", "moderate-ventriculomegaly", "normal",
    "polencephaly", "severe-ventriculomegaly", "vein-of-galen",
]

# ---------------------------------------------------------------- filenames
RF_HASH = re.compile(r"\.rf\.[0-9a-f]+$", re.I)
AUG_TAG = re.compile(r"[_-]aug[_-]?(\d+)", re.I)
EXT_TAG = re.compile(r"[_-](png|jpg|jpeg)(?=[_-]|$)", re.I)
COPY_TAG = re.compile(r"^copy[_\- ]of[_\- ]", re.I)
IMAGE_TAG = re.compile(r"[_-]image$", re.I)
PATIENT = re.compile(r"^patient0*(\d+)", re.I)
CASE_TAIL = re.compile(r"(\d+)[a-z]*$", re.I)


def parse_name(path: Path):
    """-> (scan_id, case_id, aug_index_or_None, family)"""
    stem = RF_HASH.sub("", path.stem)
    aug_m = AUG_TAG.search(stem)
    aug = int(aug_m.group(1)) if aug_m else None

    s = AUG_TAG.sub("", stem)
    s = EXT_TAG.sub("", s)
    s = COPY_TAG.sub("", s)
    s = re.sub(r"[_\s]+", "-", s).strip("-").lower()
    s = IMAGE_TAG.sub("", s).strip("-")
    scan_id = s

    pm = PATIENT.match(s)
    if pm:                      # Fetal Planes DB naming: one patient, many planes
        case_id = f"patient{int(pm.group(1)):05d}"
        family = "patient"
    else:                       # Copy-of-<class>-<case><letter>
        case_id = CASE_TAIL.sub(r"\1", s)
        family = "copy-of"
    return scan_id, case_id, aug, family


# ---------------------------------------------------------------- hashing
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def hashes(path: Path):
    """(aHash, dHash) packed as 8 bytes each."""
    try:
        g = Image.open(path).convert("L")
    except Exception:
        return np.zeros(8, np.uint8), np.zeros(8, np.uint8)
    a = np.asarray(g.resize((8, 8), Image.BILINEAR), np.float32)
    ah = np.packbits((a > a.mean()).reshape(-1).astype(np.uint8))
    d = np.asarray(g.resize((9, 8), Image.BILINEAR), np.float32)
    dh = np.packbits((d[:, 1:] > d[:, :-1]).reshape(-1).astype(np.uint8))
    return ah, dh


POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def near_pairs(H, max_dist, chunk=384):
    out = []
    n = len(H)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = POPCOUNT[np.bitwise_xor(H[s:e, None, :], H[None, :, :])].sum(2)
        ii, jj = np.where(d <= max_dist)
        out += [(int(i + s), int(j), int(d[i, j])) for i, j in zip(ii, jj) if i + s < j]
    return out


class UF:
    def __init__(s, n): s.p = list(range(n))
    def find(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb: s.p[rb] = ra


# ---------------------------------------------------------------- loading
def load_records(src: Path):
    recs = []
    for split in ("train", "valid", "test"):
        d = src / split
        if not d.is_dir():
            continue
        csv_path = d / "_classes.csv"
        labels = {}
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                rd = csv.reader(f)
                head = [h.strip() for h in next(rd)]
                for row in rd:
                    if not row:
                        continue
                    fn = row[0].strip()
                    hot = [head[i] for i, v in enumerate(row) if i and v.strip() == "1"]
                    if hot:
                        labels[fn] = hot[0]
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lab = labels.get(p.name)
            if lab is None:                       # folder-structure export
                lab = p.parent.name
            scan, case, aug, fam = parse_name(p)
            recs.append(dict(path=p, split=split, label=lab, scan=scan,
                             case=case, aug=aug, family=fam))
    return recs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("./audit"))
    ap.add_argument("--phash-dist", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    recs = load_records(a.src)
    n = len(recs)
    print(f"Loaded {n} images from {a.src}\n")

    for r in recs:
        r["sha"] = sha256(r["path"])
    AH = np.stack([h[0] for h in (hashes(r["path"]) for r in recs)])
    DH = np.stack([hashes(r["path"])[1] for r in recs])

    # ---- exact duplicates
    by_sha = defaultdict(list)
    for i, r in enumerate(recs):
        by_sha[r["sha"]].append(i)
    exact = [v for v in by_sha.values() if len(v) > 1]
    n_exact_extra = sum(len(v) - 1 for v in exact)
    exact_cross_split = sum(1 for v in exact if len({recs[i]["split"] for i in v}) > 1)
    exact_cross_label = sum(1 for v in exact if len({recs[i]["label"] for i in v}) > 1)

    # ---- near duplicates (both hashes must agree)
    ap_ = set((i, j) for i, j, _ in near_pairs(AH, a.phash_dist))
    dp_ = set((i, j) for i, j, _ in near_pairs(DH, a.phash_dist))
    near = sorted(ap_ & dp_)
    near_cross_split = [(i, j) for i, j in near if recs[i]["split"] != recs[j]["split"]]
    near_cross_label = [(i, j) for i, j in near if recs[i]["label"] != recs[j]["label"]]

    # ---- grouping
    def grouping(level):
        uf = UF(n)
        key = "scan" if level == "scan" else "case"
        buckets = defaultdict(list)
        for i, r in enumerate(recs):
            buckets[(r["label"], r[key])].append(i)
        for idx in buckets.values():
            for j in idx[1:]:
                uf.union(idx[0], j)
        for v in by_sha.values():
            for j in v[1:]:
                uf.union(v[0], j)
        for i, j in near:
            uf.union(i, j)
        return [uf.find(i) for i in range(n)]

    g_scan, g_case = grouping("scan"), grouping("case")

    def leak_stats(groups):
        spl = defaultdict(set)
        for i, g in enumerate(groups):
            spl[g].add(recs[i]["split"])
        multi = {g: s for g, s in spl.items() if len(s) > 1}
        pairs = Counter()
        for s in multi.values():
            for a_, b_ in (("train", "test"), ("train", "valid"), ("valid", "test")):
                if a_ in s and b_ in s:
                    pairs[f"{a_}<->{b_}"] += 1
        imgs_in_leaky = sum(1 for g in groups if g in multi)
        return len(set(groups)), len(multi), pairs, imgs_in_leaky

    ns_scan, leak_scan, pairs_scan, imgs_scan = leak_stats(g_scan)
    ns_case, leak_case, pairs_case, imgs_case = leak_stats(g_case)

    # ---- per class
    per_class = []
    for c in CLASS_NAMES:
        idx = [i for i, r in enumerate(recs) if r["label"] == c]
        if not idx:
            continue
        per_class.append((c, len(idx),
                          len({g_scan[i] for i in idx}),
                          len({g_case[i] for i in idx}),
                          Counter(recs[i]["family"] for i in idx)))

    # ---- report
    L = []
    P = lambda s="": (print(s), L.append(s))

    P("=" * 72); P("DATASET AUDIT — Fetal Brain Abnormalities Ultrasound v1"); P("=" * 72)
    P(f"Images                          : {n}")
    P(f"Split sizes                     : " +
      ", ".join(f"{k}={v}" for k, v in Counter(r['split'] for r in recs).items()))
    P(f"Images carrying an _aug_N tag   : {sum(1 for r in recs if r['aug'] is not None)}")
    P(f"Images with no _aug tag         : {sum(1 for r in recs if r['aug'] is None)}")
    P("")
    P(f"Unique source scans  (scan level): {ns_scan}")
    P(f"Unique cases/patients (case level): {ns_case}")
    P("")
    P("-" * 72); P("DUPLICATES"); P("-" * 72)
    P(f"Byte-identical duplicate images : {n_exact_extra} extra copies in {len(exact)} groups")
    P(f"  ... spanning >1 split         : {exact_cross_split} groups")
    P(f"  ... with conflicting labels   : {exact_cross_label} groups")
    P(f"Near-duplicate pairs (aHash+dHash <= {a.phash_dist}) : {len(near)}")
    P(f"  ... across different splits   : {len(near_cross_split)}")
    P(f"  ... across different labels   : {len(near_cross_label)}")
    P("")
    P("-" * 72); P("LEAKAGE IN THE PUBLISHED SPLIT"); P("-" * 72)
    P(f"[scan level] groups split across >1 partition : {leak_scan}/{ns_scan} "
      f"({100*leak_scan/max(ns_scan,1):.1f}%)   images affected: {imgs_scan}/{n} "
      f"({100*imgs_scan/n:.1f}%)")
    for k, v in pairs_scan.most_common():
        P(f"    {k:<16}: {v} groups")
    P(f"[case level] groups split across >1 partition : {leak_case}/{ns_case} "
      f"({100*leak_case/max(ns_case,1):.1f}%)   images affected: {imgs_case}/{n} "
      f"({100*imgs_case/n:.1f}%)")
    for k, v in pairs_case.most_common():
        P(f"    {k:<16}: {v} groups")
    P("")
    P("-" * 72); P("PER CLASS"); P("-" * 72)
    P(f"{'class':<28}{'imgs':>6}{'scans':>7}{'cases':>7}  source")
    for c, ni, nsc, nca, fam in per_class:
        src = "+".join(f"{k}:{v}" for k, v in fam.most_common())
        P(f"{c:<28}{ni:>6}{nsc:>7}{nca:>7}  {src}")
    P("")
    tot_scan = sum(x[2] for x in per_class); tot_case = sum(x[3] for x in per_class)
    P(f"{'TOTAL':<28}{n:>6}{tot_scan:>7}{tot_case:>7}")

    (a.out / "audit_report.txt").write_text("\n".join(L))
    with open(a.out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split", "label", "scan_id", "case_id", "aug", "family",
                    "scan_group", "case_group", "sha256"])
        for i, r in enumerate(recs):
            w.writerow([r["path"].name, r["split"], r["label"], r["scan"], r["case"],
                        r["aug"], r["family"], g_scan[i], g_case[i], r["sha"]])
    with open(a.out / "duplicates.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kind", "file_a", "split_a", "label_a",
                                       "file_b", "split_b", "label_b"])
        for v in exact:
            for j in v[1:]:
                w.writerow(["exact", recs[v[0]]["path"].name, recs[v[0]]["split"], recs[v[0]]["label"],
                            recs[j]["path"].name, recs[j]["split"], recs[j]["label"]])
        for i, j in near:
            w.writerow(["near", recs[i]["path"].name, recs[i]["split"], recs[i]["label"],
                        recs[j]["path"].name, recs[j]["split"], recs[j]["label"]])
    print(f"\nWrote {a.out}/audit_report.txt, manifest.csv, duplicates.csv")


if __name__ == "__main__":
    main()

