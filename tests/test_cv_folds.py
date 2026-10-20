"""
End-to-end check of 02b_make_cv_folds.py on a synthetic export: stratification,
group integrity, leakage-free splits, aug-copies only in train.
"""
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "02b_make_cv_folds.py"


def _make_export(root: Path, seed=0):
    """Roboflow-style flat export + _classes.csv. 3 classes with 12/10/6 cases;
    each case has 2 scans; train additionally holds an _aug_1 copy of each scan."""
    rng = np.random.default_rng(seed)
    spec = {"normal": 12, "cyst": 10, "tumor": 6}
    classes = list(spec)
    rows = defaultdict(list)
    n = 0
    for cls, cases in spec.items():
        for c in range(cases):
            for scan in range(2):
                base = f"patient{n:04d}_plane{scan}"
                arr = rng.integers(0, 255, (32, 32), dtype=np.uint8)
                split = "test" if c % 5 == 0 else "train"
                names = [(split, f"{base}.rf.abc{n}.jpg")]
                if split == "train":
                    names.append(("train", f"{base}_aug_1.rf.abc{n}.jpg"))
                for sp, fn in names:
                    (root / sp).mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr).save(root / sp / fn)
                    rows[sp].append((fn, cls))
            n += 1
    for sp, items in rows.items():
        with open(root / sp / "_classes.csv", "w") as f:
            f.write("filename," + ",".join(classes) + "\n")
            for fn, cls in items:
                f.write(fn + "," + ",".join("1" if c == cls else "0" for c in classes) + "\n")
    return spec


def test_stratified_grouped_folds(tmp_path):
    src, out = tmp_path / "ds", tmp_path / "cv"
    _make_export(src)
    r = subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--out", str(out),
                        "--folds", "3", "--trials", "20", "--phash-dist", "0",
                        "--min-groups", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAIL" not in r.stdout

    # manifest-level assertions
    import csv
    rows = list(csv.DictReader(open(out / "cv_manifest.csv")))
    assert {int(x["fold"]) for x in rows} == {0, 1, 2}

    test_fold_of_group = {}
    for x in rows:
        if x["split"] == "test":
            assert x["aug"] in ("", "None"), "augmented copy leaked into test"
            g = x["group"]
            assert test_fold_of_group.setdefault(g, x["fold"]) == x["fold"]

    # no group in two splits within a fold; each group tested in exactly one fold
    for k in "012":
        sp = defaultdict(set)
        for x in rows:
            if x["fold"] == k:
                sp[x["group"]].add(x["split"])
        assert all(len(v) == 1 for v in sp.values())
    all_groups = {x["group"] for x in rows}
    assert set(test_fold_of_group) == all_groups

    # stratified: every class appears in every test fold
    per = Counter((x["label"], x["fold"]) for x in rows if x["split"] == "test")
    for cls in ("normal", "cyst", "tumor"):
        for k in "012":
            assert per[(cls, k)] > 0, f"{cls} missing from test fold {k}"


def _design():
    import importlib.util
    p = Path(__file__).resolve().parent.parent / "scripts" / "02c_design_classes.py"
    spec = importlib.util.spec_from_file_location("dc", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HIER = {"always_separate": ["normal"], "catch_all": "other",
        "families": {"vent": ["mild", "moderate", "severe"], "pf": ["a", "b"], "x": ["lone"]}}


def test_design_merges_small_classes_inside_family():
    d = _design()
    g = {"normal": set(range(30)), "mild": set(range(100, 108)), "moderate": set(range(200, 202)),
         "severe": set(range(300, 303)), "a": set(range(400, 410)), "b": set(range(500, 502)),
         "lone": {900}}
    cmap = d.design(g, HIER, min_groups=5, log=lambda *_: None)
    assert cmap["normal"] == "normal"
    assert cmap["mild"] == "mild"                      # enough cases, stays separate
    assert cmap["moderate"] == cmap["severe"] != "mild"  # 2+3 cases merged together
    assert cmap["a"] == "a" or cmap["a"] == cmap["b"]
    assert cmap["lone"] == "__exclude__"               # 1 case, nowhere to merge -> dropped
    # every kept final class has >= 5 independent cases
    from collections import defaultdict
    fin = defaultdict(set)
    for k, v in cmap.items():
        fin[v] |= g[k]
    assert all(len(v) >= 5 for k, v in fin.items() if k != "__exclude__")


def test_design_confusion_merge():
    d = _design()
    rows = [("mild", "moderate")] * 6 + [("moderate", "mild")] * 6 + [("mild", "mild")] * 4 \
        + [("moderate", "moderate")] * 4 + [("severe", "severe")] * 10
    cl = d.confusion_clusters(rows, ["mild", "moderate", "severe"], 0.25)
    assert sorted(map(sorted, cl)) == [["mild", "moderate"], ["severe"]]
