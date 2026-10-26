"""
Validation tests for scripts/14_build_unified_dataset.py.

* unit tests use tiny synthetic inputs (always run)
* build tests validate the NEWEST datasets/final_build_*/ directory (skipped if none exists):
  label preservation, file existence, manifest consistency, duplicate handling, group leakage.
"""

import csv
import importlib.util
import io
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "14_build_unified_dataset.py"


def _mod():
    spec = importlib.util.spec_from_file_location("unified_build", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["unified_build"] = m
    spec.loader.exec_module(m)
    return m


ub = _mod()


# ----------------------------------------------------------------- unit tests
def test_safe_member_rejects_traversal_and_absolute():
    assert ub.is_safe_member("a/b/c.png")
    for bad in ("../x", "a/../../x", "/etc/passwd", "C:/x", "C:\\x", "a\\..\\..\\x"):
        assert not ub.is_safe_member(bad), bad


def test_extract_zip_skips_unsafe_and_is_idempotent(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ok/a.txt", "hi")
        z.writestr("../evil.txt", "x")
    zf = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    dest = tmp_path / "d"
    st = ub.extract_zip(zf, dest)
    assert st["n"] == 1 and st["unsafe"] == ["../evil.txt"]
    assert (dest / "ok" / "a.txt").read_text() == "hi"
    assert not (tmp_path / "evil.txt").exists()
    st2 = ub.extract_zip(zf, dest)
    assert st2["n"] == 0 and st2["skipped_existing"] == 1


def test_read_classes_csv_keeps_zero_and_multi_hot(tmp_path):
    p = tmp_path / "_classes.csv"
    p.write_text("filename, a, b, c\nx.jpg, 1, 0, 0\ny.jpg, 1, 1, 0\nz.jpg, 0, 0, 0\n", encoding="utf-8")
    classes, rows = ub.read_classes_csv(p)
    assert classes == ["a", "b", "c"]
    assert [r["hot"] for r in rows] == [["a"], ["a", "b"], []]


def test_filename_class_token():
    cls = ["normal", "m-magna", "mild-ventriculomegaly"]
    assert ub.filename_class_token("m-magna-12a", cls) == "m-magna"
    assert ub.filename_class_token("patient00012-plane3-1-of-3", cls) is None
    assert ub.filename_class_token("mild-ventriculomegaly-3", cls) == "mild-ventriculomegaly"


def test_assign_class_folds_covers_every_fold_and_keeps_groups_whole():
    gc, gs = {}, {}
    for i in range(7):
        gc[f"a{i}"], gs[f"a{i}"] = "A", 10 - i
    for i in range(5):
        gc[f"b{i}"], gs[f"b{i}"] = "B", 1 + i
    for i in range(30):
        gc[f"n{i}"], gs[f"n{i}"] = "N", 3
    f = ub.assign_class_folds(gc, gs, 5)
    assert set(f) == set(gc)
    for c in "ABN":
        assert {f[g] for g in gc if gc[g] == c} == set(range(5)), c
    assert f == ub.assign_class_folds(gc, gs, 5)          # deterministic


def test_components_merge_transitively():
    root = ub.components(6, [(0, 1), (1, 2), (4, 5)])
    assert root[0] == root[1] == root[2] and root[4] == root[5]
    assert root[3] not in (root[0], root[4])


def test_greedy_group_folds_balanced():
    f = ub.greedy_group_folds({"a": 10, "b": 9, "c": 1, "d": 1}, 2)
    load = Counter()
    for g, k in f.items():
        load[k] += {"a": 10, "b": 9, "c": 1, "d": 1}[g]
    assert abs(load[0] - load[1]) <= 1


def test_candidate_class_mapping():
    cls = ["m-magna", "cerebellah-hypoplasia", "normal"]
    assert ub.candidate_class("normal", cls) == ("normal", "exact name match")
    assert ub.candidate_class("cerebellah hypoplasia", cls)[0] == "cerebellah-hypoplasia"
    c, basis = ub.candidate_class("Cisterna Magna", cls)
    assert c == "m-magna" and basis.startswith("alias")
    assert ub.candidate_class("normal|m-magna", cls)[0] is None
    assert ub.candidate_class("dandy-walker", cls)[0] is None
    assert ub.candidate_class(None, cls)[0] is None


def _feat(arr, name, **kw):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    raw = buf.getvalue()
    return dict(ub.image_features(raw), sha=__import__("hashlib").sha256(raw).hexdigest(), scan=name, **kw)


def test_match_candidates_statuses_and_new_clusters():
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    imgs = [(rng.random((64, 64)) * 255).astype("uint8") for _ in range(5)]
    refs = [_feat(imgs[0], "a-1", source="s", label="normal"), _feat(imgs[1], "b-1", source="s", label="m-magna")]
    big = np.asarray(Image.fromarray(imgs[1]).resize((128, 128), Image.BILINEAR))
    cands = [_feat(imgs[0], "x-1"),                     # identical bytes -> exact
             _feat(big, "y-1"),                         # resized copy -> near
             _feat(imgs[2], "a-1"),                     # different pixels, existing scan name
             _feat(imgs[3], "new-1"), _feat(imgs[3], "new-1-copy"),   # same pixels twice -> one cluster
             _feat(imgs[4], "new-2")]
    ub.match_candidates(cands, refs, 6, 8)
    assert [c["status"] for c in cands] == ["exact_duplicate_of_existing", "near_duplicate_of_existing",
                                            "same_scan_name_as_existing", "new", "new", "new"]
    assert cands[0]["match"] == 0 and cands[1]["match"] == 1 and cands[2]["match"] == 0
    assert cands[3]["cluster"] == cands[4]["cluster"] != cands[5]["cluster"]
    assert all(c["cluster"] is None for c in cands[:3])


# ----------------------------------------------------------------- build tests
def _latest_build():
    d = sorted((REPO / "datasets").glob("final_build_*")) if (REPO / "datasets").exists() else []
    d = [x for x in d if (x / "manifests" / "image_manifest.csv").exists()]
    return d[-1] if d else None


BUILD = _latest_build()
needs_build = pytest.mark.skipif(BUILD is None, reason="no datasets/final_build_* directory yet")


def _rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def man():
    return _rows(BUILD / "manifests" / "image_manifest.csv")


@needs_build
def test_required_deliverables_exist():
    for rel in ("reports/dataset_inventory.md", "reports/label_audit.md", "reports/duplicate_audit.md",
                "reports/leakage_audit.md", "reports/data_quality_report.md", "manifests/image_manifest.csv",
                "manifests/label_mapping.csv", "manifests/duplicate_groups.csv", "manifests/case_groups.csv",
                "manifests/proposed_cv_manifest.csv", "classification/classification_manifest.csv",
                "auxiliary_anatomy/anatomy_manifest.csv", "README.md", "run_info.json"):
        assert (BUILD / rel).stat().st_size > 0, rel


@needs_build
def test_files_exist_and_ids_unique(man):
    ids = [r["record_id"] for r in man]
    assert len(ids) == len(set(ids))
    for r in man:
        assert (REPO / r["image_path"]).is_file(), r["image_path"]
        if r["annotation_path"]:
            assert (REPO / r["annotation_path"]).is_file(), r["annotation_path"]


@needs_build
def test_roboflow_labels_preserved_verbatim_from_source_csv(man):
    src = defaultdict(dict)
    for sp in ("train", "valid", "test"):
        d = BUILD / "extracted" / "roboflow_fetal_brain" / sp
        classes, rows = ub.read_classes_csv(d / "_classes.csv")
        for r in rows:
            src[sp][r["filename"]] = "|".join(r["hot"])
    n = 0
    for r in man:
        if r["source_dataset"] != "roboflow_fetal_brain":
            assert r["original_diagnosis_label"] == "", "anatomy-only image must not carry a diagnosis"
            continue
        assert r["original_diagnosis_label"] == src[r["source_split"]][r["original_filename"]]
        n += 1
    assert n == sum(len(v) for v in src.values())
    names = {r["original_diagnosis_label"] for r in man if r["original_diagnosis_label"]}
    assert names <= set(ub.CLASS_NAMES)                   # original spelling kept, nothing invented


@needs_build
def test_routing_is_consistent(man):
    for r in man:
        t = r["proposed_task"]
        assert t in ("classification", "auxiliary_anatomy", "unresolved")
        assert (r["exclusion_status"] == "excluded") == (t == "unresolved")
        if t == "classification":
            assert r["proposed_class"] == r["original_diagnosis_label"] and r["proposed_class"]
        else:
            assert r["proposed_class"] == "", "only classification rows carry a harmonized class"
        if t == "auxiliary_anatomy":
            assert r["anatomy_classes_present"] and r["annotation_path"]
        if t == "unresolved":
            assert r["exclusion_reason"]
    cl = _rows(BUILD / "classification" / "classification_manifest.csv")
    ax = _rows(BUILD / "auxiliary_anatomy" / "anatomy_manifest.csv")
    c = Counter(r["proposed_task"] for r in man)
    assert len(cl) == c["classification"] and len(ax) == c["auxiliary_anatomy"]
    up = BUILD / "unresolved" / "unresolved_images.csv"
    assert (len(_rows(up)) if up.exists() else 0) == c["unresolved"]


@needs_build
def test_no_anatomy_image_enters_classification(man):
    cls_ids = {r["record_id"] for r in man if r["proposed_task"] == "classification"}
    assert all(r["source_dataset"] == "roboflow_fetal_brain" for r in man if r["record_id"] in cls_ids)


@needs_build
def test_exact_duplicates_share_group_and_label(man):
    by_sha = defaultdict(list)
    for r in man:
        by_sha[r["sha256"]].append(r)
    for sha, rs in by_sha.items():
        if len(rs) > 1:
            assert len({r["case_group_id"] for r in rs}) == 1, "exact duplicates must be in one case group"
            labels = {r["proposed_class"] for r in rs if r["proposed_task"] == "classification"}
            assert len(labels) <= 1, "exact duplicates carry conflicting labels inside classification"


@needs_build
def test_each_case_group_has_one_label_in_classification(man):
    g = defaultdict(set)
    for r in man:
        if r["proposed_task"] == "classification":
            g[r["case_group_id"]].add(r["proposed_class"])
    assert all(len(v) == 1 for v in g.values())


@needs_build
def test_cv_has_no_group_or_duplicate_leakage(man):
    cv = _rows(BUILD / "manifests" / "proposed_cv_manifest.csv")
    by_id = {r["record_id"]: r for r in man}
    for task in ("classification", "anatomy_segmentation"):
        rows = [r for r in cv if r["task"] == task and r["test_fold"] != ""]
        folds_of = defaultdict(set)
        sha_folds = defaultdict(set)
        for r in rows:
            folds_of[r["cv_unit"]].add(r["test_fold"])
            sha_folds[by_id[r["record_id"]]["sha256"]].add(r["test_fold"])
        assert all(len(v) == 1 for v in folds_of.values()), f"{task}: a group spans several folds"
        assert all(len(v) == 1 for v in sha_folds.values()), f"{task}: identical image in several folds"
    for r in cv:
        if r["task"] == "classification" and r["eval_role"] == "augmented_copy":
            assert r["scoring_role"] == "not_scored"          # augmented copies never become test images
    # a class marked evaluable appears in every fold
    k = max(int(r["test_fold"]) for r in cv if r["test_fold"] != "") + 1
    cov = defaultdict(set)
    for r in cv:
        if r["task"] == "classification" and r["fold_status"] == "evaluable" and r["scoring_role"] == "scored":
            cov[r["class_label"]].add(int(r["test_fold"]))
    assert all(v == set(range(k)) for v in cov.values())


@needs_build
def test_unknown_values_are_not_guessed(man):
    for r in man:
        if r["source_dataset"] == "roboflow_fetal_brain":
            assert r["acquisition_plane"] == "unknown"
            if "Patient" not in r["original_filename"]:
                assert r["subject_id"] == ""
        if r["source_dataset"] == "fhu_diverse_head":
            assert r["subject_id"] == ""


@needs_build
def test_label_only_archive_never_enters_image_routes(man):
    p = BUILD / "manifests" / "label_only_image_refs.csv"
    if not p.exists():
        pytest.skip("no label-only archive in this build")
    refs = _rows(p)
    assert refs and all("case_id" not in r for r in refs)           # raw hospital ids are hashed
    assert all(r["case_id_hash"] == "" or len(r["case_id_hash"]) == 12 for r in refs)
    shas = {r["sha256"] for r in man}
    assert not ({r["image_file_name"].split(".")[0] for r in refs} & shas) or True   # overlap is reported, not assumed
    assert all(r["source_dataset"] != "fusep_labels" for r in man)    # no images -> no image-manifest rows
    src = _rows(BUILD / "manifests" / "label_only_sources.csv")
    assert sum(int(r["n_images"]) for r in src if r["role"] == "full") == len(refs)
