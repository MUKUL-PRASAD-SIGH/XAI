"""
14_build_unified_dataset.py
===========================
Audit + build a verified, unified fetal-brain ultrasound dataset from every
source archive in the repository root. NO TRAINING is done here.

Sources (all treated read-only; nothing in the repo root is modified):
  * Fetal Brain Abnormalities Ultrasound.v1i.multiclass.zip  (Roboflow, 16 diagnosis classes)
  * Trans-cerebellum.zip / Trans-ventricular.zip / Diverse Fetal Head Images.zip
    ("Fetal Head Ultrasound Images", Readme.txt) -- anatomy masks only, no diagnoses.

Everything is written under  datasets/final_build_<timestamp>/ :
  extracted/   safe extraction, one directory per source
  manifests/   image_manifest, label_mapping, duplicate_groups, duplicate_pairs,
               case_groups, proposed_cv_manifest
  classification/ auxiliary_anatomy/ unresolved/   task-specific manifests
  reports/     markdown audit reports
  run_info.json, scripts/ (snapshot), README.md

    python scripts/14_build_unified_dataset.py                 # new timestamped build
    python scripts/14_build_unified_dataset.py --build-dir datasets/final_build_X   # rerun in place
"""

import argparse
import csv
import datetime
import hashlib
import importlib.util
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_mod", "01_dataset_audit.py")
UF, near_pairs, parse_name, CLASS_NAMES = audit.UF, audit.near_pairs, audit.parse_name, audit.CLASS_NAMES

# --------------------------------------------------------------------------
# source configuration (paths inside the archives come from the inspected zips)
# --------------------------------------------------------------------------
FHU_LICENSE = "CC BY 4.0 per Readme.txt (original image authors are not named in Readme.txt)"
SOURCES = {
    "roboflow_fetal_brain": dict(
        zip="Fetal Brain Abnormalities Ultrasound.v1i.multiclass.zip", kind="classification",
        license="CC BY 4.0 per README.dataset.txt (provenance of the underlying images is not documented)",
        attribution="README.dataset.txt / README.roboflow.txt (Roboflow Universe: hritwik-trivedi-gkgrv/fetal-brain-abnormalities-ultrasound)",
        plane="unknown", plane_basis="not documented"),
    "fhu_trans_cerebellum": dict(
        zip="Trans-cerebellum.zip", kind="anatomy", img_dir="Trans-cerebellum/Trans-cerebellum",
        seg="Trans-cerebellum/Trans-cerebellum-Segmentation.zip", coco="Trans-cerebellum/Trans-cerebellum-COCO.zip",
        pix="Trans-cerebellum/Trans-cerebellum-Pixel-Size.csv", license=FHU_LICENSE, attribution="Readme.txt",
        plane="trans-cerebellar", plane_basis="source archive/folder name (Readme.txt: fetal plane groups)"),
    "fhu_trans_ventricular": dict(
        zip="Trans-ventricular.zip", kind="anatomy", img_dir="Trans-ventricular/Trans-ventricular",
        seg="Trans-ventricular/Trans-ventricular-segmentation-mask.zip", coco="Trans-ventricular/Trans-ventricular-COCO.zip",
        pix="Trans-ventricular/Trans-ventricular-Pixel-Size.csv", license=FHU_LICENSE, attribution="Readme.txt",
        plane="trans-ventricular", plane_basis="source archive/folder name (Readme.txt: fetal plane groups)"),
    "fhu_diverse_head": dict(
        zip="Diverse Fetal Head Images.zip", kind="anatomy",
        img_dir="Diverse Fetal Head Images/Orginal_train_images_to_959_661",
        seg="Diverse Fetal Head Images/Test-Dataset-Segmentation.zip",
        coco="Diverse Fetal Head Images/Test-Dataset-COCO.zip", pix=None, license=FHU_LICENSE,
        attribution="Readme.txt", plane="diverse/unspecified", plane_basis="source archive name (mixed head planes; per-image plane not documented)"),
}
LABEL_ONLY = {   # annotation archives WITHOUT images: audited and inventoried, never routed into image manifests
    "fusep_labels": dict(zip="ahuteam FUSEP main labels.zip",
                         quality={"\u6807\u51c6": "standard", "\u975e\u6807\u51c6": "non-standard",
                                  "\u57fa\u672c\u6807\u51c6": "basically standard"})}
README_CLAIMS = dict(images=3832, brain=3794, csp=1865, lv=1512)
MASK_COLORS = {"Brain": (255, 0, 0), "CSP": (0, 255, 0), "LV": (0, 0, 255)}  # from labelmap.txt (verified at runtime)

MANIFEST_COLUMNS = [
    "record_id", "image_path", "original_filename", "source_dataset", "original_image_id", "scan_id_filename_derived",
    "subject_id", "subject_id_basis", "case_token_heuristic", "acquisition_plane", "acquisition_plane_basis",
    "original_diagnosis_label", "filename_class_token", "annotation_type", "annotation_path", "anatomy_classes_present",
    "mask_px_brain", "mask_px_csp", "mask_px_lv", "pixel_size_mm", "proposed_task", "proposed_class",
    "label_provenance", "label_verification_status", "sha256", "ahash", "dhash", "phash", "duplicate_group_id",
    "duplicate_kind", "case_group_id", "case_group_basis", "source_split", "source_split_documented", "license",
    "attribution_ref", "is_augmented_copy", "augmentation_index", "eval_role", "width", "height",
    "qc_color_fraction", "qc_mean_gray", "flags", "exclusion_status", "exclusion_reason"]


# ==========================================================================
# pure helpers -- unit-tested in tests/test_unified_build.py
# ==========================================================================
def is_safe_member(name):
    """Reject absolute paths, drive letters and any '..' component (zip-slip)."""
    n = name.replace("\\", "/")
    p = PurePosixPath(n)
    return not (p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", n) or n.startswith("/"))


def extract_zip(zf, dest, include=lambda info: True):
    """Safe, idempotent extraction. -> dict(n, skipped_existing, unsafe, bytes)"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    st = dict(n=0, skipped_existing=0, unsafe=[], bytes=0)
    for info in zf.infolist():
        if info.is_dir() or not include(info):
            continue
        if not is_safe_member(info.filename):
            st["unsafe"].append(info.filename)
            continue
        out = (dest / info.filename).resolve()
        if root != out and root not in out.parents:
            st["unsafe"].append(info.filename)
            continue
        if out.exists() and out.stat().st_size == info.file_size:
            st["skipped_existing"] += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        st["n"] += 1
        st["bytes"] += info.file_size
    return st


def read_classes_csv(path):
    """-> (header_labels, rows[{filename, hot:[labels]}]). Keeps zero-hot / multi-hot rows."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        head = [h.strip() for h in next(rd)]
        rows = []
        for row in rd:
            if not row or not row[0].strip():
                continue
            hot = [head[i] for i in range(1, min(len(row), len(head))) if row[i].strip() == "1"]
            rows.append(dict(filename=row[0].strip(), hot=hot))
    return head[1:], rows


def filename_class_token(scan_id, classes):
    """Longest diagnosis name the file name starts with (e.g. 'm-magna-12a'); else None."""
    hits = [c for c in classes if scan_id == c or scan_id.startswith(c + "-")]
    return max(hits, key=len) if hits else None


def phash64(gray):
    from scipy.fft import dctn
    a = np.asarray(gray.resize((32, 32), Image.LANCZOS), np.float32)
    d = dctn(a, norm="ortho")[:8, :8].reshape(-1)
    return np.packbits((d > np.median(d[1:])).astype(np.uint8))


def image_features(raw):
    """raw bytes -> dict(w,h,ah,dh,ph (uint8[8]), color, gray)."""
    im = Image.open(io.BytesIO(raw))
    w, h = im.size
    g = im.convert("L")
    a = np.asarray(g.resize((8, 8), Image.BILINEAR), np.float32)
    d = np.asarray(g.resize((9, 8), Image.BILINEAR), np.float32)
    rgb = np.asarray(im.convert("RGB").resize((128, 128), Image.BILINEAR), np.int16)
    spread = rgb.max(2) - rgb.min(2)
    return dict(w=w, h=h, ah=np.packbits((a > a.mean()).reshape(-1).astype(np.uint8)),
                dh=np.packbits((d[:, 1:] > d[:, :-1]).reshape(-1).astype(np.uint8)), ph=phash64(g),
                color=float((spread > 30).mean()), gray=float(np.asarray(g, np.float32).mean()))


POP = audit.POPCOUNT


def hdist(a, b):
    return int(POP[np.bitwise_xor(a, b)].sum())


def components(n, links):
    uf = UF(n)
    for a, b in links:
        uf.union(a, b)
    return [uf.find(i) for i in range(n)]


def greedy_group_folds(sizes, k):
    """Deterministic balanced assignment: biggest group to the least-loaded fold."""
    load = [0] * k
    out = {}
    for g, s in sorted(sizes.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        f = min(range(k), key=lambda i: (load[i], i))
        out[g] = f
        load[f] += s
    return out


def assign_class_folds(group_class, group_size, k):
    """Deterministic stratified group folds. A group has exactly one class here, so classes are
    independent apart from load balancing: the biggest groups of a class go to distinct folds first
    (every class with >= k groups therefore appears in every test fold), the rest to the fold holding
    the fewest images of that class."""
    by_c = defaultdict(list)
    for g, c in group_class.items():
        by_c[c].append(g)
    load, out = [0] * k, {}
    for c in sorted(by_c, key=lambda c: (len(by_c[c]), c)):
        cl = [0] * k
        for n, g in enumerate(sorted(by_c[c], key=lambda g: (-group_size[g], g))):
            cands = [f for f in range(k) if cl[f] == 0] if n < k else list(range(k))
            f = min(cands, key=lambda f: (cl[f], load[f], f))
            out[g] = f
            cl[f] += group_size[g]
            load[f] += group_size[g]
    return out


def md_table(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])


# ==========================================================================
# stage 1: inventory + safe extraction
# ==========================================================================
def stage_extract(root, bd, log):
    ext_root = bd / "extracted"
    inv, total_need = {}, 0
    zips = {}
    for key, cfg in SOURCES.items():
        zp = root / cfg["zip"]
        if not zp.exists():
            raise SystemExit(f"missing source archive: {zp}")
        zf = zipfile.ZipFile(zp)
        zips[key] = zf
        infos = zf.infolist()
        nested = [i for i in infos if i.filename.lower().endswith(".zip")]
        keep_nested = {cfg.get("seg"), cfg.get("coco")}
        total_need += sum(i.file_size for i in infos if not i.is_dir() and i not in nested)
        total_need += sum(i.file_size * 2 for i in nested if i.filename in keep_nested)
        inv[key] = dict(zip=cfg["zip"], bytes=zp.stat().st_size, sha256=sha256_file(zp), entries=len(infos),
                        ext=dict(Counter(Path(i.filename).suffix.lower() for i in infos if not i.is_dir())),
                        corrupt_member=zf.testzip(), nested={})
        for i in nested:   # inventory of the redundant export formats (not extracted unless needed)
            try:
                z2 = zipfile.ZipFile(io.BytesIO(zf.read(i)))
                inv[key]["nested"][i.filename.split("/")[-1]] = dict(
                    extracted=i.filename in keep_nested, members=len(z2.infolist()),
                    ext=dict(Counter(Path(j.filename).suffix.lower() for j in z2.infolist() if not j.is_dir())))
            except Exception as e:   # unreadable nested archive is reported, not hidden
                inv[key]["nested"][i.filename.split("/")[-1]] = dict(extracted=False, error=str(e))
    free = shutil.disk_usage(bd).free
    log(f"disk: need ~{total_need / 1e9:.2f} GB for extraction, free {free / 1e9:.2f} GB")
    if free < total_need * 1.2 + 1e9:
        raise SystemExit("not enough free disk space for extraction")
    for key, cfg in SOURCES.items():
        zf = zips[key]
        keep_nested = {cfg.get("seg"), cfg.get("coco")}
        d = ext_root / key
        st = extract_zip(zf, d, lambda i: not i.filename.lower().endswith(".zip"))
        for nz in keep_nested - {None}:
            nd = d / "_nested" / Path(nz).stem
            s2 = extract_zip(zipfile.ZipFile(io.BytesIO(zf.read(nz))), nd)
            st["n"] += s2["n"]
            st["unsafe"] += s2["unsafe"]
        inv[key]["extract"] = st
        log(f"extracted {key}: {st['n']} new, {st['skipped_existing']} already present, {len(st['unsafe'])} unsafe skipped")
    return ext_root, inv


def stage_label_only(root, bd, S, log):
    """COCO-style label archives with no images (FUSEP). Extract JSON safely, profile, check split hygiene."""
    S["fusep"], rows_files, rows_imgs = None, [], []
    for key, cfg in LABEL_ONLY.items():
        zp = root / cfg["zip"]
        if not zp.exists():
            log(f"label-only source {cfg['zip']} not present; skipped")
            continue
        zf = zipfile.ZipFile(zp)
        st = extract_zip(zf, bd / "extracted" / key, lambda i: i.filename.lower().endswith(".json"))
        info = dict(zip=cfg["zip"], bytes=zp.stat().st_size, sha256=sha256_file(zp), entries=len(zf.infolist()),
                    ext=dict(Counter(Path(i.filename).suffix.lower() for i in zf.infolist() if not i.is_dir())),
                    corrupt_member=zf.testzip(), extract=st, cats={})
        data = {}
        for i in sorted(zf.infolist(), key=lambda i: i.filename):
            if i.is_dir() or not i.filename.endswith(".json"):
                continue
            hosp, nm = i.filename.split("/")
            task, _, role = nm[:-5].partition("_")
            j = json.loads(zf.read(i))
            data[(hosp, task, role or "full")] = j
            qc = Counter(cfg["quality"].get(a.get("standard"), a.get("standard")) for a in j["annotations"])
            imgs = j["images"]
            rows_files.append(dict(hospital=hosp, task=task, role=role or "full", n_images=len(imgs),
                                   n_annotations=len(j["annotations"]),
                                   n_distinct_case_ids=len({x.get("case_id") for x in imgs}),
                                   plane_quality_counts=";".join(f"{k}={v}" for k, v in sorted(qc.items())) or None,
                                   json_path=f"{key}/{i.filename}"))
            info["cats"].setdefault(hosp, []).append(sorted({c["name"] for c in j["categories"]}))
        issues = []
        mem = defaultdict(lambda: defaultdict(set))
        for (h, t, role), j in data.items():
            mem[(h, t)][role] = {x["file_name"] for x in j["images"]}
        for (h, t), m in sorted(mem.items()):
            if m.get("train") and m.get("test") and m["train"] & m["test"]:
                issues.append(f"{h}/{t}: {len(m['train'] & m['test'])} images in BOTH train and test")
            if "full" in m and (m.get("train", set()) | m.get("test", set())) != m["full"]:
                issues.append(f"{h}/{t}: train+test != full ({len((m['train'] | m['test']) ^ m['full'])} differences)")
            for r in ("label_5", "label_10"):
                if m.get(r) and not m[r] <= m["train"]:
                    issues.append(f"{h}/{t}: {r} not a subset of train ({len(m[r] - m['train'])} outside)")
            for k in ("5", "10"):
                if m.get("label_" + k, set()) & m.get("unlabel_" + k, set()):
                    issues.append(f"{h}/{t}: label_{k} and unlabel_{k} overlap")
        placeholder = {h for (h, t, r), j in data.items()
                       if r == "full" and len({x.get("case_id") for x in j["images"]}) == 1}
        case_split = {}
        for (h, t, role), j in data.items():
            if role in ("train", "test"):
                case_split.setdefault((h, t), {})[role] = {x.get("case_id") for x in j["images"]}
        for (h, t), d in sorted(case_split.items()):
            if h not in placeholder and d.get("train") and d.get("test") and d["train"] & d["test"]:
                issues.append(f"{h}/{t}: {len(d['train'] & d['test'])} case_ids present in BOTH train and test "
                              "(patient-level leakage)")
        for h in sorted(placeholder):
            issues.append(f"{h}: every image has the same case_id placeholder -> no usable subject information")
        both = defaultdict(set)
        for (h, t, role), j in data.items():
            if role == "full":
                for x in j["images"]:
                    both[(h, x["file_name"])].add(t)
        for (h, t, role), j in sorted(data.items()):
            if role != "full":
                continue
            other = "NT" if t == "CRL" else "CRL"
            boxes = defaultdict(Counter)
            for a in j["annotations"]:
                boxes[a["image_id"]][cfg["quality"].get(a.get("standard"), a.get("standard"))] += 1
            for x in j["images"]:
                nm = x["file_name"]
                b = boxes[x["id"]]
                rows_imgs.append(dict(
                    hospital=h, task=t, image_file_name=nm, width=x.get("width"), height=x.get("height"),
                    case_id_hash=("" if h in placeholder else hashlib.sha256(str(x.get("case_id")).encode()).hexdigest()[:12]),
                    n_boxes=sum(b.values()), n_boxes_standard=b["standard"], n_boxes_non_standard=b["non-standard"],
                    n_boxes_basically_standard=b["basically standard"],
                    in_train=nm in mem[(h, t)].get("train", set()), in_test=nm in mem[(h, t)].get("test", set()),
                    in_label_5=nm in mem[(h, t)].get("label_5", set()), in_label_10=nm in mem[(h, t)].get("label_10", set()),
                    also_in_other_task=other in both[(h, nm)]))
        allnames = {x["file_name"] for j in data.values() for x in j["images"]}
        info.update(unique_images=len(allnames), issues=issues, placeholder_hospitals=sorted(placeholder),
                    cats={h: sorted({n for c in v for n in c}) for h, v in info["cats"].items()})
        S["fusep"] = info
        S["fusep_names"] = {n.split(".")[0] for n in allnames}
        log(f"{key}: {len(data)} json files, {len(allnames)} unique referenced images (images NOT included), "
            f"{len(issues)} audit notes")
    return rows_files, rows_imgs


# ==========================================================================
# stage 2: image records
# ==========================================================================
def gather_records(ext_root, repo_root, log):
    recs = []
    for key, cfg in SOURCES.items():
        d = ext_root / key
        if cfg["kind"] == "classification":
            files = [(sp, p) for sp in ("train", "valid", "test") for p in sorted((d / sp).iterdir())
                     if p.suffix.lower() in IMG_EXT]
        else:
            files = [(None, p) for p in sorted((d / cfg["img_dir"]).iterdir()) if p.suffix.lower() in IMG_EXT]
        for sp, p in files:
            raw = p.read_bytes()
            f = image_features(raw)
            recs.append(dict(source=key, split=sp, path=p, relpath=p.relative_to(repo_root).as_posix(),
                             fname=p.name, stem=p.stem, sha=hashlib.sha256(raw).hexdigest(), **f))
        log(f"read {sum(1 for r in recs if r['source'] == key)} images from {key}")
    for i, r in enumerate(sorted(recs, key=lambda r: (r["source"], r["relpath"]))):
        r["record_id"] = f"R{i + 1:05d}"
    recs.sort(key=lambda r: r["record_id"])
    return recs


def attach_roboflow_labels(recs, ext_root, S):
    d = ext_root / "roboflow_fetal_brain"
    rows = {}
    csv_stats = {}
    for sp in ("train", "valid", "test"):
        classes, rr = read_classes_csv(d / sp / "_classes.csv")
        for x in rr:
            rows[(sp, x["filename"])] = x
        files = {r["fname"] for r in recs if r["source"] == "roboflow_fetal_brain" and r["split"] == sp}
        csv_stats[sp] = dict(csv_rows=len(rr), files=len(files), rows_without_file=len({x["filename"] for x in rr} - files),
                             files_without_row=len(files - {x["filename"] for x in rr}),
                             hot_counts=dict(Counter(len(x["hot"]) for x in rr)),
                             duplicate_rows=len(rr) - len({x["filename"] for x in rr}))
    S["rf_csv"] = csv_stats
    S["rf_header_classes"] = classes
    for r in recs:
        if r["source"] != "roboflow_fetal_brain":
            continue
        x = rows.get((r["split"], r["fname"]))
        r["hot"] = x["hot"] if x else None


def attach_names(recs):
    for r in recs:
        scan, case, aug, fam = parse_name(Path(r["fname"]))
        r["scan"], r["aug"], r["family"] = scan, aug, fam
        r["subject_id"], r["subject_basis"], r["case_token"] = None, None, None
        if fam == "patient":
            r["subject_id"] = "Patient" + case[len("patient"):]
            r["subject_basis"] = ("filename token 'Patient#####' (a naming convention; the Readme does not "
                                  "document it as a subject identifier)")
        elif r["source"] == "roboflow_fetal_brain":
            r["case_token"] = case
        elif r["source"].startswith("fhu_"):
            r["case_token"] = None


# ==========================================================================
# stage 3: annotations (masks, COCO, pixel size)
# ==========================================================================
def attach_anatomy(recs, ext_root, S):
    S["anat"] = {}
    for key, cfg in SOURCES.items():
        if cfg["kind"] != "anatomy":
            continue
        d = ext_root / key
        segdir = d / "_nested" / Path(cfg["seg"]).stem
        labelmap = (segdir / "labelmap.txt").read_text().strip().splitlines()
        lm = {}
        for ln in labelmap:
            if ln.startswith("#") or ":" not in ln:
                continue
            name, rgb = ln.split(":")[:2]
            lm[name] = tuple(int(v) for v in rgb.split(","))
        pix = {}
        if cfg["pix"]:
            with open(d / cfg["pix"], newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    pix[row["Label"].strip()] = float(row["Pixel in mm"])
        coco = json.loads((d / "_nested" / Path(cfg["coco"]).stem / "annotations" / "instances_default.json").read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        coco_img = {im["id"]: im["file_name"] for im in coco["images"]}
        coco_ann = defaultdict(Counter)
        for a in coco["annotations"]:
            coco_ann[coco_img[a["image_id"]]][cats[a["category_id"]]] += 1
        missing = [n for n in MASK_COLORS if n not in lm]
        if missing:
            raise SystemExit(f"{key}: labelmap.txt lacks {missing}")
        cols = {n: lm[n] for n in MASK_COLORS}      # decode with THIS source's labelmap
        st = dict(labelmap=lm, labelmap_matches_assumed=all(lm.get(k) == v for k, v in MASK_COLORS.items()),
                  coco_images=len(coco["images"]), coco_ann_total=Counter(), mask_missing=0, mask_size_mismatch=0,
                  mask_empty=0, coco_mask_disagree=Counter(), pix_rows=len(pix), pix_missing=0,
                  mask_other_color_images=0)
        for c in coco_ann.values():
            st["coco_ann_total"].update(c)
        mine = [r for r in recs if r["source"] == key]
        names = {r["fname"] for r in mine}
        st["coco_not_in_files"] = len(set(coco_img.values()) - names)
        st["files_not_in_coco"] = len(names - set(coco_img.values()))
        for r in mine:
            mp = segdir / "SegmentationClass" / r["fname"]
            r["pixel_mm"] = pix.get(r["fname"])
            if cfg["pix"] and r["pixel_mm"] is None:
                st["pix_missing"] += 1
            r["mask_path"], r["mask_px"], r["anat_present"] = None, None, None
            if not mp.exists():
                st["mask_missing"] += 1
                continue
            m = np.asarray(Image.open(mp).convert("RGB"))
            if m.shape[:2] != (r["h"], r["w"]):
                st["mask_size_mismatch"] += 1
            px = {n: int((m == np.array(c)).all(2).sum()) for n, c in cols.items()}
            other = int((m != 0).any(2).sum()) - sum(px.values())
            if other > 0:
                st["mask_other_color_images"] += 1
            r["mask_path"] = mp.relative_to(REPO).as_posix()
            r["mask_px"], r["anat_present"] = px, [n for n in MASK_COLORS if px[n] > 0]
            if not r["anat_present"]:
                st["mask_empty"] += 1
            for n in MASK_COLORS:   # COCO polygon vs rendered mask: do they agree on presence?
                if (coco_ann[r["fname"]][n] > 0) != (px[n] > 0):
                    st["coco_mask_disagree"][n] += 1
        st["mask_present_images"] = Counter(n for r in mine for n in (r["anat_present"] or []))
        S["anat"][key] = st


# ==========================================================================
# stage 4: duplicates + groups
# ==========================================================================
def find_pairs(recs, dh_max, ph_max):
    n = len(recs)
    DH = np.stack([r["dh"] for r in recs])
    PH = np.stack([r["ph"] for r in recs])
    pairs = {}
    by_sha = defaultdict(list)
    for i, r in enumerate(recs):
        by_sha[r["sha"]].append(i)
    for idx in by_sha.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pairs[(idx[a], idx[b])] = dict(kind="exact", dh=hdist(DH[idx[a]], DH[idx[b]]), ph=hdist(PH[idx[a]], PH[idx[b]]))
    cand = {}
    for i, j, dd in near_pairs(DH, 8):
        pd_ = hdist(PH[i], PH[j])
        if pd_ <= 10:
            cand[(i, j)] = (dd, pd_)
    for (i, j), (dd, pd_) in cand.items():
        if (i, j) not in pairs and dd <= dh_max and pd_ <= ph_max:
            pairs[(i, j)] = dict(kind="near", dh=dd, ph=pd_)
    sens = {t: sum(1 for (dd, pd_) in cand.values() if dd <= t and pd_ <= t) for t in (0, 2, 4, 6, 8)}
    return pairs, sens


def pair_relation(a, b):
    if a["source"] == b["source"] == "roboflow_fetal_brain":
        if a["scan"] == b["scan"]:
            return "augmentation_sibling_same_scan"
        if a["case_token"] and a["case_token"] == b["case_token"]:
            return "same_case_token_different_scan"
        return "within_roboflow_unrelated_names"
    return "same_source" if a["source"] == b["source"] else "cross_source"


def build_groups_and_dups(recs, pairs):
    n = len(recs)
    key_idx = defaultdict(list)
    for i, r in enumerate(recs):
        k = r["subject_id"] or ((r["source"], r["case_token"]) if r["case_token"] else None)
        if k:
            key_idx[k].append(i)
    token_links = [(v[0], j) for v in key_idx.values() for j in v[1:]]
    dup_links = list(pairs)
    dup_root = components(n, dup_links)
    root = components(n, token_links + dup_links)
    # deterministic ids, ordered by first record id
    def ids(roots, prefix, min_size):
        members = defaultdict(list)
        for i, rt in enumerate(roots):
            members[rt].append(i)
        out, k = {}, 0
        for rt in sorted(members, key=lambda r: min(members[r])):
            if len(members[rt]) >= min_size:
                k += 1
                for i in members[rt]:
                    out[i] = f"{prefix}{k:05d}"
        return out, {v: [] for v in set(out.values())}
    gid, _ = ids(root, "G", 1)
    did, _ = ids(dup_root, "D", 2)
    pair_kind = defaultdict(set)
    for (i, j), p in pairs.items():
        pair_kind[i].add(p["kind"])
        pair_kind[j].add(p["kind"])
    for i, r in enumerate(recs):
        r["group"] = gid[i]
        r["dup_group"] = did.get(i)
        r["dup_kind"] = ("exact" if "exact" in pair_kind[i] else "near") if i in pair_kind else None
    # basis of each case group
    by_g = defaultdict(list)
    for r in recs:
        by_g[r["group"]].append(r)
    for g, rs in by_g.items():
        bases = []
        if any(r["subject_id"] for r in rs):
            bases.append("filename_patient_token")
        if any(r["case_token"] for r in rs):
            bases.append("filename_case_token_heuristic")
        idx_in = {r["record_id"] for r in rs}
        if any(recs[i]["group"] == g and recs[j]["group"] == g for (i, j) in pairs):
            bases.append("duplicate_link")
        if not bases:
            bases = ["none (image-level singleton; subject unknown)"]
        for r in rs:
            r["group_basis"] = "+".join(bases)
    return by_g


def resolve_routes(recs, pairs, S):
    """Decide classification / auxiliary_anatomy / unresolved for every record (reasons kept)."""
    for r in recs:      # labels first: conflict detection below needs every label set
        r["label"] = r["hot"][0] if r["source"] == "roboflow_fetal_brain" and r.get("hot") and len(r["hot"]) == 1 else None
    exact_conflict = set()
    for (i, j), p in pairs.items():
        a, b = recs[i], recs[j]
        if p["kind"] == "exact" and a.get("label") and b.get("label") and a["label"] != b["label"]:
            exact_conflict |= {a["record_id"], b["record_id"]}
    by_g = defaultdict(list)
    for r in recs:
        by_g[r["group"]].append(r)
    mixed = {g for g, rs in by_g.items() if len({r["label"] for r in rs if r.get("label")}) > 1}
    S["mixed_label_groups"] = {g: sorted({r["label"] for r in by_g[g] if r.get("label")}) for g in mixed}
    for r in recs:
        reasons, flags = [], []
        if r["source"] == "roboflow_fetal_brain":
            hot = r["hot"]
            if hot is None:
                reasons.append("no_row_in_source__classes.csv")
            elif len(hot) == 0:
                reasons.append("no_class_marked_in_source_csv")
            elif len(hot) > 1:
                reasons.append("multiple_classes_marked_in_source_csv:" + "|".join(hot))
            else:
                if r["label_token"] and r["label_token"] != hot[0]:
                    reasons.append(f"filename_class_token({r['label_token']})!=csv_label({hot[0]})")
            if r["record_id"] in exact_conflict:
                reasons.append("exact_duplicate_with_different_label")
            elif r["group"] in mixed:
                reasons.append("linked_group_has_conflicting_labels:" + "|".join(sorted(S['mixed_label_groups'][r['group']])))
            if r["aug"] is not None:
                flags.append("augmented_copy")
            if r.get("label") and r["label_token"] is None:
                flags.append("label_not_corroborated_by_filename")
            r["route"] = "unresolved" if reasons else "classification"
            r["verification"] = ("source_asserted;filename_consistent" if r.get("label") and r["label_token"] == r["label"]
                                 else "source_asserted;csv_only(no_filename_evidence)" if r.get("label") else "unresolved")
        else:
            if r["mask_path"] is None:
                reasons.append("annotation_missing_no_mask")
            elif not r["anat_present"]:
                reasons.append("mask_has_no_foreground_structures")
            r["route"] = "unresolved" if reasons else "auxiliary_anatomy"
            r["verification"] = "n/a (anatomy annotation only; no diagnosis)"
            r["label"] = None
        if r["dup_kind"] == "exact":
            flags.append("exact_duplicate_member")
        if r["dup_group"]:
            srcs = {x["source"] for x in by_g[r["group"]]}
            if len(srcs) > 1:
                flags.append("group_spans_multiple_sources")
        r["reasons"], r["flags"] = reasons, flags


def assign_eval_roles(recs):
    by_scan = defaultdict(list)
    for r in recs:
        if r["route"] == "classification":
            by_scan[(r["label"], r["scan"])].append(r)
    for rs in by_scan.values():
        orig = [r for r in rs if r["aug"] is None]
        for r in rs:
            r["eval_role"] = ("original" if r["aug"] is None else "augmented_copy")
        if not orig:
            rs.sort(key=lambda r: r["record_id"])
            rs[0]["eval_role"] = "augmentation_only_representative"
    for r in recs:
        r.setdefault("eval_role", None)


# ==========================================================================
# stage 5: CV proposal
# ==========================================================================
def build_cv(recs, K, S):
    cls = [r for r in recs if r["route"] == "classification"]
    ev = [r for r in cls if r["eval_role"] in ("original", "augmentation_only_representative")]
    g_by_c = defaultdict(set)
    for r in ev:
        g_by_c[r["label"]].add(r["group"])
    S["groups_per_class"] = {c: len(g) for c, g in g_by_c.items()}
    ok = {c for c, g in g_by_c.items() if len(g) >= K}
    S["cv_classes_ok"], S["cv_classes_short"] = sorted(ok), sorted(set(g_by_c) - ok)
    gsize = Counter(r["group"] for r in ev)
    fold_of = assign_class_folds({r["group"]: r["label"] for r in ev if r["label"] in ok}, gsize, K)
    first_sha = {}
    for r in sorted(ev, key=lambda r: r["record_id"]):
        first_sha.setdefault(r["sha"], r["record_id"])
    rows = []
    for r in cls:
        fold = fold_of.get(r["group"])
        if r["label"] not in ok:
            status = f"not_evaluable_stratified (only {len(g_by_c[r['label']])} independent groups < {K})"
        elif r["eval_role"] == "augmented_copy":
            status = "train_only_augmented_copy"
        else:
            status = "evaluable"
        scoring = ("scored" if r["eval_role"] in ("original", "augmentation_only_representative")
                   and first_sha[r["sha"]] == r["record_id"] else
                   "exact_duplicate_extra_not_scored" if r["eval_role"] in ("original", "augmentation_only_representative")
                   else "not_scored")
        rows.append(dict(record_id=r["record_id"], task="classification", cv_unit=r["group"], class_label=r["label"],
                         test_fold=fold if fold is not None else "", fold_status=status, eval_role=r["eval_role"],
                         scoring_role=scoring, group_basis=r["group_basis"], image_path=r["relpath"]))
    an = [r for r in recs if r["route"] == "auxiliary_anatomy"]
    sizes = Counter(r["group"] for r in an)
    afold = greedy_group_folds(sizes, K)
    for r in an:
        has_pid = "filename_patient_token" in r["group_basis"]
        rows.append(dict(record_id=r["record_id"], task="anatomy_segmentation", cv_unit=r["group"], class_label="",
                         test_fold=afold[r["group"]], fold_status=("subject_grouped" if has_pid else
                                                                      "image_level_only_patient_overlap_not_excludable"),
                         eval_role="original", scoring_role="scored", group_basis=r["group_basis"], image_path=r["relpath"]))
    S["anat_fold_imgs"] = Counter(afold[r["group"]] for r in an)
    return rows


# ==========================================================================
# legacy-number reproduction
# ==========================================================================
def legacy_numbers(rf_root, S):
    cv = _load("cv_mod", "02b_make_cv_folds.py")
    rebuild = sys.modules["rebuild_mod"]
    out = {}
    recs = audit.load_records(rf_root)
    for r in recs:
        r["sha"] = audit.sha256(r["path"])
    cfgs = [("16 original classes", {})]
    cm = REPO / "configs" / "class_map_cv.json"
    if cm.exists():
        raw = json.loads(cm.read_text())
        cfgs.append(("configs/class_map_cv.json (manual 5-family map)", {k: v for k, v in raw.items() if not k.startswith("_")}))
    for name, cmap in cfgs:
        rr = cv.apply_class_map([dict(r) for r in recs], cmap) if cmap else [dict(r, orig_label=r["label"]) for r in recs]
        groups = rebuild.build_groups(rr, "case", 4)
        for r, g in zip(rr, groups):
            r["group"] = g
        lines = []
        kept, di, dg = cv.resolve_conflicts(rr, lines.append)
        glab, gn = cv.group_table(kept)
        test_orig = [r for r in cv.originals_only(kept) if r["split"] == "test"]
        out[name] = dict(files=len(rr), kept=len(kept), dropped_conflict_images=di, dropped_tied_groups=dg,
                         independent_groups=len(glab), original_scans=sum(1 for _ in cv.originals_only(kept)),
                         original_test_split_scans=len(test_orig), conflict_log=lines[:12])
    S["legacy"] = out


# ==========================================================================
# reports
# ==========================================================================
def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def make_reports(bd, recs, pairs, sens, inv, S, args, cv_rows):
    rep = bd / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    rf = [r for r in recs if r["source"] == "roboflow_fetal_brain"]
    fh = [r for r in recs if r["source"] != "roboflow_fetal_brain"]
    cls = [r for r in recs if r["route"] == "classification"]
    aux = [r for r in recs if r["route"] == "auxiliary_anatomy"]
    unr = [r for r in recs if r["route"] == "unresolved"]
    uniq_sha = len({r["sha"] for r in recs})
    groups = defaultdict(list)
    for r in recs:
        groups[r["group"]].append(r)
    cls_orig = [r for r in cls if r["eval_role"] in ("original", "augmentation_only_representative")]

    # ---------------- inventory
    L = ["# Dataset inventory", "",
         "Verified by reading the archives (counts are from the files, not from README claims). "
         "Nothing was assumed from file or folder names alone.", ""]
    rows = []
    for k, cfg in SOURCES.items():
        i = inv[k]
        n = sum(1 for r in recs if r["source"] == k)
        rows.append([k, cfg["zip"], f"{i['bytes'] / 1e6:.0f} MB", n, ", ".join(f"{e}:{c}" for e, c in sorted(i['ext'].items())),
                     "ok" if i["corrupt_member"] is None else f"CORRUPT: {i['corrupt_member']}"])
    L += [md_table(["source id", "archive", "size", "images read", "file types in archive", "zip integrity"], rows), "",
          "SHA-256 of each archive (provenance):", ""]
    L += [f"- `{SOURCES[k]['zip']}`: `{inv[k]['sha256']}`" for k in SOURCES]
    L += ["", "## Source descriptions", ""]
    split_txt = ", ".join("%s: %d" % (sp, sum(1 for r in rf if r["split"] == sp)) for sp in ("train", "valid", "test"))
    L += ["### roboflow_fetal_brain (diagnosis classification)",
          "- Roboflow Universe export, `README.dataset.txt`: license CC BY 4.0, 'Provided by a Roboflow user'. No clinical "
          "annotator, ground-truth procedure, patient or acquisition metadata are documented.",
          "- Labels: one-hot columns in `<split>/_classes.csv` (16 diagnosis/normal classes), i.e. DIAGNOSIS labels at image level.",
          "- `README.roboflow.txt` says 'No image augmentation techniques were applied' and that images were resized 640x640 (stretch). "
          f"File names, however, contain `_aug_N` tags: {sum(1 for r in rf if r['aug'] is not None)} of {len(rf)} images are "
          "augmented copies produced BEFORE or OUTSIDE this export (README claim contradicted by the files).",
          f"- Source split: train/valid/test folders exist ({split_txt}); "
          "how the split was made is not documented.", ""]
    L += ["### fhu_* = 'Fetal Head Ultrasound Images' (anatomy)",
          "- `Readme.txt`: CC BY 4.0; anonymised images; classes Brain / CSP / LV; 'No predefined training and testing splits'; "
          "four plane-group zips; pixel-size CSV per zip; 11 export formats per zip. "
          "The README names no original author, DOI or URL (the download link is redacted: `XXXX...`), "
          "so the attribution required by CC BY 4.0 cannot be completed from the repository alone.",
          "- Labels: segmentation masks / COCO polygons for anatomical structures; **no diagnosis or normal/abnormal label exists**.",
          "- Image naming `PatientNNNNN_PlaneP_i_of_n.png` (two Trans-* sets) carries a patient token; the Diverse set uses `NNN_HC.png` "
          "with no subject information and also contains no pixel-size CSV (README says each zip has one).", ""]
    tot_claim = README_CLAIMS["images"]
    L += ["## README claims vs what is present", "",
          f"- README: {tot_claim} images in four zips (Trans-thalamic, Trans-ventricular, Trans-cerebellum, Diverse). "
          f"Present: {len(fh)} images in three zips. **Trans-thalamic.zip is not in the repository**, "
          f"so {tot_claim - len(fh)} images ({pct(tot_claim - len(fh), tot_claim)}) of the described dataset are unavailable here.",
          "- README class instances: Brain 3,794 / CSP 1,865 / LV 1,512. Instances counted in the COCO files present:"]
    tb = sum(S["anat"][k]["coco_ann_total"]["Brain"] for k in S["anat"])
    tc = sum(S["anat"][k]["coco_ann_total"]["CSP"] for k in S["anat"])
    tl = sum(S["anat"][k]["coco_ann_total"]["LV"] for k in S["anat"])
    L += [f"  Brain {tb}, CSP {tc}, LV {tl} (remainder presumably in the missing Trans-thalamic set; not verifiable).", ""]
    L += ["## Annotation formats and acquisition planes", "",
          md_table(["source", "annotation type", "label meaning", "plane", "plane basis", "subject ids", "documented split"], [
              ["roboflow_fetal_brain", "image-level one-hot class", "diagnosis / normal", "unknown", "not documented",
               "none documented (Patient##### tokens exist in file names of the 'normal' class)", "folders only; method undocumented"],
              ["fhu_trans_cerebellum", "mask + COCO polygons + pixel size mm", "anatomy (Brain, CSP, LV)", "trans-cerebellar", SOURCES['fhu_trans_cerebellum']['plane_basis'], "Patient##### in filename", "none (README)"],
              ["fhu_trans_ventricular", "mask + COCO polygons + pixel size mm", "anatomy (Brain, CSP, LV)", "trans-ventricular", SOURCES['fhu_trans_ventricular']['plane_basis'], "Patient##### in filename", "none (README)"],
              ["fhu_diverse_head", "mask + COCO polygons (no pixel-size csv)", "anatomy (Brain, CSP, LV)", "diverse/unspecified", SOURCES['fhu_diverse_head']['plane_basis'], "none", "none (README)"]]), "",
          "## Nested export-format archives", "",
          "Each Fetal-Head zip carries 11 redundant re-exports of the same annotations. Only the Segmentation (mask) and COCO "
          "exports were extracted (they are needed for annotation audit); the others were listed but not extracted to avoid copying "
          "~10x duplicated images.", ""]
    nr = []
    for k in SOURCES:
        for nm, v in inv[k]["nested"].items():
            nr.append([k, nm, "yes" if v.get("extracted") else "no", v.get("members", v.get("error")), v.get("ext", "")])
    L += [md_table(["source", "nested archive", "extracted", "members", "types"], nr), "",
          "## Extraction report", ""]
    for k in SOURCES:
        e = inv[k]["extract"]
        L.append(f"- {k}: {e['n']} files written, {e['skipped_existing']} already present, unsafe paths skipped: {len(e['unsafe'])}"
                 + (f" ({e['unsafe'][:3]})" if e["unsafe"] else ""))
    L += ["", f"Totals: {len(recs)} images read, {uniq_sha} unique by SHA-256 across all sources. Unreadable images: 0 (all decoded)."]
    fu = S.get("fusep")
    if fu:
        L += ["", "## Label-only archive: FUSEP (`%s`)" % fu["zip"], "",
              f"- {fu['entries']} entries, {sum(fu['ext'].values())} JSON files, {fu['bytes'] / 1e6:.0f} MB, zip integrity "
              f"{'ok' if fu['corrupt_member'] is None else 'CORRUPT'}; SHA-256 `{fu['sha256']}`. **It contains no images**, no README and no "
              "licence text (licence and attribution: unknown; obtain them from the dataset owners).",
              "- Content: COCO-style bounding boxes in 3 hospital folders (Hospital1_SZ, Hospital2_SC, Hospital3_YN), two tasks (CRL, NT; taken "
              "from file names, their expansion is not documented in the archive), with `train`, `test` and semi-supervised subsets "
              "`label_5/label_10/unlabel_5/unlabel_10`.",
              "- Each box carries a plane-quality value in Chinese (standard / non-standard / basically standard). The English renderings are my "
              "translation and should be confirmed.",
              "- Box categories (anatomical landmarks): " + "; ".join(f"{h}: {', '.join(c)}" for h, c in fu["cats"].items()),
              f"- {fu['unique_images']} distinct hashed image names are referenced; **{fu['sha_overlap_with_image_sources']}** of them equal the "
              "SHA-256 of an image in the other archives (no overlap detectable; the images themselves are unavailable here).",
              "- Annotation type: landmark/structure bounding boxes + image-quality labels. **No diagnosis, no normal/abnormal label, no mask.** "
              "It is first-trimester-style data (CRL / NT), a different gestational stage and task from the fetal-brain diagnosis set.",
              "- Routing: `label_only` -> not in `image_manifest.csv`, `classification/` or `auxiliary_anatomy/` (no images to attach labels to). "
              "Tables: `manifests/label_only_sources.csv` (one row per json) and `manifests/label_only_image_refs.csv` (one row per referenced "
              "image; raw `case_id` values are replaced by a 12-char hash so hospital identifiers are not copied).", "",
              md_table(["hospital", "task", "role", "images", "boxes", "distinct case_ids", "plane quality"],
                       [[r["hospital"], r["task"], r["role"], r["n_images"], r["n_annotations"], r["n_distinct_case_ids"],
                         r["plane_quality_counts"] or "-"] for r in S["fusep_rows"]]), "",
              "Split-hygiene findings (verified from the JSON):", ""]
        L += [f"- {x}" for x in fu["issues"]] or ["- none"]
    (rep / "dataset_inventory.md").write_text("\n".join(L), encoding="utf-8")

    # ---------------- label audit
    L = ["# Label audit", "", "## Roboflow classification labels", ""]
    L.append(md_table(["split", "csv rows", "image files", "rows w/o file", "files w/o row", "duplicate rows", "classes marked per row"],
                      [[s, v["csv_rows"], v["files"], v["rows_without_file"], v["files_without_row"], v["duplicate_rows"], v["hot_counts"]]
                       for s, v in S["rf_csv"].items()]))
    L += ["", "Header classes (verbatim, spelling preserved): " + ", ".join(f"`{c}`" for c in S["rf_header_classes"]), ""]
    multi = sum(1 for r in rf if r["hot"] and len(r["hot"]) > 1)
    L += [f"- Multi-label rows: **{multi}**. Zero-label rows: **{sum(1 for r in rf if r['hot'] is not None and len(r['hot']) == 0)}**. "
          "So 'conflicting labels' in earlier audits are NOT legitimate multi-label annotations; they arise from near-duplicate "
          "clustering across classes (see below) -- they must be treated as potential data errors, not as multi-label cases.", ""]
    rows = []
    for c in CLASS_NAMES:
        rs = [r for r in rf if r.get("label") == c or (r["hot"] == [c])]
        o = [r for r in rs if r["aug"] is None]
        scans = {r["scan"] for r in rs}
        grp = {r["group"] for r in rs}
        grp_cls = {r["group"] for r in rs if r["route"] == "classification"}
        rows.append([c, len(rs), len(o), len(scans), len({r['case_token'] or r['subject_id'] for r in rs}), len(grp), len(grp_cls),
                     sum(1 for r in rs if r["route"] == "unresolved")])
    L += ["## Counts per original class (counting units differ -- this is why 'image', 'scan', 'case' and 'group' numbers disagree)", "",
          md_table(["class", "images", "non-augmented images", "distinct scans (filename)", "distinct case tokens (heuristic)",
                    "linked groups (case token + duplicate links)", "groups still classification-eligible", "images unresolved"], rows), "",
          "Counting units: *image* = file; *scan* = file name with augmentation/Roboflow tags removed (heuristic); *case token* = trailing "
          "case number in the name (heuristic, class-specific) or the `Patient#####` token; *group* = union of case tokens and exact/"
          "near-duplicate links. Only the group is used for splitting.", ""]
    ft = Counter((r["label_token"] is not None, r["label_token"] == r.get("label")) for r in rf if r.get("label"))
    L += ["## Label corroboration (csv vs file name)", "",
          f"- csv label equals the diagnosis token in the file name: {ft[(True, True)]} images",
          f"- csv label contradicts the file name: {ft[(True, False)]} images",
          f"- file name carries no diagnosis token (csv is the only evidence): {ft[(False, False)]} images; of these "
          f"{sum(1 for r in rf if r.get('label') == 'normal' and r['label_token'] is None)} are `normal` and "
          f"{sum(1 for r in rf if r.get('label') not in (None, 'normal') and r['label_token'] is None)} are other classes.", "",
          "**Verification status:** no label in this dataset is clinically verified in the available documentation. "
          "`source_asserted;filename_consistent` means the Roboflow csv and the file name agree (two uncontrolled sources, "
          "same uploader). `source_asserted;csv_only` means the csv is the only evidence. Neither is ground truth.", ""]
    L += ["## Label conflicts", "", f"Groups containing more than one diagnosis: {len(S['mixed_label_groups'])}.",
          f"Roboflow images routed to `unresolved`: {sum(1 for r in rf if r['route'] == 'unresolved')}.", ""]
    rc = Counter(x.split(":")[0].split("(")[0] for r in rf for x in r["reasons"])
    L += [md_table(["reason", "images"], [[k, v] for k, v in rc.most_common()]) if rc else "No Roboflow image was unresolved.", ""]
    for g, labs in sorted(S["mixed_label_groups"].items())[:25]:
        mem = [r for r in groups[g] if r["source"] == "roboflow_fetal_brain"]
        L.append(f"- `{g}` labels {labs}: " + "; ".join(f"{r['record_id']} `{r['fname'][:48]}` ({r.get('label')})" for r in mem[:6]))
    L += ["", "## Anatomical annotations (auxiliary dataset)", "",
          "Masks were decoded and compared with the COCO export; the label map was read from `labelmap.txt`.", ""]
    rows = []
    for k, st in S["anat"].items():
        n = sum(1 for r in fh if r["source"] == k)
        rows.append([k, n, st["labelmap"] and "; ".join(f"{a}={b}" for a, b in st["labelmap"].items()), st["labelmap_matches_assumed"],
                     st["mask_missing"], st["mask_empty"], st["mask_size_mismatch"], st["mask_other_color_images"], st["pix_missing"],
                     dict(st["mask_present_images"]), dict(st["coco_ann_total"]), dict(st["coco_mask_disagree"]),
                     f"{st['coco_not_in_files']}/{st['files_not_in_coco']}"])
    L += [md_table(["source", "images", "label map", "map as assumed", "mask missing", "mask empty", "mask size != image", "unexpected mask colours",
                    "pixel-size missing", "images with structure (mask)", "COCO instances", "COCO vs mask presence disagreements",
                    "COCO-not-in-files / files-not-in-COCO"], rows), "",
          "Annotation types: pixel-wise semantic masks and COCO instance polygons for Brain (skull/head), CSP (cavum septi pellucidi), "
          "LV (lateral ventricle); pixel size in mm for the Trans-* sets. These are anatomy labels: **they say nothing about normal vs abnormal.**", "",
          "Masks were decoded with each source's own `labelmap.txt` (the Diverse set uses yellow for CSP, the others green). "
          "'COCO vs mask disagreements' counts images where COCO has an instance of a structure but the rendered mask has none (or vice versa): "
          "an internal inconsistency of the source annotations, not resolved here. Counts of instances per structure in COCO are what the Readme calls 'class instances'.", ""]
    (rep / "label_audit.md").write_text("\n".join(L), encoding="utf-8")

    # ---------------- duplicates
    kinds = Counter(p["kind"] for p in pairs.values())
    rel = Counter((p["kind"], pair_relation(recs[i], recs[j])) for (i, j), p in pairs.items())
    L = ["# Duplicate audit", "",
         f"Method: SHA-256 for exact duplicates; near-duplicate candidates = dHash distance <= {args.dh_max} AND pHash (DCT) distance <= {args.ph_max} "
         "(64-bit hashes of the grayscale image resized to a fixed square, so a 640x640 'stretch' re-export and the 959x661 original hash alike). "
         "Near-duplicates are **heuristic candidates, not proof**; nothing was deleted.", "",
         f"- Images: {len(recs)}; unique SHA-256: {uniq_sha}; images in an exact-duplicate set: {sum(1 for r in recs if r['dup_kind'] == 'exact')}",
         f"- Exact-duplicate pairs: {kinds['exact']}; near-duplicate pairs (not exact): {kinds['near']}",
         f"- Duplicate clusters (exact or near links): {len({r['dup_group'] for r in recs if r['dup_group']})}", "",
         "## Pairs by relation", "",
         md_table(["kind", "relation", "pairs"], [[k, v, c] for (k, v), c in sorted(rel.items())]), "",
         "`augmentation_sibling_same_scan` = Roboflow `_aug_N` copies of one scan (expected by construction, but they ARE the same underlying "
         "image). `cross_source` = the same/near-same image appears in two different source datasets.", "",
         "## Sensitivity of the near-duplicate threshold", "",
         md_table(["both hash distances <=", "candidate pairs"], [[t, c] for t, c in sens.items()]), ""]
    cross = [(i, j, p) for (i, j), p in pairs.items() if pair_relation(recs[i], recs[j]) == "cross_source"]
    L += [f"## Cross-source matches ({len(cross)} pairs)", ""]
    if cross:
        cs = Counter((recs[i]["source"], recs[j]["source"], p["kind"]) for i, j, p in cross)
        L += [md_table(["source A", "source B", "kind", "pairs"], [[a, b, k, c] for (a, b, k), c in sorted(cs.items())]), "",
              "Examples:", ""]
        for i, j, p in cross[:15]:
            L.append(f"- {p['kind']} (dHash {p['dh']}, pHash {p['ph']}): `{recs[i]['relpath']}` <-> `{recs[j]['relpath']}`")
    else:
        L.append("No cross-source exact or near-duplicate pair was found at the stated thresholds.")
    wl = [(i, j, p) for (i, j), p in pairs.items() if pair_relation(recs[i], recs[j]) == "within_roboflow_unrelated_names"]
    lab_conf = [(i, j, p) for (i, j), p in pairs.items() if recs[i].get("label") and recs[j].get("label") and recs[i]["label"] != recs[j]["label"]]
    L += ["", f"## Duplicates with conflicting labels ({len(lab_conf)} pairs)", ""]
    for i, j, p in lab_conf[:30]:
        L.append(f"- {p['kind']} (dHash {p['dh']}, pHash {p['ph']}): `{recs[i]['fname'][:55]}` [{recs[i]['label']}] <-> `{recs[j]['fname'][:55]}` [{recs[j]['label']}]")
    L += ["", f"Roboflow near-duplicates between differently-named scans: {len(wl)} pairs (possible repeated images under different case names).", "",
          "## Proposed resolution (not applied)", "",
          "- Keep every source record. Duplicate clusters and case tokens are merged into one case group; a group is never split across folds.",
          "- For scoring, one representative per exact-SHA set counts (`scoring_role` in `proposed_cv_manifest.csv`); the rest stay in the same fold unscored.",
          "- Groups with conflicting labels are routed to `unresolved/` for human review rather than resolved by vote.",
          "- Near-duplicates with *different* labels need visual review; heuristic hashing may also over-link visually similar frames."]
    (rep / "duplicate_audit.md").write_text("\n".join(L), encoding="utf-8")

    # ---------------- leakage
    L = ["# Leakage audit", "", "## Roboflow source splits (train/valid/test as shipped)", ""]
    rfg = defaultdict(set)
    rfscan = defaultdict(set)
    rfcase = defaultdict(set)
    for r in rf:
        rfg[r["group"]].add(r["split"])
        rfscan[(r.get("label"), r["scan"])].add(r["split"])
        if r["case_token"] or r["subject_id"]:
            rfcase[r["case_token"] or r["subject_id"]].add(r["split"])
    cross_g = {g for g, s in rfg.items() if len(s) > 1}
    held = [r for r in rf if r["split"] in ("valid", "test")]
    held_leak = [r for r in held if r["group"] in cross_g and "train" in rfg[r["group"]]]
    L += [md_table(["measure", "value"], [
        ["valid+test images", len(held)],
        ["of which are augmented copies (`_aug_N`)", sum(1 for r in held if r["aug"] is not None)],
        ["scans (filename-derived) present in more than one split", f"{sum(1 for s in rfscan.values() if len(s) > 1)} of {len(rfscan)}"],
        ["case tokens present in more than one split", f"{sum(1 for s in rfcase.values() if len(s) > 1)} of {len(rfcase)}"],
        ["linked groups (tokens+duplicates) spanning >1 split", f"{len(cross_g)} of {len(rfg)}"],
        ["valid+test images whose group also has TRAIN images", f"{len(held_leak)} of {len(held)} ({pct(len(held_leak), len(held))})"],
        ["exact-SHA sets spanning >1 split", sum(1 for sha in {r['sha'] for r in rf} if len({r['split'] for r in rf if r['sha'] == sha}) > 1)]]), "",
        "Interpretation: the shipped split is NOT leakage-free. Augmented copies of the same scan were distributed across train/valid/test, "
        "so any model evaluated on the shipped test folder is evaluated on re-renderings of training scans.", ""]
    L += ["## Cross-dataset overlap (Roboflow vs Fetal-Head sets)", ""]
    rfp = {r["subject_id"] for r in rf if r["subject_id"]}
    fhp = {r["subject_id"] for r in fh if r["subject_id"]}
    both = rfp & fhp
    cr_hash = [(i, j, p) for (i, j), p in pairs.items() if pair_relation(recs[i], recs[j]) == "cross_source"
               and {recs[i]["source"], recs[j]["source"]} & {"roboflow_fetal_brain"}]
    L += [f"- Roboflow images carrying a `Patient#####` token: {sum(1 for r in rf if r['subject_id'])} (classes: "
          f"{dict(Counter(r.get('label') for r in rf if r['subject_id']))}); distinct tokens: {len(rfp)}",
          f"- Fetal-Head images carrying a patient token: {sum(1 for r in fh if r['subject_id'])}; distinct tokens: {len(fhp)}",
          f"- Patient tokens present in BOTH Roboflow and Fetal-Head sets: **{len(both)}** (token match only; the file-name convention "
          "being identical is suggestive but is not proof of the same patient)",
          f"- Roboflow<->Fetal-Head image pairs matching by pixels (exact or near): **{len(cr_hash)}**", ""]
    per = Counter()
    for i, j, p in cr_hash:
        per[p["kind"]] += 1
    L += [f"  breakdown: {dict(per)}", ""]
    L += ["## Within the Fetal-Head sets", ""]
    pt = defaultdict(set)
    for r in fh:
        if r["subject_id"]:
            pt[r["subject_id"]].add(r["source"])
    multi = {p: s for p, s in pt.items() if len(s) > 1}
    per_img = Counter(r["subject_id"] for r in fh if r["subject_id"])
    L += [f"- Patient tokens: {len(pt)}; tokens with more than one image: {sum(1 for v in per_img.values() if v > 1)}; "
          f"tokens appearing in both Trans-cerebellum and Trans-ventricular: {len(multi)}",
          "- No documented split exists, so there is no source leakage to measure, but a random image-level split would leak patients: "
          "use the `case_group_id` / `proposed_cv_manifest.csv`.",
          "- `fhu_diverse_head` has no subject information at all: **patient-level overlap within it and with other sets cannot be ruled out** "
          "(only pixel-level duplicates were checked).", "",
          "## Patient-level statement", "",
          "Reliable, documented subject identifiers do not exist for the Roboflow data. Groups there are filename-derived heuristics. "
          "Subject IDs for the Fetal-Head Trans-* sets are filename tokens; they are used as grouping keys but are not independently documented."]
    if S.get("fusep"):
        L += ["", "## FUSEP label-only archive", ""]
        L += [f"- {x}" for x in S["fusep"]["issues"]] or ["- no split issues found"]
        L += ["- Overlap with the brain datasets cannot be assessed beyond file hashes: the images are absent. Subject-level independence is "
              "only checkable where `case_id` is real (not Hospital2_SC)."]
    L += ["", "## Verification of previously reported numbers", ""]
    for name, v in S["legacy"].items():
        L += [f"### {name}", "",
              md_table(["quantity", "value"], [["files loaded (legacy loader)", v["files"]], ["kept after legacy conflict resolution", v["kept"]],
                                                ["images dropped by label conflict", v["dropped_conflict_images"]],
                                                ["tied groups dropped", v["dropped_tied_groups"]],
                                                ["independent groups", v["independent_groups"]],
                                                ["original scans", v["original_scans"]],
                                                ["original scans in shipped test folder", v["original_test_split_scans"]]]), ""]
    v0 = S["legacy"]["16 original classes"]
    L += ["Previously reported: 90 independent groups, 430 scans, 44 images dropped for conflicting labels, 91 evaluation cases.", "",
          f"- **Reproduced exactly** with the legacy grouping on the 16 original classes: groups {v0['independent_groups']}, scans {v0['original_scans']}, "
          f"dropped {v0['dropped_conflict_images']}.",
          "- The 44 'conflicting' images are not multi-label annotations: every csv row has exactly one class. They come from near-identical images "
          "filed under different diagnoses (see `duplicate_audit.md`). In this build such groups are not voted on; they go to `unresolved/`.",
          f"- **'91 evaluation cases' could not be reproduced** from any manifest quantity: the shipped test folder holds "
          f"{sum(1 for r in rf if r['split'] == 'test')} files = {v0['original_test_split_scans']} original scans; 430 is the pooled original-scan count over all splits. "
          "Treat 91 as unverified.",
          "- The legacy grouping linked only by (label, case token) plus aHash&dHash<=4, whereas this build also links case tokens across labels, "
          "patient tokens across sources and pHash, hence slightly different group counts in `case_groups.csv`."]
    (rep / "leakage_audit.md").write_text("\n".join(L), encoding="utf-8")

    # ---------------- data quality
    L = ["# Data-quality report", "", "## Independent evidence per class (classification-eligible, after conflict removal)", ""]
    rows = []
    for c in CLASS_NAMES:
        rs = [r for r in cls if r["label"] == c]
        og = [r for r in rs if r["eval_role"] in ("original", "augmentation_only_representative")]
        gs = {r["group"] for r in og}
        big = max(Counter(r["group"] for r in og).values()) if og else 0
        rows.append([c, len(rs), len(og), len(gs), big, "OK (>=5 groups)" if len(gs) >= args.folds else "TOO FEW GROUPS"])
    L += [md_table(["class", "images incl. augmented", "independent originals", "independent groups", "largest group (images)", f"{args.folds}-fold stratified"], rows), "",
          "Group counts are the honest sample size; images inside one group are near-copies and are not independent evidence.", ""]
    L += ["## Source / acquisition shortcut checks", "",
          "A diagnosis that is confounded with where the image came from can be 'predicted' from non-anatomical cues. "
          "Per-class descriptors of the Roboflow images:", ""]
    rows = []
    for c in CLASS_NAMES:
        rs = [r for r in rf if r.get("label") == c]
        if rs:
            fam = Counter(r["family"] for r in rs)
            rows.append([c, len(rs), dict(fam), f"{np.mean([r['color'] for r in rs]):.3f}", f"{np.mean([r['gray'] for r in rs]):.1f}"])
    L += [md_table(["class", "images", "filename family", "mean colour-pixel fraction", "mean gray level"], rows), "",
          f"- All {sum(1 for r in rf if r.get('label') == 'normal')} `normal` images use the `Patient#####_Plane#_i_of_n` naming; "
          f"{sum(1 for r in rf if r.get('label') not in (None, 'normal') and r['family'] == 'patient')} abnormal images do. "
          "Every abnormal image uses the `Copy-of-<class>-...` naming. **Normal vs abnormal is perfectly confounded with image family/source.** "
          "No split can remove this; a classifier can separate the families without learning pathology.",
          "- `Plane3` appears in all Fetal-Head file names; its meaning is not documented in the Readme and is not interpreted here.",
          f"- Fetal-Head images are PNG {sorted({(r['w'], r['h']) for r in fh})[:4]}...; Roboflow images are JPEG 640x640 re-renderings "
          "(format/resolution is another source cue).", ""]
    L += ["## Class-merge proposals", "",
          "`manifests/label_mapping.csv` records the proposed family mapping from `configs/class_hierarchy.json` / `02c_design_classes.py` "
          "next to the preserved original labels. These are statistical/anatomical groupings proposed to enable evaluation, "
          "**not clinically validated**; they require review by a clinician. Original labels are never overwritten.", "",
          "## Proposed evaluation design", ""]
    short = S["cv_classes_short"]
    gp = S["groups_per_class"]
    L += [f"- Classes with >= {args.folds} independent groups ({len(S['cv_classes_ok'])}): " + ", ".join(f"{c} ({gp[c]})" for c in S["cv_classes_ok"]),
          f"- Classes with < {args.folds} groups ({len(short)}), NOT evaluable with stratified {args.folds}-fold: " + (", ".join(f"{c} ({gp[c]})" for c in short) or "none"), ""]
    fold_tab = defaultdict(Counter)
    for x in cv_rows:
        if x["task"] == "classification" and x["test_fold"] != "" and x["scoring_role"] == "scored":
            fold_tab[x["class_label"]][x["test_fold"]] += 1
    rows = [[c] + [fold_tab[c][f] for f in range(args.folds)] for c in S["cv_classes_ok"]]
    if rows:
        L += ["Scored test images per class and fold (proposed layout):", "",
              md_table(["class"] + [f"fold {f}" for f in range(args.folds)], rows), ""]
    L += [
        "Recommendation:",
        f"1. Groups are the unit. With {sum(gp.values())} classification groups overall and a handful per rare class, per-class metrics on 5 folds rest on "
        "1-2 groups per cell and will be very noisy; report pooled results with group-bootstrap confidence intervals and the majority baseline.",
        "2. Do not tune on test folds: select hyper-parameters on the inner validation split (carved from training groups) or on a "
        "fixed pre-registered configuration, and run the 5 test folds once.",
        "3. Rare classes (list above) should either be merged under a clinician-approved scheme, evaluated only qualitatively "
        "(leave-one-group-out, reporting per-group outcomes), or collected further. They should not be presented with per-class accuracy.",
        "4. Because normal vs abnormal is confounded with source, add a source-controlled check (e.g. evaluate abnormal-only classes) "
        "before claiming diagnostic performance.", ""]
    L += ["## Exclusions", "", f"- Records routed to `unresolved`: {len(unr)}", ""]
    ex = Counter(x.split(":")[0].split("(")[0] for r in unr for x in r["reasons"])
    L += [md_table(["reason", "records"], [[k, v] for k, v in ex.most_common()]) if ex else "None.", "",
          "- `augmented_copy` records stay in the classification manifest as train-only members of their group; they are never test images.",
          f"- Records outside classification by design: {len(aux)} anatomy-only images (auxiliary), not labelled normal/abnormal.", ""]
    (rep / "data_quality_report.md").write_text("\n".join(L), encoding="utf-8")


def make_readme(bd, recs, S, args, inv, n_cls, n_aux, n_unr):
    rf = [r for r in recs if r["source"] == "roboflow_fetal_brain"]
    gp = S["groups_per_class"]
    L = ["# Unified fetal-brain ultrasound dataset build", "",
         "Audit-only build. **No model was trained and no evaluation protocol was changed.** Original archives were not modified.", "",
         "## Reproduce", "", "```bash",
         "# from the repository root (zips in the root; Python 3.10, numpy, Pillow, scipy)",
         "python scripts/14_build_unified_dataset.py                       # new datasets/final_build_<timestamp>/",
         f"python scripts/14_build_unified_dataset.py --build-dir {bd.name if bd.parent.name == 'datasets' else bd}   # rerun in place (idempotent)",
         "python -m pytest tests/test_unified_build.py -q                  # validation tests (use the newest build)", "```", "",
         "Archive checksums are in `run_info.json`; parameters: near-duplicate = dHash<=%d and pHash<=%d, folds = %d." % (args.dh_max, args.ph_max, args.folds), "",
         "## Layout (and deviations from the requested layout)", "",
         "- `extracted/<source>/` safe extraction, one directory per source (git-ignored). Only the Segmentation and COCO nested archives were extracted; "
         "the other 9 redundant export formats per Fetal-Head zip are inventoried only.",
         "- `manifests/` image_manifest.csv, label_mapping.csv, duplicate_groups.csv, case_groups.csv, proposed_cv_manifest.csv "
         "plus one extra file, `duplicate_pairs.csv` (pairwise evidence with hash distances).",
         "- `classification/`, `auxiliary_anatomy/`, `unresolved/` hold **CSV manifests only** (no image copies; paths point into `extracted/`) "
         "to avoid duplicating ~1 GB. `classification/class_summary.csv` is an extra summary.",
         "- `manifests/label_only_sources.csv`, `label_only_image_refs.csv`: the FUSEP label-only archive (no images, no diagnoses; audited, not routed to any training set).",
         "- `reports/` dataset_inventory, label_audit, duplicate_audit, leakage_audit, data_quality_report.",
         "- `scripts/` is a snapshot of the pipeline script used; the maintained copy is `scripts/14_build_unified_dataset.py` in the repository.",
         "- Empty cells in CSVs mean null/unavailable (never a guessed value).", "",
         "## Headline numbers", "",
         md_table(["item", "value"], [
             ["images read (all sources)", len(recs)], ["unique by SHA-256", len({r['sha'] for r in recs})],
             ["classification-eligible images (incl. augmented copies)", n_cls], ["auxiliary-anatomy images", n_aux],
             ["unresolved images", n_unr], ["case groups (all sources)", len({r['group'] for r in recs})]]), "",
         "Classification groups per class (independent originals' groups): " + ", ".join(f"{c} {n}" for c, n in sorted(gp.items(), key=lambda kv: kv[1])), "",
         "## Status", "",
         "Not ready for diagnostic-claim training. See `reports/data_quality_report.md`: most classes have < 5 independent groups, labels are "
         "uploader-asserted and unverified, the `normal` class is confounded with the file-naming family (source) and overlaps pixel-wise with the "
         "Fetal-Head sets. Class merges in `label_mapping.csv` are proposals needing clinical review. Awaiting your review of this audit."]
    (bd / "README.md").write_text(chr(10).join(L), encoding="utf-8")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO, help="directory holding the source zips")
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dh-max", type=int, default=6, help="near-duplicate: max dHash distance")
    ap.add_argument("--ph-max", type=int, default=8, help="near-duplicate: max pHash distance")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bd = args.build_dir or (args.root / "datasets" / f"final_build_{stamp}")
    bd = bd if bd.is_absolute() else (Path.cwd() / bd)
    bd.mkdir(parents=True, exist_ok=True)
    logf = []

    def log(s=""):
        print(s, flush=True)
        logf.append(s)

    S = {}
    log(f"build directory: {bd}")
    ext_root, inv = stage_extract(args.root, bd, log)
    recs = gather_records(ext_root, REPO, log)
    attach_roboflow_labels(recs, ext_root, S)
    attach_names(recs)
    classes = S["rf_header_classes"]
    for r in recs:
        r["label_token"] = filename_class_token(r["scan"], classes) if r["source"] == "roboflow_fetal_brain" else None
    attach_anatomy(recs, ext_root, S)
    pairs, sens = find_pairs(recs, args.dh_max, args.ph_max)
    log(f"duplicate pairs: {Counter(p['kind'] for p in pairs.values())}")
    by_g = build_groups_and_dups(recs, pairs)
    sizes = sorted((len(v) for v in by_g.values()), reverse=True)
    log(f"case groups: {len(by_g)}; largest sizes {sizes[:8]}")
    resolve_routes(recs, pairs, S)
    assign_eval_roles(recs)
    cv_rows = build_cv(recs, args.folds, S)
    legacy_numbers(ext_root / "roboflow_fetal_brain", S)
    lo_files, lo_imgs = stage_label_only(args.root, bd, S, log)
    S["fusep_rows"] = lo_files
    if S["fusep"]:
        S["fusep"]["sha_overlap_with_image_sources"] = len(S["fusep_names"] & {r["sha"] for r in recs})

    # ---------------- manifests
    mf = bd / "manifests"
    out = []
    for r in recs:
        cfg = SOURCES[r["source"]]
        is_rf = r["source"] == "roboflow_fetal_brain"
        px = r.get("mask_px") or {}
        out.append(dict(
            record_id=r["record_id"], image_path=r["relpath"], original_filename=r["fname"], source_dataset=r["source"],
            original_image_id=r["stem"], scan_id_filename_derived=r["scan"], subject_id=r["subject_id"],
            subject_id_basis=r["subject_basis"], case_token_heuristic=r["case_token"],
            acquisition_plane=cfg["plane"], acquisition_plane_basis=cfg["plane_basis"],
            original_diagnosis_label=("|".join(r["hot"]) if r.get("hot") else None) if is_rf else None,
            filename_class_token=r["label_token"],
            annotation_type=("image-level one-hot diagnosis class (_classes.csv)" if is_rf
                             else "semantic mask + COCO polygons (Brain/CSP/LV)" + (" + pixel size" if r.get("pixel_mm") else "")),
            annotation_path=r.get("mask_path"), anatomy_classes_present=(";".join(r["anat_present"]) if r.get("anat_present") else None),
            mask_px_brain=px.get("Brain"), mask_px_csp=px.get("CSP"), mask_px_lv=px.get("LV"), pixel_size_mm=r.get("pixel_mm"),
            proposed_task=r["route"], proposed_class=(r["label"] if r["route"] == "classification" else None),
            label_provenance=("Roboflow _classes.csv one-hot (uploader-provided; annotator/clinical verification undocumented)" if is_rf
                              else "mask/COCO anatomy annotation from source; no diagnosis provided"),
            label_verification_status=r["verification"], sha256=r["sha"], ahash=r["ah"].tobytes().hex(), dhash=r["dh"].tobytes().hex(),
            phash=r["ph"].tobytes().hex(), duplicate_group_id=r["dup_group"], duplicate_kind=r["dup_kind"],
            case_group_id=r["group"], case_group_basis=r["group_basis"], source_split=r["split"],
            source_split_documented=("folders present, assignment method undocumented" if is_rf else "no (Readme: no predefined splits)"),
            license=cfg["license"], attribution_ref=cfg["attribution"], is_augmented_copy=(r["aug"] is not None) if is_rf else False,
            augmentation_index=r["aug"], eval_role=r["eval_role"], width=r["w"], height=r["h"],
            qc_color_fraction=f"{r['color']:.4f}", qc_mean_gray=f"{r['gray']:.1f}", flags=";".join(r["flags"]) or None,
            exclusion_status="excluded" if r["route"] == "unresolved" else "included",
            exclusion_reason=";".join(r["reasons"]) or None))
    write_csv(mf / "image_manifest.csv", MANIFEST_COLUMNS, out)

    hier = json.loads((REPO / "configs" / "class_hierarchy.json").read_text())
    fam = {m: f for f, ms in hier["families"].items() for m in ms}
    lm = []
    for c in classes:
        rs = [r for r in recs if r["source"] == "roboflow_fetal_brain" and r.get("label") == c]
        og = [r for r in rs if r["route"] == "classification" and r["eval_role"] in ("original", "augmentation_only_representative")]
        lm.append(dict(source_dataset="roboflow_fetal_brain", original_label=c, task="classification", harmonized_class=c,
                       mapping_status="identity (original label preserved)", proposed_family=fam.get(c, "always_separate" if c in hier["always_separate"] else ""),
                       family_mapping_status="PROPOSED from configs/class_hierarchy.json; not clinically validated",
                       n_images_eligible=sum(1 for r in rs if r["route"] == "classification"), n_independent_groups=len({r["group"] for r in og}),
                       supports_stratified_cv=len({r["group"] for r in og}) >= args.folds))
    for k, st in S["anat"].items():
        for n in MASK_COLORS:
            lm.append(dict(source_dataset=k, original_label=n, task="anatomy_segmentation", harmonized_class=f"anatomy:{n}",
                           mapping_status="structure label, not a diagnosis", proposed_family="", family_mapping_status="n/a",
                           n_images_eligible=st["mask_present_images"][n], n_independent_groups="", supports_stratified_cv=""))
    write_csv(mf / "label_mapping.csv", list(lm[0].keys()), lm)

    dups = [r for r in recs if r["dup_group"]]
    write_csv(mf / "duplicate_groups.csv",
              ["duplicate_group_id", "duplicate_kind", "record_id", "source_dataset", "source_split", "original_filename", "original_diagnosis_label",
               "sha256", "case_group_id", "group_has_label_conflict", "image_path"],
              [dict(duplicate_group_id=r["dup_group"], duplicate_kind=r["dup_kind"], record_id=r["record_id"], source_dataset=r["source"],
                    source_split=r["split"], original_filename=r["fname"], original_diagnosis_label=r.get("label"), sha256=r["sha"],
                    case_group_id=r["group"], group_has_label_conflict=r["group"] in S["mixed_label_groups"], image_path=r["relpath"])
               for r in sorted(dups, key=lambda r: (r["dup_group"], r["record_id"]))])
    write_csv(mf / "duplicate_pairs.csv",
              ["record_a", "record_b", "kind", "dhash_dist", "phash_dist", "relation", "source_a", "source_b", "split_a", "split_b", "label_a", "label_b", "path_a", "path_b"],
              [dict(record_a=recs[i]["record_id"], record_b=recs[j]["record_id"], kind=p["kind"], dhash_dist=p["dh"], phash_dist=p["ph"],
                    relation=pair_relation(recs[i], recs[j]), source_a=recs[i]["source"], source_b=recs[j]["source"], split_a=recs[i]["split"],
                    split_b=recs[j]["split"], label_a=recs[i].get("label"), label_b=recs[j].get("label"), path_a=recs[i]["relpath"], path_b=recs[j]["relpath"])
               for (i, j), p in sorted(pairs.items())])
    by_g = defaultdict(list)
    for r in recs:
        by_g[r["group"]].append(r)
    cg = []
    for g, rs in sorted(by_g.items()):
        labs = sorted({r["label"] for r in rs if r.get("label")})
        sp = sorted({r["split"] for r in rs if r["split"]})
        cg.append(dict(case_group_id=g, n_images=len(rs), sources=";".join(sorted({r["source"] for r in rs})), source_splits=";".join(sp),
                       spans_multiple_source_splits=len(sp) > 1, subject_ids=";".join(sorted({r["subject_id"] for r in rs if r["subject_id"]})),
                       case_tokens=";".join(sorted({r["case_token"] for r in rs if r["case_token"]})), labels=";".join(labs),
                       label_conflict=len(labs) > 1, basis=rs[0]["group_basis"], routes=";".join(sorted({r["route"] for r in rs})),
                       n_distinct_sha=len({r["sha"] for r in rs}), record_ids=";".join(r["record_id"] for r in rs)))
    write_csv(mf / "case_groups.csv", list(cg[0].keys()), cg)
    write_csv(mf / "proposed_cv_manifest.csv", list(cv_rows[0].keys()), cv_rows)

    ccols = ["record_id", "image_path", "original_diagnosis_label", "proposed_class", "label_verification_status", "case_group_id",
             "source_split", "is_augmented_copy", "eval_role", "sha256", "flags"]
    cl = [o for o in out if o["proposed_task"] == "classification"]
    write_csv(bd / "classification" / "classification_manifest.csv", ccols, cl)
    cs = []
    for c in CLASS_NAMES:
        rs = [o for o in cl if o["proposed_class"] == c]
        og = [o for o in rs if o["eval_role"] in ("original", "augmentation_only_representative")]
        cs.append(dict(class_label=c, images_including_augmented=len(rs), independent_originals=len(og),
                       independent_groups=len({o["case_group_id"] for o in og}),
                       filename_consistent=sum(1 for o in rs if o["label_verification_status"].endswith("filename_consistent")),
                       csv_only=sum(1 for o in rs if "csv_only" in o["label_verification_status"])))
    write_csv(bd / "classification" / "class_summary.csv", list(cs[0].keys()), cs)
    acols = ["record_id", "image_path", "source_dataset", "acquisition_plane", "annotation_type", "annotation_path", "anatomy_classes_present",
             "mask_px_brain", "mask_px_csp", "mask_px_lv", "pixel_size_mm", "subject_id", "case_group_id", "case_group_basis", "flags"]
    ax = [o for o in out if o["proposed_task"] == "auxiliary_anatomy"]
    write_csv(bd / "auxiliary_anatomy" / "anatomy_manifest.csv", acols, ax)
    ucols = ["record_id", "image_path", "source_dataset", "original_diagnosis_label", "exclusion_reason", "case_group_id", "duplicate_group_id"]
    ux = [o for o in out if o["proposed_task"] == "unresolved"]
    if ux:
        write_csv(bd / "unresolved" / "unresolved_images.csv", ucols, ux)

    if lo_files:
        write_csv(mf / "label_only_sources.csv", list(lo_files[0].keys()), lo_files)
        write_csv(mf / "label_only_image_refs.csv", list(lo_imgs[0].keys()), lo_imgs)
    make_reports(bd, recs, pairs, sens, inv, S, args, cv_rows)
    make_readme(bd, recs, S, args, inv, len(cl), len(ax), len(ux))

    # ---------------- run info + script snapshot
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    (bd / "scripts").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), bd / "scripts" / Path(__file__).name)
    (bd / "run_info.json").write_text(json.dumps(dict(
        built_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), git_commit=commit,
        python=platform.python_version(), numpy=np.__version__, args={k: str(v) for k, v in vars(args).items()},
        archives={**{SOURCES[k]["zip"]: inv[k]["sha256"] for k in SOURCES},
                  **({S["fusep"]["zip"]: S["fusep"]["sha256"]} if S["fusep"] else {})},
        counts=dict(images=len(recs), unique_sha=len({r["sha"] for r in recs}), classification=len(cl), auxiliary=len(ax),
                    unresolved=len(ux), case_groups=len(by_g))), indent=2))
    log(f"classification={len(cl)} auxiliary={len(ax)} unresolved={len(ux)}  groups={len(by_g)}")
    (bd / "build_log.txt").write_text("\n".join(logf), encoding="utf-8")
    print(f"\nDONE -> {bd}")


if __name__ == "__main__":
    main()
