# Unified fetal-brain ultrasound dataset build

Audit-only build. **No model was trained and no evaluation protocol was changed.** Original archives were not modified.

## Reproduce

```bash
# from the repository root (zips in the root; Python 3.10, numpy, Pillow, scipy)
python scripts/14_build_unified_dataset.py                       # new datasets/final_build_<timestamp>/
python scripts/14_build_unified_dataset.py --build-dir final_build_20261022_010449   # rerun in place (idempotent)
python -m pytest tests/test_unified_build.py -q                  # validation tests (use the newest build)
```

Archive checksums are in `run_info.json`; parameters: near-duplicate = dHash<=6 and pHash<=8, folds = 5.

## Layout (and deviations from the requested layout)

- `extracted/<source>/` safe extraction, one directory per source (git-ignored). Only the Segmentation and COCO nested archives were extracted; the other 9 redundant export formats per Fetal-Head zip are inventoried only.
- `manifests/` image_manifest.csv, label_mapping.csv, duplicate_groups.csv, case_groups.csv, proposed_cv_manifest.csv plus one extra file, `duplicate_pairs.csv` (pairwise evidence with hash distances).
- `classification/`, `auxiliary_anatomy/`, `unresolved/` hold **CSV manifests only** (no image copies; paths point into `extracted/`) to avoid duplicating ~1 GB. `classification/class_summary.csv` is an extra summary.
- `manifests/label_only_sources.csv`, `label_only_image_refs.csv`: the FUSEP label-only archive (no images, no diagnoses; audited, not routed to any training set).
- `reports/` dataset_inventory, label_audit, duplicate_audit, leakage_audit, data_quality_report.
- `scripts/` is a snapshot of the pipeline script used; the maintained copy is `scripts/14_build_unified_dataset.py` in the repository.
- Empty cells in CSVs mean null/unavailable (never a guessed value).

## Headline numbers

| item | value |
|---|---|
| images read (all sources) | 4035 |
| unique by SHA-256 | 3912 |
| classification-eligible images (incl. augmented copies) | 1651 |
| auxiliary-anatomy images | 2267 |
| unresolved images | 117 |
| case groups (all sources) | 1751 |

Classification groups per class (independent originals' groups): holoprosencephaly 1, m-magna 1, intracranial-tumor 1, colphocephaly 2, hydracenphaly 2, vein-of-galen 2, severe-ventriculomegaly 3, intracranial-hemorrdge 3, cerebellah-hypoplasia 4, anold-chiari-malformation 4, arachnoid-cyst 5, polencephaly 5, encephalocele 7, moderate-ventriculomegaly 8, mild-ventriculomegaly 9, normal 31

## Status

Not ready for diagnostic-claim training. See `reports/data_quality_report.md`: most classes have < 5 independent groups, labels are uploader-asserted and unverified, the `normal` class is confounded with the file-naming family (source) and overlaps pixel-wise with the Fetal-Head sets. Class merges in `label_mapping.csv` are proposals needing clinical review. Awaiting your review of this audit.