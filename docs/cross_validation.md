# Stratified grouped 5-fold cross-validation

Why: a single 75-image test split (about one image per rare diagnosis) cannot support
per-class claims. Experiments 1-3 showed Xception at 57.3 % vs a 56.0 % majority baseline
(p = 0.456), train 97.8 % vs test 57.3 %, and a normal-vs-abnormal score driven by source.

## Run

```bash
python scripts/02b_make_cv_folds.py --src ./ds --out ./dataset_cv \
    --class-map configs/class_map_cv.json --link
python scripts/03_train.py    --cv ./dataset_cv --model xception --out ./models_cv --mixed-precision
python scripts/04_evaluate.py --cv ./dataset_cv --models ./models_cv --out ./results_cv --tta
```

Read `dataset_cv/cv_report.txt` BEFORE training: it lists, per class, how many
independent groups exist and how many test images land in each fold.

## What the fold builder guarantees

| Property | How |
|---|---|
| Stratified | `StratifiedGroupKFold`, best of 300 seeds scored on empty (class, fold) cells and per-class imbalance |
| Grouped | case / near-duplicate / exact-duplicate clusters built after the class map, so merged families cannot straddle folds |
| Label noise | conflicting labels in one group: majority wins, tie drops the group (logged) |
| Own validation set | carved from the training groups of each fold; test fold never used for early stopping |
| Test purity | valid/test contain true originals only; stored `_aug_N` copies are train-only |
| Verified | per-fold check of group and SHA-256 overlap between splits; run aborts on any leak |
| Every case tested once | checked at the end of the run |

Classes with fewer than `--min-groups` (3) independent groups are excluded and named in the report.

## Overfitting controls in `03_train.py` (all flags)

dropout 0.5, label smoothing 0.1, AdamW weight decay 1e-4, mixup 0.2, dampened class
weights (`--cw-power 0.5`), fewer unfrozen Xception layers (`block13`+), early stopping and
LR schedule on `val_loss` (val macro-F1 is too noisy on a few dozen images).
Training accuracy far above validation accuracy still means overfitting: raise
`--dropout`, `--mixup`, or `--unfreeze-from block14`.

## Reading the results

`results_table.md` reports pooled accuracy and macro-F1 with 95 % case-level bootstrap CIs,
the majority baseline with real precision/recall/F1, and an exact binomial p-value
for accuracy vs the baseline. `predictions_<model>.csv` holds every test prediction for further
analysis. Per-fold mean +/- sd is in the table when there are several folds.

## Limits that code cannot fix

* The normal/abnormal split is confounded with the image source (family column of
  `cv_manifest.csv`). Folds are stratified by class, so normal is still mostly one source.
  A high normal-vs-abnormal score is not evidence of clinical recognition.
* The class families in `configs/class_map_cv.json` are my grouping by anatomy, not a clinical
  ruling. They carry `_review` notes; have a clinician approve them before publishing.
