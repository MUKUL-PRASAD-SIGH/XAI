# Leakage audit

## Roboflow source splits (train/valid/test as shipped)

| measure | value |
|---|---|
| valid+test images | 350 |
| of which are augmented copies (`_aug_N`) | 264 |
| scans (filename-derived) present in more than one split | 128 of 212 |
| case tokens present in more than one split | 82 of 93 |
| linked groups (tokens+duplicates) spanning >1 split | 80 of 90 |
| valid+test images whose group also has TRAIN images | 349 of 350 (99.7%) |
| exact-SHA sets spanning >1 split | 0 |

Interpretation: the shipped split is NOT leakage-free. Augmented copies of the same scan were distributed across train/valid/test, so any model evaluated on the shipped test folder is evaluated on re-renderings of training scans.

## Cross-dataset overlap (Roboflow vs Fetal-Head sets)

- Roboflow images carrying a `Patient#####` token: 282 (classes: {'normal': 282}); distinct tokens: 31
- Fetal-Head images carrying a patient token: 1268; distinct tokens: 700
- Patient tokens present in BOTH Roboflow and Fetal-Head sets: **22** (token match only; the file-name convention being identical is suggestive but is not proof of the same patient)
- Roboflow<->Fetal-Head image pairs matching by pixels (exact or near): **42**

  breakdown: {'near': 42}

## Within the Fetal-Head sets

- Patient tokens: 700; tokens with more than one image: 345; tokens appearing in both Trans-cerebellum and Trans-ventricular: 287
- No documented split exists, so there is no source leakage to measure, but a random image-level split would leak patients: use the `case_group_id` / `proposed_cv_manifest.csv`.
- `fhu_diverse_head` has no subject information at all: **patient-level overlap within it and with other sets cannot be ruled out** (only pixel-level duplicates were checked).

## Patient-level statement

Reliable, documented subject identifiers do not exist for the Roboflow data. Groups there are filename-derived heuristics. Subject IDs for the Fetal-Head Trans-* sets are filename tokens; they are used as grouping keys but are not independently documented.

## Verification of previously reported numbers

### 16 original classes

| quantity | value |
|---|---|
| files loaded (legacy loader) | 1768 |
| kept after legacy conflict resolution | 1724 |
| images dropped by label conflict | 44 |
| tied groups dropped | 0 |
| independent groups | 90 |
| original scans | 430 |
| original scans in shipped test folder | 39 |

### configs/class_map_cv.json (manual 5-family map)

| quantity | value |
|---|---|
| files loaded (legacy loader) | 1768 |
| kept after legacy conflict resolution | 1741 |
| images dropped by label conflict | 27 |
| tied groups dropped | 0 |
| independent groups | 90 |
| original scans | 431 |
| original scans in shipped test folder | 39 |

Previously reported: 90 independent groups, 430 scans, 44 images dropped for conflicting labels, 91 evaluation cases.

- **Reproduced exactly** with the legacy grouping on the 16 original classes: groups 90, scans 430, dropped 44.
- The 44 'conflicting' images are not multi-label annotations: every csv row has exactly one class. They come from near-identical images filed under different diagnoses (see `duplicate_audit.md`). In this build such groups are not voted on; they go to `unresolved/`.
- **'91 evaluation cases' could not be reproduced** from any manifest quantity: the shipped test folder holds 176 files = 39 original scans; 430 is the pooled original-scan count over all splits. Treat 91 as unverified.
- The legacy grouping linked only by (label, case token) plus aHash&dHash<=4, whereas this build also links case tokens across labels, patient tokens across sources and pHash, hence slightly different group counts in `case_groups.csv`.