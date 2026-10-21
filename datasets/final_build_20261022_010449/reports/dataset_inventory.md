# Dataset inventory

Verified by reading the archives (counts are from the files, not from README claims). Nothing was assumed from file or folder names alone.

| source id | archive | size | images read | file types in archive | zip integrity |
|---|---|---|---|---|---|
| roboflow_fetal_brain | Fetal Brain Abnormalities Ultrasound.v1i.multiclass.zip | 57 MB | 1768 | .csv:3, .jpg:1768, .txt:2 | ok |
| fhu_trans_cerebellum | Trans-cerebellum.zip | 274 MB | 684 | .csv:1, .png:684, .zip:11 | ok |
| fhu_trans_ventricular | Trans-ventricular.zip | 232 MB | 584 | .csv:1, .png:584, .zip:11 | ok |
| fhu_diverse_head | Diverse Fetal Head Images.zip | 313 MB | 999 | .png:999, .zip:11 | ok |

SHA-256 of each archive (provenance):

- `Fetal Brain Abnormalities Ultrasound.v1i.multiclass.zip`: `761eeebd1b950d3221da5aa1aea097af159cd9211418912a685572a97b67cddb`
- `Trans-cerebellum.zip`: `00b8dcb09591a2c583ba9a7cc53195c79754119be98f6d8cf5f70cde826523a8`
- `Trans-ventricular.zip`: `f76cb9e1762a794fa8a35f5f59f893410f2ff0e797a28bdad4d4df923e4109e5`
- `Diverse Fetal Head Images.zip`: `5229b8256fb498916bc77e853b8b7509c1a1d8d3e431998ae292b3e5ae0cab52`

## Source descriptions

### roboflow_fetal_brain (diagnosis classification)
- Roboflow Universe export, `README.dataset.txt`: license CC BY 4.0, 'Provided by a Roboflow user'. No clinical annotator, ground-truth procedure, patient or acquisition metadata are documented.
- Labels: one-hot columns in `<split>/_classes.csv` (16 diagnosis/normal classes), i.e. DIAGNOSIS labels at image level.
- `README.roboflow.txt` says 'No image augmentation techniques were applied' and that images were resized 640x640 (stretch). File names, however, contain `_aug_N` tags: 1338 of 1768 images are augmented copies produced BEFORE or OUTSIDE this export (README claim contradicted by the files).
- Source split: train/valid/test folders exist (train: 1418, valid: 174, test: 176); how the split was made is not documented.

### fhu_* = 'Fetal Head Ultrasound Images' (anatomy)
- `Readme.txt`: CC BY 4.0; anonymised images; classes Brain / CSP / LV; 'No predefined training and testing splits'; four plane-group zips; pixel-size CSV per zip; 11 export formats per zip. The README names no original author, DOI or URL (the download link is redacted: `XXXX...`), so the attribution required by CC BY 4.0 cannot be completed from the repository alone.
- Labels: segmentation masks / COCO polygons for anatomical structures; **no diagnosis or normal/abnormal label exists**.
- Image naming `PatientNNNNN_PlaneP_i_of_n.png` (two Trans-* sets) carries a patient token; the Diverse set uses `NNN_HC.png` with no subject information and also contains no pixel-size CSV (README says each zip has one).

## README claims vs what is present

- README: 3832 images in four zips (Trans-thalamic, Trans-ventricular, Trans-cerebellum, Diverse). Present: 2267 images in three zips. **Trans-thalamic.zip is not in the repository**, so 1565 images (40.8%) of the described dataset are unavailable here.
- README class instances: Brain 3,794 / CSP 1,865 / LV 1,512. Instances counted in the COCO files present:
  Brain 2235, CSP 1238, LV 992 (remainder presumably in the missing Trans-thalamic set; not verifiable).

## Annotation formats and acquisition planes

| source | annotation type | label meaning | plane | plane basis | subject ids | documented split |
|---|---|---|---|---|---|---|
| roboflow_fetal_brain | image-level one-hot class | diagnosis / normal | unknown | not documented | none documented (Patient##### tokens exist in file names of the 'normal' class) | folders only; method undocumented |
| fhu_trans_cerebellum | mask + COCO polygons + pixel size mm | anatomy (Brain, CSP, LV) | trans-cerebellar | source archive/folder name (Readme.txt: fetal plane groups) | Patient##### in filename | none (README) |
| fhu_trans_ventricular | mask + COCO polygons + pixel size mm | anatomy (Brain, CSP, LV) | trans-ventricular | source archive/folder name (Readme.txt: fetal plane groups) | Patient##### in filename | none (README) |
| fhu_diverse_head | mask + COCO polygons (no pixel-size csv) | anatomy (Brain, CSP, LV) | diverse/unspecified | source archive name (mixed head planes; per-image plane not documented) | none | none (README) |

## Nested export-format archives

Each Fetal-Head zip carries 11 redundant re-exports of the same annotations. Only the Segmentation (mask) and COCO exports were extracted (they are needed for annotation audit); the others were listed but not extracted to avoid copying ~10x duplicated images.

| source | nested archive | extracted | members | types |
|---|---|---|---|---|
| fhu_trans_cerebellum | Trans-cerebellum-Cityscapes.zip | no | 2053 | {'.txt': 1, '.png': 2052} |
| fhu_trans_cerebellum | Trans-cerebellum-COCO.zip | yes | 1 | {'.json': 1} |
| fhu_trans_cerebellum | Trans-cerebellum-CVAT.zip | no | 1 | {'.xml': 1} |
| fhu_trans_cerebellum | Trans-cerebellum-Datumaro.zip | no | 1 | {'.json': 1} |
| fhu_trans_cerebellum | Trans-cerebellum-ICDAR-Localization.zip | no | 684 | {'.txt': 684} |
| fhu_trans_cerebellum | Trans-cerebellum-LabelMe.zip | no | 1365 | {'.xml': 684, '.png': 681} |
| fhu_trans_cerebellum | Trans-cerebellum-OpenImage.zip | no | 965 | {'.csv': 4, '.json': 1, '.meta': 1, '.png': 959} |
| fhu_trans_cerebellum | Trans-cerebellum-PASCAL.zip | no | 2049 | {'.txt': 3, '.png': 1362, '.xml': 684} |
| fhu_trans_cerebellum | Trans-cerebellum-Segmentation.zip | yes | 1370 | {'.txt': 2, '.png': 1368} |
| fhu_trans_cerebellum | Trans-cerebellum-TFRecord.zip | no | 2 | {'.tfrecord': 1, '.pbtxt': 1} |
| fhu_trans_cerebellum | Trans-cerebellum-YOLO.zip | no | 687 | {'.txt': 685, '.names': 1, '.data': 1} |
| fhu_trans_ventricular | Trans-ventricular-CityScapes.zip | no | 1753 | {'.txt': 1, '.png': 1752} |
| fhu_trans_ventricular | Trans-ventricular-COCO.zip | yes | 1 | {'.json': 1} |
| fhu_trans_ventricular | Trans-ventricular-CVAT.zip | no | 1 | {'.xml': 1} |
| fhu_trans_ventricular | Trans-ventricular-Datumaro.zip | no | 1 | {'.json': 1} |
| fhu_trans_ventricular | Trans-ventricular-ICDAR-Localization.zip | no | 584 | {'.txt': 584} |
| fhu_trans_ventricular | Trans-ventricular-LabelMe.zip | no | 1139 | {'.xml': 584, '.png': 555} |
| fhu_trans_ventricular | Trans-ventricular-Open-Images.zip | no | 1285 | {'.csv': 4, '.json': 1, '.meta': 1, '.png': 1279} |
| fhu_trans_ventricular | Trans-ventricular-PASCAL-VOC .zip | no | 1697 | {'.txt': 3, '.png': 1110, '.xml': 584} |
| fhu_trans_ventricular | Trans-ventricular-segmentation-mask.zip | yes | 1170 | {'.txt': 2, '.png': 1168} |
| fhu_trans_ventricular | Trans-ventricular-TFRecord.zip | no | 2 | {'.tfrecord': 1, '.pbtxt': 1} |
| fhu_trans_ventricular | Trans-ventricular-YOLO-1.1-.zip | no | 587 | {'.txt': 585, '.names': 1, '.data': 1} |
| fhu_diverse_head | Test-Dataset-CitySpaces.zip | no | 2998 | {'.txt': 1, '.png': 2997} |
| fhu_diverse_head | Test-Dataset-COCO.zip | yes | 1 | {'.json': 1} |
| fhu_diverse_head | Test-Dataset-CVAT.zip | no | 1 | {'.xml': 1} |
| fhu_diverse_head | Test-Dataset-Datumaro.zip | no | 1 | {'.json': 1} |
| fhu_diverse_head | Test-Dataset-ICDAR-Localization.zip | no | 999 | {'.txt': 999} |
| fhu_diverse_head | Test-Dataset-LabelMe.zip | no | 1998 | {'.xml': 999, '.png': 999} |
| fhu_diverse_head | Test-Dataset-OpenImage.zip | no | 1816 | {'.csv': 4, '.json': 1, '.meta': 1, '.png': 1810} |
| fhu_diverse_head | Test-Dataset-PASCAL.zip | no | 3000 | {'.txt': 3, '.png': 1998, '.xml': 999} |
| fhu_diverse_head | Test-Dataset-Segmentation.zip | yes | 2000 | {'.txt': 2, '.png': 1998} |
| fhu_diverse_head | Test-Dataset-TFRecord.zip | no | 2 | {'.tfrecord': 1, '.pbtxt': 1} |
| fhu_diverse_head | Test-Dataset-YOLO.zip | no | 1002 | {'.txt': 1000, '.names': 1, '.data': 1} |

## Extraction report

- roboflow_fetal_brain: 0 files written, 1773 already present, unsafe paths skipped: 0
- fhu_trans_cerebellum: 0 files written, 685 already present, unsafe paths skipped: 0
- fhu_trans_ventricular: 0 files written, 585 already present, unsafe paths skipped: 0
- fhu_diverse_head: 0 files written, 999 already present, unsafe paths skipped: 0

Totals: 4035 images read, 3912 unique by SHA-256 across all sources. Unreadable images: 0 (all decoded).

## Label-only archive: FUSEP (`ahuteam FUSEP main labels.zip`)

- 45 entries, 42 JSON files, 30 MB, zip integrity ok; SHA-256 `4ca9ca60b24a16a0964a649534d10ae08c376997affb5e4bdc65f05d66814b25`. **It contains no images**, no README and no licence text (licence and attribution: unknown; obtain them from the dataset owners).
- Content: COCO-style bounding boxes in 3 hospital folders (Hospital1_SZ, Hospital2_SC, Hospital3_YN), two tasks (CRL, NT; taken from file names, their expansion is not documented in the archive), with `train`, `test` and semi-supervised subsets `label_5/label_10/unlabel_5/unlabel_10`.
- Each box carries a plane-quality value in Chinese (standard / non-standard / basically standard). The English renderings are my translation and should be confirmed.
- Box categories (anatomical landmarks): Hospital1_SZ: AB, B, C, DP, G, H, LV, MDS, MLS, MX, NB, NT, NTAPS, RBP; Hospital2_SC: AB, B, C, DP, G, H, LV, MDS, MLS, MX, NB, NT, NTAPS, RBP; Hospital3_YN: AB, B, C, DP, G, H, LV, MDS, MLS, MX, NB, NT, NTAPS, RBP
- 4017 distinct hashed image names are referenced; **0** of them equal the SHA-256 of an image in the other archives (no overlap detectable; the images themselves are unavailable here).
- Annotation type: landmark/structure bounding boxes + image-quality labels. **No diagnosis, no normal/abnormal label, no mask.** It is first-trimester-style data (CRL / NT), a different gestational stage and task from the fetal-brain diagnosis set.
- Routing: `label_only` -> not in `image_manifest.csv`, `classification/` or `auxiliary_anatomy/` (no images to attach labels to). Tables: `manifests/label_only_sources.csv` (one row per json) and `manifests/label_only_image_refs.csv` (one row per referenced image; raw `case_id` values are replaced by a 12-char hash so hospital identifiers are not copied).

| hospital | task | role | images | boxes | distinct case_ids | plane quality |
|---|---|---|---|---|---|---|
| Hospital1_SZ | CRL | full | 766 | 6875 | 756 | basically standard=43;non-standard=4327;standard=2505 |
| Hospital1_SZ | CRL | label_10 | 61 | 539 | 61 | non-standard=358;standard=181 |
| Hospital1_SZ | CRL | label_5 | 30 | 273 | 30 | non-standard=190;standard=83 |
| Hospital1_SZ | CRL | test | 154 | 1395 | 154 | basically standard=12;non-standard=811;standard=572 |
| Hospital1_SZ | CRL | train | 612 | 5480 | 606 | basically standard=31;non-standard=3516;standard=1933 |
| Hospital1_SZ | CRL | unlabel_10 | 551 | 0 | 547 | - |
| Hospital1_SZ | CRL | unlabel_5 | 582 | 0 | 576 | - |
| Hospital1_SZ | NT | full | 766 | 6610 | 753 | basically standard=17;non-standard=4271;standard=2322 |
| Hospital1_SZ | NT | label_10 | 61 | 518 | 61 | non-standard=325;standard=193 |
| Hospital1_SZ | NT | label_5 | 30 | 253 | 30 | non-standard=187;standard=66 |
| Hospital1_SZ | NT | test | 154 | 1311 | 154 | non-standard=877;standard=434 |
| Hospital1_SZ | NT | train | 612 | 5299 | 607 | basically standard=17;non-standard=3394;standard=1888 |
| Hospital1_SZ | NT | unlabel_10 | 551 | 0 | 546 | - |
| Hospital1_SZ | NT | unlabel_5 | 582 | 0 | 578 | - |
| Hospital2_SC | CRL | full | 737 | 7323 | 1 | basically standard=40;non-standard=2341;standard=4942 |
| Hospital2_SC | CRL | label_10 | 58 | 562 | 1 | non-standard=197;standard=365 |
| Hospital2_SC | CRL | label_5 | 29 | 270 | 1 | non-standard=93;standard=177 |
| Hospital2_SC | CRL | test | 148 | 1464 | 1 | basically standard=11;non-standard=420;standard=1033 |
| Hospital2_SC | CRL | train | 589 | 5859 | 1 | basically standard=29;non-standard=1921;standard=3909 |
| Hospital2_SC | CRL | unlabel_10 | 531 | 0 | 1 | - |
| Hospital2_SC | CRL | unlabel_5 | 560 | 0 | 1 | - |
| Hospital2_SC | NT | full | 759 | 6958 | 1 | basically standard=11;non-standard=3320;standard=3627 |
| Hospital2_SC | NT | label_10 | 60 | 531 | 1 | non-standard=288;standard=243 |
| Hospital2_SC | NT | label_5 | 30 | 271 | 1 | non-standard=92;standard=179 |
| Hospital2_SC | NT | test | 152 | 1417 | 1 | non-standard=729;standard=688 |
| Hospital2_SC | NT | train | 607 | 5541 | 1 | basically standard=11;non-standard=2591;standard=2939 |
| Hospital2_SC | NT | unlabel_10 | 547 | 0 | 1 | - |
| Hospital2_SC | NT | unlabel_5 | 577 | 0 | 1 | - |
| Hospital3_YN | CRL | full | 486 | 9044 | 482 | basically standard=96;non-standard=5104;standard=3844 |
| Hospital3_YN | CRL | label_10 | 38 | 706 | 38 | non-standard=346;standard=360 |
| Hospital3_YN | CRL | label_5 | 19 | 344 | 19 | non-standard=204;standard=140 |
| Hospital3_YN | CRL | test | 98 | 1858 | 98 | basically standard=36;non-standard=948;standard=874 |
| Hospital3_YN | CRL | train | 388 | 7186 | 385 | basically standard=60;non-standard=4156;standard=2970 |
| Hospital3_YN | CRL | unlabel_10 | 350 | 0 | 348 | - |
| Hospital3_YN | CRL | unlabel_5 | 369 | 0 | 367 | - |
| Hospital3_YN | NT | full | 503 | 9010 | 488 | basically standard=36;non-standard=6562;standard=2412 |
| Hospital3_YN | NT | label_10 | 40 | 642 | 40 | basically standard=20;non-standard=474;standard=148 |
| Hospital3_YN | NT | label_5 | 20 | 340 | 20 | non-standard=224;standard=116 |
| Hospital3_YN | NT | test | 101 | 1836 | 101 | non-standard=1382;standard=454 |
| Hospital3_YN | NT | train | 402 | 7172 | 388 | basically standard=36;non-standard=5178;standard=1958 |
| Hospital3_YN | NT | unlabel_10 | 362 | 0 | 354 | - |
| Hospital3_YN | NT | unlabel_5 | 382 | 0 | 371 | - |

Split-hygiene findings (verified from the JSON):

- Hospital1_SZ/CRL: 4 case_ids present in BOTH train and test (patient-level leakage)
- Hospital1_SZ/NT: 8 case_ids present in BOTH train and test (patient-level leakage)
- Hospital3_YN/CRL: 1 case_ids present in BOTH train and test (patient-level leakage)
- Hospital3_YN/NT: 1 case_ids present in BOTH train and test (patient-level leakage)
- Hospital2_SC: every image has the same case_id placeholder -> no usable subject information