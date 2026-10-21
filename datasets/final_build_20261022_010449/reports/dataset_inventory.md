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