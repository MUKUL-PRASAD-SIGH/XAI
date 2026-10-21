# Data-quality report

## Independent evidence per class (classification-eligible, after conflict removal)

| class | images incl. augmented | independent originals | independent groups | largest group (images) | 5-fold stratified |
|---|---|---|---|---|---|
| anold-chiari-malformation | 50 | 4 | 4 | 1 | TOO FEW GROUPS |
| arachnoid-cyst | 139 | 17 | 5 | 6 | OK (>=5 groups) |
| cerebellah-hypoplasia | 83 | 13 | 4 | 4 | TOO FEW GROUPS |
| colphocephaly | 73 | 13 | 2 | 7 | TOO FEW GROUPS |
| encephalocele | 146 | 24 | 7 | 6 | OK (>=5 groups) |
| holoprosencephaly | 12 | 3 | 1 | 3 | TOO FEW GROUPS |
| hydracenphaly | 20 | 2 | 2 | 1 | TOO FEW GROUPS |
| intracranial-hemorrdge | 58 | 7 | 3 | 5 | TOO FEW GROUPS |
| intracranial-tumor | 14 | 1 | 1 | 1 | TOO FEW GROUPS |
| m-magna | 38 | 5 | 1 | 5 | TOO FEW GROUPS |
| mild-ventriculomegaly | 246 | 41 | 9 | 7 | OK (>=5 groups) |
| moderate-ventriculomegaly | 261 | 36 | 8 | 11 | OK (>=5 groups) |
| normal | 282 | 210 | 31 | 13 | OK (>=5 groups) |
| polencephaly | 98 | 20 | 5 | 6 | OK (>=5 groups) |
| severe-ventriculomegaly | 126 | 20 | 3 | 9 | TOO FEW GROUPS |
| vein-of-galen | 5 | 5 | 2 | 3 | TOO FEW GROUPS |

Group counts are the honest sample size; images inside one group are near-copies and are not independent evidence.

## Source / acquisition shortcut checks

A diagnosis that is confounded with where the image came from can be 'predicted' from non-anatomical cues. Per-class descriptors of the Roboflow images:

| class | images | filename family | mean colour-pixel fraction | mean gray level |
|---|---|---|---|---|
| anold-chiari-malformation | 50 | {'copy-of': 50} | 0.000 | 72.9 |
| arachnoid-cyst | 139 | {'copy-of': 139} | 0.000 | 63.3 |
| cerebellah-hypoplasia | 130 | {'copy-of': 130} | 0.000 | 59.2 |
| colphocephaly | 90 | {'copy-of': 90} | 0.000 | 60.4 |
| encephalocele | 146 | {'copy-of': 146} | 0.000 | 51.1 |
| holoprosencephaly | 12 | {'copy-of': 12} | 0.000 | 73.7 |
| hydracenphaly | 20 | {'copy-of': 20} | 0.000 | 89.6 |
| intracranial-hemorrdge | 58 | {'copy-of': 58} | 0.000 | 69.3 |
| intracranial-tumor | 14 | {'copy-of': 14} | 0.000 | 61.0 |
| m-magna | 38 | {'copy-of': 38} | 0.000 | 61.4 |
| mild-ventriculomegaly | 246 | {'copy-of': 246} | 0.000 | 79.4 |
| moderate-ventriculomegaly | 314 | {'copy-of': 314} | 0.000 | 66.2 |
| normal | 282 | {'patient': 282} | 0.000 | 45.8 |
| polencephaly | 98 | {'copy-of': 98} | 0.000 | 72.7 |
| severe-ventriculomegaly | 126 | {'copy-of': 126} | 0.000 | 62.9 |
| vein-of-galen | 5 | {'copy-of': 5} | 0.000 | 58.4 |

- All 282 `normal` images use the `Patient#####_Plane#_i_of_n` naming; 0 abnormal images do. Every abnormal image uses the `Copy-of-<class>-...` naming. **Normal vs abnormal is perfectly confounded with image family/source.** No split can remove this; a classifier can separate the families without learning pathology.
- `Plane3` appears in all Fetal-Head file names; its meaning is not documented in the Readme and is not interpreted here.
- Fetal-Head images are PNG [(959, 661)]...; Roboflow images are JPEG 640x640 re-renderings (format/resolution is another source cue).

## Class-merge proposals

`manifests/label_mapping.csv` records the proposed family mapping from `configs/class_hierarchy.json` / `02c_design_classes.py` next to the preserved original labels. These are statistical/anatomical groupings proposed to enable evaluation, **not clinically validated**; they require review by a clinician. Original labels are never overwritten.

## Proposed evaluation design

- Classes with >= 5 independent groups (6): arachnoid-cyst (5), encephalocele (7), mild-ventriculomegaly (9), moderate-ventriculomegaly (8), normal (31), polencephaly (5)
- Classes with < 5 groups (10), NOT evaluable with stratified 5-fold: anold-chiari-malformation (4), cerebellah-hypoplasia (4), colphocephaly (2), holoprosencephaly (1), hydracenphaly (2), intracranial-hemorrdge (3), intracranial-tumor (1), m-magna (1), severe-ventriculomegaly (3), vein-of-galen (2)

Scored test images per class and fold (proposed layout):

| class | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| arachnoid-cyst | 6 | 5 | 4 | 1 | 1 |
| encephalocele | 4 | 5 | 4 | 5 | 6 |
| mild-ventriculomegaly | 7 | 9 | 9 | 7 | 9 |
| moderate-ventriculomegaly | 6 | 6 | 7 | 3 | 6 |
| normal | 43 | 43 | 41 | 42 | 41 |
| polencephaly | 3 | 3 | 4 | 6 | 4 |

Recommendation:
1. Groups are the unit. With 88 classification groups overall and a handful per rare class, per-class metrics on 5 folds rest on 1-2 groups per cell and will be very noisy; report pooled results with group-bootstrap confidence intervals and the majority baseline.
2. Do not tune on test folds: select hyper-parameters on the inner validation split (carved from training groups) or on a fixed pre-registered configuration, and run the 5 test folds once.
3. Rare classes (list above) should either be merged under a clinician-approved scheme, evaluated only qualitatively (leave-one-group-out, reporting per-group outcomes), or collected further. They should not be presented with per-class accuracy.
4. Because normal vs abnormal is confounded with source, add a source-controlled check (e.g. evaluate abnormal-only classes) before claiming diagnostic performance.

## Exclusions

- Records routed to `unresolved`: 117

| reason | records |
|---|---|
| linked_group_has_conflicting_labels | 117 |

- `augmented_copy` records stay in the classification manifest as train-only members of their group; they are never test images.
- Records outside classification by design: 2267 anatomy-only images (auxiliary), not labelled normal/abnormal.
