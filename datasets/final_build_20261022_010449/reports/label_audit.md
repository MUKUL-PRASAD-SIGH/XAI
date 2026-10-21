# Label audit

## Roboflow classification labels

| split | csv rows | image files | rows w/o file | files w/o row | duplicate rows | classes marked per row |
|---|---|---|---|---|---|---|
| train | 1418 | 1418 | 0 | 0 | 0 | {1: 1418} |
| valid | 174 | 174 | 0 | 0 | 0 | {1: 174} |
| test | 176 | 176 | 0 | 0 | 0 | {1: 176} |

Header classes (verbatim, spelling preserved): `anold-chiari-malformation`, `arachnoid-cyst`, `cerebellah-hypoplasia`, `colphocephaly`, `encephalocele`, `holoprosencephaly`, `hydracenphaly`, `intracranial-hemorrdge`, `intracranial-tumor`, `m-magna`, `mild-ventriculomegaly`, `moderate-ventriculomegaly`, `normal`, `polencephaly`, `severe-ventriculomegaly`, `vein-of-galen`

- Multi-label rows: **0**. Zero-label rows: **0**. So 'conflicting labels' in earlier audits are NOT legitimate multi-label annotations; they arise from near-duplicate clustering across classes (see below) -- they must be treated as potential data errors, not as multi-label cases.

## Counts per original class (counting units differ -- this is why 'image', 'scan', 'case' and 'group' numbers disagree)

| class | images | non-augmented images | distinct scans (filename) | distinct case tokens (heuristic) | linked groups (case token + duplicate links) | groups still classification-eligible | images unresolved |
|---|---|---|---|---|---|---|---|
| anold-chiari-malformation | 50 | 4 | 4 | 4 | 4 | 4 | 0 |
| arachnoid-cyst | 139 | 17 | 9 | 5 | 5 | 5 | 0 |
| cerebellah-hypoplasia | 130 | 16 | 8 | 5 | 5 | 4 | 47 |
| colphocephaly | 90 | 14 | 6 | 3 | 3 | 2 | 17 |
| encephalocele | 146 | 24 | 12 | 7 | 7 | 7 | 0 |
| holoprosencephaly | 12 | 3 | 1 | 1 | 1 | 1 | 0 |
| hydracenphaly | 20 | 2 | 2 | 2 | 2 | 2 | 0 |
| intracranial-hemorrdge | 58 | 7 | 5 | 3 | 3 | 3 | 0 |
| intracranial-tumor | 14 | 1 | 1 | 1 | 1 | 1 | 0 |
| m-magna | 38 | 5 | 3 | 1 | 1 | 1 | 0 |
| mild-ventriculomegaly | 246 | 40 | 21 | 10 | 9 | 9 | 0 |
| moderate-ventriculomegaly | 314 | 42 | 19 | 10 | 10 | 8 | 53 |
| normal | 282 | 210 | 102 | 31 | 31 | 31 | 0 |
| polencephaly | 98 | 20 | 8 | 5 | 5 | 5 | 0 |
| severe-ventriculomegaly | 126 | 20 | 8 | 3 | 3 | 3 | 0 |
| vein-of-galen | 5 | 5 | 3 | 2 | 2 | 2 | 0 |

Counting units: *image* = file; *scan* = file name with augmentation/Roboflow tags removed (heuristic); *case token* = trailing case number in the name (heuristic, class-specific) or the `Patient#####` token; *group* = union of case tokens and exact/near-duplicate links. Only the group is used for splitting.

## Label corroboration (csv vs file name)

- csv label equals the diagnosis token in the file name: 1344 images
- csv label contradicts the file name: 0 images
- file name carries no diagnosis token (csv is the only evidence): 424 images; of these 282 are `normal` and 142 are other classes.

**Verification status:** no label in this dataset is clinically verified in the available documentation. `source_asserted;filename_consistent` means the Roboflow csv and the file name agree (two uncontrolled sources, same uploader). `source_asserted;csv_only` means the csv is the only evidence. Neither is ground truth.

## Label conflicts

Groups containing more than one diagnosis: 2.
Roboflow images routed to `unresolved`: 117.

| reason | images |
|---|---|
| linked_group_has_conflicting_labels | 117 |

- `G01693` labels ['cerebellah-hypoplasia', 'moderate-ventriculomegaly']: R02285 `Copy-of-cerebellah-hypoplasia-20a_aug_1_png_jpg.` (cerebellah-hypoplasia); R02286 `Copy-of-cerebellah-hypoplasia-20a_aug_3_png_jpg.` (cerebellah-hypoplasia); R02287 `Copy-of-cerebellah-hypoplasia-20a_aug_5_png_jpg.` (cerebellah-hypoplasia); R02288 `Copy-of-cerebellah-hypoplasia-20b_aug_0_png_jpg.` (cerebellah-hypoplasia); R02289 `Copy-of-cerebellah-hypoplasia-20b_aug_1_png_jpg.` (cerebellah-hypoplasia); R02290 `Copy-of-cerebellah-hypoplasia-20b_aug_2_png_jpg.` (cerebellah-hypoplasia)
- `G01698` labels ['colphocephaly', 'moderate-ventriculomegaly']: R02302 `Copy-of-colphocephaly-36c_aug_2_png_jpg.rf.5a10c` (colphocephaly); R02303 `Copy-of-colphocephaly-36c_aug_5_png_jpg.rf.77506` (colphocephaly); R02741 `Copy-of-colphocephaly-36c_aug_0_png_jpg.rf.24754` (colphocephaly); R02742 `Copy-of-colphocephaly-36c_aug_0_png_jpg.rf.78696` (colphocephaly); R02743 `Copy-of-colphocephaly-36c_aug_0_png_jpg.rf.ec336` (colphocephaly); R02744 `Copy-of-colphocephaly-36c_aug_1_png_jpg.rf.1ce9e` (colphocephaly)

## Anatomical annotations (auxiliary dataset)

Masks were decoded and compared with the COCO export; the label map was read from `labelmap.txt`.

| source | images | label map | map as assumed | mask missing | mask empty | mask size != image | unexpected mask colours | pixel-size missing | images with structure (mask) | COCO instances | COCO vs mask presence disagreements | COCO-not-in-files / files-not-in-COCO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fhu_trans_cerebellum | 684 | Brain=(255, 0, 0); CSP=(0, 255, 0); LV=(0, 0, 255); background=(0, 0, 0) | True | 0 | 0 | 0 | 0 | 0 | {'Brain': 681, 'CSP': 352, 'LV': 2} | {'Brain': 681, 'CSP': 373, 'LV': 2} | {'CSP': 21} | 0/0 |
| fhu_trans_ventricular | 584 | Brain=(255, 0, 0); CSP=(0, 255, 0); LV=(0, 0, 255); background=(0, 0, 0) | True | 0 | 0 | 0 | 0 | 0 | {'Brain': 555, 'CSP': 347, 'LV': 459} | {'Brain': 555, 'CSP': 374, 'LV': 464} | {'CSP': 27, 'LV': 5} | 0/0 |
| fhu_diverse_head | 999 | Brain=(255, 0, 0); CSP=(255, 255, 0); LV=(0, 0, 255); background=(0, 0, 0) | False | 0 | 0 | 0 | 0 | 0 | {'Brain': 999, 'CSP': 408, 'LV': 510} | {'Brain': 999, 'CSP': 491, 'LV': 526} | {'CSP': 83, 'LV': 16} | 0/0 |

Annotation types: pixel-wise semantic masks and COCO instance polygons for Brain (skull/head), CSP (cavum septi pellucidi), LV (lateral ventricle); pixel size in mm for the Trans-* sets. These are anatomy labels: **they say nothing about normal vs abnormal.**

Masks were decoded with each source's own `labelmap.txt` (the Diverse set uses yellow for CSP, the others green). 'COCO vs mask disagreements' counts images where COCO has an instance of a structure but the rendered mask has none (or vice versa): an internal inconsistency of the source annotations, not resolved here. Counts of instances per structure in COCO are what the Readme calls 'class instances'.
