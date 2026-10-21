# Duplicate audit

Method: SHA-256 for exact duplicates; near-duplicate candidates = dHash distance <= 6 AND pHash (DCT) distance <= 8 (64-bit hashes of the grayscale image resized to a fixed square, so a 640x640 'stretch' re-export and the 959x661 original hash alike). Near-duplicates are **heuristic candidates, not proof**; nothing was deleted.

- Images: 4035; unique SHA-256: 3912; images in an exact-duplicate set: 246
- Exact-duplicate pairs: 123; near-duplicate pairs (not exact): 713
- Duplicate clusters (exact or near links): 284

## Pairs by relation

| kind | relation | pairs |
|---|---|---|
| exact | augmentation_sibling_same_scan | 81 |
| exact | cross_source | 1 |
| exact | same_source | 41 |
| near | augmentation_sibling_same_scan | 546 |
| near | cross_source | 44 |
| near | same_case_token_different_scan | 48 |
| near | same_source | 26 |
| near | within_roboflow_unrelated_names | 49 |

`augmentation_sibling_same_scan` = Roboflow `_aug_N` copies of one scan (expected by construction, but they ARE the same underlying image). `cross_source` = the same/near-same image appears in two different source datasets.

## Sensitivity of the near-duplicate threshold

| both hash distances <= | candidate pairs |
|---|---|
| 0 | 321 |
| 2 | 557 |
| 4 | 656 |
| 6 | 745 |
| 8 | 938 |

## Cross-source matches (45 pairs)

| source A | source B | kind | pairs |
|---|---|---|---|
| fhu_trans_cerebellum | fhu_trans_ventricular | exact | 1 |
| fhu_trans_cerebellum | fhu_trans_ventricular | near | 2 |
| fhu_trans_cerebellum | roboflow_fetal_brain | near | 20 |
| fhu_trans_ventricular | roboflow_fetal_brain | near | 22 |

Examples:

- exact (dHash 0, pHash 0): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient01537_Plane3_3_of_9.png` <-> `datasets/final_build_20261022_010449/extracted/fhu_trans_ventricular/Trans-ventricular/Trans-ventricular/Patient01537_Plane3_6_of_9.png`
- near (dHash 2, pHash 4): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00710_Plane3_3_of_6.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00710_Plane3_3_of_6_png_jpg.rf.9b8c5f47315303bbd9dcac2ae3fcd164.jpg`
- near (dHash 3, pHash 6): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00712_Plane3_3_of_4.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00712_Plane3_3_of_4_png_jpg.rf.b663f4c34804074cd532eb6611708efa.jpg`
- near (dHash 0, pHash 2): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00712_Plane3_4_of_4.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00712_Plane3_4_of_4_png_jpg.rf.b0d01c07409cf627e81063f4a1e47bc9.jpg`
- near (dHash 3, pHash 0): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00713_Plane3_1_of_4.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/test/Patient00713_Plane3_1_of_4_png_jpg.rf.da29e291fb9bd39841f431d5ba6911de.jpg`
- near (dHash 1, pHash 2): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00714_Plane3_3_of_8.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00714_Plane3_3_of_8_png_jpg.rf.337036a4db10bce63f5e94525b0aa4f0.jpg`
- near (dHash 2, pHash 4): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00715_Plane3_2_of_2.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00715_Plane3_2_of_2_png_jpg.rf.9428713c2e5f436cd4cf8b87bb7ffbdd.jpg`
- near (dHash 1, pHash 2): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00716_Plane3_6_of_6.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00716_Plane3_2_of_6_png_jpg.rf.21163f7c888052f95ab873fd30d85222.jpg`
- near (dHash 3, pHash 6): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00717_Plane3_3_of_3.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00717_Plane3_3_of_3_png_jpg.rf.28f65d5ef39482a49e7719fbd74ea33b.jpg`
- near (dHash 0, pHash 2): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00718_Plane3_5_of_5.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00718_Plane3_5_of_5_png_jpg.rf.c3e69f302f74fa7e8e14fa45bd14e308.jpg`
- near (dHash 1, pHash 2): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00720_Plane3_1_of_2.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/test/Patient00720_Plane3_1_of_2_png_jpg.rf.99b4ef35ee3db07e2c2c40808e7a9966.jpg`
- near (dHash 1, pHash 2): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00721_Plane3_3_of_3.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00721_Plane3_3_of_3_png_jpg.rf.b30ddd4a6fe406aa0ca1332985429bf3.jpg`
- near (dHash 3, pHash 4): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00722_Plane3_5_of_5.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00722_Plane3_5_of_5_png_jpg.rf.95c61e1162bacd7bd1d326b9880be9a0.jpg`
- near (dHash 3, pHash 0): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00725_Plane3_3_of_4.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00725_Plane3_3_of_4_png_jpg.rf.ccef46f3da262b26764ee5d6f5dc5838.jpg`
- near (dHash 1, pHash 0): `datasets/final_build_20261022_010449/extracted/fhu_trans_cerebellum/Trans-cerebellum/Trans-cerebellum/Patient00726_Plane3_1_of_3.png` <-> `datasets/final_build_20261022_010449/extracted/roboflow_fetal_brain/train/Patient00726_Plane3_1_of_3_png_jpg.rf.1d6cbf7f4cfe7fbc3178545860aa8a42.jpg`

## Duplicates with conflicting labels (44 pairs)

- near (dHash 0, pHash 2): `Copy-of-colphocephaly-36c_aug_5_png_jpg.rf.775069690c61` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_1_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 0, pHash 2): `Copy-of-colphocephaly-36c_aug_5_png_jpg.rf.775069690c61` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_1_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 1, pHash 2): `Copy-of-colphocephaly-36c_aug_5_png_jpg.rf.775069690c61` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_4_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 1, pHash 2): `Copy-of-colphocephaly-36c_aug_5_png_jpg.rf.775069690c61` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_4_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 1, pHash 2): `Copy-of-colphocephaly-36c_aug_5_png_jpg.rf.775069690c61` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_5_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 1, pHash 2): `Copy-of-colphocephaly-36c_aug_5_png_jpg.rf.775069690c61` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_5_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 0, pHash 2): `Copy-of-colphocephaly-36c_aug_5_png_jpg.rf.775069690c61` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_png_jpg.rf.6f1d69` [moderate-ventriculomegaly]
- near (dHash 0, pHash 2): `Copy-of-colphocephaly-36c_aug_5_png_jpg.rf.775069690c61` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_png_jpg.rf.e7aa00` [moderate-ventriculomegaly]
- near (dHash 1, pHash 0): `Copy-of-moderate-ventriculomegaly_20c_aug_1_png_jpg.rf.` [moderate-ventriculomegaly] <-> `Copy-of-cerebellah-hypoplasia_20c_aug_1_png_jpg.rf.9d46` [cerebellah-hypoplasia]
- near (dHash 0, pHash 0): `Copy-of-moderate-ventriculomegaly_20c_aug_1_png_jpg.rf.` [moderate-ventriculomegaly] <-> `Copy-of-cerebellah-hypoplasia_20c_aug_5_png_jpg.rf.68ab` [cerebellah-hypoplasia]
- near (dHash 1, pHash 0): `Copy-of-moderate-ventriculomegaly_20c_aug_1_png_jpg.rf.` [moderate-ventriculomegaly] <-> `Copy-of-cerebellah-hypoplasia_20c_png_jpg.rf.9fc7170436` [cerebellah-hypoplasia]
- near (dHash 1, pHash 0): `Copy-of-moderate-ventriculomegaly_20c_aug_1_png_jpg.rf.` [moderate-ventriculomegaly] <-> `Copy-of-cerebellah-hypoplasia_20c_aug_1_png_jpg.rf.9d46` [cerebellah-hypoplasia]
- near (dHash 0, pHash 0): `Copy-of-moderate-ventriculomegaly_20c_aug_1_png_jpg.rf.` [moderate-ventriculomegaly] <-> `Copy-of-cerebellah-hypoplasia_20c_aug_5_png_jpg.rf.68ab` [cerebellah-hypoplasia]
- near (dHash 1, pHash 0): `Copy-of-moderate-ventriculomegaly_20c_aug_1_png_jpg.rf.` [moderate-ventriculomegaly] <-> `Copy-of-cerebellah-hypoplasia_20c_png_jpg.rf.9fc7170436` [cerebellah-hypoplasia]
- near (dHash 1, pHash 4): `Copy-of-moderate-ventriculomegaly_20c_aug_3_png_jpg.rf.` [moderate-ventriculomegaly] <-> `Copy-of-cerebellah-hypoplasia_20c_aug_3_png_jpg.rf.ae83` [cerebellah-hypoplasia]
- near (dHash 1, pHash 4): `Copy-of-moderate-ventriculomegaly_20c_aug_3_png_jpg.rf.` [moderate-ventriculomegaly] <-> `Copy-of-cerebellah-hypoplasia_20c_aug_3_png_jpg.rf.ae83` [cerebellah-hypoplasia]
- near (dHash 1, pHash 0): `Copy-of-cerebellah-hypoplasia_20c_aug_1_png_jpg.rf.9d46` [cerebellah-hypoplasia] <-> `Copy-of-moderate-ventriculomegaly_20c_aug_5_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 1, pHash 0): `Copy-of-cerebellah-hypoplasia_20c_aug_1_png_jpg.rf.9d46` [cerebellah-hypoplasia] <-> `Copy-of-moderate-ventriculomegaly_20c_aug_5_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 0, pHash 0): `Copy-of-cerebellah-hypoplasia_20c_aug_5_png_jpg.rf.68ab` [cerebellah-hypoplasia] <-> `Copy-of-moderate-ventriculomegaly_20c_aug_5_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 0, pHash 0): `Copy-of-cerebellah-hypoplasia_20c_aug_5_png_jpg.rf.68ab` [cerebellah-hypoplasia] <-> `Copy-of-moderate-ventriculomegaly_20c_aug_5_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 1, pHash 0): `Copy-of-cerebellah-hypoplasia_20c_png_jpg.rf.9fc7170436` [cerebellah-hypoplasia] <-> `Copy-of-moderate-ventriculomegaly_20c_aug_5_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 1, pHash 0): `Copy-of-cerebellah-hypoplasia_20c_png_jpg.rf.9fc7170436` [cerebellah-hypoplasia] <-> `Copy-of-moderate-ventriculomegaly_20c_aug_5_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 4, pHash 6): `Copy-of-colphocephaly-36c_aug_0_png_jpg.rf.24754af78a5b` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_4_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 4, pHash 6): `Copy-of-colphocephaly-36c_aug_0_png_jpg.rf.24754af78a5b` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_4_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 0, pHash 0): `Copy-of-colphocephaly-36c_aug_0_png_jpg.rf.ec3360174401` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_0_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 0, pHash 0): `Copy-of-colphocephaly-36c_aug_0_png_jpg.rf.ec3360174401` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_0_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 2, pHash 8): `Copy-of-colphocephaly-36c_aug_1_png_jpg.rf.1ce9e4eadc62` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_0_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 2, pHash 8): `Copy-of-colphocephaly-36c_aug_1_png_jpg.rf.1ce9e4eadc62` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_0_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 0, pHash 2): `Copy-of-colphocephaly-36c_aug_1_png_jpg.rf.2efefba97f3a` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_1_png_jpg.rf.` [moderate-ventriculomegaly]
- near (dHash 0, pHash 2): `Copy-of-colphocephaly-36c_aug_1_png_jpg.rf.2efefba97f3a` [colphocephaly] <-> `Copy-of-moderate-ventriculomegaly-36b_aug_1_png_jpg.rf.` [moderate-ventriculomegaly]

Roboflow near-duplicates between differently-named scans: 49 pairs (possible repeated images under different case names).

## Proposed resolution (not applied)

- Keep every source record. Duplicate clusters and case tokens are merged into one case group; a group is never split across folds.
- For scoring, one representative per exact-SHA set counts (`scoring_role` in `proposed_cv_manifest.csv`); the rest stay in the same fold unscored.
- Groups with conflicting labels are routed to `unresolved/` for human review rather than resolved by vote.
- Near-duplicates with *different* labels need visual review; heuristic hashing may also over-link visually similar frames.