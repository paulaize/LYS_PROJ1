# RatLesNetV2 External Mouse Datasets

Last updated: 2026-07-01

This note records the public mouse T2w stroke MRI datasets identified for the
RatLesNetV2 transfer-learning branch, how they overlap, what files should be
used, and how they fit into the LYS training plan.

The goal is not to make external data the final target. The goal is to use
external mouse data to move RatLesNetV2 away from its original rat domain, then
fine-tune and validate on manually reviewed LYS masks.

## Source Datasets

| Short name | Record | License | Useful contents | Training use |
|---|---|---|---|---|
| `An2022` | <https://zenodo.org/records/6379879> | CC-BY-4.0 | Mouse T2w NIfTI scans, manual lesion masks, automated masks | Primary public mouse pretraining source |
| `Koch2017` | <https://zenodo.org/records/842677> | CC-BY-4.0 | Mouse stroke T2w NIfTI scans, manual lesion masks, atlas-registered copies, group metadata | Add native-space manual cases after QC |
| `Mulder2017` | <https://datadryad.org/dataset/doi:10.5061/dryad.1m528> | CC0 | Mouse tMCAO MRI data, ImageJ/manual segmentations from two observers, automated segmentations, T2 maps | Use later if maximum public data are needed; requires extra conversion/QC |
| `Knab2025` | <https://zenodo.org/records/14709930> | CC-BY-4.0 | Mouse T2w NIfTI scans, native lesion masks, atlas-space lesion masks, behavioral/outcome data | Add unique native-space manual cases not already in `An2022` |

## RatLesNetV2 Starting Point

RatLesNetV2 itself was trained on rat data, not these mouse datasets. The paper
reports 916 T2-weighted rat brain MRI scans from 671 rats. That makes the
pretrained model a useful rodent T2w lesion prior, but not a ready mouse/LYS
segmenter.

Training plan:

```text
RatLesNetV2 pretrained rat weights
  -> public mouse native-space manual masks
  -> LYS mouse native-space manual masks
  -> held-out LYS validation
  -> prediction used only as draft mask for human review
```

The final scientific lesion volume still comes from the reviewed/corrected
mask, not from an unreviewed model output.

## Geometry Summary

These values are based on metadata plus representative header inspection. A
full per-case geometry report is still required after download.

| Dataset | Representative image shape | Representative voxel size | Format/space notes |
|---|---:|---:|---|
| `LYS` | `256 x 256 x 18` | `0.07 x 0.07 x 0.5 mm` | Native Bruker T2 RARE target domain |
| `An2022` Charite sample | `256 x 256 x 32` | `0.1 x 0.1 x 0.5 mm` | Native NIfTI `t2.nii` |
| `Knab2025` sample | `256 x 256 x 32` | `0.1 x 0.1 x 0.5 mm` | Native NIfTI `t2.nii`; overlapping sample matches `An2022` geometry |
| `Koch2017` sample | `256 x 256 x 32` | approx. `0.1 x 0.1 x 0.5 mm` | Native NIfTI; sample affine has slight in-plane rotation |
| `Mulder2017` Leiden T2-map sample | `128 x 128 x 16` | `0.117188 x 0.117188 x 0.5 mm` | MHD/T2-map style data; not same geometry as LYS or An/Knab/Koch |

Implications:

- External data are not geometry-matched to LYS.
- The public mouse data are still useful for mouse-domain adaptation because
  they teach mouse T2w lesion appearance.
- LYS fine-tuning and held-out LYS validation remain mandatory.
- Do not resample everything blindly. First generate a geometry/provenance
  report for all cases, then decide whether a training stage should run in
  native geometry or on a chosen common grid.

### Orientation Note

Visual ITK-SNAP checks on representative non-Mulder public cases
`An2022__20190320CH_Exp4_M20`, `Koch2017__20170428AR_TTC_M06`, and
`Knab2025__20170207CH_SC01` showed the same orientation problem: the 32-slice
axis is anatomically a coronal stack, but the NIfTI headers declare it as the
superior/inferior axis. Opening these files therefore places the coronal stack
in the transverse viewer. Relabeling the header orientation to `LSP` puts the
stack in the expected coronal viewer.

Use the separate orientation-normalized copy before importing external data for
training:

```bash
make ratlesnetv2-orient-external-lsp \
  RUN_ARGS="--external-root /Volumes/Untitled/external_datasets --target-axcodes LSP --overwrite"
```

This writes `ratlesnetv2_clean_source_LSP_oriented/` and
`manifests/external_dataset_manifest_LSP_oriented.csv`. The operation is a
header/affine relabel only: voxel arrays are not permuted or resampled, and
the same affine is applied to each lesion mask. The 2026-07-01 verification
found 426 `LSP` scan/mask pairs with unchanged shapes, binary masks, and
unchanged lesion voxel counts. All rows remain marked
`needs_visual_qc_lsp_orientation` until inspected.

## Overlap Findings

The following overlap checks were made from README/metadata files, ZIP central
directory indices, and small spreadsheet/index files, without downloading all
image payloads.

| Pair | Overlap status |
|---|---|
| `An2022` vs `Mulder2017` | Direct overlap. `An2022` includes `data_mulder_et_al_2017/`, a 19-case subset from Mulder et al. |
| `An2022` vs `Knab2025` | Direct overlap. 73 usable `Knab2025` native `t2.nii` + `masklesion.nii` case IDs also appear in `An2022` Charite cases. |
| `Koch2017` vs `An2022` / `Knab2025` / `Mulder2017` | No exact case-ID overlap found in the archive indices inspected. Treat as mostly independent, pending final hash/metadata checks after download. |
| `Koch2017` internal | Contains `all/dat/` native data and `mcao_cropped_to_20slices/dat/` cropped MCAO copies. Do not treat cropped and uncropped copies of the same mouse as independent training cases. |

Observed counts from archive indices:

| Dataset/cohort | Count observed | Manual label availability |
|---|---:|---|
| `An2022/data_charite` | 382 case folders | 331 with `masklesion_manual.nii` |
| `An2022/data_mulder_et_al_2017` | 19 case folders | 19 with manual observer masks |
| `Knab2025/repository/dat` | 153 folders with both `t2.nii` and `masklesion.nii` | 153 usable native pairs |
| `Koch2017/all/dat` | 38 folders with native `t2.nii` + `masklesion.nii` | 38 native pairs in `all/dat` |
| `Koch2017/mcao_cropped_to_20slices/dat` | 17 cropped MCAO folders with `t2.nii` + `masklesion.nii` | likely cropped copies; handle separately |

Cleaned run on `/Volumes/Untitled/external_datasets` on 2026-07-01:

| Dataset | Exported pairs | Notes |
|---|---:|---|
| `An2022` | 331 | Native Charite manual masks only |
| `Knab2025` | 80 | 73 duplicate Charite case IDs skipped |
| `Koch2017` | 15 | Public zip is a malformed >4 GB non-Zip64 archive; cleaner repairs/falls back where possible and records unreadable pairs |
| `Mulder2017` | 121 | Included with `--include-mulder`; T2-map/IAM manual-mask pairs |

## What To Use From Each Dataset

### An2022

Use:

```text
data_repository/data_charite/<case_id>/t2.nii
data_repository/data_charite/<case_id>/masklesion_manual.nii
```

Skip as ground truth:

```text
masklesion_auto.nii
```

For the included Mulder subset, use only if the full `Mulder2017` dataset is
not also imported:

```text
data_repository/data_mulder_et_al_2017/<case_id>/t2.nii
data_repository/data_mulder_et_al_2017/<case_id>/MANUAL_IAM.nii
data_repository/data_mulder_et_al_2017/<case_id>/MANUAL_SdJ.nii
```

Do not use both `MANUAL_IAM.nii` and `MANUAL_SdJ.nii` as two independent
training cases for the same image. Pick one primary observer for training and
keep the other for inter-rater/QC analysis.

### Knab2025

Use only native-space pairs:

```text
repository/dat/<case_id>/t2.nii
repository/dat/<case_id>/masklesion.nii
```

Skip for native RatLesNetV2 training:

```text
repository/dat/<case_id>/x_masklesion.nii
repository/dat/<case_id>/ix_ANO.nii
```

Those atlas-space files are useful for atlas/outcome analyses, not for native
T2w lesion segmentation. When `case_id` is already present in `An2022`, keep a
single copy. The simplest rule is:

```text
If An2022 has case_id with masklesion_manual.nii, keep An2022 and skip Knab duplicate.
If Knab case_id is not in An2022, add Knab.
```

### Koch2017

Use native-space pairs:

```text
all/dat/<case_id>/t2.nii
all/dat/<case_id>/masklesion.nii
```

Do not use atlas-space versions for native training:

```text
x_t2.nii
x_masklesion.nii
```

Treat these as a separate option rather than independent new animals:

```text
mcao_cropped_to_20slices/dat/<case_id>/t2.nii
mcao_cropped_to_20slices/dat/<case_id>/masklesion.nii
```

Start with `all/dat/` native cases. Use cropped copies only if a deliberate
experiment needs them, and deduplicate against the corresponding uncropped
case.

Archive caveat: the downloaded Koch zip is malformed for random access because
it is larger than 4 GB but not cleanly Zip64 encoded. The downloader/cleaner
tries a `zip -FF` repair and system `unzip` fallback, then skips any remaining
unreadable native pairs with `reason=unreadable_pair` in `skipped_cases.csv`.

### Mulder2017

Use full `Mulder2017` only if additional public data are needed after
`An2022`, `Knab2025`, and `Koch2017` are integrated. It needs more import work
because the public archive includes MHD/T2-map data and ImageJ ROI/manual
segmentation folders.

Prefer one manual observer as the training label, for example:

```text
*_MHD_Manual_Segmentations_IAM
```

Keep the other observer for inter-rater/QC:

```text
*_MHD_Manual_Segmentations_SdJ
```

Skip as training ground truth unless explicitly doing a pseudo-label study:

```text
*_MHD_Automated_Segmentations
AUTOMATED.nii
```

If the full `Mulder2017` archive is imported, drop `An2022`'s
`data_mulder_et_al_2017/` subset to avoid duplication.

## Recommended Training Set Construction

Initial public-mouse pretraining set:

```text
An2022 Charite manual native cases
+ Knab2025 native manual cases not already present in An2022
+ Koch2017 all/dat native manual cases
```

Optional later expansion:

```text
+ full Mulder2017 primary-observer manual cases
- An2022 data_mulder_et_al_2017 duplicates
```

LYS fine-tuning set:

```text
All available LYS T2w scans with human-reviewed native lesion masks
```

LYS validation:

```text
Held-out LYS animals only
```

Do not report public-dataset validation as evidence that the model is good for
LYS. Public validation is useful for debugging/pretraining. Held-out LYS
performance is the target-domain performance.

## Split Policy

Recommended split logic:

1. Deduplicate public datasets by normalized `source_dataset`, `source_case_id`,
   and eventually file hashes after download.
2. Keep public mouse data in the public pretraining stage.
3. Keep LYS cases separate for fine-tuning and final validation.
4. If LYS has few cases, use leave-one-out or repeated small holdouts rather
   than a large fixed validation split.
5. Controls/no-lesion scans can be included with all-zero masks, but they
   should not dominate the training set.

## Required Provenance Columns

Every imported public or LYS case should carry:

```text
source_dataset
source_record_url
source_case_id
study
timepoint
species
stroke_model
scan_path
mask_path
label_source          # manual / automated / pseudo_label
label_observer        # IAM / SdJ / unknown / LYS reviewer
image_space           # native / atlas / cropped_native
shape
spacing_mm
affine_hash
mask_voxels
lesion_volume_mm3
duplicate_of
qc_flag
```

For training, default filters should be:

```text
label_source == manual
image_space == native
duplicate_of is null
mask_voxels > 0, unless explicitly including no-lesion controls
```

## Final Pipeline Placement

The full training pipeline should become:

```text
1. Download external archives outside git and clean them with:
   `make ratlesnetv2-download-external RUN_ARGS="--output-root /Volumes/Untitled/external_datasets --include-mulder --delete-archives"`.
2. Use `/Volumes/Untitled/external_datasets/ratlesnetv2_clean_source/` as the
   external source-folder import. It contains only selected image/mask pairs
   plus manifests under `/Volumes/Untitled/external_datasets/manifests/`.
3. Convert LYS Fiji RoiSet.zip annotations to *_lesion_mask.nii.gz.
4. Build a geometry/provenance report for all candidates.
5. Deduplicate public records and select manual native labels.
6. Add selected source folders to the RatLesNetV2 YAML plan.
7. Export the harmonized RatLesNetV2 dataset under work/.
8. Upload prepared dataset + repo branch + optional pretrained weights to Colab.
9. Cloud smoke test: 1 case, 1 epoch.
10. Public mouse training/adaptation stage.
11. LYS fine-tuning stage.
12. Held-out LYS evaluation.
13. Bring predictions back locally.
14. Human-review predicted masks.
15. Use reviewed masks for lesion volume and downstream v1/v2 outputs.
```

Google Colab free tier should be adequate for smoke tests and small fine-tunes.
Long public-dataset runs may need checkpointing to Google Drive, smaller
subsets, resumable runs, or a paid/stable GPU runtime.
