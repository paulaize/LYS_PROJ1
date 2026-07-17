# RatLesNetV2 External Mouse Datasets

Last updated: 2026-07-15

This document defines which public mouse MRI data can be used for RatLesNetV2
adaptation. It is a reference for dataset selection and provenance, not the
training runbook. The active training protocol lives in
[ratlesnetv2_lys_v1_kaggle_workflow.md](ratlesnetv2_lys_v1_kaggle_workflow.md).

## Current Decision

Direct rat-pretrained -> LYS fine-tuning is the main baseline. Public mouse
adaptation is a controlled comparator. Its source checkpoint must be selected
using external train/validation only, then fine-tuned with the selected loss on
the same five LYS development folds as the direct baseline. Keep it only if the
paired OOF comparison shows a repeatable downstream LYS benefit.

External data are never the final target-domain test. Held-out LYS cases are
the only valid final performance claim.

## Sources

| Short name | Record | Useful contents | Current use |
|---|---|---|---|
| `An2022` | <https://zenodo.org/records/6379879> | Mouse T2w NIfTI scans, manual lesion masks | Main public mouse source |
| `Knab2025` | <https://zenodo.org/records/14709930> | Mouse T2w scans, native and atlas-space masks | Use unique native manual cases not already in `An2022` |
| `Koch2017` | <https://zenodo.org/records/842677> | Mouse stroke T2w scans, manual masks, atlas copies | Use native manual cases after QC |
| `Mulder2017` | <https://datadryad.org/dataset/doi:10.5061/dryad.1m528> | Mouse tMCAO MRI, ImageJ/manual segmentations | Later optional expansion; extra conversion/QC needed |

## Current Prepared External Source

Use:

```text
/Volumes/Untitled/external_datasets/ratlesnetv2_clean_source_LSP_SI_flipped/
```

This folder is the current non-Mulder external source after:

1. cleaning native scan/mask pairs
2. relabeling headers to `LSP`
3. applying the confirmed superior/inferior voxel-array flip

Current prepared count: 426 scan/mask pairs.

The current upload archive is
`External_Mouse_T2w_manual_LSP_SI_v0.tar.gz`. Its prepared manifest contains
426 distinct `animal_id` values. The active Kaggle workflow uses those IDs for
an external-only 80/20 train/validation split and records the limitation that
richer source-animal metadata is not currently available.

| Dataset | Shape | Count |
|---|---:|---:|
| `An2022` | `256 x 256 x 32` | 331 |
| `Knab2025` | `256 x 256 x 32` | 45 |
| `Knab2025` | `192 x 192 x 32` | 35 |
| `Koch2017` | `256 x 256 x 32` | 15 |

All outputs are `LSP`, scan/mask affines match, shapes are unchanged, masks are
binary, lesion voxel counts are unchanged, and every output is exactly the
axis-1 flip of the corresponding `ratlesnetv2_clean_source_LSP_oriented/`
input. Keep representative visual QC before relying on a run.

## Geometry

| Dataset | Representative shape | Representative voxel size | Notes |
|---|---:|---:|---|
| `LYS` | `256 x 256 x 18` | `0.07 x 0.07 x 0.5 mm` | Target domain |
| `An2022` | `256 x 256 x 32` | `0.1 x 0.1 x 0.5 mm` | Native T2w |
| `Knab2025` | `256 x 256 x 32` | `0.1 x 0.1 x 0.5 mm` | Some duplicate Charite cases |
| `Koch2017` | `256 x 256 x 32` | about `0.1 x 0.1 x 0.5 mm` | Slight in-plane rotation in sample affine |
| `Mulder2017` | `128 x 128 x 16` | `0.117188 x 0.117188 x 0.5 mm` | Later optional; not geometry-matched |

Implications:

- public mouse data can support mouse-domain adaptation
- LYS fine-tuning and held-out LYS validation remain mandatory
- do not report public validation as LYS performance
- do not blindly resample everything without a deliberate training-grid choice

## Inclusion Rules

Use for training:

- manual labels only
- native-space images/masks only
- one label per image
- deduplicated cases
- positive lesion masks unless no-lesion controls are deliberately included

Skip as ground truth:

- automated masks
- atlas-space masks such as `x_masklesion.nii`
- duplicate copies of the same animal/image
- cropped/native duplicates unless a run explicitly studies cropping

Dataset-specific rules:

- `An2022`: use Charite `t2.nii` with `masklesion_manual.nii`.
- `Knab2025`: use native `t2.nii` with `masklesion.nii` only when the case is
  not already present in `An2022`.
- `Koch2017`: use `all/dat` native pairs; treat `mcao_cropped_to_20slices` as a
  separate optional experiment.
- `Mulder2017`: use later only if more public data are needed; choose one
  manual observer for training and keep the other for inter-rater/QC.

## Known Overlap

| Pair | Policy |
|---|---|
| `An2022` vs `Mulder2017` | `An2022` includes a Mulder subset. Do not include both. |
| `An2022` vs `Knab2025` | 73 usable Knab native case IDs overlap with An Charite cases. Keep one copy. |
| `Koch2017` vs others | No exact case-ID overlap found in inspected indices; still keep provenance/hash checks. |
| `Koch2017` internal | Native and cropped MCAO folders can contain the same animal. Do not treat both as independent by default. |

## Cleaning Commands

Download and clean external archives outside git:

```bash
make ratlesnetv2-download-external RUN_ARGS="--output-root /Volumes/Untitled/external_datasets --include-mulder --delete-archives"
```

Create the LSP header-normalized intermediate:

```bash
make ratlesnetv2-orient-external-lsp \
  RUN_ARGS="--external-root /Volumes/Untitled/external_datasets --target-axcodes LSP --overwrite"
```

Create the current S/I-flipped source:

```bash
make ratlesnetv2-flip-external-si \
  RUN_ARGS="--external-root /Volumes/Untitled/external_datasets --output-source-root /Volumes/Untitled/external_datasets/ratlesnetv2_clean_source_LSP_SI_flipped --output-manifest /Volumes/Untitled/external_datasets/manifests/external_dataset_manifest_LSP_SI_flipped.csv --all --overwrite"
```

## Training Set Policy

Public mouse adaptation set:

```text
An2022 manual native cases
+ Knab2025 unique native manual cases
+ Koch2017 native manual cases
```

Optional later:

```text
+ full Mulder2017 primary-observer manual cases
- An2022 Mulder subset duplicates
```

LYS fine-tuning set:

```text
LYS T2w scans with human-reviewed native lesion masks
```

Final evaluation:

```text
held-out LYS test only, after strategy selection
```

## Required Provenance

Every imported case should carry:

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
label_source
label_observer
image_space
shape
spacing_mm
affine_hash
mask_voxels
lesion_volume_mm3
duplicate_of
qc_flag
```

Default training filter:

```text
label_source == manual
image_space == native
duplicate_of is null
mask_voxels > 0
```

## Pipeline Placement

Use this document to decide and clean external data. Then use the active Kaggle
workflow to:

1. split external records by manifest `animal_id` into train and validation;
2. select one source checkpoint using external validation only;
3. fine-tune that checkpoint on the same five LYS folds and with the same
   selected loss/settings as the direct baseline;
4. calibrate a separate OOF threshold;
5. compare direct versus external-initialized predictions on identical LYS
   cases;
6. reserve the locked LYS test until initialization, threshold, and ensemble
   are frozen.
