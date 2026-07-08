# DL-First Future Pipeline

This is a future design note. It is not the active v1 implementation plan.
v1 must finish first with one animal, no atlas, no DL, and reviewed masks.

## Principle

DL outputs are drafts. Every pixel-interpretation model must pass through a
human correction gate before its result is used as scientific data.

The corrected outputs become training data for the next model version.

## What DL Can Replace

Appropriate DL tasks:

- brain extraction
- lesion draft segmentation
- section-to-atlas initialization
- nucleus/cell detection
- cell-type classification where transparent thresholds fail

Do not replace deterministic steps:

- voxel-volume arithmetic
- Swanson/indirect edema correction
- physical dilation rings
- midline mirroring
- table joins
- provenance and QC logic

## MRI DL Track

Inputs remain Scan 2 T2 RARE with anisotropic volume math.

Draft lesion options:

1. pretrained mouse T2 lesion model, if practical
2. RatLesNetV2 transfer learning
3. nnU-Net after enough corrected LYS masks exist

Every draft mask goes through a human correction gate:

```text
model draft -> ITK-SNAP / 3D Slicer / napari review -> corrected mask
```

Record:

- model version
- reviewer
- edit Dice/IoU
- QC flag
- corrected mask path

Corrected masks remain the source of truth for volume and future training.

## RatLesNetV2 Status

RatLesNetV2 is the active DL side branch:

- branch: `dl-ratlesnetv2-finetune`
- local prep and tests in `lys-bbb`
- Kaggle preferred for free GPU training
- Colab fallback
- upstream rat weights are a rodent prior, not a finished LYS model

Current strategy:

```text
main baseline: rat pretrained -> LYS fine-tuning
comparator:    rat pretrained -> external mouse adaptation -> LYS fine-tuning
```

The direct LYS run is promising:

- validation Dice about `0.53` by epoch 10
- recall improved
- predicted lesion voxels approached target lesion voxels
- not background collapse

Continue only from best validation-Dice checkpoints with early stopping,
validation overlays, and LR reduction on plateau. Do not select the final epoch
automatically. Do not use the held-out LYS test split until the training
strategy is chosen.

Command runbook:

- [../ratlesnetv2_finetune/README.md](../ratlesnetv2_finetune/README.md)

Strategy/reference docs:

- [ratlesnetv2_finetuning_branch.md](ratlesnetv2_finetuning_branch.md)
- [ratlesnetv2_external_datasets.md](ratlesnetv2_external_datasets.md)

## IHC DL Track

Future IHC DL work should start after the v1 positive-area path is reliable.

Likely order:

1. keep QuPath/Bio-Formats `.vsi` as the working format
2. confirm channel maps from YAML
3. use ABBA/DeepSlice for section-to-Allen initialization
4. use InstanSeg or StarDist for DAPI nuclei/cell detection
5. review/edit sampled tiles in QuPath
6. train/refine object classifiers for cell type
7. keep GFAP/IBA1 area fractions as primary morphology readouts

IgG-FITC threshold calibration remains required. DL does not resolve positivity
thresholding or specificity provenance.

## Active-Learning Loop

For lesions and cells:

```text
model runs
-> human edits
-> edit metric + corrected output saved
-> high-edit cases prioritized for fine-tuning
-> new model version
```

Adopt a DL backend only if it reduces human edit burden against corrected LYS
data. Public mouse or public histology performance is not enough.

## QC Gates

- NIfTI spacing/header validated
- corrected masks used for volume
- model version recorded
- edit metric recorded
- validation overlays reviewed
- IgG-FITC threshold approved
- object-level and area-level IHC readouts checked for gross disagreement
- held-out LYS test used only once for final model evaluation

## Open Decisions

- when enough corrected LYS masks exist for nnU-Net or further RatLesNetV2
  fine-tuning
- final mask editor standard for careful MRI border work
- AIDAmri vs ANTs/SyN for MRI->Allen
- perilesional ring width
- whether direct QuPath CLI remains acceptable or a project/cached workflow is
  required for `.vsi` scale
