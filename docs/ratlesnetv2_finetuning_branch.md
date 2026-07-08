# RatLesNetV2 Finetuning Branch

Branch: `dl-ratlesnetv2-finetune`

This branch is an independent MRI lesion deep-learning track. It is not the
active v1 segmentation backend. v1 still uses human-reviewed masks as the
scientific source of truth; RatLesNetV2 outputs are draft masks that must return
through the same review gate.

Command details live in [../ratlesnetv2_finetune/README.md](../ratlesnetv2_finetune/README.md).

## Current Position

RatLesNetV2 was trained on rat T2w stroke MRI. The LYS target is mouse T2w MRI,
so upstream weights are a rodent lesion prior, not a finished LYS segmenter.

The current direct LYS fine-tuning run is promising and is now the baseline to
beat:

- validation Dice reached about `0.53` by epoch 10
- recall improved from about `0.51` to `0.61`
- predicted lesion voxels approached target lesion voxels
- this does not look like background collapse

The curve is beginning to plateau and validation loss rises after about epoch
4. Continue from `best_by_validation_dice.model` with lower LR or
`reduce-on-plateau`, early stopping, and validation overlays. Do not select the
final epoch automatically.

## Experiment Order

Use this order:

1. **Rat baseline.** Evaluate upstream rat weights on LYS validation without
   training.
2. **Direct LYS fine-tuning.** Train rat weights on LYS train and select by LYS
   validation Dice. This is the main baseline.
3. **Short external mouse adaptation.** Train rat weights briefly on external
   mouse train while monitoring LYS validation.
4. **LYS fine-tuning after external adaptation.** Continue from the best
   external checkpoint and compare against direct LYS fine-tuning.

Keep external mouse adaptation only if:

```text
rat -> external mouse adaptation -> LYS fine-tuning
```

beats:

```text
rat -> LYS fine-tuning
```

on LYS validation and later confirms on the held-out LYS test.

Do not use the held-out LYS test split until the final training strategy is
chosen from validation Dice and overlay QC.

## Local vs Cloud Responsibilities

Local MacBook:

- convert and review LYS masks
- prepare RatLesNetV2 folder contracts
- validate manifests/shapes/spacing
- run tests

Cloud GPU runtime:

- clone upstream RatLesNetV2
- train/fine-tune
- write checkpoints, plots, overlays, and prediction masks

Kaggle is currently the preferred free GPU runtime. Colab remains a fallback.

## Folder Contents

`ratlesnetv2_finetune/` contains:

- `dataset.py`: deterministic conversion from configured scan/mask pairs to the
  RatLesNetV2 folder contract.
- `roiset_to_nifti_mask.py`: Fiji/ImageJ RoiSet -> NIfTI lesion mask.
- `source_folders.py`: scans local source folders for scan/mask pairs.
- `scripts/prepare_lys_roiset_dataset.py`: bulk LYS RoiSet/Bruker cleanup.
- `scripts/review_lys_masks_itksnap.py`: opens editable mask copies in
  ITK-SNAP.
- `scripts/add_source_folder.py`: updates a local dataset YAML plan.
- `scripts/prepare_dataset.py`: builds prepared RatLesNetV2 folders.
- `scripts/split_prepared_dataset.py`: creates train/validation/test splits
  from a prepared dataset.
- `scripts/plan_cloud_run.py`: prints notebook/cloud commands.
- `scripts/finetune_ratlesnetv2.py`: imports upstream RatLesNetV2 and runs
  evaluation/fine-tuning with metrics, checkpoints, LR scheduling, and
  prediction exports.
- `requirements-colab.txt`: notebook GPU runtime requirements. The filename is
  historical; it is used for Kaggle and Colab.

## Current Data State

Prepared upload archives:

- `LYS_T2w_manual_v0.tar.gz`
- `External_Mouse_T2w_manual_LSP_SI_v0.tar.gz`

Kaggle dataset path:

```text
/kaggle/input/datasets/paaulaiz/ratlesnet-training-tarballs/
```

Current counts:

- LYS: 262 cases, mostly `256 x 256 x 18 x 1`
- external mouse: 426 cases, mostly `256 x 256 x 32 x 1`

Kaggle input datasets are read-only. Write split folders, checkpoints, plots,
and overlays under `/kaggle/working`.

If a Kaggle copy exposes only `scan.nii` / `scan_lesionIAM.nii`, create
compressed `.nii.gz` aliases in `/kaggle/working` before training.

## Metrics To Trust

Read these first:

- Dice
- recall
- precision
- predicted lesion voxels vs target lesion voxels
- validation overlays

Overall accuracy is background-dominated for sparse lesion masks and should not
drive decisions.

## Required Run Artifacts

Real comparison runs should write:

- `best_by_validation_dice.model`
- `best_by_validation_loss.model`
- `last.model`
- `lr_history.csv`
- `metrics_epoch.csv`
- `metrics_cases.csv`
- `final_metrics.json`
- validation overlays and voxel-count curves

Interrupted runs should write `interrupted.model` and `run_status.json`.

## Limits

- One-case/one-epoch runs are smoke tests only.
- All-train prepared exports must be split before real evaluation.
- Public mouse validation is not evidence of LYS performance.
- Passing `--ratlesnet-repo` only imports upstream code; real fine-tuning also
  needs `--pretrained-model ... --require-pretrained`.
- Predictions are not scientific outputs until reviewed/corrected.
