# RatLesNetV2 Finetuning Branch

Branch: `dl-ratlesnetv2-finetune`

This branch starts the independent MRI lesion deep-learning track requested by
Paul. It is deliberately separate from the active v1 spine: v1 still produces
human-reviewed lesion masks, while this branch turns those masks into training
data for a future neural-network draft segmenter.

## Why RatLesNetV2 Here

RatLesNetV2 is a rodent T2w lesion segmentation network with public code at
<https://github.com/jmlipman/RatLesNetv2> and paper record
<https://arxiv.org/abs/2001.09138>. It is not the active v1 segmentation
backend. It is a candidate transfer-learning branch for later MRI lesion
drafts once enough corrected LYS masks exist.

## Local vs Cloud Responsibilities

Local MacBook Pro M1 8 GB:

- prepare NIfTI folders,
- validate shapes/spacing/labels,
- write manifests,
- run unit tests,
- plan cloud commands.

Cloud GPU runtime:

- clone upstream RatLesNetV2,
- run the finetuning/training script,
- save checkpoints outside git,
- export predictions for later human review.

No local CUDA setup is expected. If PyTorch is used locally, the code selects
MPS when available rather than CUDA, but real training should happen on cloud.

## Added Folder

`ratlesnetv2_finetune/` contains:

- `dataset.py`: deterministic conversion from configured NIfTI scan/manual mask
  pairs to RatLesNetV2 folders.
- `scripts/prepare_dataset.py`: local dataset-preparation CLI.
- `scripts/plan_cloud_run.py`: prints Colab/cloud commands.
- `scripts/finetune_ratlesnetv2.py`: cloud training loop that imports the
  upstream model, optionally loads pretrained weights, and continues training.
- `configs/dataset_template.yml`: first dataset plan, seeded with
  `BD_08_5D`.
- `requirements-colab.txt`: cloud runtime dependency list.

## Current Limitations

- One case is enough to smoke-test the path, not to estimate model performance.
- Finetuned predictions must still pass through the human mask review gate.
- Dataset export is single-modality T2w only.
- The upstream code expects `scan_lesionIAM.nii.gz`; the exporter also writes
  `scan_lesion.nii.gz` for README compatibility.

## Acceptance For This Branch Setup

- `main` includes the committed `ihc` history.
- `ihc` remains as a branch for future work.
- `dl-ratlesnetv2-finetune` branches from updated `main`.
- Local tests cover dataset preparation and cloud-command planning without
  requiring PyTorch, model downloads, or network calls.

