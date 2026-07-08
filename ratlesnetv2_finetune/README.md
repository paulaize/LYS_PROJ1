# RatLesNetV2 Finetuning Runbook

This folder is a side track for MRI lesion deep learning. It does not replace
the active v1 rule that final lesion volumes come from human-reviewed masks.

## Current Strategy

Treat direct rat-pretrained -> LYS fine-tuning as the main baseline.

```text
A. rat pretrained -> LYS fine-tuning
B. rat pretrained -> external mouse adaptation -> LYS fine-tuning
```

Keep external mouse Stage 1 only if it improves downstream LYS validation
against A. Do not use the held-out LYS test split until the final strategy is
chosen.

Current direct LYS signal:

- validation Dice reached about `0.53` by epoch 10
- recall improved from about `0.51` to `0.61`
- predicted lesion voxels approached target lesion voxels
- this is not background collapse

Next runs should continue from `best_by_validation_dice.model`, use validation
overlays, early stopping, and LR reduction on plateau. Do not select the final
epoch automatically.

## Data Rules

- Inputs under `data/`, `/Volumes/...`, and `/kaggle/input/...` are read-only.
- Local intermediates go under `work/`.
- Kaggle split folders, checkpoints, plots, and overlays go under
  `/kaggle/working`.
- Model weights, prepared tarballs, checkpoints, predictions, `work/`, and
  `outputs/` stay out of git.
- Training outputs are draft masks only; predictions must go through human
  review before any scientific volume calculation.

## Local Commands

All local commands use the existing `lys-bbb` environment through Makefile
targets.

Convert a Fiji/ImageJ `RoiSet.zip` to a NIfTI mask:

```bash
make ratlesnetv2-roiset-to-mask RUN_ARGS="--nifti /path/to/scan.nii.gz --roiset /path/to/RoiSet.zip --output /path/to/case_lesion_mask.nii.gz"
```

Prepare LYS RoiSet/Bruker source folders in bulk:

```bash
make ratlesnetv2-prepare-lys-roisets RUN_ARGS="--source-drive /Volumes/Untitled --output-root ~/Desktop/LYS_RatLesNetV2_clean_source --overwrite"
```

Review editable LYS mask copies in ITK-SNAP:

```bash
make ratlesnetv2-review-lys-masks RUN_ARGS="--source-root ~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source --output-root ~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source_reviewed --limit 1"
```

Create or update a local dataset YAML from a source folder:

```bash
cp ratlesnetv2_finetune/configs/dataset_from_folders_template.yml \
  ratlesnetv2_finetune/configs/local_dataset.yml

make ratlesnetv2-add-source \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_dataset.yml \
  RUN_ARGS="--source-root ~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source_reviewed --split train --study LYS --timepoint mixed --scan-subdir T2w --mask-subdir masks"
```

Build a RatLesNetV2-ready dataset:

```bash
make ratlesnetv2-prepare \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_dataset.yml \
  RUN_ARGS=--overwrite
```

Split an existing prepared dataset:

```bash
make ratlesnetv2-split-prepared RUN_ARGS="--input work/ratlesnetv2_finetune/datasets/LYS_T2w_manual_v0 --output work/ratlesnetv2_finetune/datasets/LYS_T2w_manual_v0_split --validation-fraction 0.15 --test-fraction 0.15 --seed 20260706 --overwrite"
```

Print a cloud command plan:

```bash
make ratlesnetv2-cloud-plan \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_dataset.yml \
  RUN_ARGS="--cloud-dataset-root /kaggle/working/LYS_T2w_manual_v0_split --cloud-output /kaggle/working/ratlesnet_runs --pretrained-model /kaggle/working/RatLesNetv2/trained_models/Table2-3/RatLesNetv2/homogeneous/model-1 --epochs 10 --lr 5e-5 --require-pretrained"
```

## Kaggle Setup

Kaggle is currently the preferred free GPU runtime. Use `nvidia-smi` every
session because GPU assignment can change; Paul observed a Kaggle GPU labelled
`T100` in the current setup.

Minimal setup:

```python
%cd /kaggle/working
!git clone --branch dl-ratlesnetv2-finetune https://github.com/paulaize/LYS_PROJ1.git LYS_PROJ1
%cd /kaggle/working/LYS_PROJ1
!pip install -q -r ratlesnetv2_finetune/requirements-colab.txt matplotlib

%cd /kaggle/working
!test -d RatLesNetv2 || git clone --depth 1 https://github.com/jmlipman/RatLesNetv2.git RatLesNetv2
%cd /kaggle/working/LYS_PROJ1
!nvidia-smi
!ls -lh /kaggle/working/RatLesNetv2/trained_models/Table2-3/RatLesNetv2/homogeneous/model-1
```

Optional P100-only PyTorch reset if the Kaggle image's default torch/CUDA
combination fails. Do not run this by default on T4 x2 until it is validated.

```python
!pip uninstall -y torch torchvision torchaudio
!pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Kaggle Dataset Split

Current Kaggle dataset root:

```text
/kaggle/input/datasets/paaulaiz/ratlesnet-training-tarballs/
```

Create explicit LYS train/validation/test split and external train-only split:

```python
%cd /kaggle/working/LYS_PROJ1

!python -m ratlesnetv2_finetune.scripts.split_prepared_dataset \
  --input /kaggle/input/datasets/paaulaiz/ratlesnet-training-tarballs/LYS_T2w_manual_v0/LYS_T2w_manual_v0 \
  --output /kaggle/working/LYS_T2w_manual_v0_split \
  --validation-fraction 0.15 \
  --test-fraction 0.15 \
  --seed 20260706 \
  --overwrite

!python -m ratlesnetv2_finetune.scripts.split_prepared_dataset \
  --input /kaggle/input/datasets/paaulaiz/ratlesnet-training-tarballs/External_Mouse_T2w_manual_LSP_SI_v0/External_Mouse_T2w_manual_LSP_SI_v0 \
  --output /kaggle/working/External_Mouse_T2w_manual_LSP_SI_v0_split \
  --validation-fraction 0 \
  --test-fraction 0 \
  --seed 20260706 \
  --overwrite
```

Check expected files:

```python
from pathlib import Path

for p in [
    "/kaggle/working/LYS_T2w_manual_v0_split/train",
    "/kaggle/working/LYS_T2w_manual_v0_split/validation",
    "/kaggle/working/LYS_T2w_manual_v0_split/test",
    "/kaggle/working/External_Mouse_T2w_manual_LSP_SI_v0_split/train",
]:
    p = Path(p)
    print(
        p,
        "scan:", sum(1 for _ in p.rglob("scan.nii.gz")),
        "label:", sum(1 for _ in p.rglob("scan_lesionIAM.nii.gz")),
    )
```

If a Kaggle dataset copy exposes `.nii` files without `.nii.gz` aliases, create
compressed aliases in `/kaggle/working` only:

```python
from pathlib import Path
import nibabel as nib

def gzip_nifti_alias(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    img = nib.load(str(src))
    nib.save(img, str(dst))

for root in [
    Path("/kaggle/working/LYS_T2w_manual_v0_split"),
    Path("/kaggle/working/External_Mouse_T2w_manual_LSP_SI_v0_split"),
]:
    converted = 0
    for case_dir in root.rglob("*"):
        if not case_dir.is_dir():
            continue
        for src, dst in [
            (case_dir / "scan.nii", case_dir / "scan.nii.gz"),
            (case_dir / "scan_lesionIAM.nii", case_dir / "scan_lesionIAM.nii.gz"),
        ]:
            if src.exists() and not dst.exists():
                gzip_nifti_alias(src, dst)
                converted += 1
    print(root, "created", converted, "gz aliases")
```

## Current Direct LYS Baseline

This run uses LYS train/validation only and does not touch the held-out test
split.

```python
%cd /kaggle/working/LYS_PROJ1

!python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /kaggle/working/RatLesNetv2 \
  --input /kaggle/working/LYS_T2w_manual_v0_split/train \
  --validation /kaggle/working/LYS_T2w_manual_v0_split/validation \
  --output /kaggle/working/ratlesnet_runs_lys_direct \
  --pretrained-model /kaggle/working/RatLesNetv2/trained_models/Table2-3/RatLesNetv2/homogeneous/model-1 \
  --require-pretrained \
  --epochs 10 \
  --lr 5e-5 \
  --gpu 0 \
  --loadMemory 0 \
  --save-every 1 \
  --eval-every 1 \
  --early-stop-patience 8 \
  --export-predictions validation \
  --export-prediction-limit 1 \
  --export-prediction-epochs all
```

Inspect metrics and overlays:

```python
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

root = Path("/kaggle/working/ratlesnet_runs_lys_direct")
run = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
             key=lambda p: int(p.name))[-1]

display(pd.read_csv(run / "metrics_epoch.csv").tail(10))

for name in [
    "latest_qc_overlay.png",
    "validation_metric_curves.png",
    "voxel_count_curves.png",
]:
    path = run / name
    if path.exists():
        display(Image(filename=str(path)))
```

## An et al. 2023 Quick Comparator

An et al. 2023 released an inference notebook and TorchScript weights for
mouse stroke lesion segmentation:

- code/weights: <https://github.com/scalableminds/stroke-lesion-segmentation>
- paper: <https://www.nature.com/articles/s41598-023-39826-8>

The local wrapper `finetune_an2023.py` uses the same prepared split folders as
RatLesNetV2. It fine-tunes the released `lesion_model.pt` directly and exports
the same style of validation metrics, full-volume NIfTI predictions, and QC
overlays. This is a quick transfer-learning comparator; do not use the LYS test
split until the final strategy is selected.

Kaggle setup after cloning `LYS_PROJ1`:

```python
%cd /kaggle/working
!test -d stroke-lesion-segmentation || git clone --depth 1 https://github.com/scalableminds/stroke-lesion-segmentation.git
%cd /kaggle/working/LYS_PROJ1
!ls -lh /kaggle/working/stroke-lesion-segmentation/lesion_model.pt
```

Direct An-pretrained -> LYS fine-tuning:

```python
%cd /kaggle/working/LYS_PROJ1

!python -m ratlesnetv2_finetune.scripts.finetune_an2023 \
  --input /kaggle/working/LYS_T2w_manual_v0_split/train \
  --validation /kaggle/working/LYS_T2w_manual_v0_split/validation \
  --output /kaggle/working/an2023_runs_lys_direct \
  --model-path /kaggle/working/stroke-lesion-segmentation/lesion_model.pt \
  --require-pretrained \
  --epochs 10 \
  --lr 1e-5 \
  --gpu 0 \
  --save-every 1 \
  --eval-every 1 \
  --early-stop-patience 8 \
  --lr-scheduler reduce-on-plateau \
  --lr-scheduler-metric validation_dice \
  --lr-plateau-patience 3 \
  --lr-plateau-factor 0.5 \
  --min-lr 1e-6 \
  --metrics-threshold 0.8 \
  --export-predictions validation \
  --export-prediction-limit 8 \
  --export-prediction-epochs all
```

Important differences from RatLesNetV2:

- The released An repository is inference-oriented, so this wrapper fine-tunes
  the TorchScript module rather than importing a full training package.
- Preprocessing follows the notebook: scale each volume by its max to a
  uint8-like `0..255` range, then apply `(x - 127.5) / 127.5`.
- The model expects a fixed center crop/pad of `152 x 196 x 30` in model order.
  LYS `256 x 256 x 18` scans are center-cropped in-plane and zero-padded in
  slice depth; predictions are restored to the original NIfTI grid for export.
- Best checkpoints are `.pt` TorchScript files:
  `best_by_validation_dice.pt`, `best_by_validation_loss.pt`, `last.pt`, and
  `an2023_finetuned.pt`.

Inspect An metrics and overlays:

```python
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

root = Path("/kaggle/working/an2023_runs_lys_direct")
run = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
             key=lambda p: int(p.name))[-1]

display(pd.read_csv(run / "metrics_epoch.csv").tail(10))

for name in [
    "latest_qc_overlay.png",
    "validation_metric_curves.png",
    "voxel_count_curves.png",
]:
    path = run / name
    if path.exists():
        display(Image(filename=str(path)))
```

## Continue Direct LYS Baseline

Continue from the best validation-Dice checkpoint, not the final epoch. Replace
`/1` with the selected run directory.

```python
DIRECT_RUN = "/kaggle/working/ratlesnet_runs_lys_direct/1"

!python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /kaggle/working/RatLesNetv2 \
  --input /kaggle/working/LYS_T2w_manual_v0_split/train \
  --validation /kaggle/working/LYS_T2w_manual_v0_split/validation \
  --output /kaggle/working/ratlesnet_runs_lys_direct_continue \
  --pretrained-model {DIRECT_RUN}/best_by_validation_dice.model \
  --require-pretrained \
  --epochs 30 \
  --lr 1e-5 \
  --gpu 0 \
  --loadMemory 0 \
  --save-every 5 \
  --eval-every 1 \
  --early-stop-patience 8 \
  --lr-scheduler reduce-on-plateau \
  --lr-scheduler-metric validation_dice \
  --lr-plateau-patience 3 \
  --lr-plateau-factor 0.5 \
  --min-lr 1e-6 \
  --export-predictions validation \
  --export-prediction-limit 8 \
  --export-prediction-epochs all
```

## External Mouse Comparator

Use external adaptation only as a comparator against the direct LYS baseline.
Validation remains LYS target-domain validation.

```python
!python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /kaggle/working/RatLesNetv2 \
  --input /kaggle/working/External_Mouse_T2w_manual_LSP_SI_v0_split/train \
  --validation /kaggle/working/LYS_T2w_manual_v0_split/validation \
  --output /kaggle/working/ratlesnet_runs_external_short_5e5 \
  --pretrained-model /kaggle/working/RatLesNetv2/trained_models/Table2-3/RatLesNetv2/homogeneous/model-1 \
  --require-pretrained \
  --epochs 5 \
  --lr 5e-5 \
  --gpu 0 \
  --loadMemory 0 \
  --save-every 1 \
  --eval-every 1 \
  --lr-scheduler reduce-on-plateau \
  --lr-scheduler-metric validation_dice \
  --lr-plateau-patience 2 \
  --lr-plateau-factor 0.5 \
  --min-lr 1e-6 \
  --export-predictions validation \
  --export-prediction-limit 8 \
  --export-prediction-epochs all
```

If the external run improves LYS validation, fine-tune LYS from that run's
`best_by_validation_dice.model` and compare against the direct LYS baseline.

## Final Test Use

Do not use:

```bash
--test /kaggle/working/LYS_T2w_manual_v0_split/test
```

until the final training strategy is chosen from LYS validation Dice and
overlay QC. Then run one final evaluation/fine-tune command and report only the
held-out LYS test rows from `final_metrics.json` or `metrics_epoch.csv`.

## Run Outputs

Each run directory writes:

```text
training_loss
validation_loss
lr_history.csv
metrics_epoch.csv
metrics_cases.csv
final_metrics.json
loss_curves.png
validation_metric_curves.png
final_metric_summary.png
voxel_count_curves.png
latest_qc_overlay.png
latest_validation_qc_overlay.png
latest_qc_overlay.json
best_by_validation_dice.model
best_by_validation_loss.model
last.model
interrupted.model
run_status.json
RatLesNetv2.model
```

`KeyboardInterrupt` writes `interrupted.model`, refreshes `last.model`, updates
plots/status where possible, and exits with status `130`.

## Colab Fallback

Colab remains a fallback. Use the same commands with `/content/...` paths and
stage data/checkpoints in Google Drive. The nibabel compatibility patch is in
`scripts/finetune_ratlesnetv2.py`; do not solve nibabel errors by downgrading
the notebook runtime.

## Local Verification

After code changes:

```bash
make lint
make test
git diff --check
```
