# LYS v2 architecture comparator: Kaggle protocol

The executable notebook is
`notebooks/lys_v2_architecture_comparator_kaggle.ipynb`. This protocol is a
new architecture-comparison stage; it does not replace or silently modify the
RatLesNetV2 v1 evidence.

## Question and order of operations

1. Reuse the exact preserved 201-case LYS development membership and five
   grouped folds from RatLesNetV2 v1.
2. Compare two direct-target candidates on pooled OOF predictions:
   `an2023_adapted_direct` and `nnunetv2_resencm_250_direct`.
3. Select the direct candidate by mean per-case OOF Dice, with paired CI,
   failures, volume error, HD95, surface Dice, fold consistency, and cohort
   consistency retained as diagnostics.
4. Train an external-source checkpoint only for that architecture, initialize
   the same five LYS folds from it, and compare direct versus external
   initialization on the same OOF cases.
5. Freeze the selected development setup and package the evidence. Do not
   expose a locked test in this notebook.

If `nnUNetTrainer_250epochs` wins the compute-budget screen, a definitive
nnU-Net claim should repeat all five folds with canonical `nnUNetTrainer`
(1,000 epochs) before final freezing. Mixing 250-epoch and 1,000-epoch fold
outputs in one OOF comparison is forbidden.

## Candidate definitions

### An et al. (2023)-inspired candidate

The paper describes an anisotropic 3-D U-Net variant with 8/16/32 feature
maps, two in-plane pooling levels, mostly `3 x 3 x 1` convolutions, no batch
normalization, inverse-frequency weighted categorical cross entropy, Adam at
`1e-4`, Gaussian noise with standard deviation `0.45`, and Glorot
initialization. The transparent implementation in this repository has 42,442
parameters.

This is explicitly a paper-inspired full-field adaptation, not an exact
reproduction. The authors' public repository is inference-only, the training
crop implementation is unavailable, and the released TorchScript model has
three logits while the paper describes two classes. The adaptation uses two
classes, native full fields, robust per-volume percentile scaling to `[-1,1]`,
physical batch size 1 with gradient accumulation 8, and validation-Dice
checkpointing.

Sources:

- <https://www.nature.com/articles/s41598-023-39826-8>
- <https://github.com/scalableminds/stroke-lesion-segmentation>

### nnU-Net v2 candidate

The notebook pins nnU-Net v2.8.1 source commit
`468cf803df9b267150ae2b6c0c59b8ac84f16227`, uses the official
`nnUNetPlannerResEncM` preset for a P100 16 GB GPU, and initially uses the
official `nnUNetTrainer_250epochs` compute-budget trainer. The same preserved
folds are installed as `splits_final.json`; nnU-Net is never allowed to make a
new split. Validation probabilities come from `checkpoint_best.pth` through
`--npz --val_best` and are converted back to the common native-space OOF
contract with an orientation assertion.

Sources:

- <https://pypi.org/project/nnunetv2/>
- <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/reference/dataset-format.md>
- <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md>
- <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md>
- <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md>

## Required inputs

- `LYS_T2w_manual_v1` (258 cases)
- `External_Mouse_T2w_manual_LSP_SI_v0` (426 cases)
- one preserved RatLesNetV2 v1 `split_assignments.csv`, normally supplied by
  the prior final artifact bundle
- this repository branch

The notebook refuses to regenerate target splits. It also omits all 57
locked-test images from both new architecture workspaces.

## Interpretation boundary

The paper reported approximately 0.89 Dice internally and 0.76 on an external
dataset, but those values are not acceptance thresholds for LYS because the
cohort, split, labels, and evaluation protocol differ. Select from paired LYS
OOF evidence. External validation is source-adaptation evidence only.

If the old 57-case locked test has already been evaluated, it is historical
evidence and cannot be reused to develop or select these new candidates. A new
untouched test version is required for a final claim. Model outputs remain
draft masks until independent validation and an explicit change to the
human-review policy.
