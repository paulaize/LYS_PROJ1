# LYS mouse T2w lesion-model training

Branch `dl-ratlesnetv2-finetune` develops and selects a RatLesNetV2 model for
mouse T2-weighted ischemic-lesion segmentation using human-reviewed LYS masks.
The inherited MRI/IHC pipeline remains in the repository for compatibility and
context, but it is not the active development target on this branch.

Read [AGENTS.md](AGENTS.md) before changing code.

## Active experiment

```text
upstream rat checkpoint
  -> direct LYS Tversky, folds 0-4
  -> direct LYS CE+Dice, folds 0-4
  -> paired OOF comparison and loss selection

upstream rat checkpoint
  -> external mouse adaptation selected on external validation only
  -> selected LYS setup, folds 0-4
  -> paired OOF direct-versus-external comparison

freeze five models + OOF threshold + mean-probability ensemble
  -> evaluate the locked LYS test once
```

The exact Kaggle cells and decision gates are in
[docs/ratlesnetv2_lys_v1_kaggle_workflow.md](docs/ratlesnetv2_lys_v1_kaggle_workflow.md).
External-source inclusion and provenance are in
[docs/ratlesnetv2_external_datasets.md](docs/ratlesnetv2_external_datasets.md).

After the RatLesNetV2 OOF experiment, the explicit v2 architecture-comparator
notebook can compare a paper-inspired An et al. (2023) network with nnU-Net v2
on the same preserved development folds. It is a separate protocol and does
not reopen the locked test:
[notebooks/lys_v2_architecture_comparator_kaggle.ipynb](notebooks/lys_v2_architecture_comparator_kaggle.ipynb).

The separate LYS v3 deployment comparator screens the official standard 3-D
nnU-Net against the ResEnc-M deployment cost without using external data or
materializing the locked test:
[notebooks/lys_v3_standard_nnunet_kaggle.ipynb](notebooks/lys_v3_standard_nnunet_kaggle.ipynb).

## Current datasets

- `LYS_T2w_manual_v1`: 258 corrected target cases, expected spacing
  `0.07 x 0.07 x 0.5 mm`.
- `External_Mouse_T2w_manual_LSP_SI_v0`: 426 manual external records used only
  as a controlled initialization comparator.

Kaggle may alter NIfTI suffixes while exposing dataset contents. The active
workflow normalizes inputs by payload type under `/kaggle/working`; never
rename files under `/kaggle/input`.

## Local checks

Use the existing `lys-bbb` environment:

```bash
make env-check
make lint
make test
```

Useful training-data commands:

```bash
make ratlesnetv2-audit RUN_ARGS="--help"
make ratlesnetv2-grouped-cv RUN_ARGS="--help"
make ratlesnetv2-split-prepared RUN_ARGS="--help"
make ratlesnetv2-normalize RUN_ARGS="--help"
make ratlesnetv2-calibrate-threshold RUN_ARGS="--help"
make ratlesnetv2-evaluate-ensemble RUN_ARGS="--help"
```

The Kaggle guide calls the trainer and evaluation utilities directly with
fully recorded options. Local preparation/rebuild commands are summarized in
[ratlesnetv2_finetune/README.md](ratlesnetv2_finetune/README.md).
Frozen five-model inference on unlabeled scans is documented in
[docs/ratlesnetv2_mac_inference.md](docs/ratlesnetv2_mac_inference.md).

## Downstream Allen atlas mapping

The first MRI-to-atlas adapter is documented in
[docs/t2w_allen_atlas_mapping.md](docs/t2w_allen_atlas_mapping.md). It stages
frozen inference inputs for AIDAmri v3, validates subject-space atlas labels,
generates registration QC overlays, and reports native-space lesion overlap by
atlas region. Atlas analysis is downstream of the frozen lesion model and must
not feed back into model selection or the locked-test protocol.

## Active package

```text
ratlesnetv2_finetune/
├── dataset.py                         # canonical prepared-folder writer
├── source_folders.py                  # local source-plan support
├── roiset_to_nifti_mask.py            # Fiji RoiSet conversion
├── configs/                            # portable templates; local plans ignored
└── scripts/
    ├── normalize_prepared_dataset.py  # Kaggle filename/payload normalization
    ├── audit_prepared_dataset.py      # scan/mask QC and overlays
    ├── create_grouped_cv.py           # locked test + five OOF folds
    ├── split_prepared_dataset.py      # external train/validation split
    ├── finetune_ratlesnetv2.py        # active trainer/exporter
    ├── train_an2023_unet_adapted.py   # versioned paper-inspired comparator
    ├── prepare_architecture_comparator.py
    ├── compare_oof_candidates.py
    ├── calibrate_probability_threshold.py
    ├── evaluate_probability_ensemble.py
    └── ...                             # dataset rebuild/provenance utilities
```

Generated datasets and runs belong under `work/`, `outputs/`, or
`/kaggle/working`; they are not committed.

## Scientific boundary

The model is intended to reduce manual lesion-drawing burden. Until a frozen
version is independently validated and the review policy changes explicitly,
its predictions remain drafts and human-reviewed masks remain the scientific
source of truth for lesion volume.

The locked test is not a development dashboard. It is opened only after the
loss, initialization, five checkpoints, OOF threshold, ensemble, and
postprocessing rule are frozen.
