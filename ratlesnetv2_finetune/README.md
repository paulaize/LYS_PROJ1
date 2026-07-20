# RatLesNetV2 training utilities

This package contains the active T2w lesion dataset preparation, QC, grouped
splitting, RatLesNetV2 fine-tuning, OOF threshold calibration, and locked-test
ensemble tools.

The executable experiment is
[docs/ratlesnetv2_lys_v1_kaggle_workflow.md](../docs/ratlesnetv2_lys_v1_kaggle_workflow.md).
Do not substitute ad hoc commands for its recorded folds, seeds, checkpoint
rules, or test gate.

## Active experiment commands

| Module | Purpose |
|---|---|
| `normalize_prepared_dataset` | recover Kaggle-altered NIfTI names into a validated canonical tree |
| `audit_prepared_dataset` | validate geometry, labels, artifacts, and review overlays |
| `create_grouped_cv` | create subject-disjoint LYS test/fold assignments |
| `split_prepared_dataset` | create grouped external train/validation data |
| `finetune_ratlesnetv2` | train/evaluate RatLesNetV2 and export probabilities |
| `calibrate_probability_threshold` | select a threshold from validation-only OOF maps |
| `evaluate_probability_ensemble` | evaluate the frozen five-model locked-test ensemble once |
| `infer_ratlesnetv2_ensemble` | run the frozen mean-probability ensemble on unlabeled scans |

## Versioned architecture comparator

The separate post-RatLesNetV2 comparator is executed by
[`notebooks/lys_v2_architecture_comparator_kaggle.ipynb`](../notebooks/lys_v2_architecture_comparator_kaggle.ipynb).
It reuses the preserved LYS development folds, omits locked-test images, and
adds three focused modules:

| Module | Purpose |
|---|---|
| `train_an2023_unet_adapted` | train the transparent paper-inspired full-field comparator |
| `prepare_architecture_comparator` | stage preserved folds, convert nnU-Net data, and export native OOF probabilities |
| `compare_oof_candidates` | produce paired case/fold/cohort comparisons from calibrated OOF reports |
| `create_oof_qc_contact_sheet` | render development-only scan/manual/prediction/error PNG QC |

This is an explicit v2 protocol, not an alternative path inside the active v1
RatLesNetV2 experiment. It does not evaluate a locked test.

Run help locally through the existing environment:

```bash
make ratlesnetv2-normalize RUN_ARGS="--help"
make ratlesnetv2-audit RUN_ARGS="--help"
make ratlesnetv2-grouped-cv RUN_ARGS="--help"
make ratlesnetv2-split-prepared RUN_ARGS="--help"
make ratlesnetv2-calibrate-threshold RUN_ARGS="--help"
make ratlesnetv2-evaluate-ensemble RUN_ARGS="--help"
make ratlesnetv2-infer RUN_ARGS="--help"
```

## Dataset rebuild utilities

These are not called during every Kaggle run, but they are required to recreate
the reviewed inputs and their provenance:

- `prepare_lys_roiset_dataset.py` and `roiset_to_nifti_mask.py`: stage LYS
  Bruker scans and convert Fiji lesion RoiSets.
- `review_lys_masks_itksnap.py`: queue editable mask copies without modifying
  source data.
- `dataset.py`, `source_folders.py`, `add_source_folder.py`, and
  `prepare_dataset.py`: build the RatLesNetV2 case-folder contract.
- `download_external_datasets.py`, `orient_external_dataset_lsp.py`, and
  `flip_external_si_axis.py`: recreate the external mouse source.
- `package_prepared_dataset.py`: create portable versioned Kaggle archives.

Portable configuration examples are under `configs/`. Files named
`local_*.yml` are ignored because they may contain workstation paths.

## Data contract

Each prepared case contains:

```text
scan.nii.gz               # 4-D X x Y x slices x 1
scan_lesionIAM.nii.gz     # 3-D binary label used by upstream loader
scan_lesion.nii.gz        # identical compatibility alias
```

The root `manifest.csv` carries case identity, split, study, timepoint, paths,
shape, spacing, and lesion-volume provenance. Generated datasets, run folders,
weights, and predictions stay outside Git.

The current 258-case `LYS_T2w_manual_v1` archive excludes exactly:

```text
Thrombin_09_PhIND__JD_TH09_C1S1bis_2
Thrombin09_C1S2
Thrombin_09_C1S4
Thrombin_09_C2S2
```

Changing that list requires a new dataset/split version.

## Development rule

Run `make lint` and `make test` after code changes. Tests use synthetic NIfTI
fixtures and must not download datasets or model weights.
