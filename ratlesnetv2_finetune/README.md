# RatLesNetV2 fine-tuning side track

This folder contains the lesion-segmentation dataset preparation, QC, training,
OOF calibration, and final ensemble utilities. It does not change the active
v1 scientific rule: lesion volumes used for the study remain based on
human-reviewed masks until the frozen model has been independently validated.

## Active LYS v1 workflow

The exact Kaggle cells and decision gates are in
[`docs/ratlesnetv2_lys_v1_kaggle_workflow.md`](../docs/ratlesnetv2_lys_v1_kaggle_workflow.md).

The active experiment is:

```text
corrected LYS_v1 + automatic QC
  -> conservative same-animal grouping
  -> locked 20% test + five development folds
  -> direct Tversky on folds 0-4
  -> direct CE+Dice on folds 0-4
  -> separate OOF threshold calibration and paired loss comparison
  -> one external-only train/validation adaptation checkpoint
  -> same selected LYS loss on folds 0-4 from that checkpoint
  -> external OOF threshold calibration
  -> paired direct-versus-external comparison
  -> freeze five models, threshold, mean-probability ensemble, no postprocessing
  -> evaluate the locked test once
```

Do not select Tversky versus CE+Dice from fold 0. Do not use LYS validation to
select the external source checkpoint. Do not inspect the locked test before
all model and threshold decisions are frozen.

“Direct” means the upstream RatLesNetV2 rat checkpoint is fine-tuned on LYS
without the external dataset. “External-pretrained” uses the same starting
checkpoint, adds external-only adaptation, then uses the identical LYS target
configuration. External data are retained as a controlled comparator, not
assumed to be beneficial.

nnU-Net remains a useful later benchmark but is not part of this active
RatLesNetV2 experiment.

## Current prepared archive

The corrected archive is:

```text
outputs/ratlesnetv2_finetune/LYS_T2w_manual_v1.tar.gz
```

It contains 258 cases and excludes:

```text
Thrombin_09_PhIND__JD_TH09_C1S1bis_2
Thrombin09_C1S2
Thrombin_09_C1S4
Thrombin_09_C2S2
```

The conservative subject inference groups only clear filename-supported D1/D7
pairs and treats every other case as a singleton. Its output
`inferred_subject_groups.csv` and the final `split_assignments.csv` are required
provenance artifacts.

## Main utilities

- `audit_prepared_dataset.py`: scan/mask shape, affine, spacing, lesion geometry,
  connected components, flags, and overlay gallery.
- `create_grouped_cv.py`: locked subject groups and five development folds.
- `split_prepared_dataset.py`: external-only train/validation split grouped by
  manifest `animal_id`.
- `finetune_ratlesnetv2.py`: controlled losses, validation-Dice checkpointing,
  early stopping, LR reduction, and NIfTI probability export.
- `calibrate_probability_threshold.py`: pooled OOF threshold sweep with case,
  surface, volume, failure, and subgroup metrics; rejects test predictions.
- `evaluate_probability_ensemble.py`: averages five frozen test probability
  maps using an OOF-derived threshold and writes the one-time locked-test report.
- `package_prepared_dataset.py`: versioned portable `.tar.gz` packaging with
  explicit case exclusions.

The existing An-2023 and historical training-grid utilities are optional
comparators. They are not required for the active workflow. nnU-Net is deferred
and has no active conversion/training script in this branch.

## Local commands

Use the existing `lys-bbb` environment:

```bash
make test

make ratlesnetv2-audit RUN_ARGS="--help"
make ratlesnetv2-grouped-cv RUN_ARGS="--help"
make ratlesnetv2-split-prepared RUN_ARGS="--help"
make ratlesnetv2-calibrate-threshold RUN_ARGS="--help"
make ratlesnetv2-evaluate-ensemble RUN_ARGS="--help"
```

Dataset inputs are read-only. Local intermediates belong under `work/`, final
archives and reports under `outputs/`, and neither model weights nor prepared
datasets belong in git.

## Historical results

Results from the old 39-case `LYS_v0` validation split are preliminary only.
They motivated testing RatLesNetV2, Tversky, CE+Dice, and external adaptation,
but they must not be used to select the final `LYS_v1` model because labels and
subject grouping changed. Historical notebooks, predictions, and checkpoints
should be preserved as `LYS_v0`, not mixed with the v1 output folders.
