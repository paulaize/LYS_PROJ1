# RatLesNetV2 fine-tuning branch

Branch: `dl-ratlesnetv2-finetune`

This is an independent MRI lesion deep-learning track. It is not the current
v1 scientific segmentation backend: model outputs remain draft masks until the
frozen model has been independently validated and the human-review policy is
reconsidered explicitly.

The active executable protocol is
[`ratlesnetv2_lys_v1_kaggle_workflow.md`](ratlesnetv2_lys_v1_kaggle_workflow.md).
Supporting command details live in
[`../ratlesnetv2_finetune/README.md`](../ratlesnetv2_finetune/README.md).

## Current position

Historical `LYS_v0` runs showed that RatLesNetV2 learns the target task and
made Tversky, CE+Dice, and external mouse adaptation credible candidates. They
are not final evidence because labels and subject grouping have changed.

The corrected `LYS_v1` experiment therefore starts over with:

- 258 corrected cases after four explicit exclusions;
- automatic scan/mask QC;
- conservative grouping of clear same-animal D1/D7 pairs;
- approximately 20% locked test subjects;
- five grouped development folds;
- pooled OOF probability maps and configuration-specific thresholds.

## Experiment order

```text
direct upstream-rat checkpoint -> LYS Tversky, folds 0-4
direct upstream-rat checkpoint -> LYS CE+Dice, folds 0-4
  -> compare paired OOF results and select the loss

upstream-rat checkpoint -> external train
  -> checkpoint selected on external validation only
  -> selected LYS loss, folds 0-4
  -> compare paired OOF results against direct LYS

freeze initialization, five fold checkpoints, OOF threshold,
mean-probability ensemble, and postprocessing=none
  -> evaluate locked LYS test once
```

The external stage must never monitor LYS while choosing its source
checkpoint. Only the later LYS fine-tuning folds measure whether the external
initialization helped.

Do not select the loss from fold 0. Do not use the locked test for threshold,
checkpoint, preprocessing, postprocessing, or visual model decisions.

## Local and cloud responsibilities

Local MacBook:

- convert and review LYS masks;
- prepare and package portable datasets;
- validate manifests, shapes, affines, and spacing;
- preserve correction, grouping, and split provenance;
- run the test suite.

Kaggle GPU:

- run automatic dataset QC and inspect the report;
- freeze the grouped split;
- train 10 direct target folds, one external source model, and five
  external-initialized target folds;
- calibrate three OOF thresholds and create two paired comparison reports;
- run the final five-model test ensemble once.

## Required artifacts

- `LYS_v1_qc/`;
- `inferred_subject_groups.csv` and `split_assignments.csv`;
- each run's `RatLesNetv2.model`, `selected_checkpoint.json`, metrics, status,
  and final validation probabilities;
- three `selected_threshold.json` files and case/subgroup reports;
- paired loss and initialization comparisons;
- the external source checkpoint selected using external validation only;
- `lys_v1_final_frozen_spec.json` with checkpoint hashes;
- the final locked-test ensemble probability maps, masks, and report.

## Metrics

Primary comparison evidence is per-case OOF Dice. Also inspect paired fold and
cohort behavior, precision, recall, complete failures, lesion detection,
absolute volume error, HD95, surface Dice, and Bland–Altman volume agreement.
Overall voxel accuracy is background dominated and must not select a model.

## Limits

- One-case or one-epoch runs are smoke tests only.
- External validation is not evidence of LYS performance.
- A small mean Dice difference that is inconsistent across folds or cohorts is
  not decisive evidence for a more complicated path.
- nnU-Net, additional losses, augmentation, normalization, N4, sampling, and
  postprocessing remain later controlled ablations, not changes to improvise
  after the locked-test gate.
