# LYS v3 standard 3-D nnU-Net: Kaggle protocol

The executable notebook is
`notebooks/lys_v3_standard_nnunet_kaggle.ipynb`.

This is a deployment-oriented development experiment. It does not replace the
RatLesNetV2 v1 protocol or the LYS v2 ResEnc-M comparator. It uses LYS target
data only; external mouse data are deliberately excluded.

## Objective

Measure whether the official standard 3-D full-resolution nnU-Net provides a
better inference-size/speed tradeoff than ResEnc-M without an unacceptable loss
of target-domain development performance.

The fixed candidate is:

- dataset: `Dataset701_LYSDevelopmentV1`;
- planner: official `ExperimentPlanner`;
- plans: `nnUNetPlans`;
- configuration: `3d_fullres`;
- architecture gate: `PlainConvUNet`;
- direct LYS initialization only;
- postprocessing: none.

The five-epoch fold-0 run is a runtime and packaging benchmark only. It is not
model-selection evidence. The optional 250-epoch five-fold stage is a
compute-budget screen. A final nnU-Net claim still requires a separately
declared canonical training budget and a new untouched test version.

## Required Kaggle inputs

Attach:

1. `LYS_T2w_manual_v1` as an exposed directory or
   `LYS_T2w_manual_v1.tar.gz`;
2. one preserved RatLesNetV2 artifact containing the 258-row
   `split_assignments.csv`, or the CSV itself;
3. Internet access so the pinned repository and nnU-Net source can be fetched.

Enable a Kaggle GPU. The notebook pins nnU-Net v2.8.1 at source commit
`468cf803df9b267150ae2b6c0c59b8ac84f16227`.

The notebook refuses to regenerate target splits. It normalizes Kaggle inputs
under `/kaggle/working`, materializes only the 201 development cases in
nnU-Net format, and asserts that all 57 locked-test cases remain omitted.

## Run order

1. Leave `RUN_BENCHMARK_5E=True` and `RUN_FULL_250=False`.
2. Run all cells. Download
   `LYS_v3_standard3d_5epoch_benchmark.zip`.
3. Test that model locally with
   `nnUNetv2_predict_from_modelfolder`, fold 0, and
   `checkpoint_best.pth`.
4. If the size and CPU runtime are acceptable, start a new Kaggle session,
   attach the resume archive, set `RUN_FULL_250=True`, and choose folds through
   `FOLDS_TO_RUN`.
5. Download a new resume archive at the end of every Kaggle session.

Interrupted nnU-Net runs resume from `checkpoint_latest.pth` with `--c`.
Completed runs are verified by checkpoint SHA-256 and validation case IDs.

## Interpretation

The five-epoch benchmark answers only:

- whether the standard plan is truly smaller;
- whether training and inference execute;
- checkpoint size and parameter count;
- approximate target hardware inference cost.

It must not be compared scientifically with completed RatLesNetV2 or ResEnc-M
experiments. For the 250-epoch stage, pooled OOF probabilities from all five
preserved folds are required before accuracy claims or TTA/postprocessing
decisions. Model predictions remain draft masks requiring human review.
