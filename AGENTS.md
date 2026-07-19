# AGENTS.md — RatLesNetV2 training branch brief

Read this before changing the branch. When a scientific fact is missing, fail
helpfully or ask Paul; do not infer it from a filename or image appearance.

## Branch objective

Branch `dl-ratlesnetv2-finetune` develops and selects a T2-weighted mouse brain
lesion segmentation model from human-reviewed masks. The active protocol is:

1. compare Tversky with CE+Dice on the same five grouped LYS development folds;
2. select the loss from pooled out-of-fold (OOF) evidence;
3. test whether external-mouse adaptation improves that selected LYS setup;
4. freeze initialization, five fold models, OOF threshold, mean-probability
   ensemble, and `postprocessing=none`;
5. evaluate the locked LYS test once.

The executable source of truth is
`docs/ratlesnetv2_lys_v1_kaggle_workflow.md`. Dataset provenance and inclusion
rules are in `docs/ratlesnetv2_external_datasets.md`. Do not revive historical
notebooks, grids, or preliminary `LYS_v0` results as current evidence.

## Scientific rules that must not drift

- Human-reviewed native lesion masks are the target labels.
- Model predictions remain draft masks until the frozen model is independently
  validated and the human-review policy is changed explicitly.
- The locked test cannot influence loss, initialization, checkpoint,
  probability threshold, preprocessing, postprocessing, or visual decisions.
- Do not inspect fold-specific locked-test metrics while exporting the five
  final probability maps. Evaluate their ensemble once.
- Select checkpoints using validation Dice only. The active target schedule is
  at most 50 epochs, validation every epoch, early stopping 12, and reduce-LR
  plateau patience 4.
- Overall voxel accuracy is background dominated and must not select a model.
  Compare Dice, precision/recall, failures, volume error, HD95, surface Dice,
  and fold/cohort consistency.
- External validation measures source adaptation only. It is never evidence of
  target-domain LYS performance.
- Keep `postprocessing=none` unless a future version validates a rule entirely
  on OOF development predictions before opening a new locked test.

## Active datasets

### LYS target domain

- prepared dataset: `LYS_T2w_manual_v1`
- 258 cases after the four approved exclusions recorded in the packaging
  workflow
- T2 geometry expected by QC: `0.07 x 0.07 x 0.5 mm`
- native scans are anisotropic/2.5-D
- one RatLesNetV2 scan per case: `X x Y x slices x 1`
- one binary 3-D label: `scan_lesionIAM.nii.gz`
- `scan_lesion.nii.gz` is a required identical compatibility alias

Subject-disjoint grouping is mandatory. Keep only clear filename-supported
longitudinal pairs together; otherwise use conservative singleton grouping.
Do not manually rewrite case identities during model selection without an
explicit metadata correction and a new split version.

### External mouse comparator

- prepared dataset: `External_Mouse_T2w_manual_LSP_SI_v0`
- 426 manual native scan/mask records from An2022, unique Knab2025 cases, and
  Koch2017
- current source split: 80/20 train/validation grouped by manifest
  `animal_id`
- external data may initialize a target model but never enter the final LYS
  test claim

Kaggle has exposed some `.nii.gz` inputs with plain `.nii` or truncated
suffixes. Never rename those files blindly. Use
`ratlesnetv2_finetune.scripts.normalize_prepared_dataset`, which detects gzip
from the payload, fully loads the arrays, validates both mask copies, and
writes canonical files under `/kaggle/working`.

## Active code map

Core experiment:

- `audit_prepared_dataset.py`: geometry, affine, mask, and overlay QC
- `create_grouped_cv.py`: locked test plus five grouped development folds
- `split_prepared_dataset.py`: external-only train/validation split
- `finetune_ratlesnetv2.py`: training, validation checkpointing, status,
  metrics, and probability exports
- `calibrate_probability_threshold.py`: validation-only OOF threshold selection
- `evaluate_probability_ensemble.py`: one-time five-model locked-test ensemble
- `normalize_prepared_dataset.py`: Kaggle input repair and validation

Dataset rebuild/provenance utilities are also active and must be kept:

- `dataset.py`, `source_folders.py`, and `prepare_dataset.py`
- `prepare_lys_roiset_dataset.py`, `roiset_to_nifti_mask.py`, and
  `review_lys_masks_itksnap.py`
- `download_external_datasets.py`, `orient_external_dataset_lsp.py`, and
  `flip_external_si_axis.py`
- `package_prepared_dataset.py`

Do not add alternative architectures, generic experiment grids, or historical
wrappers to the active path while the controlled RatLesNetV2 experiment is in
progress. A future comparator belongs in a new, explicit protocol version.

## Data and repository rules

1. Treat `data/`, `/Volumes/...`, and `/kaggle/input` as read-only.
2. Local intermediates go under `work/`; final local deliverables go under
   `outputs/`; Kaggle intermediates go under `/kaggle/working`.
3. Never commit data, model weights, predictions, run directories, secrets,
   local absolute paths, QuPath projects, or large binaries.
4. Use `pathlib.Path`; do not concatenate path strings.
5. Preserve manifests, normalization reports, QC reports, split assignments,
   seeds, Git revisions, checkpoint hashes, and threshold provenance.
6. Fail loudly on missing files, unexpected counts, shape/affine mismatches,
   non-binary labels, duplicate prediction rows, or split leakage.
7. Tests must use synthetic fixtures and must not download models or data.

Ignored `local_*.yml` files can contain workstation paths. They are local
rebuild inputs, not portable branch documentation.

## Environment and commands

Use the existing `lys-bbb` conda environment on the Mac; do not create a new
environment unless Paul asks. Python is 3.11.x on macOS arm64.

```bash
make env-check
make lint
make test
```

After every code change, run at least `make lint` and `make test`.

Local PyTorch code must support MPS/CPU rather than assuming CUDA. The active
Kaggle training command uses its NVIDIA GPU through the upstream RatLesNetV2
interface and records the exact upstream Git revision in the frozen spec.

Before calling an external tool or upstream repository method, inspect its
installed help/source. Do not invent APIs.

## Outputs required for completion

Preserve:

- both normalization reports and portable manifests;
- LYS QC report and reviewed overlays;
- `inferred_subject_groups.csv`, `split_assignments.csv`, and split summaries;
- 10 direct target fold runs;
- one external source checkpoint and five external-initialized target runs;
- three OOF threshold reports;
- paired loss and initialization comparisons;
- frozen specification with model hashes and Git revisions;
- five locked-test probability exports and final ensemble report;
- the final reproducibility artifact bundle from the Kaggle guide.

A training run is not complete merely because it wrote a model file. Its
status, selected checkpoint, metrics, probability manifest, and required
provenance must also exist.

## Non-training project context

This branch inherits the MRI/IHC v1 implementation from `main`. It is context,
not the active development target:

- MRI scientific volume still comes from a human-corrected mask and NIfTI
  header spacing.
- IHC readouts are named `IgG-FITC`, not direct LYS241 concentration.
- Anti-human IgG specificity is resolved, but the positivity threshold still
  requires control-based calibration and human sign-off.
- MRI and IHC will later register independently to Allen CCFv3; direct MRI/IHC
  registration is figure-only, not a quantification route.

Do not expand atlas, compartments, IHC, or batch-processing work on this branch
unless Paul explicitly redirects the task. Avoid deleting inherited production
code merely because the active RatLesNetV2 guide does not call it.

## Coding style

- Prefer small, explicit changes over rewrites of the active trainer.
- Keep inputs/outputs visible and deterministic.
- Use config/manifest values rather than hidden globals or filename guesses.
- Keep scientific uncertainty explicit in names and reports.
- Add focused tests for validation gates and failure cases.
- Remove superseded code instead of maintaining parallel undocumented paths;
  Git history is the archive.
