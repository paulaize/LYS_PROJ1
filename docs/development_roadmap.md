# Development Roadmap

The project is built in milestones. The active target is still v1: a thin,
reviewed, one-animal spine. Later milestones add quality, atlas labels,
compartments, batching, and eventually active learning.

## Principles

1. Build the end-to-end spine before improving any one stage.
2. Keep stable interfaces so backends can change without rewriting callers.
3. Treat automatic MRI masks as drafts; human-reviewed masks are truth.
4. Keep v1 free of atlas, DL, compartments, and batch processing.
5. Record uncertainty and provenance instead of hiding it in variable names.

## Current v1 Decisions

- v1 animal: `BD_08_5D`
- control animal for IHC threshold/background calibration: `C6S5`
- MRI lesion side: `image_right`
- T2 voxel spacing: `0.07 x 0.07 x 0.5 mm`
- MRI draft segmentation: threshold seed only, disposable if wrong
- MRI source of truth: `lesion_corrected.nii.gz`
- IHC readout name: `IgG-FITC` / `igg_fitc_*`
- anti-IgG specificity: resolved as anti-human IgG specific to humanized LYS241
- IgG-FITC positivity threshold: still open; must be calibrated and approved

## Stable Interfaces

Keep these signatures stable:

```python
# src/mri/segment.py
def segment_lesion(volume, brain_mask, *, method="threshold", **kw):
    """Return a binary lesion mask. v1: method='threshold'. Later: method='dl'."""

# src/mri/edit.py
def review_mask(volume, draft_mask, out_path):
    """Open editor, save corrected mask, return dict(edited, dice, reviewer)."""

# src/ihc/ingest.py
def ingest_qupath(csv_path):
    """QuPath measurement export -> list of tidy measurement rows."""
```

## Milestone 1 - v1 Thin Spine

Goal: one reviewed animal, no atlas, no DL.

MRI:

- load configured T2 NIfTI
- validate spacing/header
- N4 bias correction
- rough brain mask
- threshold draft lesion mask
- napari review/correction
- raw and Swanson/indirect corrected lesion volume

IHC:

- read channel/section facts from YAML
- create/review `tissue_v1`
- generate exploratory IgG-FITC threshold candidates from target/control images
- record threshold sign-off before final export
- measure IgG-FITC positive area inside reviewed tissue ROI

Deliverable:

```text
outputs/BD_08_5D/BD_08_5D_v1.csv
```

Required columns include:

```text
animal_id, timepoint, panel, modality, measure, value, unit,
area_mm2, n_cells, edited, edit_dice, reviewer, qc_flag, model_version
```

Active checklist: [v1_next_session_todo.md](v1_next_session_todo.md).

## Milestone 2 - Better Drafts

Goal: improve the draft masks/measurements without changing the v1 review rule.

- evaluate pretrained mouse T2 lesion models against corrected masks
- evaluate RatLesNetV2 and/or nnU-Net fine-tuning once enough masks exist
- keep predictions as drafts that require human review
- add cell/nucleus detection in QuPath with StarDist or InstanSeg
- keep area-level GFAP/IBA1 readouts for morphology-heavy markers

RatLesNetV2 branch status:

- Kaggle is the preferred free GPU runtime
- direct rat-pretrained -> LYS fine-tuning is the main baseline
- public mouse adaptation is only a comparator
- held-out LYS test remains untouched until strategy selection

RatLesNetV2 docs:

- [../ratlesnetv2_finetune/README.md](../ratlesnetv2_finetune/README.md)
- [ratlesnetv2_finetuning_branch.md](ratlesnetv2_finetuning_branch.md)
- [ratlesnetv2_external_datasets.md](ratlesnetv2_external_datasets.md)

## Milestone 3 - Atlas Labels

Goal: register MRI and IHC independently to Allen CCFv3.

MRI:

- AIDAmri first, ANTs/SyN fallback
- inspect registration overlays, especially large 5d lesions
- feed corrected lesion masks where the registration tool can use them

IHC:

- QuPath/Fiji/ABBA with DeepSlice as a starting point
- manual BigWarp refinement where needed
- import Allen region annotations back into QuPath

Do not register MRI directly to IHC for quantification. Direct MRI/IHC overlays
are figure-only.

## Milestone 4 - Compartments And Join

Goal: full tidy table shape.

- core = corrected lesion mask
- peri = physical dilation ring minus core
- contra = atlas-midline mirror
- early IHC-only animals get a separate `compartment_method`
- join MRI and IHC by `animal_id`, `allen_region`, `hemisphere`, and
  `compartment`

Deliverable shape:

```text
animal x region x compartment x cell_type x measure
```

## Milestone 5 - Batch And QC

Goal: all animals with reproducible outputs.

- loop configured animals
- skip/resume completed stages
- write QC overlays and index tables
- include controls
- prepare final CSVs for R

## Milestone 6 - Active Learning

Only after the pipeline is useful:

- collect corrected lesion masks/cell edits
- fine-tune lesion and cell models
- version model outputs
- track edit burden over time

## Decision Gates

- v1 complete: one reviewed animal produces a joined CSV with no guessed facts.
- DL adoption: a model reduces edit burden against corrected masks.
- atlas adoption: registration overlays pass QC on difficult 5d lesions.
- threshold adoption: IgG-FITC threshold has documented control-based review and
  sign-off.
- batch adoption: one animal can be rerun deterministically without duplicate
  measurements.

## Risks And Fallbacks

| Risk | Fallback |
|---|---|
| MRI threshold draft targets artifact | redraw during review; do not block v1 |
| QuPath direct `.vsi` opens are too slow | move to a QuPath project/cached-import workflow |
| no approved IgG-FITC threshold | keep outputs exploratory until sign-off exists |
| public mouse adaptation hurts LYS validation | discard that stage and keep direct LYS baseline |
| AIDAmri/ANTs setup is slow | defer atlas; v1 does not need it |
