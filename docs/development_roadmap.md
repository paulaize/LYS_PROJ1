# Development Roadmap

This roadmap is the project-level milestone map. It should stay short. Active
IHC details belong in `docs/IHC_v1_plan.md`; the exact next checklist belongs in
`docs/v1_next_session_todo.md`.

## Operating Principles

- Build the thin spine before adding atlas, DL, compartments, or batch logic.
- Keep MRI and IHC development separable while the MRI model work runs on
  `dl-ratlesnetv2-finetune`.
- Use stable interfaces so later backends do not rewrite callers.
- Let humans review scientific gates: lesion mask, section QC, tissue ROI,
  threshold approval, and later atlas/cell outputs.
- Never hide unknown facts in code. Unknown channel, threshold, spacing,
  section-pairing, or control facts must fail clearly or stay in config as
  `null`.

## Current Active Milestone: v1 Thin Spine

Goal: one reviewed animal-level output for `BD_08_5D`.

MRI:

- Bruker scan 2/reco 1 T2w converted to NIfTI.
- Header spacing validated.
- N4/brain mask run.
- Automatic lesion mask is only a draft.
- Human-reviewed `lesion_corrected.nii.gz` is required before scientific use.
- Lesion volume uses NIfTI header spacing.

IHC:

- Target/control `.vsi` files configured.
- QuPath diagnostics, threshold sweeps, and threshold review thumbnails exist.
- Section-QC thumbnail/manifest export exists.
- Panel-specific section selections and exploratory threshold tooling exist.
- The updated C6S5 Panel B bundle is technically available as of 2026-07-22;
  all eight section series require freshly generated QC and remain
  `technical_open_only_unreviewed` until human selection.
- The active target is one isolated two-panel exploratory run with complete
  bundle checksums, per-section failures, separate automatic candidates, and
  deterministic tidy outputs.
- A clean real-data run completed on 2026-07-22. Panel A produced an automatic
  exploratory candidate and target rows; Panel B produced a complete
  sensitivity table but no passing candidate. Remaining blockers are human
  section/tissue/artifact review, acquisition compatibility, and separate
  threshold approval—not technical file access.

v1 acceptance:

- selected usable IHC sections
- approved IgG-FITC threshold
- reviewed `tissue_v1`
- reviewed `artifact_exclude` annotations where folds/tears/debris would
  inflate FITC signal
- deterministic IHC export
- reviewed MRI lesion mask
- animal-level joined CSV
- `make test` and `make lint` pass

## Milestone 1: IHC v1 Completion

Build the IHC side to a defensible first result.

Tasks:

1. Keep section QC thumbnails and manifest as provenance.
2. Use the configured panel-specific section selections.
3. IgG-FITC threshold sweeps and review thumbnails on selected sections.
4. Threshold sign-off record.
5. Reviewed `tissue_v1` gate.
6. Duplicate-safe final QuPath export.
7. Python ingestion to deterministic combined IHC CSV.

Do not add cell segmentation, atlas, compartments, or cross-panel ROI transfer
in this milestone.

## Milestone 2: MRI v1 Completion

Complete the MRI side independently of the IHC branch.

Tasks:

1. Human correction/redraw of the T2w lesion mask.
2. Raw and Swanson/indirect edema-corrected lesion volume.
3. Per-slice lesion area profile.
4. QC metadata: edited, edit Dice, reviewer, flag.
5. Optional T2*/TOF QC exploration only if it does not delay v1.

The MRI DL finetuning branch can improve draft masks later, but v1 uses the
human-corrected mask as truth.

## Milestone 3: v1 Join

Join IHC and MRI at animal level only.

Join key:

```text
animal_id + timepoint
```

Do not use section-level or atlas-region joins until those mappings exist.

Deliverable:

```text
outputs/BD_08_5D/BD_08_5D_v1.csv
```

Minimum columns:

```text
animal_id, timepoint, panel, modality, measure, value, unit,
area_mm2, n_cells, edited, edit_dice, reviewer, qc_flag, model_version
```

## Milestone 4: Region And Hemisphere Layer

Add region-first IHC structure without yet requiring full core/peri/contra.

Tasks:

1. Section orientation QC.
2. Hemisphere split and ipsi/contra labels.
3. Per-section ROI manifests.
4. Optional simple whole-ipsi vs whole-contra marker summaries.

Panel A/B are still not treated as exact section matches unless the physical
sectioning scheme is confirmed.

## Milestone 5: Atlas Registration

Bring both tracks into Allen space independently.

MRI:

- AIDAmri or ANTs-based MRI-to-Allen registration.
- Corrected lesion mask and optional edema map in atlas space.

IHC:

- QuPath/ABBA/DeepSlice or manual ABBA registration from DAPI/anatomy.
- Per-section region labels.

QC:

- registration overlay review
- failed/moderate/good registration quality labels
- no direct MRI-to-IHC quantification

## Milestone 6: Compartments

Build core/peri/contra once atlas and MRI lesion masks are reliable.

Rules:

- core = reviewed MRI lesion mask
- peri = physical distance ring around core
- contra = atlas-midline mirror
- early IHC-only animals need a different `compartment_method`
- Panel A NeuroTrace-derived histology core is a later fallback/proxy, not v1

Output:

```text
animal x region x compartment x hemisphere
```

## Milestone 7: Cell/Object-Level IHC

Add cell and object analysis after area-level v1 works.

Likely path:

- DAPI nuclei segmentation with StarDist/InstanSeg/Cellpose candidates.
- Panel A object/area metrics for NeuroTrace and Podocalyxin.
- Panel B area-first metrics for IBA1 and GFAP.
- IgG-FITC continuous intensity plus positive-area metrics.
- Human correction/spot-check gate before model outputs are trusted.

Do not make activation claims from IBA1/GFAP morphology until validated.

## Milestone 8: Batch And Reporting

After one animal is reliable:

- loop configured animals
- handle IHC-only early timepoints
- generate QC index tables
- generate HTML/PDF review report
- write final tidy long table for R

## Branch Strategy

Keep active branches separated by responsibility:

- `ihc`: QuPath/Python IHC workflow, section QC, threshold calibration,
  IHC export, documentation.
- `dl-ratlesnetv2-finetune`: MRI lesion segmentation model preparation and
  finetuning.
- Merge after both sides expose stable outputs that join at animal level.

Avoid broad refactors while both branches are active.

## Stable Interfaces

These interfaces should remain stable:

```python
def segment_lesion(volume, brain_mask, *, method="threshold", **kw):
    ...

def review_mask(volume, draft_mask, out_path):
    ...

def ingest_qupath(csv_path):
    ...
```

Optional keyword-only additions are fine. Breaking callers is not.

## Active Open Decisions

- Which IHC sections are good enough for threshold calibration and final export?
- What IgG-FITC threshold is approved for each panel/acquisition scope?
- Are Panel A and Panel B physically adjacent/matched sections?
- What is the inter-section spacing/z-position?
- Are additional controls available: single-stain, unstained, no-primary, or
  secondary-only?
- Which peri-ring width is biologically meaningful for later compartments?
- Which atlas registration path should become primary after v1?
