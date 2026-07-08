# MRI + IHC Pipeline Plan

This is the high-level project plan. The active IHC implementation plan is
`docs/IHC_v1_plan.md`. The short executable checklist is
`docs/v1_next_session_todo.md`.

## Project Goal

Build a reproducible analysis pipeline for a mouse ischemic-stroke study that
eventually emits a tidy table suitable for R:

```text
animal x region x compartment x cell_type x measure
```

The pipeline has two independent imaging tracks:

- MRI: Bruker T2w lesion volume and later atlas/compartment labels.
- IHC: Olympus `.vsi` multiplex fluorescence quantification.

MRI and IHC should not be directly registered to each other for quantification.
The long-term bridge is atlas space: MRI goes to Allen, IHC goes to Allen, and
tables join on animal, region, hemisphere, and compartment.

## Active v1 Scope

v1 is a thin spine on one animal:

- one target animal: `BD_08_5D`
- one IHC control: `C6S5`
- no atlas
- no deep learning
- no core/peri/contra compartments
- no cell-level detection
- no batch processing

v1 output is an animal-level table joining:

- reviewed MRI T2w lesion volume
- selected-section IHC IgG-FITC positive-area metrics

The immediate IHC plan is in `docs/IHC_v1_plan.md`.

## Study Shape

Known current protocol:

- cross-sectional timepoints: 1h, 3h, 6h, 24h, 48h, 5d
- MRI exists only for 24h, 48h, and 5d animals
- IHC exists for early and late animals
- about 10 stroke animals and 3 controls
- each IHC panel has 8 section series plus overview/label series

Configured current files:

- Panel A: DAPI, IgG-FITC, Podocalyxin, NeuroTrace
- Panel B: DAPI, IgG-FITC, GFAP, IBA1

Do not assume Panel A and Panel B sections are matched. Until the physical
sectioning scheme is confirmed, cross-panel comparisons are animal-level or
later atlas-region-level, not exact pixel/cell colocalization.

## MRI Track

v1 MRI input:

- Scan 2, RARE `T2_haute_resolution_Turbo`
- T2w lesion-volume scan
- 256 x 256, FOV 17.92 mm
- expected spacing: `0.07 x 0.07 x 0.5 mm`
- 18 slices, no gap

v1 MRI workflow:

1. Convert Bruker scan 2/reco 1 to NIfTI.
2. Validate header spacing.
3. Run N4 bias correction.
4. Create brain/hemisphere support.
5. Generate an automatic lesion draft.
6. Review/correct the lesion mask manually.
7. Compute raw and Swanson/indirect edema-corrected lesion volume from header
   spacing.

The automatic mask is only a draft. The reviewed `lesion_corrected.nii.gz` is
the v1 source of truth and future model-reference data.

Optional MRI adjuncts for later:

- T2*: hemorrhage/susceptibility QC or covariate.
- TOF: vascular/MCA-flow context or covariate.

They are not v1 blockers.

## IHC Track

v1 IHC uses a hybrid workflow:

- QuPath/Bio-Formats for `.vsi` reading, review, overlays, and measurement
  export.
- Python for orchestration, validation, CSV ingestion, provenance, and joins.

v1 IHC workflow:

1. Build a section-QC manifest and thumbnails.
2. Select usable sections.
3. Calibrate IgG-FITC thresholds from selected target/control sections.
4. Review threshold overlays visually.
5. Record threshold sign-off.
6. Review/correct `tissue_v1`.
7. Export final IgG-FITC positive area inside reviewed tissue.
8. Ingest to Python and write deterministic IHC CSVs.

Important IHC rules:

- Keep the marker/readout name `IgG-FITC`.
- Anti-human IgG specificity to LYS241 is resolved, but threshold calibration
  is still required.
- C6S5 is a no-LYS241/background control, not a complete spectral correction
  control.
- Do not claim full spectral unmixing without single-stain, unstained,
  no-primary, and secondary-only controls.
- Section thickness is `10 um`, but inter-section spacing/z-position is
  unknown; do not compute whole-lesion histology volume in v1.
- For `BD_08_5D`, IHC image-right is recorded as ipsilateral with visual
  orientation QC still pending.

## Join Strategy

Use three join levels, in this order.

### 1. Animal-Level Join

This is v1.

```text
animal_id + timepoint
```

MRI contributes:

- T2w lesion volume
- edema-corrected lesion volume
- review metadata

IHC contributes:

- selected-section IgG-FITC metrics per panel
- section/tissue/threshold QC metadata

This join is safe even when Panel A/B sections are not matched and IHC sections
are not mapped to MRI slices.

### 2. Section/Slice-Level Join

Use only after a section-to-MRI-slice mapping exists.

```text
animal_id + section_id + approximate_mri_slice
```

Possible outputs:

- MRI slice lesion area vs IHC sampled lesion/marker burden.
- MRI local intensity ratio vs IHC marker intensity.

Do not implement this in v1.

### 3. Atlas-Region Join

This is the long-term target.

```text
animal_id + hemisphere + allen_region + compartment
```

MRI contributes:

- regional lesion burden
- core/peri/contra compartments
- edema or hemorrhage/vessel covariates

IHC contributes:

- regional marker metrics
- cell-type metrics once cell detection exists
- section/registration QC

This avoids fragile direct MRI-to-IHC image registration.

## Future Region-First Architecture

The long-term IHC data model should be:

```text
animal -> panel -> section -> reviewed ROI -> hemisphere/region/compartment -> marker metrics
```

Later ROI sources:

- whole tissue
- ipsi/contra hemisphere
- atlas region
- MRI-derived core
- MRI-derived peri rings
- mirrored contralateral ROIs
- validated histology-derived lesion proxy when MRI is absent

Rules for future ROI work:

- Do not define a biological measurement region using the same marker that is
  later interpreted in that region.
- Panel A NeuroTrace can be explored as a histology lesion anchor, but not as a
  v1 dependency.
- Panel B lesion/core ROI transfer from Panel A requires confirmed section
  matching or atlas-mediated registration.
- Cross-panel colocalization requires confirmed section matching or atlas-level
  aggregation.

## Output Tables

v1 minimal table:

```text
animal_id, timepoint, panel, modality, measure, value, unit,
area_mm2, n_cells, edited, edit_dice, reviewer, qc_flag, model_version
```

Later ROI-level table:

```text
animal_id, group, dose_mgkg, timepoint, panel, section_id,
roi_id, roi_type, side, distance_bin_um, roi_source,
marker, metric, value, unit, qc_status
```

Later object-level table:

```text
animal_id, panel, section_id, object_id, object_type,
centroid_x_um, centroid_y_um, roi_id, atlas_region_optional,
area_um2, mean_intensity, shape_features,
distance_to_core_um, distance_to_vessel_um
```

Later animal-level summary table:

```text
animal_id, group, dose_mgkg, timepoint,
mri_t2w_lesion_volume_mm3,
ihc_igg_fitc_panel_a_summary,
ihc_igg_fitc_panel_b_summary,
qc_status
```

## Milestone Summary

1. v1: one animal, reviewed MRI lesion volume, selected-section IgG-FITC area.
2. v1 cleanup: section QC, threshold sign-off, deterministic export.
3. v2: atlas registration and region labels.
4. v3: core/peri/contra compartments and atlas-mediated MRI/IHC join.
5. v4: cell/object-level IHC and marker-specific models.
6. v5: batch processing and QC reporting.

The roadmap is intentionally staged so MRI model finetuning can proceed on its
own branch without blocking IHC v1.
