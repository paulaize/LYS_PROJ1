# MRI + IHC Full-Pipeline Design

This is the compact design reference for the eventual full workflow. It is not
the active v1 checklist. For current work, use
[v1_next_session_todo.md](v1_next_session_todo.md).

## Current v1 Override

v1 intentionally stops before this full design:

- one animal: `BD_08_5D`
- no atlas
- no deep learning
- no compartments
- no cell-level IHC
- no batch processing

v1 produces a reviewed MRI lesion volume, reviewed/approved IgG-FITC positive
area, and one minimal joined CSV.

## Study Shape

- cross-sectional mouse thrombin-stroke study
- timepoints: 1h, 3h, 6h, 24h, 48h, 5d
- MRI exists only for 24h, 48h, and 5d animals
- IHC exists across early and late animals
- each IHC animal has Panel A and Panel B `.vsi` slides

Early IHC-only animals must carry a different `compartment_method` later if
compartments are estimated from histology rather than MRI.

## Common-Space Rule

Quantification joins through Allen CCFv3:

```text
MRI -> Allen
IHC -> Allen
join by animal, hemisphere, Allen region, compartment
```

Do not register MRI directly to IHC for quantification. Direct MRI/IHC overlays
are figure-only.

## MRI Track

Inputs:

- Scan 1: localizer, discard
- Scan 2: RARE T2 high-resolution lesion-volume scan
- Scan 3: FcFLASH T2* optional hemorrhage readout
- Scan 4: TOF angio optional vessel context

Geometry:

```text
256 x 256, FOV 17.92 mm -> 0.07 mm in-plane
18 slices x 0.5 mm
anisotropic / 2.5-D
volume = voxel_count x prod(header_spacing)
```

v1 MRI flow:

```text
Bruker -> NIfTI -> header check -> N4 -> brain mask -> threshold draft
-> human review/correction -> raw volume -> Swanson/indirect correction
```

Later MRI additions:

- AIDAmri or ANTs/SyN for MRI->Allen
- atlas-based edema/Jacobian maps
- core/peri/contra masks in Allen space
- optional T2* hemorrhage covariate

## IHC Track

Inputs:

- Olympus `.vsi`, fluorescence, 4 channels
- Panel A: DAPI, NeuroTrace, Podocalyxin, IgG-FITC
- Panel B: DAPI, IBA1, GFAP, IgG-FITC
- channel order comes from YAML, never assumption

Current interpretation:

- readout name remains IgG-FITC
- anti-human IgG specificity to humanized LYS241 is resolved
- Panel A NeuroTrace 640/660 deep-red is accepted for v1
- positivity threshold is not resolved and must be calibrated/reviewed

v1 IHC flow:

```text
.vsi -> QuPath/Bio-Formats -> reviewed tissue_v1 ROI
-> exploratory target/control threshold candidates
-> threshold sign-off
-> IgG-FITC positive area export
-> Python ingest
```

Later IHC additions:

- QuPath project/cached import if direct CLI `.vsi` opens are too slow
- ABBA/DeepSlice section->Allen registration
- StarDist/InstanSeg cell/nucleus detection
- QuPath object classifier or transparent marker thresholds
- area-level GFAP/IBA1 and IgG-FITC readouts
- cell-level IgG-FITC by cell type where segmentation is defensible

## Compartments

Later compartment definitions:

- **core:** corrected MRI lesion mask
- **peri:** physical dilation ring minus core
- **contra:** mirror of core/peri across atlas midline

Compartments are computed in MRI/Allen space and transferred to IHC through the
shared atlas. For IHC-only early animals, use a separately flagged histology
proxy if needed; do not pool it silently with MRI-derived compartments.

## Output Table

Final long-table shape:

```text
animal_id
timepoint
panel
modality
hemisphere
allen_region
compartment
compartment_method
cell_type
measure
value
unit
area_mm2
n_cells
edited
edit_dice
reviewer
qc_flag
model_version
```

v1 emits only the subset that is available before atlas/compartments/cell-level
analysis.

## Automation Boundaries

Automate:

- Bruker conversion
- NIfTI IO/header checks
- N4 and draft mask generation
- volume math
- QuPath command orchestration
- CSV ingest and joins

Human gates:

- MRI lesion mask correction
- IHC `tissue_v1` review
- threshold approval
- later ABBA alignment review
- later cell detection/classifier QC

## Build Order

1. v1 thin spine: reviewed MRI + reviewed/approved IHC positive area.
2. Atlas labels: MRI->Allen and IHC->Allen.
3. Compartments and full join.
4. Better draft models and cell-level IHC.
5. Batch processing and QC index.

The v1 completion gate in [v1_next_session_todo.md](v1_next_session_todo.md)
must pass before atlas or batch work becomes useful.

## Open Parameters

- IgG-FITC threshold approval per panel/acquisition batch
- perilesional ring width in mm
- MRI->Allen path: AIDAmri wholesale vs leaner ANTs/SyN
- final 3-D mask editor standard: napari, ITK-SNAP, or 3D Slicer
