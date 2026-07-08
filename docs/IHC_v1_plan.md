# IHC v1 Plan

This is the active plan for the IHC side of `LYS_PROJ1`. It is intentionally
practical: what is true now, what v1 should produce, and what is deferred.

For the exact next coding tasks, use `docs/v1_next_session_todo.md`.

## Goal

Build a reproducible IHC workflow for the current test animal that can:

1. Open configured Olympus `.vsi` files through QuPath/Bio-Formats.
2. Process only real tissue section series, not label or overview series.
3. Select sections that are usable for analysis.
4. Quantify IgG-FITC positive area inside reviewed tissue ROIs.
5. Calibrate the first IgG-FITC positivity threshold from configured controls.
6. Export tidy rows with enough provenance to join later with MRI outputs.

v1 is deliberately not atlas registration, cell detection, compartments, or
batch processing. Those are later milestones.

## Current Inputs

Target animal:

- `BD_08_5D`
- timepoint: `5d`
- group: stroke
- panels: A and B
- MRI: present

IHC control:

- `C6S5`
- role: `no_lys241_control`
- panels: A and B
- MRI: not needed for v1 threshold/background calibration

Configured section series:

- Bio-Formats exposes 10 series per `.vsi`: label, overview, then 8 section
  series.
- v1 processes only series `2..9`, mapped to `section_01..section_08`.
- Section thickness is recorded as `10 um`.
- The physical spacing or z-interval between sections is not recorded.
- Panel A/B section matching is recorded as unconfirmed.

Because z-spacing is unknown, IHC v1 must not report whole-lesion histology
volume from `sum(area) * 10 um`. It may report per-section area, mean section
area, sampled burden, or fraction of tissue/hemisphere affected once those ROIs
exist.

## Active Assumptions And Guardrails

Panel A and Panel B are not assumed to be section-matched.

Until section pairing is confirmed, do not transfer ROIs directly between Panel
A and Panel B and do not claim exact cross-panel microscopic colocalization.
Allowed v1 comparisons are within a panel and at animal-level summary:

- Panel A: IgG-FITC with NeuroTrace and Podocalyxin context.
- Panel B: IgG-FITC with IBA1 and GFAP context.
- Across panels: compare animal-level or later atlas-region summaries, not exact
  pixel/cell coordinates.

QuPath and Python have separate roles:

- QuPath/Groovy: `.vsi` reading, review surfaces, ROI/threshold overlays, and
  measurement export.
- Python: config loading, QuPath command orchestration, CSV ingestion,
  validation, provenance, joins, and final deliverables.

Quantification must use raw channel values. Display LUTs or RGB composites are
for review only.

The marker/readout name remains `IgG-FITC` / `igg_fitc_*`. Specificity is
resolved as anti-human IgG-FITC targeting humanized LYS241, but positivity
thresholding is still unresolved.

C6S5 is useful as a no-LYS241 negative FITC/background control. It is not enough
to claim full spectral correction or rigorous unmixing, because single-stain,
unstained, no-primary, and secondary-only controls are not configured.

For `BD_08_5D`, the working IHC side convention is recorded as image-right
ipsilateral and image-left contralateral, with `orientation_qc_status:
pending_visual_qc`. Future overlay exports should label ipsi, contra, and
midline before any ipsi/contra masks are trusted.

## Current Implementation State

Implemented:

- Configured target/control animals.
- Confirmed per-panel IgG-FITC channel index for current v1 files.
- `make ihc-diagnose`: metadata-only QuPath/Bio-Formats smoke test.
- `make ihc-section-qc`: low-resolution section QC thumbnails and manifest.
- `make calibrate-ihc`: exploratory threshold calibration manifest.
- `make ihc-threshold-sweeps`: CSV sweep of candidate IgG-FITC thresholds.
- `make ihc-threshold-review`: raw FITC and threshold-overlay PNGs.
- `make ihc-threshold-dashboard`: manual review/sign-off dashboard.
- `make ihc-threshold-signoff`: validates dashboard decisions and writes
  canonical threshold sign-off files.
- `make ihc-quantify`: runs selected-section IgG-FITC quantification and writes
  raw plus tidy CSVs.
- `export_measurements.groovy`: final measurement exporter requiring
  `tissue_v1` and supporting `artifact_exclude` subtraction.
- `ingest_qupath(csv_path)`: QuPath CSV to tidy Python rows.

Generated so far:

- Panel A `section_01` target/control threshold sweep CSV.
- Panel A `section_01` target/control threshold-review PNGs.
- Section-QC manifest and thumbnails for Panel A target/control.
- Section-QC manifest and thumbnails for Panel B target plus C6S5 Panel B
  `section_01`.
- Failure rows for C6S5 Panel B `section_02..section_08`, which do not open
  through the current direct Bio-Formats CLI series mapping.
- The section-QC manifest has one row for every expected target/control panel
  section combination: 32 rows total, with failed or suspect sections flagged.
- Section selections are now recorded per panel in YAML:
  - BD_08_5D Panel A: `section_01`, `section_03`, `section_06`
  - C6S5 Panel A: `section_05`, `section_06`, `section_07`
  - BD_08_5D Panel B: `section_05`, `section_06`, `section_08`
  - C6S5 Panel B: unavailable/corrupted, excluded from calibration

Important caveat:

- The tested C6S5 Panel A `section_01` is visibly damaged. It should be treated
  as a smoke test, not as a final calibration section.
- BD_08_5D Panel A configured `section_08` opens as a macro image with invalid
  pixel calibration and 3 channels instead of the configured 4. The target
  Panel A section-series mapping needs review.
- BD_08_5D Panel B configured `section_04` and `section_07` opened with one
  readable channel instead of the configured 4. Paul confirmed those images
  were not saved properly, so they are excluded.
- C6S5 Panel B `section_01` opens as a 600 x 207, 3-channel macro-sized image.
  C6S5 Panel B `section_02..section_08` do not open through direct CLI series
  selection. Paul confirmed the C6S5 Panel B file is corrupted/unavailable for
  now, so Panel B must proceed without this control and stay exploratory unless
  a replacement control or approved fallback threshold is added.

## Required v1 Workflow

### 1. Section QC

Before more threshold work, run section QC. The command exports one review
thumbnail per section for each target/control panel and writes a manifest with:

```bash
make ihc-section-qc CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --run --keep-going"
```

```text
animal_id
source_role
panel
section_id
series_index
channels
expected_channels
thumbnail_path
section_qc_status
section_qc_notes
selected_for_threshold_calibration
selected_for_final_export
qc_flag
```

Outputs:

- `work/BD_08_5D/ihc_section_qc_manifest.csv`
- `work/BD_08_5D/ihc_section_qc/panel_<panel>/<animal>_<role>/<section>/section_qc_composite.png`

Manual review should mark folds, holes, tears, missing tissue, severe
background, saturation, or focus problems. Record selections per panel:

If a configured Bio-Formats series cannot be opened, `--keep-going` records a
manifest row with `section_qc_status=failed_qupath_open` and continues. Treat
that section as excluded until the series mapping is corrected.

```yaml
selected_section_ids: ["section_03", "section_06"]
```

or:

```yaml
excluded_section_ids: ["section_01", "section_02"]
```

The threshold sweep, review-thumbnail, and diagnostic runners respect
panel-level `section_selection` when no explicit `--section` is passed.
`make ihc-section-qc` still surveys all configured series so QC can be
regenerated independently of the selected subset.

### 2. Threshold Calibration

Run threshold sweeps only on selected usable target/control sections.

Current command shapes:

```bash
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A"
make ihc-threshold-review CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A"
```

Panel A schedules six selected sections: three BD_08_5D target sections and
three C6S5 no-LYS241 control sections.

Panel B currently schedules only BD_08_5D target sections:

```bash
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel B"
make ihc-threshold-review CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel B"
```

Because C6S5 Panel B is corrupted/unavailable, Panel B IgG-FITC positive-area
outputs must be labeled `exploratory_no_panel_b_control` until a valid Panel B
control or explicit fallback calibration rule is approved.

Outputs:

- CSV percent-positive sweep by threshold.
- Raw FITC thumbnails.
- Threshold-overlay PNGs.
- Manual threshold dashboard HTML and sign-off template:
  - `work/<animal>/ihc_manual_review/panel_<panel>_threshold_review.html`
  - `work/<animal>/ihc_manual_review/panel_<panel>_threshold_signoff_template.json`
- A manifest tying each threshold candidate to source animal, panel, section,
  tissue ROI status, and QC status.

No threshold becomes final from a CSV alone. The reviewer must inspect the
images.

The manual dashboard is generated with:

```bash
make ihc-threshold-dashboard CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A"
```

It shows all section-QC thumbnails, pre-checks configured selected sections, and
only prompts for thresholds that pass simple plausibility criteria:

```text
control mean <= 1%
target mean >= 0.05%
target/control fold >= 5
```

The downloaded decision JSON becomes a formal sign-off with:

```bash
make ihc-threshold-signoff CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --decision work/BD_08_5D/ihc_manual_review/panel_A_threshold_review_decision.json"
```

This writes:

```text
work/BD_08_5D/ihc_threshold_signoff_panel_A.json
work/BD_08_5D/ihc_threshold_signoff_panel_A.csv
```

Fold artifacts can be semi-automated later, but v1 does not trust a fully
automatic fold detector for final measurements. If folds, tears, saturated
edges, or debris are annotated as `artifact_exclude`, the threshold sweep and
review-thumbnail exporters subtract them from the sampled tissue. Rows then
record:

```text
artifact_annotation_count
tissue_area_um2
artifact_excluded_area_um2
```

If no artifact annotations exist, rows are flagged
`no_artifact_exclusion_annotations`.

Important QuPath execution rule:

- Bare `--image <file.vsi>` runs are useful for exploratory direct-file
  processing, but they do not load saved project annotations.
- Reviewed `tissue_v1` and `artifact_exclude` annotations require running
  scripts against a QuPath project with `--project <project.qpproj>` and the
  corresponding project image name, or from the open QuPath project.

### 3. Threshold Sign-Off

Add a sign-off record before final export:

```text
animal_id
panel
threshold_scope
igg_fitc_threshold
reviewer
review_date
approved
target_sections_used
control_sections_used
control_type
notes
```

If a threshold is exploratory, final positive-area measurements must remain
blocked or clearly flagged as not final.

### 4. Tissue ROI Review

`tissue_v1` is the denominator for area measurements. The automatic tissue ROI
is only a draft.

For final v1 measurements:

- `tissue_v1` must be present.
- `tissue_v1` must be reviewed/corrected in QuPath.
- Folds, tears, holes, saturated edges, and debris that should not contribute
  to measurement must be annotated as `artifact_exclude`.
- The export must measure inside `tissue_v1 - artifact_exclude`, not the full
  image rectangle.
- The output row must carry tissue ROI QC/provenance.

### 5. Final IHC Export

For first-draft validation data, direct `.vsi` quantification is allowed only
when it is explicitly marked exploratory:

```bash
make ihc-quantify CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --threshold 250 --exploratory --tissue-mode auto_if_missing --run"
```

This writes:

```text
work/BD_08_5D/ihc_quantification_panel_A.csv
work/BD_08_5D/ihc_quantification_panel_A_tidy.csv
```

Rows from this route carry `threshold_exploratory_not_final` and
`rough_tissue_auto_unreviewed` QC flags.

For selected sections with reviewed `tissue_v1` and approved threshold, the
exporter must be run with project annotations available and the reviewed tissue
gate required. The Python runner can already enforce the sign-off and
`require_reviewed` mode, but project-image orchestration is still the next
small integration step; until then, direct `.vsi` runs will fail helpfully if
`tissue_v1` is not loaded:

```bash
make ihc-quantify CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --signoff work/BD_08_5D/ihc_threshold_signoff_panel_A.json --tissue-mode require_reviewed --run"
```

The underlying Groovy exporter accepts these arguments when run from a QuPath
project/open image context:

```bash
QuPath script --project "<project.qpproj>" \
  --image "<project image name>" \
  --args "panel,out_csv,igg_fitc_channel_index,approved_threshold,downsample,tissue_v1,section_id,approved,artifact_exclude,require_reviewed,animal_id,target" \
  src/ihc/qupath/export_measurements.groovy
```

The final combined IHC CSV should be deterministic. Re-running a section must
not silently duplicate rows; the runner overwrites raw/tidy CSVs by default
unless `--append` is explicitly passed.

### 6. Python Ingestion

Python ingests QuPath CSV rows and carries:

```text
animal_id
timepoint
panel
section_id
modality
measure
value
unit
area_mm2
n_cells
threshold
threshold_status
tissue_qc_status
section_qc_status
anti_igg_specificity_resolved
fitc_specific_to_lys241
fitc_igg_specificity
fitc_spectral_bleedthrough
qc_flag
model_version
```

## v1 Output

The v1 output is a minimal animal-level table, not the final atlas table.

Required IHC measures for v1:

- IgG-FITC positive area.
- Total valid reviewed tissue area after artifact exclusion.
- Original reviewed tissue area.
- Artifact-excluded area.
- IgG-FITC percent positive area.

Recommended additional v1 measures if cheap:

- Mean IgG-FITC intensity inside reviewed tissue.
- Median IgG-FITC intensity inside reviewed tissue.

Positive-area metrics are threshold-sensitive, so keeping continuous intensity
metrics gives a useful cross-check.

## MRI Join In v1

MRI is being developed on a separate branch. The IHC branch should avoid
modifying MRI model-training code.

The first safe join is animal-level:

```text
animal_id + timepoint
MRI: reviewed T2w lesion volume, raw and edema-corrected
IHC: selected-section IgG-FITC summary metrics per panel
```

Do not join Panel A and Panel B by exact section ID unless section matching is
confirmed. Do not join IHC sections to MRI slices unless a section-to-slice
mapping exists.

## Deferred To v2 Or Later

These are useful but not v1 tasks:

- Hemisphere masks and mirrored contra ROIs.
- Core/peri/contra compartments.
- Peri-lesion rings.
- Panel A NeuroTrace-derived histology lesion core.
- Panel B lesion ROI transfer from Panel A.
- Cross-panel colocalization.
- Atlas registration with ABBA/DeepSlice.
- Cell detection/classification.
- Podo vessel skeleton metrics.
- IBA1/GFAP morphology features.
- T2* hemorrhage and TOF vessel covariates.

Future direction:

- Region-first quantification is the right long-term architecture:
  `animal -> panel -> section -> reviewed ROI -> region/hemisphere/compartment
  -> marker metrics`.
- Panel A NeuroTrace can become a histology lesion anchor after validation.
- Panel B should use whole ipsi/contra or atlas-region summaries unless Panel
  A/B section matching is confirmed.
- Core/peri/contra should ultimately be generated from reviewed MRI masks and
  transferred through atlas space, not by direct MRI-to-IHC registration.

## Immediate Coding Queue

1. Review `work/BD_08_5D/ihc_section_qc_manifest.csv` and the section-QC
   thumbnails.
2. Fix or document the suspect series mappings exposed by section QC.
3. Select usable sections in config.
4. Re-run threshold sweeps/review thumbnails on selected Panel A sections.
5. Resolve Panel B control section availability before Panel B calibration.
6. Add threshold sign-off records.
7. Make final IHC export deterministic and duplicate-safe.
8. Export one selected Panel A and one selected Panel B section with approved
   thresholds.
9. Ingest final IHC rows and join with the reviewed MRI animal-level row.
