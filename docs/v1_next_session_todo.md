# v1 Next-Session TODO

Last cleaned: 2026-07-08

This is the short execution checklist for the next development session. The
IHC strategy lives in `docs/IHC_v1_plan.md`. The full project roadmap lives in
`docs/development_roadmap.md`.

Keep v1 limited to one animal, no atlas, no deep learning, no compartments, and
no batch processing.

## Current State

Branch:

- Current branch: `ihc`.
- MRI lesion-model finetuning is being developed separately on
  `dl-ratlesnetv2-finetune`.
- Keep IHC edits isolated from `ratlesnetv2_finetune/`, `src/mri/` model code,
  and cloud-training artifacts unless Paul explicitly asks otherwise.

Target data:

- v1 animal: `BD_08_5D`.
- IHC control: `C6S5`, role `no_lys241_control`.
- Panels: A and B.
- Each configured `.vsi` has label, overview, then 8 section series
  (`section_01..section_08`, Bio-Formats series `2..9`).

Implemented:

- `make ihc-diagnose`
- `make ihc-section-qc`
- `make calibrate-ihc`
- `make ihc-threshold-sweeps`
- `make ihc-threshold-review`
- QuPath exporters for threshold sweep, threshold review thumbnails, tissue ROI
  draft, and final tissue-only measurement export.
- Python ingestion of QuPath CSVs.

Generated:

- Panel A `section_01` target/control threshold sweep CSV.
- Panel A `section_01` target/control review PNGs under
  `work/BD_08_5D/ihc_threshold_review/panel_A/`.
- Section-QC manifest:
  `work/BD_08_5D/ihc_section_qc_manifest.csv`.
- Section-QC thumbnails:
  `work/BD_08_5D/ihc_section_qc/`.
- The manifest has 32 rows: target/control x Panel A/B x 8 configured
  sections. There are 25 thumbnails because C6S5 Panel B sections 02-08 fail
  before thumbnail export.
- Usable sections are now recorded per panel in config:
  - BD_08_5D Panel A: `section_01`, `section_03`, `section_06`
  - C6S5 Panel A: `section_05`, `section_06`, `section_07`
  - BD_08_5D Panel B: `section_05`, `section_06`, `section_08`
  - C6S5 Panel B: unavailable/corrupted, excluded from threshold calibration

Important caveat:

- The C6S5 Panel A `section_01` control image is visibly damaged. Treat those
  outputs as smoke-test artifacts only, not calibration evidence.
- Section QC exposed likely series/config problems:
  - BD_08_5D Panel A `section_08` opens as a macro image with invalid pixel
    calibration and only 3 channels instead of the configured 4.
  - BD_08_5D Panel B `section_04` and `section_07` opened with only one
    readable channel instead of the configured 4. Paul confirmed these images
    were not saved properly, so they are excluded.
  - C6S5 Panel B `section_01` produced a macro-sized 600 x 207, 3-channel
    thumbnail.
  - C6S5 Panel B `section_02..section_08` failed to open through the configured
    Bio-Formats series indices. Paul confirmed the Panel B control file is
    corrupted/unavailable for now.

## Confirmed Rules

- Do not assume Panel A and Panel B sections are matched.
- Do not do cross-panel colocalization in v1.
- Do not estimate whole-lesion histology volume: section thickness is `10 um`,
  but inter-section spacing/z-position is unknown.
- Treat IHC image-right ipsilateral for `BD_08_5D` as pending visual
  orientation QC until overlays label ipsi, contra, and midline.
- Do not use unreviewed `tissue_v1` as final denominator.
- Do not approve an IgG-FITC threshold from CSV values alone.
- Keep the measured marker name `IgG-FITC`; LYS241 interpretation lives in
  provenance.
- C6S5 is a no-LYS241/background control, not a full spectral-unmixing control.
- QuPath handles `.vsi` IO/review/export; Python handles orchestration,
  validation, provenance, and joins.

## Next Work, In Order

### 1. Keep Section QC As Provenance

Section QC has been reviewed enough to select v1 working sections. Keep the
manifest and thumbnails as provenance. Regenerate only if the section mapping or
source files change.

Output one low-resolution review thumbnail and one manifest row per section:

```text
animal_id, source_role, panel, section_id, series_index,
channels, expected_channels, thumbnail_path,
section_qc_status, section_qc_notes,
selected_for_threshold_calibration, selected_for_final_export, qc_flag
```

Suggested command shape:

```bash
make ihc-section-qc CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --run --keep-going"
```

Output paths:

```text
work/BD_08_5D/ihc_section_qc_manifest.csv
work/BD_08_5D/ihc_section_qc/panel_<panel>/<animal>_<role>/<section>/section_qc_composite.png
```

Manual review should identify folds, holes, tears, missing tissue, saturation,
bad focus, severe background, macro/overview images, failed series, and wrong
channel counts. The current selected/excluded sections are recorded under each
panel's `section_selection` block.
If a configured series cannot be opened, the command should record
`section_qc_status=failed_qupath_open`; treat that section as excluded until
the series layout is corrected.

### 2. Accept Panel B Control Unavailable For v1

For now, do not block development on C6S5 Panel B. It is marked
`control_status: corrupted_unavailable` and
`exclude_from_threshold_calibration: true`.

Panel B target processing may continue, but Panel B IgG-FITC positive-area
outputs must be flagged as exploratory/no Panel B control until a valid control
or approved fallback threshold rule exists.

### 3. Run Panel A Calibration On Selected Sections

Panel A is now the clean control-calibrated path:

```bash
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --run"
make ihc-threshold-review CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --run"
```

Dry-run check already schedules:

```text
BD_08_5D Panel A: section_01, section_03, section_06
C6S5 Panel A: section_05, section_06, section_07
```

### 4. Add Artifact Exclusion Annotations

The Panel A threshold review showed that folds can be marked as IgG-FITC
positive. Do not solve this only by raising the threshold. In QuPath, add
annotations named `artifact_exclude` for folds, tears, holes, saturated edges,
and debris that should not contribute to measurement.

The sweep, review-thumbnail, and final measurement exporters now subtract
`artifact_exclude` from `tissue_v1` and record:

```text
artifact_annotation_count, tissue_area_um2, artifact_excluded_area_um2
```

If no artifact annotations exist, rows are flagged
`no_artifact_exclusion_annotations`.

The current direct `.vsi` commands do not load saved QuPath annotations. For
reviewed `tissue_v1` and `artifact_exclude`, create/use a QuPath project and
run scripts with:

```bash
QuPath script --project "<project.qpproj>" --image "<project image name>" ...
```

After adding artifact annotations, rerun review/sweeps for the affected
sections before threshold sign-off.

### 5. Build Manual Threshold Dashboard

Generate the manual review dashboard:

```bash
make ihc-threshold-dashboard CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A"
```

It writes:

```text
work/BD_08_5D/ihc_manual_review/panel_A_threshold_review.html
work/BD_08_5D/ihc_manual_review/panel_A_threshold_signoff_template.json
```

The dashboard shows all section-QC thumbnails, pre-checks the configured
selected sections, summarizes the target/control threshold table, and only
prompts for thresholds that pass simple plausibility criteria:

```text
control mean <= 1%
target mean >= 0.05%
target/control fold >= 5
```

For the current direct-image Panel A sweep, this proposes `250` and `500`.
Use the page to download a manual decision JSON after reviewing thumbnails.

### 6. Run Panel B Target-Only Exploratory Sweeps

Panel B has no usable configured control for now:

```bash
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel B --run"
make ihc-threshold-review CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel B --run"
```

Dry-run check already schedules only:

```text
BD_08_5D Panel B: section_05, section_06, section_08
```

These outputs are useful for development and review, but not final
control-calibrated Panel B IgG-FITC claims.

### 7. Add Threshold Sign-Off

After the dashboard decision JSON is downloaded, store it as:

```text
work/BD_08_5D/ihc_manual_review/panel_A_threshold_review_decision.json
```

Then validate and approve it:

```bash
make ihc-threshold-signoff CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --decision work/BD_08_5D/ihc_manual_review/panel_A_threshold_review_decision.json"
```

The sign-off command refuses:

- unapproved decisions,
- thresholds outside dashboard candidates,
- thresholds no longer plausible under the current sweep CSV,
- animal/panel mismatches,
- missing reviewer/date/notes,
- missing target or control sections.

It writes:

```text
work/BD_08_5D/ihc_threshold_signoff_panel_A.json
work/BD_08_5D/ihc_threshold_signoff_panel_A.csv
```

Required fields:

```text
animal_id, panel, threshold_scope, igg_fitc_threshold,
reviewer, review_date, approved,
target_sections_used, control_sections_used, control_type, notes
```

Final exports must require an approved threshold, or else write rows clearly
flagged as exploratory/not final.

### 8. Finalize Tissue ROI Gate

Before final measurement export:

- `tissue_v1` exists.
- `tissue_v1` was reviewed/corrected in QuPath.
- `artifact_exclude` annotations have been added where folds/tears/debris
  would otherwise inflate IgG-FITC area.
- Pixel calibration is valid.
- IgG-FITC channel index is confirmed.
- Threshold is approved.

Failures should be helpful and explicit.

### 9. Make Final Export Deterministic

Initial deterministic export wiring is in place:

```bash
make ihc-quantify CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --threshold 250 --exploratory --tissue-mode auto_if_missing --run"
```

This writes a raw QuPath measurement CSV and a tidy Python CSV under
`work/BD_08_5D/`. Re-running overwrites by default unless `--append` is passed.
Exploratory direct-VSI rows are clearly flagged as not final.

For final exports, ensure re-exporting a section does not silently duplicate
rows. Either overwrite combined CSVs for v1 or deduplicate by:

```text
animal_id, panel, section_id, image, region, measure
```

Also ensure final exports carry:

```text
image, section_id, panel, channel_index, threshold, downsample,
tissue_qc_status, section_qc_status, threshold_status, qc_flag
```

### 10. Produce The First Final IHC CSV

Current immediate path:

- Run first-draft Panel A quantification with the threshold chosen from the
  dashboard.
- Inspect the tidy CSV and raw overlays/sections in parallel.
- Add/review `tissue_v1` and `artifact_exclude` in QuPath for selected sections.

After selected sections, reviewed `tissue_v1`, project annotations, and
threshold sign-off:

- Export one selected Panel A section.
- Export one selected Panel B section.
- Ingest both into Python.
- Write one deterministic combined IHC CSV under `work/BD_08_5D/`.

Remaining integration item: make the Python quantification runner address
QuPath project image names directly, so reviewed `tissue_v1`/`artifact_exclude`
annotations are loaded automatically instead of relying on direct `.vsi`
exploratory mode.

### 11. Join With MRI At Animal Level

For v1, join only at animal level:

```text
animal_id, timepoint, panel, modality, measure, value, unit,
area_mm2, n_cells, edited, edit_dice, reviewer, qc_flag, model_version
```

Do not perform section-to-MRI-slice joins or Panel A/B section joins until
section matching/z-position is confirmed.

## MRI Parallel Track

MRI work may continue on `dl-ratlesnetv2-finetune`, but v1 still requires a
reviewed human lesion mask before the joined CSV is considered scientific.

The automatic MRI lesion draft is not final output. The reviewed
`lesion_corrected.nii.gz` is the source of truth for v1 volume and edema
correction.

## Validation Commands

After documentation-only edits:

```bash
make test
make lint
```

After IHC code edits:

```bash
make test
make lint
make ihc-diagnose CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --target-only --limit 1"
```

Real QuPath runs may need unsandboxed execution on macOS.

## v1 Completion Gate

v1 is complete when:

- No channel, threshold, path, section-matching, spacing, or control fact was
  guessed.
- Usable IHC sections are selected with QC provenance.
- IgG-FITC threshold is calibrated from selected target/control sections and
  approved.
- `tissue_v1` is reviewed before final measurement.
- Final IHC export is deterministic and duplicate-safe.
- MRI lesion mask has documented human review.
- One command produces a reviewed animal-level CSV joining MRI and IHC.
- `make test` and `make lint` pass.
