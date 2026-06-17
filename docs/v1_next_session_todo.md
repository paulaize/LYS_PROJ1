# v1 next-session TODO

Last audited: 2026-06-17

This is the current execution checklist for the next programming session. It
overrides the older "Immediate next actions" at the end of
`docs/development_roadmap.md`. Keep v1 limited to one animal, no atlas, no DL,
no compartments, and no batch processing.

## Zero-context orientation

- Current v1 animal: `BD_08_5D` (`5d`, stroke), configured in
  `config/animals/BD_08_5D.yml`.
- Current priority: finish the first end-to-end implementation, not improve the
  MRI segmentation backend.
- MRI truth policy: `lesion_draft.nii.gz` is an untrusted automatic seed. The
  reviewed `lesion_corrected.nii.gz` is the source of truth for v1 lesion
  volume and Swanson/indirect edema correction.
- The current threshold draft can target bright peripheral MRI artifact instead
  of the true image-right isocortical lesion. That is not a reason to add
  atlas, DL, or an isocortex ROI heuristic during v1. If the draft is bad,
  erase/redraw it by hand in the mask editor and keep the correction metadata.
- IHC direction: the script performs the analysis. The human gate is review/sign
  off of `tissue_v1` and threshold/calibration provenance, not hand analysis
  done before scripting.
- IHC threshold reality: no approved IgG-FITC positivity threshold exists yet,
  and the team has not previously analyzed these images in QuPath. v1 must
  create the first calibration/exploration workflow; do not expect Paul to
  provide a pre-existing QuPath threshold.
- IHC control reality: control animal `C6S5` has Panel A and Panel B `.vsi`
  files under `data/C6S5/IHC/`. It has no MRI in v1 and is configured as an
  IHC threshold/background control.

## Current status

- [x] Existing `lys-bbb` environment is usable on macOS arm64.
- [x] Required and optional Python imports pass.
- [x] QuPath 0.7.0, brkraw, and napari command-line checks pass.
- [x] `make test` passes: 34 tests.
- [x] MRI and IHC-ingest unit tests use synthetic data and pass.
- [x] `make lint` passes.
- [x] One real v1 animal is configured: `BD_08_5D`.
- [x] Bruker Scan 2 / reco 1 T2 RARE was converted to
  `work/BD_08_5D/mri/t2_scan2.nii.gz`.
- [x] Converted T2 header spacing validates as `0.07 x 0.07 x 0.5 mm`.
- [x] v1 MRI lesion side is explicit in config as `lesion_side: image_right`.
- [x] Napari MRI mask editor display now uses 18-slice `(slice, y, x)` order
  while saving masks back to the original NIfTI `(x, y, slice)` layout.
- [x] First no-editor MRI technical run wrote draft/corrected masks and
  `outputs/BD_08_5D/BD_08_5D_v1.csv`; QC is correctly flagged
  `needs_human_review`.
- [x] Current MRI threshold draft is known to be anatomically unreliable on
  `BD_08_5D` because it can select bright image-right peripheral signal. Treat
  it as a disposable seed; the human-corrected mask is the v1 output.
- [x] IHC QuPath/Bio-Formats diagnostics work for one Panel A target section.
- [x] First exploratory Panel A threshold sweep completed for
  `BD_08_5D` section_01 and matching `C6S5` control section_01 at
  `ihc.threshold_calibration.downsample=32.0`. Output:
  `work/BD_08_5D/ihc_threshold_sweep_panel_A.csv`.
- [ ] Git working tree is reviewed and clean. Recheck before editing; the
  current setup changes are intentionally uncommitted.
- [ ] One real v1 animal has completed the full reviewed MRI + IHC path.

## Immediate handoff for the next session

Do not start by running the full panel. The next IHC goal is to turn the
exploratory threshold machinery into a reviewable calibration workflow, then
scale only to selected good sections.

1. Inspect the first smoke-test CSV:
   `work/BD_08_5D/ihc_threshold_sweep_panel_A.csv`. It contains Panel A
   section_01 target/control threshold percentages and is explicitly not final.
2. Add a visual threshold-review artifact, preferably exported overlay or PNG
   thumbnails for the candidate thresholds, so Paul can approve/reject a
   threshold using images rather than CSV values alone.
3. Run the same one-section smoke test for Panel B target and C6S5 control.
   Use `ihc-diagnose` first, then `ihc-threshold-sweeps`; keep
   `--section section_01 --limit 1` until it succeeds.
4. Add section QC/selection: record the best sections in
   `selected_section_ids` or bad sections in `excluded_section_ids` before any
   multi-section export.
5. Add the threshold sign-off record: panel/batch scope, threshold value,
   reviewer, date, target/control images used, and `approved` vs exploratory.
6. Only after those gates, run deterministic IHC exports for the selected
   sections and join them with the reviewed MRI row into the v1 CSV.

## Facts Paul must provide

Do not guess these values in code or config.

- [x] Choose the one v1 test animal: `BD_08_5D` (`5d`, stroke), with MRI and
  Panel A/B IHC data present.
- [x] Record the T2 RARE Scan 2 NIfTI path and confirm the brkraw output name:
  `work/BD_08_5D/mri/t2_scan2.nii.gz`.
- [x] Record the available top-level `.vsi` paths for Panel A and Panel B in
  `config/animals/BD_08_5D.yml`.
- [x] Confirm how QuPath/Bio-Formats exposes the 8 sections/series inside each
  `.vsi`: each file has 10 images/series: label, whole-slide overview, then 8
  section series. v1 should process section series only, with later manual
  selection/QC for tears and folds.
- [x] Confirm the exact zero-based channel map and IgG-FITC channel index for
  each panel in QuPath/Bio-Formats.
- [x] Identify the NeuroTrace product/emission for v1: accepted as
  `NeuroTrace 640/660 Deep-Red Fluorescent Nissl Stain` (Paul, 2026-06-17).
  Panel A FITC spectral bleed-through is not flagged for v1. Paul may still do
  a later full fluorochrome audit.
- [x] Resolve what the anti-IgG-FITC reagent binds and whether it detects
  endogenous mouse IgG: confirmed anti-human IgG secondary, specific to
  humanized LYS241, not endogenous mouse IgG (Paul, 2026-06-17).
- [ ] Build, tune, and approve the first IgG-FITC positive threshold for each
  panel or acquisition batch using negative and positive controls. No prior
  QuPath threshold exists for these images. Specificity and positivity
  threshold are orthogonal; anti-human specificity does not unblock
  thresholding.
- [x] Record the v1 lesion side convention: lesion is expected on image-right
  in the MRI viewer. For the current Bruker T2 NIfTI orientation this maps to
  the higher-index half of array axis 0 as displayed in Fiji and is encoded as
  `image_right`.
- [ ] Draw/review an authoritative MRI lesion mask. Tuning MRI threshold `k` is
  optional/later; `k: 2.5` is only a provisional technical seed and is not a v1
  scientific gate if the corrected human mask is used.
- [x] Confirm napari as the initial v1 MRI mask editor. Keep the interface open
  to later switch to ITK-SNAP / 3D Slicer, and keep 3D Slicer + MONAI Label in
  mind for a later improved annotation workflow.
- [x] Confirm napari editing workflow: the draft lesion is a labels layer edited
  by paint/erase, not a movable shape. Closing napari saves the corrected mask.

## Next programming session

Complete these in order.

### P0: remove guessed configuration

- [x] Restore unresolved channel maps and channel indexes in
  `config/animals/TEMPLATE.yml` to `null`.
- [x] Remove the unconfirmed global `default_igg_fitc_channel_index` from
  `config/pipeline.yml`, or keep it `null` and fail helpfully when unset.
- [x] Keep global IgG-FITC threshold `null` until it has been approved.
- [x] Add validation tests proving unresolved channel indexes and thresholds
  cannot silently run.

### P0: make the v1 IHC export scientifically usable

The current exporter now requires `tissue_v1` and measures inside that
annotation. The generated `tissue_v1` is rough and must be reviewed/corrected
in QuPath before final measurements are trusted.

- [x] Confirm whether the configured Panel A/B `.vsi` files contain all
  section series and decide the exact QuPath export iteration unit.
- [x] Standardize the v1 tissue ROI name. The script should perform the tissue
  analysis/export; `tissue_v1` is the review/sign-off gate for the ROI used by
  the script, not hand analysis performed before scripting.
  Annotation name: `tissue_v1`.
- [x] Update `detect_cells.groovy` to create a rough signal-derived
  `tissue_v1` annotation that Paul can correct/review in QuPath.
- [x] Update `export_measurements.groovy` to require and measure only the
  reviewed tissue annotation, excluding off-tissue background.
- [x] Calculate `total_area_um2` from the measured tissue ROI, not the full
  image rectangle.
- [ ] Fail helpfully when the tissue annotation, pixel calibration, confirmed
  channel index, or approved final threshold is missing. Exploratory threshold
  runs are allowed only when clearly marked as calibration/not final.
- [x] Decide duplicate-row behavior: overwrite combined CSV for v1.
- [ ] Prevent duplicate rows when a section is re-exported, or make overwrite
  behavior explicit and deterministic.
- [ ] Record the image/section identifier, panel, channel index, threshold,
  downsample, and QC status in the QuPath export.
- [ ] Test the revised exporter on one Panel A and one Panel B section before
  processing all sections.
- [ ] Add section-selection review: process only the best few sections for v1,
  with `selected_section_ids` / `excluded_section_ids` recorded in config.
  If multiple good sections are selected, quantify each and aggregate later;
  do not silently choose one section without QC provenance.

### P0: connect config, QuPath, and final provenance

- [ ] Add a small QuPath orchestration command/script that reads one animal
  YAML and runs all configured Panel A and Panel B sections with their confirmed
  indexes and thresholds. It must skip label/overview series and iterate only
  section series `2..9` for `BD_08_5D`.
- [ ] Write one deterministic combined IHC CSV under `work/<animal_id>/`.
- [x] Carry resolved anti-IgG specificity facts into v1 output provenance:
  `anti_igg_specificity_resolved=true`, `fitc_specific_to_lys241=true`,
  `fitc_igg_specificity=anti_human_confirmed`, and source
  `confirmed_anti_human_IgG (Paul, 2026-06-17)`.
- [x] Carry panel-specific `fitc_spectral_bleedthrough` into the v1 output
  provenance.
- [ ] Ensure unresolved spectral bleed-through and thresholds produce clear QC
  flags rather than being treated as resolved. Do not add a QC warning for the
  now-resolved anti-human specificity fact.
- [x] Add tests for the new provenance and config-validation behavior.
- [x] Add `make calibrate-ihc` / `src/ihc/calibrate.py` as a fail-loud
  placeholder that documents specificity vs threshold as orthogonal.
- [ ] Add threshold-calibration workflow or helper using vehicle/control images;
  do not hardcode a provisional analysis threshold as final data. This helper
  is how v1 should create the first threshold candidates because there is no
  prior QuPath threshold to reuse. A no-LYS241 / vehicle / secondary-only
  control should provide clean background for background mean + k*SD
  calibration because the secondary is anti-human, but control paths must still
  come from config.
- [x] Configure control animal `C6S5` for IHC threshold calibration. MRI is not
  required for this control in v1.
- [x] Add `export_threshold_sweep.groovy` so the first review output can be a
  CSV sweep of candidate IgG-FITC thresholds inside `tissue_v1`.
- [x] Add `make ihc-threshold-sweeps`, a Python/QuPath runner that can loop
  BD_08_5D and C6S5 across configured section series for exploratory threshold
  sweeps. Default is dry-run. First real execution should be one section only:
  `RUN_ARGS="--panel A --section section_01 --target-only --limit 1 --run"`.
- [x] Add `make ihc-diagnose`, a metadata-only QuPath/Bio-Formats opener for
  one configured section. Run this before a real sweep to separate slow `.vsi`
  series opening from slow pixel reads.
- [ ] Resolve the current direct-CLI QuPath performance gate. Paul observed
  that `RUN_ARGS="--panel A --run"` was too slow and had to be interrupted;
  one-section diagnostics should decide whether v1 should continue with direct
  `--image/--server` calls or switch to a QuPath project/cached-import path.
  Status: direct CLI is usable for a single section at
  `ihc.threshold_calibration.downsample=32.0`; Panel A section_01 target and
  C6S5 control sweeps completed and wrote
  `work/BD_08_5D/ihc_threshold_sweep_panel_A.csv`. Whole-panel execution is
  still intentionally not validated and should wait until selected/best
  sections are chosen.
- [ ] Decide how to review threshold-sweep outputs visually. The current CSV
  gives target/control percentages by threshold, but final v1 threshold approval
  still needs either QuPath overlays, exported thumbnails, or another clear
  review artifact attached to the approval record.
- [x] Replace the fail-loud calibration placeholder with
  `ihc_threshold_calibration_manifest.csv`, an exploratory manifest listing
  panels, target/control files, candidate thresholds, and review status.
- [ ] Add a review/sign-off record for the chosen IHC threshold: reviewer,
  date, control/source images, threshold value, panel/batch scope, and whether
  the result is exploratory or approved.

### P1: validate the MRI path on the real test animal

- [x] Create `config/animals/BD_08_5D.yml` from the cleaned template.
- [x] Add a configured Bruker-to-NIfTI conversion command:
  `make convert-mri CONFIG=config/animals/BD_08_5D.yml`.
- [x] Confirm header spacing is `0.07 x 0.07 x 0.5 mm`.
- [x] Configure segmentation to use the known image-right lesion side instead
  of inferring side from hyperintensity.
- [x] Fix napari editor display: NIfTI data is `(x, y, slice)`, but napari
  review uses `(slice, y, x)` so the user sees 18 non-sideways slices. The
  edited labels are transformed back before saving.
- [x] Run a first no-editor technical MRI pass:
  `make run CONFIG=config/animals/BD_08_5D.yml RUN_ARGS=--no-mask-editor`.
- [ ] Visually inspect the N4-corrected volume and generated brain mask.
- [ ] Review/edit `work/BD_08_5D/lesion_corrected.nii.gz` with the chosen mask
  editor; replace the technical fallback with a documented human-corrected
  mask.
- [ ] If the draft is badly wrong, clear/redraw it rather than trying to rescue
  the threshold seed. Preserve the corrected human mask.
- [ ] Confirm the CSV records the reviewed mask as `reviewed_edited` or
  `reviewed_no_changes`, never `needs_human_review`.
- [ ] Later, after v1 is running, optionally run candidate `k` values against
  the reference human mask. Do not let this block the first implementation.
- [ ] Confirm raw and Swanson-corrected lesion volumes are plausible. The
  current no-editor technical run produced raw `2.171 mm3` and
  Swanson-corrected `2.115 mm3`, but these are not final until mask review.
- [x] Confirm technical MRI rows carry reviewer, edit Dice, and QC status.
- [ ] Confirm reviewed final MRI rows carry reviewer, edit Dice, and QC status.

### P2: MRI model-preparation notes, not v1 blockers

- [ ] Collect corrected 3D lesion masks as future model reference data. One
  mask is enough to test the v1 pipeline mechanically.
- [ ] With `3-5` corrected stroke masks, test a pretrained mouse T2 lesion model
  or the current threshold backend honestly by Dice/edit burden.
- [ ] With `8-12` corrected stroke masks, consider a small transfer-learning or
  nnU-Net fine-tuning attempt if the pretrained model is close.
- [ ] With `15-25` corrected masks, fine-tuning becomes much more defensible.
  Training from scratch would need substantially more data and is not the
  current plan.

### P1: repository cleanup

- [x] Fix current Ruff errors and trailing whitespace without unrelated
  refactors.
- [x] Reconcile the README with the actual repository layout; `.codex/` is not
  required in this checkout.
- [ ] Add a tiny real NIfTI/header fixture only if it can be safely
  de-identified and kept small.
- [ ] Review existing uncommitted changes, then commit the intended setup and
  v1 fixes.

## Validation sequence

After every code change:

```bash
make test
```

`make test` disables external pytest plugin autoloading; direct pytest currently
tries to import napari's pytest plugin in this environment.

Before calling v1 complete:

```bash
make env-check
make test
make lint
make convert-mri CONFIG=config/animals/BD_08_5D.yml
make run CONFIG=config/animals/BD_08_5D.yml IHC=work/BD_08_5D/ihc.csv
```

Inspect manually:

- `work/BD_08_5D/lesion_draft.nii.gz`
- `work/BD_08_5D/lesion_corrected.nii.gz`
- revised tissue-only QuPath measurements
- `outputs/BD_08_5D/BD_08_5D_v1.csv`

MRI mask-review notes:

- Run `make run CONFIG=config/animals/BD_08_5D.yml` to open napari.
- Select the `lesion (edit me)` labels layer and use paint/erase; the mask is
  not a movable shape.
- Closing napari saves `work/BD_08_5D/lesion_corrected.nii.gz` and rewrites the
  CSV with `reviewed_edited` or `reviewed_no_changes`.
- `RUN_ARGS=--no-mask-editor` is for technical runs only and overwrites the
  corrected mask with a draft flagged `needs_human_review`.

## v1 completion gate

- [ ] No channel, threshold, path, specificity, or overlap fact was guessed.
- [ ] IHC thresholds were produced by the documented calibration workflow and
  explicitly approved; no pre-existing QuPath threshold was assumed.
- [ ] MRI header spacing and anisotropic volume calculation were verified.
- [ ] MRI lesion mask received documented human review/correction, and the
  unreviewed draft was not used as final data.
- [ ] IHC positive area uses reviewed tissue area as its denominator.
- [ ] Re-running IHC export does not silently duplicate measurements.
- [ ] Interpretation flags and QC provenance survive into deliverables.
- [ ] One configured animal produces a reviewed joined CSV.
- [ ] `make test` and `make lint` pass.
