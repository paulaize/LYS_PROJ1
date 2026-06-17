# v1 next-session TODO

Last audited: 2026-06-17

This is the current execution checklist for the next programming session. It
overrides the older "Immediate next actions" at the end of
`docs/development_roadmap.md`. Keep v1 limited to one animal, no atlas, no DL,
no compartments, and no batch processing.

## Current status

- [x] Existing `lys-bbb` environment is usable on macOS arm64.
- [x] Required and optional Python imports pass.
- [x] QuPath 0.7.0, brkraw, and napari command-line checks pass.
- [x] `make test` passes: 19 tests.
- [x] MRI and IHC-ingest unit tests use synthetic data and pass.
- [x] `make lint` passes.
- [x] One real v1 animal is configured: `BD_08_5D`.
- [x] Bruker Scan 2 / reco 1 T2 RARE was converted to
  `work/BD_08_5D/mri/t2_scan2.nii.gz`.
- [x] Converted T2 header spacing validates as `0.07 x 0.07 x 0.5 mm`.
- [x] First no-editor MRI technical run wrote draft/corrected masks and
  `outputs/BD_08_5D/BD_08_5D_v1.csv`; QC is correctly flagged
  `needs_human_review`.
- [ ] Git working tree is reviewed and clean. Recheck before editing; the
  current setup changes are intentionally uncommitted.
- [ ] One real v1 animal has completed the full reviewed MRI + IHC path.

## Facts Paul must provide

Do not guess these values in code or config.

- [x] Choose the one v1 test animal: `BD_08_5D` (`5d`, stroke), with MRI and
  Panel A/B IHC data present.
- [x] Record the T2 RARE Scan 2 NIfTI path and confirm the brkraw output name:
  `work/BD_08_5D/mri/t2_scan2.nii.gz`.
- [x] Record the available top-level `.vsi` paths for Panel A and Panel B in
  `config/animals/BD_08_5D.yml`.
- [ ] Confirm how QuPath/Bio-Formats exposes the 8 sections/series inside each
  `.vsi`, and whether the v1 export should run once per top-level file or once
  per series/section.
- [ ] Confirm the exact zero-based channel map and IgG-FITC channel index for
  each panel in QuPath/Bio-Formats.
- [ ] Identify the NeuroTrace product/catalog number and emission. Record
  whether a NeuroTrace-only control produces signal in the FITC channel.
- [ ] Resolve what the anti-IgG-FITC reagent binds and whether it detects
  endogenous mouse IgG. Record the result in the interpretation flags.
- [ ] Tune and approve the IgG-FITC positive threshold for each panel or
  acquisition batch using negative and positive controls.
- [ ] Draw/review a reference MRI lesion mask and tune MRI threshold `k`.
  `k: 2.5` may be used only as a provisional first technical run.
- [ ] Confirm napari as the standardized v1 MRI mask editor, or select an
  alternative.

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

The current exporter is preliminary: it counts the full rectangular image,
including off-tissue background. The annotation created by
`detect_cells.groovy` is not used by the exporter.

- [ ] Confirm whether the configured Panel A/B `.vsi` files contain all
  section series and decide the exact QuPath export iteration unit.
- [ ] Standardize a manually reviewed whole-tissue annotation for v1. Do not
  invent an automatic tissue threshold.
- [ ] Update `export_measurements.groovy` to require and measure only the
  reviewed tissue annotation, excluding off-tissue background.
- [ ] Calculate `total_area_um2` from the measured tissue ROI, not the full
  image rectangle.
- [ ] Fail helpfully when the tissue annotation, pixel calibration, confirmed
  channel index, or approved threshold is missing.
- [ ] Prevent duplicate rows when a section is re-exported, or make overwrite
  behavior explicit and deterministic.
- [ ] Record the image/section identifier, panel, channel index, threshold,
  downsample, and QC status in the QuPath export.
- [ ] Test the revised exporter on one Panel A and one Panel B section before
  processing all sections.

### P0: connect config, QuPath, and final provenance

- [ ] Add a small QuPath orchestration command/script that reads one animal
  YAML and runs all configured Panel A and Panel B sections with their confirmed
  indexes and thresholds.
- [ ] Write one deterministic combined IHC CSV under `work/<animal_id>/`.
- [ ] Carry `anti_igg_specificity_resolved`, `fitc_specific_to_lys241`, and
  `fitc_overlap_risk` into the final joined output or an attached provenance
  table.
- [ ] Ensure unresolved specificity/overlap produces a clear QC flag rather
  than being treated as resolved.
- [ ] Add tests for the new provenance and config-validation behavior.

### P1: validate the MRI path on the real test animal

- [x] Create `config/animals/BD_08_5D.yml` from the cleaned template.
- [x] Add a configured Bruker-to-NIfTI conversion command:
  `make convert-mri CONFIG=config/animals/BD_08_5D.yml`.
- [x] Confirm header spacing is `0.07 x 0.07 x 0.5 mm`.
- [x] Run a first no-editor technical MRI pass:
  `make run CONFIG=config/animals/BD_08_5D.yml RUN_ARGS=--no-mask-editor`.
- [ ] Visually inspect the N4-corrected volume and generated brain mask.
- [ ] Review/edit `work/BD_08_5D/lesion_corrected.nii.gz` with the chosen mask
  editor; replace the technical fallback with a documented human-corrected
  mask.
- [ ] Run candidate `k` values against the reference human lesion mask.
- [ ] Select and record the approved `k`; preserve the corrected human mask.
- [ ] Confirm raw and Swanson-corrected lesion volumes are plausible. The
  technical run produced raw `6.468 mm3` and Swanson-corrected `6.637 mm3`, but
  these are not final until mask review.
- [x] Confirm technical MRI rows carry reviewer, edit Dice, and QC status.
- [ ] Confirm reviewed final MRI rows carry reviewer, edit Dice, and QC status.

### P1: repository cleanup

- [x] Fix current Ruff errors and trailing whitespace without unrelated
  refactors.
- [ ] Reconcile the README with the actual repository layout; `.codex/` is
  currently described but was absent at the latest audit.
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

## v1 completion gate

- [ ] No channel, threshold, path, specificity, or overlap fact was guessed.
- [ ] MRI header spacing and anisotropic volume calculation were verified.
- [ ] MRI lesion mask received documented human review.
- [ ] IHC positive area uses reviewed tissue area as its denominator.
- [ ] Re-running IHC export does not silently duplicate measurements.
- [ ] Interpretation flags and QC provenance survive into deliverables.
- [ ] One configured animal produces a reviewed joined CSV.
- [ ] `make test` and `make lint` pass.
