# v1 Next-Session TODO

Last audited: 2026-07-07

This is the active execution checklist. It overrides older roadmap notes when
they conflict.

v1 remains deliberately small:

- one animal: `BD_08_5D`
- no atlas
- no deep learning
- no compartments
- no batch processing

## Current Facts

- `BD_08_5D` is configured in `config/animals/BD_08_5D.yml`.
- `C6S5` is configured as the IHC threshold/background control.
- Bruker scan 2 / reco 1 is the T2 RARE lesion-volume scan.
- Converted T2 spacing validates as `0.07 x 0.07 x 0.5 mm`.
- Lesion side is explicit in config as `image_right`.
- The automatic MRI threshold mask is an untrusted draft. If it targets
  artifact, erase/redraw it during human review.
- `lesion_corrected.nii.gz` is the v1 MRI source of truth.
- The IHC marker/readout is `IgG-FITC` / `igg_fitc_*`, not a direct
  `lys241_*` concentration measure.
- Anti-IgG specificity is resolved as anti-human IgG specific to humanized
  LYS241, but no IgG-FITC positivity threshold has been approved.
- Panel A NeuroTrace is accepted for v1 as 640/660 deep-red, so Panel A FITC
  spectral bleed-through is not flagged for the current animal.

## Current Status

- `make env-check`, `make lint`, and `make test` pass in `lys-bbb`.
- `make convert-mri CONFIG=config/animals/BD_08_5D.yml` converts the configured
  T2 scan under `work/BD_08_5D/mri/`.
- A no-editor technical MRI run has written draft/corrected masks and a v1 CSV
  row flagged `needs_human_review`.
- Napari review displays the T2 stack as `(slice, y, x)` while saving masks back
  to the original NIfTI layout.
- QuPath/Bio-Formats diagnostics work for one Panel A target section.
- A first exploratory Panel A threshold sweep completed for `BD_08_5D`
  `section_01` and matching `C6S5` control `section_01` at
  `ihc.threshold_calibration.downsample=32.0`.
- The full reviewed MRI + IHC path has not yet completed for one animal.

## Do Not Start Yet

- Allen registration
- MRI/IHC compartments
- cell-level IHC classification
- multi-animal batch processing
- replacing the v1 MRI draft with a DL backend
- using the LYS held-out RatLesNetV2 test split for strategy decisions

## Immediate Order

1. **Make threshold review visual.** The current sweep CSV is not enough for
   scientific sign-off. Add exported overlays/thumbnails or another clear
   visual artifact for candidate IgG-FITC thresholds.
2. **Repeat the one-section IHC path for Panel B.** Diagnose first, then run
   target/control threshold sweeps with `--section section_01 --limit 1`.
3. **Add section QC/selection.** Record selected or excluded section IDs in
   config/provenance before scaling beyond one section.
4. **Add threshold sign-off.** Store panel/batch scope, threshold value,
   reviewer, date, target/control images used, and `approved` vs exploratory.
5. **Run deterministic IHC export only after review gates pass.** Final export
   must use reviewed `tissue_v1` and an approved IgG-FITC threshold.
6. **Review the MRI lesion mask.** Open napari with
   `make run CONFIG=config/animals/BD_08_5D.yml`, redraw/correct as needed, and
   ensure final CSV rows are `reviewed_edited` or `reviewed_no_changes`.
7. **Join reviewed MRI and IHC rows into the v1 CSV.**

## Commands

Environment and tests:

```bash
make env-check
make lint
make test
```

MRI:

```bash
make convert-mri CONFIG=config/animals/BD_08_5D.yml
make run CONFIG=config/animals/BD_08_5D.yml
```

Technical MRI run only:

```bash
make run CONFIG=config/animals/BD_08_5D.yml RUN_ARGS=--no-mask-editor
```

IHC calibration manifest:

```bash
make calibrate-ihc CONFIG=config/animals/BD_08_5D.yml
```

IHC diagnostics and one-section sweeps:

```bash
make ihc-diagnose CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --target-only --limit 1 --run --timeout-seconds 180"
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --target-only --limit 1 --run --timeout-seconds 600"
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --controls-only --limit 1 --run --append"
```

## Open Implementation Items

- Fail helpfully when final export is missing `tissue_v1`, pixel calibration,
  confirmed channel index, or approved threshold.
- Prevent duplicate IHC rows on re-export, or make overwrite behavior explicit
  and deterministic.
- Record image/section ID, panel, channel index, threshold, downsample,
  threshold status, tissue ROI status, and QC status in QuPath exports.
- Add a config-driven QuPath orchestration command that iterates only selected
  section series and skips labels/overviews.
- Resolve whether repeated direct QuPath CLI opens are acceptable or whether v1
  should switch to a QuPath project/cached-import workflow.
- Confirm reviewed MRI rows carry reviewer, edit Dice, and QC status.

## Completion Gate

v1 is complete only when:

- no channel, threshold, path, specificity, or spacing fact was guessed
- MRI header spacing and anisotropic volume math were verified
- MRI lesion mask received documented human review/correction
- unreviewed draft masks are not treated as final data
- IHC positive area uses reviewed tissue area as the denominator
- IgG-FITC threshold was produced by the documented calibration workflow and
  explicitly approved
- re-running IHC export does not silently duplicate measurements
- interpretation flags and QC provenance survive into deliverables
- one configured animal produces a reviewed joined CSV
- `make test` and `make lint` pass
