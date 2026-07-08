# LYS_PROJ1 stroke MRI + IHC pipeline

Semi-automated analysis of a preclinical ischemic-stroke study in mice:

- **MRI:** lesion volume from Bruker → NIfTI T2 RARE scans.
- **IHC:** Olympus `.vsi` multiplex fluorescence quantification, starting with IgG-FITC % positive area.
- **Later:** independent MRI→Allen and IHC→Allen registration, compartments, cell-level measurements, and a full tidy table for stats in R.

> **Agents and humans should read [`AGENTS.md`](AGENTS.md) first.** It is the canonical operating brief: hard constraints, domain facts not to invent, v1 scope, and definition of done.

## Status: v1 thin spine

v1 is deliberately small: one test animal, simplest reliable method per step, **no atlas and no deep learning yet**. It produces:

1. a human-corrected MRI lesion volume,
2. a basic QuPath/IHC IgG-FITC positive-area measurement,
3. one minimal joined CSV.

The current v1 test animal is `BD_08_5D`. The automatic MRI lesion mask is
only an untrusted draft. If it targets artifact or the wrong anatomy, erase or
redraw it during review; `lesion_corrected.nii.gz` is the source of truth for
v1 lesion volume.

Atlas registration, DL models, compartments, cell-level detection, and batch processing are later milestones described in `docs/development_roadmap.md`.

For the current audited blockers and the exact next programming-session order,
read [`docs/v1_next_session_todo.md`](docs/v1_next_session_todo.md).

## Use your existing conda env

This starter is configured for Paul’s existing environment:

```bash
conda run -n lys-bbb python scripts/check_env.py
conda run -n lys-bbb python -m pytest -q
```

Optional small dev additions:

```bash
conda env update -n lys-bbb -f env/lys-bbb-dev-additions.yml
```

`env/environment.yml` is kept as a reference/export target only; the default workflow does **not** create a new env.

## Quickstart

```bash
# 1. Check env
make env-check

# 2. Run pure tests
make test

# 3. Use the current v1 animal config
ls config/animals/BD_08_5D.yml

# For a future animal, copy the template and fill paths/channel facts.
# cp config/animals/TEMPLATE.yml config/animals/<animal_id>.yml

# 4. Convert configured Bruker T2 scan 2 to NIfTI under work/
make convert-mri CONFIG=config/animals/BD_08_5D.yml

# 5. Optional: run MRI + ingest an existing IHC export
make run CONFIG=config/animals/BD_08_5D.yml RUN_ARGS=--no-mask-editor IHC=work/BD_08_5D/ihc_A.csv
```

If you do not yet have an IHC export CSV, the MRI track can still run. If you do not yet have MRI for an early timepoint, set `mri.has_mri: false` in that animal config.
`RUN_ARGS=--no-mask-editor` is for a first technical run only; it writes the
draft mask as `needs_human_review` instead of opening napari.

## MRI v1 mask review

For the current v1 test animal:

```bash
make convert-mri CONFIG=config/animals/BD_08_5D.yml
make run CONFIG=config/animals/BD_08_5D.yml
```

Running without `RUN_ARGS=--no-mask-editor` opens napari for human review.
Napari shows the T2 volume as an 18-slice stack in `(slice, y, x)` display
order. The saved NIfTI mask remains in the original `(x, y, slice)` layout, so
Fiji still opens `work/BD_08_5D/lesion_corrected.nii.gz` as 18 images.

In napari, select the `lesion (edit me)` labels layer. The automatic lesion is
a label mask, not a movable shape. Use the labels-layer paint brush to add
lesion pixels/voxels and the eraser to remove them, slice by slice. Close the
napari window to save.

The current threshold draft can be anatomically wrong, especially by selecting
bright peripheral signal instead of the image-right isocortical lesion. That is
not a v1 blocker. The reviewer may clear the draft and draw the lesion by hand;
the corrected mask is what volume and edema correction use.

The run writes:

- `work/BD_08_5D/lesion_draft.nii.gz`
- `work/BD_08_5D/lesion_corrected.nii.gz`
- `outputs/BD_08_5D/BD_08_5D_v1.csv`

QC flags in the CSV:

- `reviewed_edited`: napari opened and the corrected mask differs from the draft.
- `reviewed_no_changes`: napari opened and the reviewer accepted the draft.
- `needs_human_review`: napari did not run, or `--no-mask-editor` was used.

Do not run `RUN_ARGS=--no-mask-editor` after making a manual correction unless
you intentionally want to overwrite `lesion_corrected.nii.gz` with the
unreviewed draft again.

## IHC v1

The active IHC plan is [`docs/IHC_v1_plan.md`](docs/IHC_v1_plan.md). The exact
next checklist is [`docs/v1_next_session_todo.md`](docs/v1_next_session_todo.md).

Current v1 IHC rule set:

- QuPath handles `.vsi` IO, review surfaces, overlays, and measurement export.
- Python handles config, orchestration, CSV ingestion, provenance, and joins.
- Panel A and Panel B are not assumed to be section-matched.
- `C6S5` is a no-LYS241/background control, not a complete spectral correction
  control.
- No IgG-FITC positivity threshold is approved yet.
- Final measurements require selected good sections, reviewed `tissue_v1`, and
  threshold sign-off.
- Folds/tears/debris should be annotated in QuPath as `artifact_exclude`; final
  measurements use `tissue_v1 - artifact_exclude`.
- Direct `--image <file.vsi>` QuPath commands are exploratory and do not load
  saved project annotations. Reviewed final runs need a QuPath project
  (`--project <project.qpproj>`).

Useful current commands:

```bash
make calibrate-ihc CONFIG=config/animals/BD_08_5D.yml
make ihc-diagnose CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --target-only --limit 1"
make ihc-section-qc CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--keep-going"
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A"
make ihc-threshold-review CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A"
make ihc-threshold-dashboard CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A"
make ihc-threshold-signoff CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --decision work/BD_08_5D/ihc_manual_review/panel_A_threshold_review_decision.json"
make ihc-quantify CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --threshold 250 --exploratory --tissue-mode auto_if_missing"
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel B"
```

Real QuPath runs add `--run` inside `RUN_ARGS`; on macOS they may need
unsandboxed execution.

The `ihc-quantify` example above is a first-draft direct-VSI export. It writes
rows flagged as exploratory because it uses rough auto tissue and cannot see
saved QuPath project annotations. Reviewed/final exports should use an approved
threshold sign-off and `tissue_mode=require_reviewed`.

Selected sections are recorded per panel in the animal configs. Panel A uses
BD_08_5D sections `01`, `03`, `06` and C6S5 control sections `05`, `06`, `07`.
Panel B currently runs BD_08_5D sections `05`, `06`, `08` only because the
C6S5 Panel B control file is corrupted/unavailable; Panel B IgG-FITC outputs
remain exploratory until a valid control or approved fallback threshold exists.

The manual threshold dashboard writes:

```text
work/BD_08_5D/ihc_manual_review/panel_A_threshold_review.html
work/BD_08_5D/ihc_manual_review/panel_A_threshold_signoff_template.json
```

After manual review, place the downloaded decision at
`work/BD_08_5D/ihc_manual_review/panel_A_threshold_review_decision.json` and
run `make ihc-threshold-signoff` to create:

```text
work/BD_08_5D/ihc_threshold_signoff_panel_A.json
work/BD_08_5D/ihc_threshold_signoff_panel_A.csv
```

## Layout

```text
AGENTS.md, CLAUDE.md       # operating brief + Claude pointer
config/                    # global pipeline + per-animal configs
src/mri/                   # io, preprocess, segment, edit, edema, volume
src/ihc/                   # ingest.py + QuPath Groovy scripts
src/atlas/                 # later-milestone stubs
compartments.py, join.py   # later-milestone stubs
scripts/check_env.py       # confirms lys-bbb imports
legacy/context_code/       # old/reference scripts only, not production

data/                      # gitignored read-only inputs
work/                      # gitignored intermediates
outputs/                   # gitignored final outputs
```

## Codex

This repo is currently driven by `AGENTS.md`. A `.codex/` directory is not
required in this checkout. Keep provider/auth/secrets in your user-level Codex
config, not this repository.

Good first Codex prompt:

```text
Read AGENTS.md and docs/v1_next_session_todo.md. Validate the repo and report the next smallest v1 task without adding atlas, DL, compartments, or batch.
```

## Critical interpretation note

The v1 IHC marker is named **IgG-FITC** in code and outputs. Specificity is now
resolved: the secondary is confirmed anti-human IgG, and LYS241 is humanized
Glunomab, so IgG-FITC is interpreted as LYS241-associated signal. Keep the
physical measurement name `IgG-FITC` / `igg_fitc_*`; the LYS241 interpretation
lives in provenance flags.

This confirmation does not set the positivity threshold. IgG-FITC threshold
calibration still needs to be built and reviewed because no prior QuPath
threshold exists. Panel A NeuroTrace is accepted as 640/660 deep-red for v1, so
Panel A FITC spectral bleed-through is not flagged at this stage.
