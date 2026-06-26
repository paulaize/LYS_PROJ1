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

## RatLesNetV2 finetuning branch

Branch `dl-ratlesnetv2-finetune` starts a separate MRI lesion deep-learning
track for RatLesNetV2 transfer learning. It is cloud-oriented: local code
prepares and validates NIfTI/manual-mask datasets, while actual finetuning runs
on a GPU runtime such as Google Colab.

See [`ratlesnetv2_finetune/README.md`](ratlesnetv2_finetune/README.md) and
[`docs/ratlesnetv2_finetuning_branch.md`](docs/ratlesnetv2_finetuning_branch.md).

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

## QuPath v1 export

v1 Groovy scripts do **not** assume channel order. Pass the IgG-FITC channel
index from the animal YAML/config.

There is currently **no approved IgG-FITC positivity threshold** for these IHC
images. This is the first quantitative analysis of the slides, and the team
does not have an existing QuPath threshold to reuse. The pipeline should create
candidate thresholds from configured controls/image statistics, write review
outputs, and require human sign-off before final positive-area measurements are
accepted. QuPath is the viewer/export engine here; it is not assumed to provide
a scientifically valid threshold automatically.

The current control animal for threshold/background calibration is `C6S5`:

- Panel A: `data/C6S5/IHC/panel_A/BD_08_C6S5 C-_01.vsi`
- Panel B: `data/C6S5/IHC/panel_B/BD_08_C6S5 C- IBA1 GFAP IgG_01.vsi`

Create the first threshold-calibration manifest:

```bash
make calibrate-ihc CONFIG=config/animals/BD_08_5D.yml
```

This writes `work/BD_08_5D/ihc_threshold_calibration_manifest.csv`. It is a
review plan, not an approved threshold.

Generate all exploratory threshold-sweep QuPath commands without running them:

```bash
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml
```

Run only Panel A commands:

```bash
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A"
```

Before running a real threshold sweep, diagnose one configured section. This
opens the `.vsi` series and writes metadata only; it does not read image pixels
or compute thresholds.

```bash
make ihc-diagnose CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --target-only --limit 1 --run --timeout-seconds 180"
```

If this times out, the bottleneck is direct QuPath/Bio-Formats opening of the
`.vsi` series and the next step is to move through a QuPath project/cached
import path rather than repeated direct CLI image opens.

If diagnostics succeeds, execute one threshold-sweep test command:

```bash
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --target-only --limit 1 --run --timeout-seconds 600"
```

If that succeeds, run the matching C6S5 control section:

```bash
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --controls-only --limit 1 --run --append"
```

Only after the target/control one-section smoke test succeeds and the best
sections have been selected should you scale up. Do not use the old
`downsample=8.0` command for exploratory sweeps; the v1 config uses
`ihc.threshold_calibration.downsample=32.0` for this first pass.

## RatLesNetV2 local commands

Prepare a local RatLesNetV2 dataset from reviewed T2w masks:

```bash
make ratlesnetv2-prepare RATLESNET_CONFIG=ratlesnetv2_finetune/configs/dataset_template.yml
```

Print cloud/Colab commands for the prepared dataset:

```bash
make ratlesnetv2-cloud-plan RATLESNET_CONFIG=ratlesnetv2_finetune/configs/dataset_template.yml
```

Then scale up to the chosen sections, or to the whole panel if needed:

```bash
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --run"
```

The sweep runner processes BD_08_5D and the C6S5 control across configured
section series `2..9`, writes
`work/BD_08_5D/ihc_threshold_sweep_panel_<panel>.csv`, and flags results as
`threshold_sweep_not_final`. It uses a rough automatic tissue ROI in memory at
the exploratory downsample configured in `config/pipeline.yml`, so this is for
threshold exploration only; final export still needs reviewed `tissue_v1`.

The current exporter requires `tissue_v1` and measures inside that annotation,
not the full rectangular image. Do not treat its positive-area result as final
until `tissue_v1` has been reviewed/corrected and the IgG-FITC threshold has
been approved.

Example shape:

```bash
QuPath script --image "/path/to/section.vsi" \
  --args "A,tissue_v1,16,200" \
  src/ihc/qupath/detect_cells.groovy

QuPath script --image "/path/to/section.vsi" \
  --args "A,$PWD/work/BD_08_5D/ihc_threshold_sweep_A.csv,1,100;250;500;1000;2000;4000;8000;16000,8,tissue_v1,section_01" \
  src/ihc/qupath/export_threshold_sweep.groovy

QuPath script --image "/path/to/section.vsi" \
  --args "A,$PWD/work/BD_08_5D/ihc_A.csv,1,<approved_threshold>,8,tissue_v1,section_01,approved" \
  src/ihc/qupath/export_measurements.groovy
```

Argument order for `export_measurements.groovy`:

```text
panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold,downsample,tissue_annotation_name,section_id,threshold_status
```

Do not trust the placeholder threshold above. Fill it only after the calibration
workflow has produced a reviewed/approved threshold.

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
