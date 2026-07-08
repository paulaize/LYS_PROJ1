# LYS_PROJ1 stroke MRI + IHC pipeline

Semi-automated analysis for a preclinical ischemic-stroke study in mice:

- **MRI:** Bruker T2 RARE -> NIfTI -> human-reviewed lesion volume.
- **IHC:** Olympus `.vsi` fluorescence -> QuPath/Python -> IgG-FITC positive-area readouts.
- **Later:** independent MRI->Allen and IHC->Allen registration, compartments, cell-level measurements, and a tidy table for R.

Read [AGENTS.md](AGENTS.md) first. It is the operating brief for agents and humans: data rules, domain facts, v1 scope, and definition of done.

## Current Status

The active production target is **v1 thin spine**:

- one test animal: `BD_08_5D`
- no atlas, no deep learning, no compartments, no batch processing
- MRI lesion volume from a reviewed/corrected mask
- IHC IgG-FITC positive area inside a reviewed `tissue_v1` ROI
- one minimal joined CSV

The automatic MRI lesion mask is only a draft. For `BD_08_5D` it can target bright peripheral artifact instead of the expected image-right isocortical lesion. The corrected `work/<animal_id>/lesion_corrected.nii.gz` mask is the source of truth for v1.

The current execution checklist is [docs/v1_next_session_todo.md](docs/v1_next_session_todo.md).

## Documentation Map

- [AGENTS.md](AGENTS.md): canonical rules and facts.
- [docs/v1_next_session_todo.md](docs/v1_next_session_todo.md): active v1 task list.
- [docs/development_roadmap.md](docs/development_roadmap.md): milestone plan after v1.
- [docs/MRI_IHC_pipeline_plan.md](docs/MRI_IHC_pipeline_plan.md): compact full-pipeline design reference.
- [docs/MRI_IHC_pipeline_plan_DL.md](docs/MRI_IHC_pipeline_plan_DL.md): later DL/human-in-loop design.
- [ratlesnetv2_finetune/README.md](ratlesnetv2_finetune/README.md): RatLesNetV2 local/Kaggle command runbook.
- [docs/ratlesnetv2_finetuning_branch.md](docs/ratlesnetv2_finetuning_branch.md): RatLesNetV2 branch strategy and current baseline.
- [docs/ratlesnetv2_external_datasets.md](docs/ratlesnetv2_external_datasets.md): public mouse dataset policy for RatLesNetV2 adaptation.

## Environment

Use the existing conda environment:

```bash
make env-check
make test
make lint
```

The default environment name is `lys-bbb`. `env/environment.yml` is a reference/export target, not a request to create a new environment.

Small dev additions, only if needed:

```bash
conda env update -n lys-bbb -f env/lys-bbb-dev-additions.yml
```

## v1 Quickstart

```bash
# Check the environment and tests.
make env-check
make test

# Convert configured Bruker scan 2 / reco 1 to NIfTI under work/.
make convert-mri CONFIG=config/animals/BD_08_5D.yml

# Technical run only: writes an unreviewed draft mask flagged needs_human_review.
make run CONFIG=config/animals/BD_08_5D.yml RUN_ARGS=--no-mask-editor

# Real v1 MRI run: opens napari for mask review/correction.
make run CONFIG=config/animals/BD_08_5D.yml
```

Optional IHC ingest, after a valid QuPath export exists:

```bash
make run CONFIG=config/animals/BD_08_5D.yml IHC=work/BD_08_5D/ihc.csv
```

Important: `RUN_ARGS=--no-mask-editor` is for technical runs only. It overwrites `lesion_corrected.nii.gz` with the draft and flags the row as `needs_human_review`.

## IHC Threshold Work

No approved IgG-FITC positivity threshold exists yet. The pipeline must create candidate thresholds from configured controls/image statistics, generate review artifacts, and require human sign-off before final positive-area rows are accepted.

Current control animal: `C6S5`.

Useful commands:

```bash
# Build the exploratory calibration manifest.
make calibrate-ihc CONFIG=config/animals/BD_08_5D.yml

# Dry-run configured threshold-sweep commands.
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml

# Diagnose one target section before pixel work.
make ihc-diagnose CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --target-only --limit 1 --run --timeout-seconds 180"

# Run one target sweep, then the matching control sweep.
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --target-only --limit 1 --run --timeout-seconds 600"
make ihc-threshold-sweeps CONFIG=config/animals/BD_08_5D.yml RUN_ARGS="--panel A --section section_01 --controls-only --limit 1 --run --append"
```

The sweep CSV is exploratory and must remain flagged as not final. Final measurement export still requires a reviewed `tissue_v1` ROI and an approved threshold.

## RatLesNetV2 Branch

Branch `dl-ratlesnetv2-finetune` is an independent MRI lesion DL track. It does not replace v1. It consumes human-reviewed masks and produces draft masks that still require human review before scientific volume calculations.

Current model strategy:

```text
main baseline: rat pretrained -> LYS fine-tuning
comparator:    rat pretrained -> external mouse adaptation -> LYS fine-tuning
```

The direct LYS run is now the baseline to beat: validation Dice reached about `0.53` by epoch 10 without background collapse. Continue from `best_by_validation_dice.model`, use validation overlays, reduce LR on plateau, and do not touch the held-out LYS test split until the final strategy is chosen.

Kaggle is currently the preferred free GPU runtime. The command runbook is [ratlesnetv2_finetune/README.md](ratlesnetv2_finetune/README.md).

## Make Targets

| Target | Purpose |
|---|---|
| `make env-check` | check core environment imports/tools |
| `make test` | run pytest with plugin autoload disabled |
| `make lint` | run Ruff |
| `make convert-mri CONFIG=...` | Bruker T2 scan 2 -> NIfTI |
| `make run CONFIG=...` | run one v1 animal |
| `make calibrate-ihc CONFIG=...` | write exploratory threshold manifest |
| `make ihc-diagnose CONFIG=... RUN_ARGS="..."` | QuPath/Bio-Formats metadata diagnostic |
| `make ihc-threshold-sweeps CONFIG=... RUN_ARGS="..."` | dry-run or execute exploratory threshold sweeps |
| `make ratlesnetv2-prepare RATLESNET_CONFIG=...` | prepare RatLesNetV2 folder contract locally |
| `make ratlesnetv2-split-prepared RUN_ARGS="..."` | split prepared RatLesNetV2 dataset |
| `make ratlesnetv2-cloud-plan RATLESNET_CONFIG=...` | print cloud training commands |

## Layout

```text
AGENTS.md, CLAUDE.md       # operating brief + Claude pointer
config/                    # global pipeline + per-animal configs
src/mri/                   # io, preprocess, segment, edit, edema, volume
src/ihc/                   # ingest/calibrate + QuPath Groovy scripts
src/atlas/                 # later milestones
src/compartments.py        # later milestones
src/join.py                # later milestones
scripts/                   # command wrappers and diagnostics
ratlesnetv2_finetune/      # independent RatLesNetV2 branch tooling
legacy/context_code/       # reference-only old scripts

data/                      # gitignored read-only inputs
work/                      # gitignored intermediates
outputs/                   # gitignored final deliverables
```

## Interpretation Note

The IHC marker/readout is named **IgG-FITC** in code and outputs. Specificity is resolved: the secondary is anti-human IgG, and LYS241 is humanized Glunomab, so IgG-FITC is interpreted as LYS241-associated signal through provenance flags.

This does not set the positivity threshold. Threshold calibration and sign-off remain open v1 gates.
