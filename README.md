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

Atlas registration, DL models, compartments, cell-level detection, and batch processing are later milestones described in `docs/development_roadmap.md`.

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

# 3. Configure one animal
cp config/animals/TEMPLATE.yml config/animals/M07.yml
# Fill every TODO: paths, timepoint, channel map, thresholds, reviewer.

# 4. Optional: run MRI + ingest an existing IHC export
make run CONFIG=config/animals/M07.yml IHC=work/M07/ihc_A.csv
```

If you do not yet have an IHC export CSV, the MRI track can still run. If you do not yet have MRI for an early timepoint, set `mri.has_mri: false` in that animal config.

## QuPath v1 export

v1 Groovy scripts do **not** assume channel order. Pass the IgG-FITC channel index and threshold after confirming them in the animal YAML/config.

Example shape:

```bash
QuPath script --image "/path/to/section.vsi" \
  --args "A" \
  src/ihc/qupath/detect_cells.groovy

QuPath script --image "/path/to/section.vsi" \
  --args "A,$PWD/work/M07/ihc_A.csv,3,500,8" \
  src/ihc/qupath/export_measurements.groovy
```

Argument order for `export_measurements.groovy`:

```text
panel,out_csv,igg_fitc_channel_index,igg_fitc_threshold,downsample
```

Do not trust `3` or `500` above; they are examples only. Fill them from your confirmed config.

## Layout

```text
AGENTS.md, CLAUDE.md       # operating brief + Claude pointer
.codex/                    # Codex config, agents, reusable prompts
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

This repo includes project-scoped Codex files:

- `AGENTS.md`
- `.codex/config.toml`
- `.codex/agents/*.toml`
- `.codex/prompts/*.md`

Keep provider/auth/secrets in your user-level Codex config, not this repository.

Good first Codex prompt:

```text
Read AGENTS.md and .codex/prompts/03-run_validate.md. Validate the repo and report the next smallest v1 task without adding atlas, DL, compartments, or batch.
```

## Critical interpretation note

The v1 IHC marker is named **IgG-FITC** in code and outputs. Do not interpret it as direct LYS241 concentration until anti-IgG specificity and endogenous IgG leakage confounding are resolved.
