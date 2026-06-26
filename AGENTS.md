# AGENTS.md — operating brief for LYS_PROJ1

> Read this before changing code. Codex reads `AGENTS.md` at the start of a task; Claude Code can use `CLAUDE.md`, which points here. When this file conflicts with assumptions, **this file wins**. When a needed fact is marked **TODO**, do not guess it: add a config placeholder, fail helpfully, or ask Paul.

---

## 1. What this project is

Semi-automated Python pipeline for a preclinical ischemic-stroke study in mice (thrombin model, Drieu et al. 2020).

- **MRI** (Bruker → NIfTI): measure ischemic **lesion volume** from T2-weighted scans.
- **IHC** (Olympus `.vsi`, 4-channel fluorescence): quantify anti-IgG-FITC signal associated with candidate drug **LYS241** across neurons, endothelium, astrocytes, and microglia — object-level later, area-level in v1.

Both modalities eventually register **independently** to the **Allen Mouse Brain CCFv3**, which is the common frame and source of region labels. Final deliverable: tidy long table `animal × region × compartment × cell_type × measure`, plus QC overlays, for stats in R.

Rationale lives in `docs/`:

- `docs/MRI_IHC_pipeline_plan.md`
- `docs/MRI_IHC_pipeline_plan_DL.md`
- `docs/development_roadmap.md`

This file is the operational summary. Do not duplicate the full docs into code comments.

---

## 2. Current target: v1 thin spine only

**Phase: v1 development.** Build one test animal, simplest reliable method per step, **no atlas, no DL, no compartments, no batch**.

The current prioritized programming checklist is
`docs/v1_next_session_todo.md`. Read it before starting the next v1 session.

v1 includes:

- NIfTI load + header/spacing sanity.
- N4 bias correction.
- simple brain mask / hemisphere support.
- an automatic draft lesion mask, currently from threshold segmentation.
- human mask review/edit gate.
- anisotropic lesion volume from NIfTI header spacing.
- Swanson/indirect edema correction on the **human-corrected** mask.
- QuPath `.vsi` export for basic **IgG-FITC % positive area**.
- Python ingestion of QuPath CSV.
- one minimal joined CSV.

v1 deliberately excludes:

- Allen registration / AIDAmri / ABBA.
- deep-learning lesion or cell models.
- core / peri / contra compartments.
- cell-level detection/classification.
- multi-animal batch processing.

Later features must be added behind stable interfaces rather than by rewriting the v1 spine.

Current v1 MRI policy: the automatic lesion mask is only a draft and may be
very wrong. For `BD_08_5D` it can be pulled toward bright peripheral artifacts
even though the true lesion is expected on image-right in the isocortex above
the corpus callosum. Do not spend v1 time adding atlas/DL/isocortex ROI
heuristics unless Paul explicitly redirects the work. The reviewed/corrected
human mask is the source of truth for v1 volume, and a full manual redraw is
acceptable when the draft is a poor seed.

---

## 3. Environment: use the existing `lys-bbb` conda env

Use Paul’s current conda env. Do **not** create a new env, including the older
planned name `stroke-pipeline`, unless Paul explicitly asks.

- env name: `lys-bbb`
- typical env path: `/opt/anaconda3/envs/lys-bbb`
- platform: macOS arm64 / Apple Silicon
- Python in the env: Python 3.11.x

Preferred commands:

```bash
conda run -n lys-bbb python scripts/check_env.py
conda run -n lys-bbb python -m pytest -q
conda run -n lys-bbb python -m src.run_animal --config config/animals/<id>.yml
```

If dev tools are missing, install only small additions into the existing env:

```bash
conda env update -n lys-bbb -f env/lys-bbb-dev-additions.yml
```

GPU rules:

- Apple Silicon GPU = **MPS**, not CUDA.
- Never write `.cuda()`.
- If PyTorch is added later, use:

```python
torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

External apps are not Python packages:

- QuPath, Fiji, ABBA, ITK-SNAP, 3D Slicer, and FSL are desktop/system installs.
- Do not `pip install qupath` or `conda install abba`.
- Drive QuPath with CLI/Groovy or `paquo`; use ABBA through the QuPath extension on macOS.

---

## 4. Hard data rules

1. **Never edit files under `data/` or `/Volumes/...` in place.** Inputs are read-only.
2. Outputs go to `work/` for intermediates and `outputs/` for final deliverables.
3. Never commit `data/`, `work/`, `outputs/`, secrets, local external-drive paths, model weights, QuPath projects, or large binaries.
4. Use `pathlib.Path` for paths. No string-concatenated paths.
5. Scan/channel/path facts come from YAML config. Never hardcode `.vsi` channel order, scan paths, or absolute user paths.
6. Fail loudly on bad assumptions, especially wrong voxel spacing, missing channel maps, unknown thresholds, or missing reviewer/QC provenance.

### Large IHC file policy

- Raw `.vsi` files are large whole-slide inputs and must remain in their
  original read-only location, usually `data/` or an external `/Volumes/...`
  drive.
- Do not copy, rewrite, or convert full `.vsi` files into TIFF/OME-TIFF as part
  of the default workflow unless Paul explicitly asks.
- IHC code should process one animal, panel, section, or tile at a time rather
  than loading whole panels or whole batches into memory.
- Use downsampled pyramid levels for diagnostics, threshold-review thumbnails,
  and visual QC whenever full resolution is not required.
- Temporary caches must be written under `work/<animal_id>/`, be rebuildable,
  and be safe to delete. Final deliverables belong under `outputs/`.

---

## 5. Domain facts that must not be invented

### MRI

- Scan 1 = FLASH localizer → discard.
- **Scan 2 = RARE `T2_haute_resolution_Turbo` → T2w lesion-volume scan.**
- Scan 3 = FcFLASH `T2s_rapide` → optional T2* hemorrhage readout.
- Scan 4 = FLASH TOF angio → optional vessel context, not lesion volume.
- T2 geometry: 256×256, FOV 17.92 mm → **0.07 mm in-plane**; 18 slices × **0.5 mm**, no gap.
- Voxels are anisotropic / 2.5-D.
- Lesion volume = `voxel_count × prod(header_spacing)`, equivalently Σ slice area × slice thickness.
- Always read spacing from the NIfTI header and check it against expected values from config.

### Study design

- About 10 stroke animals + 3 controls.
- Timepoints: 1h, 3h, 6h, 24h, 48h, 5d.
- Each animal = one timepoint; cross-sectional, not longitudinal.
- **MRI exists only for 24h, 48h, and 5d.** Early 1h/3h/6h animals are IHC-only; later join/compartment code must handle no-MRI animals and flag `compartment_method`.

### IHC

- `.vsi`: Olympus fluorescence WSI, uint16, 20×, ~0.325 µm/px, pyramidal, ~3 GB each, 4 channels, 8 sections/animal/panel + 1 overview.
- Panel A expected markers: DAPI, NeuroTrace, Podocalyxin, anti-IgG-FITC.
- Panel B expected markers: DAPI, IBA1, GFAP, anti-IgG-FITC.
- The current v1 animal `BD_08_5D` has confirmed panel channel maps in
  `config/animals/BD_08_5D.yml`. Future animals still need their own confirmed
  channel maps. Read channel order from config; never assume `channel 0 = DAPI`.
- Current v1 Panel A NeuroTrace is accepted as **NeuroTrace 640/660 Deep-Red
  Fluorescent Nissl Stain** (Paul, 2026-06-17), so Panel A FITC spectral
  bleed-through is not flagged for v1. Paul may still do a later full
  fluorochrome audit.

### FITC / LYS241 confound

Confirmed fact (Paul, 2026-06-17): the anti-IgG-FITC secondary is an
**anti-human IgG** antibody. LYS241 is humanized Glunomab, so the secondary
specifically targets LYS241 and does **not** bind endogenous mouse IgG. The
post-stroke endogenous-IgG-leakage biological confound is resolved.

This resolves interpretation specificity, not positivity thresholding. The
IgG-FITC positive threshold still needs calibration/sign-off from configured
controls or an approved calibration rule.

No approved IgG-FITC positivity threshold exists yet for the current IHC
images. This is the first quantitative analysis of these slides, and Paul's
team has not already established QuPath thresholds. Do not assume QuPath will
provide a scientifically valid "basic threshold" by itself. v1 must include a
calibration/exploration workflow that proposes candidate thresholds from
configured controls or documented image statistics, then requires human
sign-off before final positive-area measurements are accepted. Candidate
thresholds may be exported for review only if clearly marked exploratory/not
final.

Therefore:

- In code and tables, call the marker/readout **`IgG-FITC`** or `igg_fitc_*`, not direct `LYS241 concentration`.
- Do not name v1 measures `lys241_*`.
- Carry `anti_igg_specificity_resolved=true`,
  `fitc_specific_to_lys241=true`, `fitc_igg_specificity="anti_human_confirmed"`,
  and source `"confirmed_anti_human_IgG (Paul, 2026-06-17)"` in provenance.
- Keep spectral bleed-through separate from biological specificity. For
  `BD_08_5D` v1, Paul accepted Panel A NeuroTrace as 640/660 deep-red
  (2026-06-17), so `fitc_spectral_bleedthrough="none"` is allowed for the v1
  config. Future panels/animals still need this fact recorded rather than
  assumed. Panel B has no NeuroTrace and can carry
  `fitc_spectral_bleedthrough="none"`.

### Registration and compartments

- Do not register MRI directly to IHC for quantification.
- Later: MRI→Allen and IHC→Allen independently; join through atlas region/compartment/hemisphere keys.
- Direct MRI↔IHC overlay is figure-only, never quantification.
- Compartments later: core = corrected lesion mask; peri = physical dilation ring minus core; contra = atlas-midline mirror.

---

## 6. Stable interfaces

Keep these signatures stable. Callers should depend on these interfaces, not backend details.

```python
# src/mri/segment.py
def segment_lesion(volume, brain_mask, *, method="threshold", **kw):
    """Return a binary lesion mask. v1: method='threshold'. Later: method='dl'."""

# src/mri/edit.py
def review_mask(volume, draft_mask, out_path):
    """Open editor, save corrected mask, return dict(edited, dice, reviewer)."""

# src/ihc/ingest.py
def ingest_qupath(csv_path):
    """QuPath measurement export -> list of tidy measurement rows."""
```

Adding optional keyword-only arguments is acceptable when needed for orchestration, but do not break existing callers.

---

## 7. Repo layout to preserve

```text
LYS_PROJ1/
├── AGENTS.md
├── CLAUDE.md
├── env/
├── config/
│   ├── pipeline.yml
│   └── animals/<id>.yml
├── src/
│   ├── mri/{io,preprocess,segment,edit,edema,volume}.py
│   ├── ihc/{ingest.py,qupath/*.groovy}
│   ├── atlas/                  # later milestones only
│   ├── compartments.py          # later milestones only
│   ├── join.py                  # later milestones only
│   └── run_animal.py
├── scripts/
├── tests/
├── docs/
├── legacy/context_code/         # reference-only old scripts, not production
├── data/                        # gitignored read-only inputs
├── work/                        # gitignored intermediates
├── outputs/                     # gitignored final outputs
└── notebooks/                   # # %% scratch .py only
```

A `.codex/` directory is optional and is not required in the current checkout.

---

## 8. Build/test commands

```bash
make env-check
make test
make run CONFIG=config/animals/<id>.yml IHC=work/<id>/ihc_A.csv
```

These must call `conda run -n lys-bbb ...` by default.

After every code change, run:

```bash
conda run -n lys-bbb python -m pytest -q
```

Do not add network calls or model downloads to tests.

---

## 9. Definition of done for v1 tasks

MRI code is done only when:

- NIfTI loads from a configured path.
- Header spacing is read and validated.
- Volume math uses header spacing, not isotropic assumptions.
- A reviewed/corrected binary lesion mask produces a plausible mm³ number.
- Human edit metadata is carried: `edited`, `edit_dice`, `reviewer`, `qc_flag`.
- Unreviewed fallback masks are clearly flagged `needs_human_review` and are not treated as final QC-pass data.
- The automatic draft mask is not treated as scientific output. If it targets
  artifact or the wrong anatomy, the reviewer should erase/redraw it and the
  corrected mask is what volume and edema correction use.

IHC code is done only when:

- QuPath export CSV is parsed into tidy long rows.
- IgG-FITC positive area is produced by the scripted analysis against a
  reviewed/approved tissue ROI (`tissue_v1`), not by hand analysis and not from
  the full rectangular image including off-tissue background.
- FITC readouts are labeled as IgG-FITC, not direct LYS241 concentration.
- Channel/threshold facts are config-driven or explicitly passed from config.
- Missing channel order or threshold produces a helpful error, not a guessed result.
- Because no prior QuPath threshold exists, v1 includes creating the first
  calibration/sign-off path; it must not require Paul to supply a pre-existing
  threshold from earlier analysis.
- Re-running an export does not silently duplicate section measurements.
- Specificity, spectral bleed-through, and IHC threshold provenance are
  carried into deliverables or an attached provenance table.

Joined v1 output is done only when one command writes a CSV with at least:

```text
animal_id, timepoint, panel, modality, measure, value, unit,
area_mm2, n_cells, edited, edit_dice, reviewer, qc_flag, model_version
```

Full later table schema lives in `src/join.py`.

---

## 10. External tools — use real APIs, do not invent methods

Before calling external tools, check installed help/docs/source for real signatures.

| Tool | Use |
|---|---|
| brkraw | Bruker → NIfTI |
| nibabel | NIfTI IO / headers |
| SimpleITK | N4, resampling, IO |
| ANTsPy (`antspyx`) | registration later |
| napari | optional v1 mask editing |
| QuPath | `.vsi` analysis, headless Groovy |
| paquo | optional QuPath orchestration from Python |
| ITK-SNAP / 3D Slicer | external manual mask editing |
| StarDist / InstanSeg | later cell/nuclei detection |
| ABBA / DeepSlice | later IHC→Allen |
| AIDAmri | later MRI→Allen; FSL/LIP orientation gotchas |
| nnU-Net / An et al. model | later DL lesion backend after QC |

Known macOS/arm64 cautions:

- `abba_python` GUI does not run reliably on macOS; use ABBA through QuPath extension.
- AIDAmri is Milestone 3, not v1; expects LIP orientation and depends on FSL.
- `antspyx` can be finicky on arm64; verify `import ants` early, but do not block v1 on it.

---

## 11. Legacy code policy

Files under `legacy/context_code/` are reference-only scripts from earlier local work. They may contain hardcoded paths, exploratory atlas code, older TIFF/ND workflows, and outdated FITC/LYS241 naming. Do not import them from `src/`. Extract ideas only through small, reviewed refactors that obey this file.

---

## 12. TODO facts only Paul can provide

- The prioritized and actionable version of this list is maintained in
  `docs/v1_next_session_todo.md`.
- [x] Current v1 animal/channel/path facts are recorded for `BD_08_5D` in
  `config/animals/BD_08_5D.yml`. New animals must still provide their own
  config facts.
- [x] NeuroTrace variant / emission accepted for v1:
  `NeuroTrace 640/660 Deep-Red Fluorescent Nissl Stain` (Paul, 2026-06-17).
- [x] anti-IgG-FITC specificity to LYS241 vs generic IgG: resolved as
  anti-human IgG specific to humanized LYS241 (Paul, 2026-06-17).
- [ ] Péri-lesional ring width in mm, later milestone.
- [ ] IgG-FITC positive threshold after control-based calibration.
- [ ] MRI threshold `k` only if the draft-threshold backend is tuned later;
  it is not a gate for v1 if the corrected human mask is used.
- [x] Initial v1 mask editor is napari. Keep the interface open to ITK-SNAP,
  3D Slicer, or Fiji/Labkit later.

---

## 13. Coding style

- Small targeted diffs over rewrites.
- Clear hand-written code; minimal cleverness.
- Comments only where intent/domain assumptions are not obvious.
- Explicit inputs/outputs per stage, no hidden global state.
- Use `pathlib.Path`.
- Do not hide scientific uncertainty in variable names.
- Prefer deterministic tests with synthetic arrays and tiny fixtures.
