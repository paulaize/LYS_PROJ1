# Development roadmap — stroke MRI + IHC pipeline (fast v1 first)

**How to read this.** Milestones are ordered for *fastest time-to-a-working-number*. v1 (Milestone 0+1) is a thin end-to-end spine on **one animal** using the simplest reliable method at each step — no atlas, no DL yet — so you have real outputs in ~3–4 focused days. Everything after v1 is a **drop-in upgrade behind a fixed interface**, so nothing built early gets thrown away. Estimates are rough focused-days for one person; treat them as relative, not contractual.

**Guiding principles for a fast v1**
1. **Spine before polish.** Get NIfTI→volume and .vsi→measurement→one table working before improving any single step.
2. **Fix interfaces, swap backends.** Define `segment_lesion()`, `detect_cells()`, etc. once; v1 fills them with threshold/built-in methods, v2 swaps in DL — same signature, no caller changes.
3. **Defer the atlas.** Region/compartment labels are Milestone 3–4. v1 reports whole-section / whole-hemisphere numbers. This removes the single biggest time sink (AIDAmri + ABBA setup) from the critical path to first output.
4. **Don't let model wrangling block the spine.** If the pretrained lesion model isn't runnable in an hour, v1 uses thresholding and the model becomes a v2 task.
5. **Human gate from day one** — even v1's lesion mask gets a manual edit pass; that's cheap and it's the habit the whole design depends on.

---

## Repo & data layout (set up in Milestone 0)

```
stroke-pipeline/
├── env/
│   └── environment.yml
├── config/
│   ├── pipeline.yml            # global params (peri-ring width, thresholds, model versions)
│   └── animals/
│       └── M07.yml             # per-animal: paths, timepoint, panel, channel map
├── src/
│   ├── mri/
│   │   ├── io.py               # load/save NIfTI, header sanity (pathlib)
│   │   ├── preprocess.py       # N4, brain mask
│   │   ├── segment.py          # segment_lesion(vol) -> mask   [threshold | dl backends]
│   │   ├── edit.py             # launch editor, log edit-Dice
│   │   ├── edema.py            # Swanson / atlas correction
│   │   └── volume.py           # mask -> mm^3 + per-slice profile
│   ├── ihc/
│   │   ├── qupath/             # Groovy scripts run headless
│   │   │   ├── detect_cells.groovy
│   │   │   └── export_measurements.groovy
│   │   └── ingest.py           # read QuPath CSV exports -> tidy rows
│   ├── atlas/                  # (Milestone 3+) AIDAmri / ABBA wrappers
│   ├── compartments.py         # (Milestone 4) core/peri/contra from mask+reg
│   ├── join.py                 # (Milestone 4) merge MRI+IHC -> long table
│   └── run_animal.py           # entry point: one config -> outputs
├── data/                       # (gitignored) inputs, never edited in place
├── work/                       # intermediates (masks, CSVs) per animal
├── outputs/                    # final masks, tables, figures
└── notebooks/                  # # %% cell-style .py scratch, not committed
```

Use `# %%` cell-style `.py` in `notebooks/` for exploration; promote stable code into `src/` modules. All paths via `pathlib`. One private GitHub repo from the start; `data/`, `work/`, `outputs/` gitignored.

---

## Environment (Milestone 0, M1-aware)

```yaml
# env/environment.yml  — keep MRI-Python light; QuPath/Fiji/ABBA are SEPARATE desktop apps
name: stroke-pipeline
channels: [conda-forge]
dependencies:
  - python=3.11
  - nibabel
  - simpleitk            # N4 bias correction, resampling
  - antspyx              # registration + rodent brain extraction (check arm64 wheel)
  - napari               # 3D mask editing in-Python (or use ITK-SNAP/3D Slicer apps)
  - pyqt                 # napari backend
  - pandas
  - pyyaml
  - scikit-image
  - pip
  - pip: [paquo]         # touch QuPath projects from Python
```

Separate, heavier env only when you add the DL lesion model (PyTorch with **MPS** for M1 GPU; keep it isolated to avoid breaking `antspyx`):
```bash
conda create -n stroke-dl python=3.11 pytorch torchvision -c pytorch
# + the An et al. / nnU-Net package once you confirm it runs
```
Desktop apps installed separately (not conda): **QuPath** (+ InstanSeg & StarDist extensions), **Fiji** (+ ABBA), and **ITK-SNAP** or **3D Slicer** for 3-D mask editing.

> **Setup checkpoint:** confirm `antspyx` imports on arm64 and `napari` opens. If `antspyx` fights you on M1, defer it — it's only needed at Milestone 3, not for v1.

---

## Stable interfaces (write these signatures in Milestone 1, never change them)

```python
# src/mri/segment.py
def segment_lesion(volume, brain_mask, *, method="threshold", **kw):
    """Return a binary lesion mask. v1: method='threshold'. v2: method='dl'."""

# src/mri/edit.py
def review_mask(volume, draft_mask, out_path):
    """Open editor, save corrected mask, return dict(edited, dice, reviewer)."""

# src/ihc/ingest.py
def ingest_qupath(csv_path):
    """QuPath measurement export -> list of tidy measurement rows."""
```
Because callers depend only on these signatures, the DL upgrade in Milestone 2 changes *only* the function body.

---

## Milestone 0 — Setup & data audit  ·  ~0.5–1 day  ·  CRITICAL PATH

| Task | Action | Done when |
|---|---|---|
| Repo + env | Create repo skeleton above; build `stroke-pipeline` env | `import nibabel, SimpleITK, napari, pandas` all succeed |
| Pick the test animal | Choose one 24h or 48h animal with clear MRI + both IHC panels | Paths recorded in `config/animals/M07.yml` |
| MRI header sanity | Load the T2 NIfTI; verify spacing = 0.07×0.07×0.5 mm, orientation | `io.py` prints correct voxel size; a slice renders |
| Channel map | Open one `.vsi` in QuPath; confirm channel→marker per panel; note NeuroTrace emission | Written into the animal config; FITC-overlap risk noted |
| Model availability check | Find An et al. 2023 code/weights (GitHub) + Zenodo data; try to run inference once | **Fork:** runs → DL is viable for v2. Doesn't run easily → v1 stays threshold, plan nnU-Net fine-tune later |

**Defer:** anti-IgG specificity resolution can run in parallel (it's a wet-lab/records question, not a code blocker) — but it gates *interpretation*, so flag it now.

---

## Milestone 1 — v1 thin spine, one animal, no atlas  ·  ~2–3 days  ·  CRITICAL PATH → **this is "first version up"**

### 1A. MRI → lesion volume (~1 day)
| Task | Action | Output |
|---|---|---|
| Preprocess | `preprocess.py`: N4 bias correction + brain mask (Otsu/morphology is fine for v1) | clean volume + mask |
| Segment (threshold backend) | `segment.py` method='threshold': voxels > contra-mean+k·SD inside ipsi hemisphere; connected components; size filter | draft lesion mask |
| **✎ Edit** | `edit.py`: open volume+draft in napari (or ITK-SNAP); hand-correct borders; save; log Dice(draft,corrected) | corrected mask + edit record |
| Volume | `volume.py`: Σ(area)×0.5 mm; raw + per-slice profile | volume number (mm³) |

### 1B. IHC → basic measurement (~1 day)
| Task | Action | Output |
|---|---|---|
| QuPath project | Create project, import the `.vsi` (Bio-Formats) | openable project |
| % positive area (no segmentation yet) | `detect_cells.groovy` v1 = simple intensity threshold on FITC → % positive area per whole section; DAPI threshold → rough density | measurement CSV |
| Export + ingest | `export_measurements.groovy` → `ingest.py` parses to rows | tidy IHC rows |

> **Why %-area first:** it needs no cell segmentation, so it proves the IHC spine in hours. Cell-level detection is Milestone 2.

### 1C. Glue (~0.5 day)
| Task | Action | Output |
|---|---|---|
| Entry point | `run_animal.py`: read config → run 1A + 1B → write a minimal table | `outputs/M07_v1.csv` |
| Minimal table | Columns: animal, timepoint, panel, modality, measure, value, edited, edit_dice | **v1 deliverable** |

**✅ v1 acceptance:** one command on one animal produces a lesion volume (human-corrected) **and** an IHC measurement CSV, joined into one table. No atlas, no regions, no DL required.

**v1 deliberately omits:** atlas regions, core/peri/contra compartments, cell-level counts, multi-animal batch, DL models. All added next, behind the interfaces above.

---

## Milestone 2 — Quality swap: DL + cell-level + correction gates  ·  ~3–5 days

| Task | Action | Replaces |
|---|---|---|
| DL lesion backend | Implement `segment.py` method='dl' calling An et al. weights; QC vs ~5 hand masks (Dice) | threshold backend (same signature) |
| Fork on transfer | Good Dice → adopt DL. Poor → keep threshold for now, schedule nnU-Net fine-tune on accumulated corrected masks | — |
| Cell detection (DL) | Install StarDist **and** InstanSeg in QuPath; `detect_cells.groovy` → DAPI nuclei + cell expansion; compare counts on one slide | FITC-threshold-only IHC |
| **✎ Cell edit gate** | Manual add/delete/reclassify on sampled tiles in QuPath; log edit rate | — |
| Cell-type classify | QuPath trainable object classifier (neuron/endo/astro/microglia); thresholds as sanity check | — |
| Readouts | Add per-cell LYS241 intensity + %LYS241⁺ per type; keep %-area for LYS241/GFAP/IBA1 | — |

**✅ M2 acceptance:** lesion masks come from DL (or a documented fallback), and IHC yields both object-level (per cell type) and area-level numbers — still on one animal, still no atlas.

---

## Milestone 3 — Atlas / regions on both halves  ·  ~4–6 days (the finicky one)

| Task | Action | Risk |
|---|---|---|
| MRI→Allen | Install + run **AIDAmri** on the test animal (feed it the corrected lesion mask); inspect overlay | AIDAmri setup is fiddly → ANTs SyN fallback via `antspyx` |
| Region labels (MRI) | Map Allen regions onto lesion/hemisphere | — |
| IHC→Allen | **ABBA** in Fiji: DeepSlice auto-position (off DAPI) → spline → BigWarp refine → import regions into QuPath | ABBA learning curve; budget manual refine on 5d sections |
| Per-region IHC | Re-export measurements grouped by Allen region | — |

**✅ M3 acceptance:** both modalities carry Allen region labels; IHC numbers are now per-region.

---

## Milestone 4 — Compartments + join + full tidy table  ·  ~3–4 days

| Task | Action | Output |
|---|---|---|
| Compartments | `compartments.py`: core = corrected mask; peri = dilation ring (width from config); contra = atlas-midline mirror | per-animal label image in Allen space |
| Transfer to IHC | Pull MRI compartment labels onto IHC sections via shared atlas; histo-proxy fallback for 1h/3h/6h (flag `compartment_method`) | per-cell compartment tags |
| Join | `join.py`: merge on (animal, region, compartment, hemisphere) | long table |
| Provenance | Add model_version, edited, edit_dice, reviewer, qc_flag columns | full schema |

**✅ M4 acceptance:** the full `animal × region × compartment × cell_type × measure` tidy table for the test animal — the real deliverable shape.

---

## Milestone 5 — Batch + QC  ·  ~3–5 days

| Task | Action |
|---|---|
| Batch driver | Loop all animals/configs; resume-safe (skip completed stages) |
| Controls | Run the 3 control animals through the same path |
| QC outputs | Per-animal overlay PNGs (lesion on T2, ABBA alignment, edit-Dice summary); a QC index table |
| Sanity stats | Object-vs-area agreement check; lesion volume vs timepoint trend |

**✅ M5 acceptance:** one command produces the full multi-animal table + a QC folder; ready to hand to R for stats.

---

## Milestone 6 (later, only if reused) — Active-learning loop  ·  ~3–5 days
Collect corrected masks/cells → fine-tune nnU-Net (lesions) and StarDist/InstanSeg (cells) → version models → confirm edit-rate drops. Do this only once a second protocol justifies it.

---

## Critical path & timeline (focused days)

```
M0 ▓ (1)        setup/audit
M1 ▓▓▓ (2–3)    ◄── v1 "first version up" lands here (~day 3–4)
M2 ▓▓▓▓ (3–5)   DL + cell-level
M3 ▓▓▓▓▓ (4–6)  atlas  ◄── biggest schedule risk
M4 ▓▓▓ (3–4)    compartments + join + full table
M5 ▓▓▓▓ (3–5)   batch + QC
M6 ▓▓▓ (later)  fine-tuning
```
- **v1 in ~3–4 days.** Usable per-animal table (regions + compartments) in **~2.5–3.5 weeks**. Fully batched + QC'd in **~3–4 weeks** of focused work.
- Only M0→M1 are strictly serial-critical to v1. M2 and the *start* of M3 (installing AIDAmri/ABBA) can overlap once v1 exists.

---

## Decision gates (write the outcome into the config when you hit each)
1. **End of M0:** does the pretrained lesion model run? → sets whether M1 uses threshold or DL.
2. **M2:** DL Dice vs manual ≥ your threshold? → adopt DL or stay threshold + queue fine-tuning.
3. **M3:** AIDAmri registration QCs cleanly on a large 5d lesion? → AIDAmri or fall back to ANTs SyN.
4. **M3:** DeepSlice auto-alignment acceptable on fluorescence/DAPI? → auto+refine or mostly-manual ABBA.
5. **Anytime:** anti-IgG specificity resolved? → gates whether core LYS241 numbers are interpretable (not whether code runs).

---

## Risks & fallbacks

| Risk | Trigger | Fallback |
|---|---|---|
| Pretrained model doesn't transfer | Low Dice on your data | Threshold/classifier now; fine-tune nnU-Net on corrected masks later |
| `antspyx` / AIDAmri arm64 pain | Install fails on M1 | Run MRI registration step in a Linux container/VM, or ANTs CLI; v1 doesn't need it |
| ABBA fluorescence alignment shaky | DeepSlice misplaces sections | Manual ABBA positioning; only a subset needs it |
| .vsi files huge (~3 GB ×160) | Disk/RAM pressure | Work off the pyramid's lower-res levels for detection QC; process per-section, not whole-batch in RAM |
| Scope creep delays v1 | Tempted to add atlas/DL early | Hold the line: v1 = no atlas, no DL; they are M2–M4 |

---

## Immediate next actions (today)
1. Create the repo skeleton + `environment.yml`; build and smoke-test the env.
2. Pick the one test animal; fill `config/animals/<id>.yml` (paths, panel, channel map).
3. Run the MRI header sanity check and open one `.vsi` in QuPath to lock the channel map.
4. Attempt An et al. model inference once → record the M0 decision-gate outcome.
5. Start Milestone 1A.
