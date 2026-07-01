# Development roadmap — stroke MRI + IHC pipeline (fast v1 first)

**How to read this.** Milestones are ordered for *fastest time-to-a-working-number*. v1 (Milestone 0+1) is a thin end-to-end spine on **one animal** using the simplest reliable method at each step — no atlas, no DL yet — so you have real outputs in ~3–4 focused days. Everything after v1 is a **drop-in upgrade behind a fixed interface**, so nothing built early gets thrown away. Estimates are rough focused-days for one person; treat them as relative, not contractual.

**Guiding principles for a fast v1**
1. **Spine before polish.** Get NIfTI→volume and .vsi→measurement→one table working before improving any single step.
2. **Fix interfaces, swap backends.** Define `segment_lesion()`, `detect_cells()`, etc. once; v1 fills them with threshold/built-in methods, v2 swaps in DL — same signature, no caller changes.
3. **Defer the atlas.** Region/compartment labels are Milestone 3–4. v1 reports whole-section / whole-hemisphere numbers. This removes the single biggest time sink (AIDAmri + ABBA setup) from the critical path to first output.
4. **Don't let model wrangling block the spine.** If the pretrained lesion model isn't runnable in an hour, v1 uses thresholding and the model becomes a v2 task.
5. **Human gate from day one** — even v1's lesion mask gets a manual edit pass; that's cheap and it's the habit the whole design depends on.

**Current v1 MRI decision.** The threshold mask is an untrusted draft, not the
scientific result. For `BD_08_5D` it can select bright peripheral artifact
instead of the image-right isocortical lesion. Do not block v1 on adding an
isocortex ROI, Allen registration, or DL. If the draft is bad, the reviewer
erases/redraws it, and the corrected mask is the v1 source of truth and future
model-reference data.

**Current v1 IHC threshold decision.** No approved QuPath/IgG-FITC positivity
threshold exists yet. This project is the first quantitative analysis of these
IHC images, so v1 must build the threshold-calibration/review workflow rather
than requiring Paul or the team to provide a number from previous QuPath work.
Exploratory threshold sweeps can be generated, but final positive-area rows
need an approved threshold recorded in config/provenance.

---

## Repo & data layout (set up in Milestone 0)

```
LYS_PROJ1/
├── env/
│   └── environment.yml
├── config/
│   ├── pipeline.yml            # global params (peri-ring width, thresholds, model versions)
│   └── animals/
│       └── BD_08_5D.yml        # current v1 animal: paths, timepoint, panel, channel map
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
# env/environment.yml  — reference/export target only; current work uses lys-bbb
name: lys-bbb
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

DL training is not a local v1 requirement. For the RatLesNetV2 branch, keep the
local `lys-bbb` env for dataset preparation/tests and run actual training on a
cloud GPU runtime. If PyTorch is ever used locally on Apple Silicon, use MPS
when available rather than CUDA.
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
| Repo + env | Use repo skeleton above with the existing `lys-bbb` env | `import nibabel, SimpleITK, napari, pandas` all succeed |
| Pick the test animal | Current v1 uses `BD_08_5D`, a 5d stroke animal with MRI + both IHC panels | Paths recorded in `config/animals/BD_08_5D.yml` |
| MRI header sanity | Load the T2 NIfTI; verify spacing = 0.07×0.07×0.5 mm, orientation | `io.py` prints correct voxel size; a slice renders |
| Channel map | Open one `.vsi` in QuPath; confirm channel→marker per panel; record NeuroTrace emission | Written into the animal config; Panel A NeuroTrace accepted as 640/660 deep-red for v1 |
| Model availability check | Later only: check An et al. mouse model/resources and RatLesNetV2 transfer-learning path | **Not a v1 blocker.** Runs → DL is viable for v2. Doesn't run easily → keep manual-corrected masks and train/fine-tune only after enough masks accumulate |
| RatLesNetV2 branch setup | `dl-ratlesnetv2-finetune`: prepare corrected T2w/manual-mask folders, track external mouse dataset overlap/geometry, and print cloud finetuning commands | Independent DL track exists without changing v1 |

**Updated:** anti-IgG specificity is resolved (Paul, 2026-06-17): the secondary
is anti-human IgG-FITC and LYS241 is humanized Glunomab. Keep IgG-FITC naming
and provenance flags; remaining IHC gates are threshold calibration and Panel A
NeuroTrace/FITC spectral bleed-through.
There is no prior QuPath threshold to reuse; calibration is a v1 task.

---

## Milestone 1 — v1 thin spine, one animal, no atlas  ·  ~2–3 days  ·  CRITICAL PATH → **this is "first version up"**

### 1A. MRI → lesion volume (~1 day)
| Task | Action | Output |
|---|---|---|
| Preprocess | `preprocess.py`: N4 bias correction + brain mask (Otsu/morphology is fine for v1) | clean volume + mask |
| Segment (threshold backend) | `segment.py` method='threshold': voxels > contra-mean+k·SD inside ipsi hemisphere; connected components; size filter | untrusted draft lesion mask |
| **✎ Edit** | `edit.py`: open volume+draft in napari (or ITK-SNAP); hand-correct or redraw; save; log Dice(draft,corrected) | authoritative corrected mask + edit record |
| Volume | `volume.py`: Σ(area)×0.5 mm; raw + per-slice profile | volume number (mm³) |

### 1B. IHC → basic measurement (~1 day)
| Task | Action | Output |
|---|---|---|
| QuPath project | Create project, import the `.vsi` (Bio-Formats) | openable project |
| Calibration | Build an exploratory IgG-FITC threshold helper from controls/image statistics; reviewer signs off before final export | approved threshold provenance |
| % positive area (no segmentation yet) | `detect_cells.groovy`/export v1 = approved IgG-FITC intensity threshold inside reviewed tissue ROI → % positive area | measurement CSV |
| Export + ingest | `export_measurements.groovy` → `ingest.py` parses to rows | tidy IHC rows |

> **Why %-area first:** it needs no cell segmentation, so it proves the IHC spine in hours. Cell-level detection is Milestone 2.

### 1C. Glue (~0.5 day)
| Task | Action | Output |
|---|---|---|
| Entry point | `run_animal.py`: read config → run 1A + 1B → write a minimal table | `outputs/BD_08_5D/BD_08_5D_v1.csv` |
| Minimal table | Columns: animal, timepoint, panel, modality, measure, value, edited, edit_dice | **v1 deliverable** |

**✅ v1 acceptance:** one command on one animal produces a lesion volume (human-corrected) **and** an IHC measurement CSV, joined into one table. No atlas, no regions, no DL required.

**v1 deliberately omits:** atlas regions, core/peri/contra compartments, cell-level counts, multi-animal batch, DL models. All added next, behind the interfaces above.

---

## Milestone 2 — Quality swap: DL + cell-level + correction gates  ·  ~3–5 days

| Task | Action | Replaces |
|---|---|---|
| DL lesion backend | Implement `segment.py` method='dl' using the best proven draft backend: An et al. inference, RatLesNetV2 public-mouse + LYS fine-tune, or nnU-Net; QC vs corrected masks before trusting it | threshold backend (same signature) |
| Fork on transfer | Good Dice → adopt DL as draft backend. Poor → keep manual-corrected masks as truth and fine-tune only after enough masks accumulate | — |
| Cell detection (DL) | Install StarDist **and** InstanSeg in QuPath; `detect_cells.groovy` → DAPI nuclei + cell expansion; compare counts on one slide | FITC-threshold-only IHC |
| **✎ Cell edit gate** | Manual add/delete/reclassify on sampled tiles in QuPath; log edit rate | — |
| Cell-type classify | QuPath trainable object classifier (neuron/endo/astro/microglia); thresholds as sanity check | — |
| Readouts | Add per-cell IgG-FITC intensity + % IgG-FITC-positive per type; keep IgG-FITC/GFAP/IBA1 area fractions | — |

**✅ M2 acceptance:** lesion masks come from DL (or a documented fallback), and IHC yields both object-level (per cell type) and area-level numbers — still on one animal, still no atlas.

Mask-count expectation: one corrected mask is enough to test v1 mechanically;
`3-5` corrected masks are enough to evaluate a pretrained model; `8-12` can
support a small transfer-learning attempt; `15-25` is a better fine-tuning
target. Training from scratch would need substantially more data and is not the
default plan.

RatLesNetV2-specific note: the branch `dl-ratlesnetv2-finetune` contains local
dataset conversion, public mouse dataset notes, and a cloud-oriented
finetuning loop. The intended sequence is rat-trained RatLesNetV2 weights →
public mouse native-space manual masks → LYS manual masks → held-out LYS
validation. Predictions from that model are still drafts. They become
scientific data only after the same manual correction/provenance path used by
v1. Details are in `docs/ratlesnetv2_external_datasets.md`.

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
Collect corrected masks/cells → fine-tune RatLesNetV2 or nnU-Net (lesions) and
StarDist/InstanSeg (cells) → version models → confirm edit-rate drops. Public
mouse datasets may support lesion-model pretraining/adaptation, but held-out
LYS cases remain the target-domain validation set. Do this only once a second
protocol justifies it.

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
1. **End of M1:** did one animal run end-to-end with a documented corrected MRI
   mask? → v1 can proceed even if the draft threshold mask was poor.
2. **M2:** DL Dice vs manual ≥ your threshold? → adopt DL as the draft backend
   or keep manual-corrected masks as truth and queue fine-tuning.
3. **M3:** AIDAmri registration QCs cleanly on a large 5d lesion? → AIDAmri or fall back to ANTs SyN.
4. **M3:** DeepSlice auto-alignment acceptable on fluorescence/DAPI? → auto+refine or mostly-manual ABBA.
5. **Anytime:** threshold/spectral gates resolved? → specificity is resolved,
   but threshold calibration and Panel A NeuroTrace/FITC bleed-through still
   gate whether IgG-FITC numbers are interpretable.

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

## Immediate next actions

The environment and repository skeleton are now in place. The current audited
task order, including the IHC tissue-area correction and required provenance,
is maintained in [`v1_next_session_todo.md`](v1_next_session_todo.md).

Do not start atlas or DL work until that v1 completion gate passes.
