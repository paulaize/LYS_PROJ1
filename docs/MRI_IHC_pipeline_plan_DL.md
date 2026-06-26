# Ischemic-stroke MRI + IHC pipeline — deep-learning-first design (with human-in-the-loop correction)

**Goal.** Same end-to-end workflow as the baseline plan — lesion volume from mouse T2 MRI, multiplexed IHC quantification of LYS241 across cell types, both mapped into the Allen CCFv3, one tidy table — but rebuilt so that **every perception task is a deep-learning model**, and **every model output passes through a human correction gate** before it is trusted. The correction edits are versioned and recycled as training data, so the models improve each protocol (active learning).

**Current status note.** This is not the active v1 implementation plan. v1
should finish with one animal, no atlas, no DL, and a manually reviewed MRI
lesion mask. The corrected v1 masks become reference data for this later DL
track.

**Implementation guardrail for Codex:** do not implement tasks from this DL plan
unless `docs/v1_next_session_todo.md` or `docs/development_roadmap.md`
explicitly activates the corresponding milestone. During v1, this document is
reference-only; the active implementation remains one animal, no atlas, no DL,
no compartments, and no batch processing.

**What "deep-learning-first" means here (and what it does *not*).** DL replaces every step that involves *interpreting pixels*: brain extraction, lesion segmentation, cell/nucleus detection, cell-type classification, and section-to-atlas alignment. DL does **not** replace the deterministic steps — physical voxel-volume integration, geometric core/peri/contra construction, midline mirroring, the region/compartment join, and table assembly. Forcing a network onto those would add opacity and error for no gain. So this is *deep-learning-first*, not *deep-learning-only*, and that distinction is deliberate.

---

## 0. Architecture at a glance (◆ = DL model, ✎ = human correction gate)

```
 MRI ┌──────────────────────────────────────────────────────────────────────────────┐
     │ Bruker→NIfTI → ◆brain-extract → ◆LESION SEG → ✎BORDER EDIT → edema → volume    │
     │   (brkraw)      (antspynet)      (An et al./   (ITK-SNAP/    (Swanson/  (Σ area │
     │                                   nnU-Net)      3D Slicer)    Jacobian)  ×0.5mm)│
     │                                       │              │                          │
     │                                       └── corrected mask → ◆MRI→Allen reg ──────┤
     │                                                            (AIDAmri/ANTs/        │
     │                                                             learned)             │
     │                                              core/peri/contra (geometric+mirror) │
     └──────────────────────────────────────────────────────────────┬─────────────────┘
                                                                      ▼  (Allen + compartment labels)
 IHC ┌──────────────────────────────────────────────────────────────────────────────┐
     │ .vsi → QuPath → ◆DeepSlice+ABBA reg → ◆CELL SEG → ✎CELL EDIT → ◆classify →      │
     │ (BioFormats)     (→Allen CCF)          (InstanSeg/  (QuPath     (cell type)      │
     │                                         StarDist)    review)                     │
     │                                              │            │          │           │
     │                                         per-cell + per-region measurements ──────┤
     └──────────────────────────────────────────────────────────────┬─────────────────┘
                                                                      ▼
                          JOIN on (Allen region, compartment, hemisphere)  →  tidy long table
                                              │
                  every ✎ edit ──────────────┴──────────────► fine-tuning set (active learning)
```

Both tracks still reach the **Allen CCF** independently and join through it (the bridge principle is unchanged — DL doesn't make a direct MRI↔IHC warp any more trustworthy).

---

## 1. Human-in-the-loop design principle (the core of this version)

Each DL model is treated as a **fast first draft, never the final word.** After every model runs, a human reviews, edits if needed, and the system records:

- the **corrected** mask/detections (the ground truth that goes forward),
- an **edit-magnitude metric** — Dice (or IoU) between the model output and the corrected version, per case,
- a **reviewer + timestamp** for provenance.

The edit-magnitude does triple duty: (1) per-case QC flag (a heavily edited case is suspect biology *or* a model miss), (2) a model-health signal over time (edits should shrink as the model fine-tunes), and (3) a **prioritization queue** — cases with the largest edits are the highest-value additions to the next fine-tuning round (this is active learning). Target: a human touches *fewer* cases each protocol as the models converge on your specific data.

> Two correction gates exist: **✎ lesion border editing** (MRI, your explicit request, §2.3) and **✎ cell-detection editing** (IHC, §3.3). Both follow the same record-edit-recycle pattern.

---

## 2. MRI track (deep-learning-first)

### 2.1 Inputs
Lesion-volume workhorse = **Scan 2, RARE T2** (256², FOV 17.92 mm → 70 µm in-plane; 18 slices × 0.5 mm; anisotropic → 2.5-D). T2\* (Scan 3) optional for hemorrhage; TOF (Scan 4) optional for vessels. Bruker→NIfTI via brkraw (you have it).

### 2.2 Brain extraction (◆ DL)
| Option | Tool | Notes |
|---|---|---|
| **A. Learned rodent brain extraction** (recommended) | `antspynet` rodent brain extraction (U-Net) | Robust mouse skull-strip; one call |
| B. Skip it | — | The An et al. lesion model needs little preprocessing and can run on near-raw scans; brain mask still useful later for normalization/mirroring |
| C. Classical | Otsu + morphology | Non-DL fallback |

Run **N4 bias correction** (SimpleITK/ANTs) regardless — surface-coil RARE has strong intensity gradients; the lesion model is more stable on flattened images even if it tolerates raw.

### 2.3 Lesion segmentation (◆ DL) → then ✎ **human border correction**

**Segmentation options:**

| Option | Model | Training cost | Fit to your data |
|---|---|---|---|
| **A. Pretrained mouse-T2 model** (recommended first) | An et al. 2023 (3D U-Net variant for mouse T2w stroke); open weights + Zenodo dataset | **Zero** — apply as-is | Domain shift risk: built on MCAO/other scanners; your thrombin + surface-coil RARE + anisotropic voxels may differ → must QC |
| **B. nnU-Net, fine-tuned** | nnU-Net self-configuring 3D framework, initialized from A or trained on your accumulating masks | Low–med (handful of annotated volumes via transfer learning) | Best long-term fit; nnU-Net is the default winner for biomedical 3D seg |
| C. RatLesNetv2 | Rodent T2w CNN | Med (requires training on your own data) | Only if A and B underperform |
| D. From-scratch | any U-Net | High; needs large n you don't have | Not worth it at your scale |

**Recommended path:** run **A** on every volume → QC against `3-5` hand-drawn
masks. If Dice is high, you're nearly done. If it drifts, move to **B** after
enough corrected masks accumulate. As a practical floor, `8-12` corrected
stroke masks can support a small transfer-learning attempt; `15-25` is a better
fine-tuning target. Training from scratch would require substantially more data
and is not the default plan. Either way, the model output is a *draft* lesion
mask.

**✎ Human border-correction gate (your explicit step).** Every draft mask is opened in a 3-D label editor; the human refines borders, fills holes, deletes spurious blobs, then saves. Tool options:

| Option | Tool | Pros | Cons |
|---|---|---|---|
| **A. ITK-SNAP** (recommended for ergonomics) | Loads NIfTI + mask as a segmentation layer; paintbrush, active-contour, 3-orthogonal-view editing | Purpose-built for exactly this; fast brush; free | Separate app from Fiji |
| **B. 3D Slicer** | Segment Editor (brush, scissors, islands, threshold-paint, smoothing) | Most powerful 3-D editing; scriptable | Heavier UI |
| **C. Fiji + Labkit** | Edit labels slice-by-slice | **Keeps your collaborators in Fiji** | Less smooth for true 3-D borders |
| **D. napari** | Labels layer + brush, in Python | **Native to the Python orchestration**; scriptable correction logging | Editing ergonomics below ITK-SNAP |

Recommendation: **ITK-SNAP or 3D Slicer** for the person doing careful border work; offer **Labkit** so the lab can stay in Fiji; use **napari** if you want the correction step embedded in the Python loop with automatic Dice logging. Whichever you pick, the gate must: save the corrected mask to a versioned path, compute Dice(draft, corrected), and tag the case `edited / unedited` with the editor's name. Those corrected masks are the fine-tuning set for option **B** next round.

### 2.4 Edema correction & volume (deterministic, off the corrected mask)
- **Swanson/indirect** corrected volume (headline number, matches your prior work), and/or **atlas-Jacobian** edema map (Koch et al. 2019) once registration runs.
- Volume = Σ(corrected lesion area per slice) × 0.5 mm. Carry raw + corrected + per-slice profile.
- Note: edema correction runs on the **corrected** mask, so the human edit propagates into the final number — which is the point.

### 2.5 MRI → Allen registration (◆ DL-assisted, or classical)
| Option | Approach | Notes |
|---|---|---|
| **A. AIDAmri** (recommended start) | Purpose-built mouse-T2→Allen, stroke-validated, ships MRI-resolution label atlas | Least custom code; inherits edema map |
| B. ANTs SyN (`antspyx`) to a mouse MRI template linked to Allen | Classical nonlinear; gold-standard reliability on lesioned brains | You assemble template↔Allen link |
| C. Learned registration (SynthMorph / VoxelMorph-style) | DL deformable registration | Emerging; **less reliable than SyN on large lesions** — keep as experiment, not primary |

Recommendation: classical/AIDAmri stays primary here — lesion-induced deformation is exactly where learned registration is least trustworthy, so this is a place to resist "DL everywhere." Feed the registrar the **corrected lesion mask** so it can down-weight lesioned tissue (registration in the presence of a lesion is only reliable when informed by a lesion mask).

### 2.6 Compartments (deterministic, from the corrected mask + registration)
- **Core** = corrected lesion mask. **Péri** = dilation ring (width a tunable parameter, e.g. 0.5–1.0 mm) minus core. **Contra** = core/peri mirrored across the atlas midline. Optionally tag each voxel with its Allen region. Output a per-animal label image in Allen space → travels to the IHC side.

---

## 3. IHC track (deep-learning-first)

### 3.1 .vsi + channel map (QC gate, unchanged)
Bio-Formats reads `.vsi` natively in QuPath/Fiji — keep as working format.
**Confirm channel→marker map per panel** (Panel A:
DAPI/NeuroTrace/Podo/FITC; Panel B: DAPI/IBA1/GFAP/FITC). For v1, Panel A
NeuroTrace is accepted as 640/660 deep-red, so FITC spectral bleed-through is
not flagged; Paul may still do a later full fluorochrome audit. Anti-IgG
specificity is resolved as anti-human IgG-FITC specific to humanized LYS241. No
prior QuPath/IgG-FITC positivity threshold exists for these images, so
threshold calibration/review remains a pipeline task. DL changes none of the
remaining threshold QC requirements.

### 3.2 Section → Allen registration (◆ DL)
| Option | Flow | Notes |
|---|---|---|
| **A. DeepSlice (◆) + ABBA, manual BigWarp refine** (recommended) | DeepSlice CNN auto-predicts plane/angle → ABBA affine+spline → human refine on DAPI/white-matter | The DL-native registration path; keeps everyone in QuPath/Fiji |
| B. ABBA manual | Hand-position | For torn / lesion-distorted sections |

DeepSlice favors brightfield; drive it off the DAPI channel and budget manual refinement on the 5d sections. This *is* a human-in-the-loop registration gate (the BigWarp refine), checked by two people on a subset.

### 3.3 Cell/nucleus segmentation (◆ DL) → then ✎ **human cell correction**
| Option | Model (in QuPath) | Notes |
|---|---|---|
| **A. InstanSeg** (recommended on M1) | Fluorescence/multiplex-aware; Apple-Silicon GPU; strong benchmarks | Verify whole-slide tiling vs a region (some WSI under-detection reports) |
| B. StarDist | Battle-tested DAPI nuclei; fluorescence model | TensorFlow; CPU-slower on Mac |
| C. Cellpose | Irregular cells | Heavier setup |

Detect nuclei on **DAPI**, expand a few µm for cytoplasmic/membrane markers.

**✎ Human cell-correction gate (mirror of the lesion gate).** In QuPath, the reviewer adds missed cells, deletes false positives, and fixes obvious mis-segmentations on a sampled set of tiles/regions; QuPath logs the edits. These corrected detections (a) become local ground truth and (b) form a fine-tuning set to specialize StarDist/InstanSeg to your staining — same record-edit-recycle pattern, same Dice/edit-rate logging.

### 3.4 Cell-type classification (◆ DL or simpler)
| Option | Approach | Notes |
|---|---|---|
| **A. QuPath object classifier (trainable)** (recommended) | Train on a few annotated cells per type (neuron/endo/astro/microglia) | Handles overlap; the corrections from §3.3 seed it |
| B. Threshold per channel | Simple, transparent | Good baseline / sanity check |
| C. DL patch classifier | A small CNN on cell crops | Overkill unless thresholds fail |

Reminder: **GFAP/IBA1 are morphological** → prefer **area-fraction** (intensity threshold, not DL) over nucleus-based counts; use object counts only for soma density.

### 3.5 Readouts (deterministic measurement)
Same as baseline, both levels, matched to biology:
- **Object-level**: per-cell-type IgG-FITC mean intensity + % IgG-FITC-positive
  of each type; DAPI density (nuclei/mm²). LYS241 association is carried by
  provenance, not by renaming the measurement.
- **Area-level**: % positive area for IgG-FITC, GFAP, IBA1 per
  region×compartment.
Report both; their disagreement is itself a QC signal.

### 3.6 Compartments in IHC
Atlas regions from ABBA (per cell). Lesion-derived core/peri/contra pulled in from the **MRI corrected** compartment masks via the shared atlas. For animals without MRI (1h/3h/6h), fall back to a histology-intrinsic core proxy (IgG-leakage / NeuroTrace-loss footprint) + mirrored contra, **flagged as a different `compartment_method`** so the two definitions aren't silently pooled.

---

## 4. Integration & output table (deterministic, unchanged)
Join on `(animal, Allen_region, compartment, hemisphere)`. One **long** row per `animal × region × compartment × cell_type × measure`, with `compartment_method`, `n_cells`, `area_mm2`, and a `qc_flag` (carrying the edit-magnitude signal). Schema identical to the baseline plan — DL changes how cells/lesions are *found*, not how the table is *built*.

| key columns | measure columns | provenance columns |
|---|---|---|
| animal_id, timepoint, panel, hemisphere, allen_region, compartment, compartment_method, cell_type | measure, value, unit, n_cells, area_mm2 | model_version, edited(bool), edit_dice, reviewer, qc_flag |

The provenance columns are new vs the baseline and matter here: with DL in the loop you want every number traceable to *which model version* produced it and *whether a human edited it*.

---

## 5. The active-learning loop (what makes "DL-first" pay off)
1. Models run → drafts.
2. Humans correct at the ✎ gates → corrected ground truth + edit metrics.
3. Corrected cases (prioritizing high-edit ones) accumulate into per-task fine-tuning sets.
4. Periodically fine-tune (nnU-Net for lesions; StarDist/InstanSeg for cells) → new model version.
5. Edit-rate should fall each cycle; when it plateaus near zero, the human gate becomes a light spot-check rather than full review.

This is the mechanism that turns the upfront DL cost into a declining per-protocol cost — and it's why the correction step you asked for is not just QC, but the engine of improvement.

---

## 6. Automation architecture
**Scripted (no clicks):** brkraw conversion, N4, brain extraction, lesion inference (A/B), edema/volume, registration batches, QuPath/ABBA headless runs of saved detection+classifier scripts, table assembly, **and the edit-logging/fine-tuning bookkeeping**. Orchestrate QuPath via CLI/Groovy or `paquo`; orchestrate the MRI DL with the model's own inference script wrapped in Python.

**Interactive (the gates):** channel-map confirmation; **✎ lesion border edit**; ABBA refine; **✎ cell edit**; classifier spot-check; anti-IgG resolution. Everything heavy is automated; the *judgment* lives in the GUIs your lab knows (Fiji/QuPath) plus ITK-SNAP/Slicer/napari for 3-D mask editing.

A per-animal YAML config (paths, panel, timepoint, channel map, peri-ring width, **model versions**) drives one Python entry point. New protocol = new config + possibly a fine-tune, not new code.

---

## 7. Build order (DL-first, time-boxed)
**Phase 1 — prove the DL spine after v1 exists.** MRI: brkraw → N4 →
**An et al. inference** → open mask in ITK-SNAP/Slicer/napari, hand-correct →
volume. IHC: one `.vsi` → QuPath → **InstanSeg** on DAPI → manual fix a few
tiles → % area + per-cell IgG-FITC/LYS241-associated readout. No atlas yet.
Mini table. *Confirms whether pretrained models transfer to your data before
you build anything around them.*

**Phase 2 — atlas on both halves.** AIDAmri (MRI→Allen) + DeepSlice/ABBA (IHC→Allen). Regions enter the table.

**Phase 3 — compartments + edema + join + provenance.** Core/peri/contra, Swanson correction, MRI→IHC compartment transfer, the join, edit-logging columns.

**Phase 4 — close the active-learning loop.** Fine-tune nnU-Net + StarDist/InstanSeg on the corrected masks/cells from Phases 1–3; batch all animals; YAML configs; QC dashboards. This is where the DL investment compounds for the *next* protocol.

Stop where accuracy is good enough for the biology. For a 10-animal exploratory study, Phases 1–3 with solid correction gates are likely the right stopping point; Phase 4 is the payoff when this becomes a recurring assay.

---

## 8. QC gates
1. NIfTI header/spacing correct (0.07×0.07×0.5 mm).
2. Channel map confirmed; Panel A NeuroTrace/FITC spectral overlap ruled out.
3. N4 flattening visibly OK.
4. Pretrained lesion model QC'd vs `3-5` corrected manual masks (Dice)
   **before** trusting it.
5. **✎ lesion borders reviewed/edited; edit-Dice logged.**
6. MRI→Allen overlay inspected (esp. large 5d lesions).
7. DeepSlice/ABBA alignment double-checked on a subset.
8. Cell detection: InstanSeg vs StarDist agreement on one slide.
9. **✎ cell detections reviewed/edited on sampled tiles; edit-rate logged.**
10. anti-IgG specificity resolved as anti-human IgG-FITC specific to LYS241.
11. Object vs area readouts broadly agree.
12. Model-version + edited-flag present on every output row.

---

## 9. Tools / dependencies
| Layer | Tool | Role |
|---|---|---|
| Bruker IO | brkraw | raw → NIfTI |
| MRI brain extract | `antspynet` | DL rodent skull-strip |
| MRI lesion seg | **An et al. 2023 weights** (start) → **nnU-Net** (fine-tune); RatLesNetv2 (fallback) | DL lesion masks |
| MRI mask editing | **ITK-SNAP / 3D Slicer** (primary), Fiji-Labkit (lab-familiar), napari (Python-native) | ✎ border correction |
| MRI registration | AIDAmri / `antspyx` (SyN) | → Allen, edema |
| Histology IO | QuPath + Bio-Formats | `.vsi`, projects |
| Section→atlas | **DeepSlice** (◆) + ABBA + BigWarp | IHC → Allen |
| Cell seg | **InstanSeg** (M1-GPU) + StarDist (validation) | DL detection |
| Cell classify | QuPath trainable object classifier | by marker |
| Atlas | BrainGlobe API / Allen CCFv3 | common space |
| Orchestration | Python (`pathlib`, pandas, PyYAML), `paquo`, `subprocess` | glue, edit-logging, fine-tune bookkeeping |
| Stats (downstream) | R | mixed models, plots |

**Key references:** An et al. 2023 *Sci Rep* (mouse T2w DL lesion seg, open weights + Zenodo data); Valverde et al. (RatLesNetv2); Isensee et al. (nnU-Net); Koch et al. 2019 (atlas edema correction); Pallast et al. 2019 (AIDAmri); Chiaruttini et al. 2025 (ABBA+BraiAn); Carey et al. 2023 (DeepSlice); Goldsborough et al. 2024 (InstanSeg); Schmidt et al. 2018 (StarDist); Bankhead et al. 2017 (QuPath); Drieu et al. 2020 (thrombin model).

---

## 10. Where this plan deliberately resists DL
Being honest so you can defend the choices: **volume integration, geometric compartments, midline mirroring, the region/compartment join, and the table** stay deterministic — they're exact arithmetic/geometry where a network only adds error and opacity. **Atlas registration of lesioned brains** stays classical (SyN/AIDAmri) because learned deformable registration is least reliable exactly where deformation is largest. And **GFAP/IBA1 quantification** stays area-fraction, not cell-DL, because the biology is morphological. "Deep-learning-first" means DL owns every pixel-interpretation task and every one gets a human gate — not that DL is bolted onto steps that don't need it.

### Open items to finalize (same as baseline)
1. NeuroTrace variant/emission: accepted for v1 as 640/660 deep-red; later full
   fluorochrome audit optional.
2. Exact `.vsi` channel order per panel for new animals. The current
   `BD_08_5D` maps are in its animal YAML.
3. IgG-FITC positivity threshold calibration from configured controls / approved
   rule. There is no prior QuPath threshold to reuse.
4. AIDAmri wholesale vs leaner ANTs-only for MRI→Allen.
5. Péri-lesional ring width.
6. **Which 3-D mask editor** the lesion reviewer will standardize on (ITK-SNAP / 3D Slicer / Fiji-Labkit / napari).
