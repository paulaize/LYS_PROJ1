# Ischemic-stroke MRI + IHC analysis pipeline — design plan

**Goal.** A semi-automated, reusable Python-orchestrated workflow that (1) measures ischemic lesion volume from Bruker→NIfTI mouse T2 MRI, (2) quantifies multiplexed fluorescence IHC (LYS241 distribution across neurons, endothelium, astrocytes, microglia) at cell- and area-level, (3) maps both modalities into the Allen Mouse Brain CCFv3 so they share a coordinate frame, and (4) emits one tidy table `animal × region × compartment × cell type × signal`, while keeping QuPath + Fiji as the tools your collaborators actually open.

**Study shape (current protocol).** ~10 stroke animals + 3 controls. Cross-sectional design: each animal is one timepoint (1h, 3h, 6h, 24h, 48h, 5d after thrombin-model induction, Drieu et al. 2020). MRI exists only for 24h/48h/5d (later timepoints where T2 lesion is established). Per animal: 8 IHC sections × 2 marker panels; ~18 coronal T2 slices.

---

## 0. Architecture at a glance

```
              ┌────────────────────────── ALLEN CCFv3 (common space) ──────────────────────────┐
              │                                                                                  │
   MRI track  │  Bruker  →  NIfTI  →  preprocess  →  lesion seg  →  edema corr  →  MRI→Allen reg │
   (T2 RARE)  │  (brkraw)   (you)     (denoise,        (semi-auto,    (Swanson /     (AIDAmri /  │
              │                        N4, strip)       Fiji QC)       atlas-based)   ANTs)       │
              │                                              │                          │         │
              │                                       lesion mask in Allen ────────────►│         │
              │                                                                          ▼         │
              │                                              core / peri / contra compartments    │
              │                                              (geometric + mirrored)               │
              │                                                                                    │
   IHC track  │  .vsi  →  QuPath project  →  ABBA reg →  cell seg  →  classify  →  per-cell &      │
              │ (BioFormats)  (per section)  (→Allen)    (InstanSeg/  (by marker)   per-region     │
              │                                           StarDist)                  measurements   │
              │                                              │                          │           │
              └──────────────────────────────────────────────┼──────────────────────────┼──────────┘
                                                              ▼                          ▼
                                              ┌─────────────────────────────────────────────────┐
                                              │  JOIN on (Allen region, compartment) via atlas  │
                                              │  →  tidy long table  →  stats (R) + overlays     │
                                              └─────────────────────────────────────────────────┘
```

**Bridge principle.** MRI and IHC are *never* registered directly to each other for quantification. Each is registered to the Allen CCF; the atlas region label + the lesion-derived compartment label are the join keys. Direct per-animal MRI↔IHC overlay is a separate, figure-only deliverable (§6) built from the shared atlas coordinates.

---

## 1. Common-space strategy (decide once, everything hangs off this)

| Option | What it is | Pros | Cons | Verdict |
|---|---|---|---|---|
| **A. Atlas-bridge** (recommended) | MRI→Allen and IHC→Allen independently; join on region+compartment | Robust; gives region names for free; each modality uses its native best tool; future-proof | Overlay of the two is approximate (atlas-mediated) | **Default** |
| B. Direct MRI↔IHC per animal | Warp each IHC section onto its matching MRI slice | True per-animal overlay | Fragile: different sampling, histology shrinkage/tears, no 1:1 slice correspondence; hard to automate | Use only for selected figures |
| C. Hybrid | A for quantification + B (atlas-seeded) for figures | Best of both | More build effort | **Adopt once A works** |

Adopt **A now, layer C later.** Compartment definitions (core/peri/contra) are computed in MRI/Allen space and *carried into* IHC via the shared atlas, so the lesion geometry measured on MRI labels the histology even though histology can't see the lesion directly.

---

## 2. MRI pipeline (lesion volume + atlas mapping)

### 2.1 Which scan does what
- **Scan 2, RARE `T2_haute_resolution_Turbo`** → **the lesion-volume workhorse.** T2 hyperintensity = edema/infarct. 256², FOV 17.92 mm → **70 µm in-plane**, 18 slices × **0.5 mm**, no gap. Anisotropic: treat as 2.5-D (volume = Σ slice-area × 0.5 mm), not isotropic 3-D.
- **Scan 3, FcFLASH `T2s_rapide`** (T2*) → optional **hemorrhage** readout (susceptibility hypointensity). Relevant: thrombin model can be partly hemorrhagic, and hemorrhage corrupts T2 lesion borders.
- **Scan 4, TOF angio** → optional **vessel/perfusion-territory** context (MCA patency). Not for volume.
- Scan 1 (localizer) → discard.

### 2.2 Preprocessing (Python, automated)
You already do Bruker→NIfTI (brkraw). Then, per volume:
1. **Reorient** to a canonical mouse orientation (RAS) and fix voxel spacing in the header (0.07×0.07×0.5 mm) — critical, downstream volume + registration trust this.
2. **Denoise** (optional): non-local means / MP-PCA. RARE is usually clean enough; skip unless borders look noisy.
3. **Bias-field correction**: N4 (ANTs/SimpleITK). Surface-coil RARE has strong intensity gradients that wreck any global threshold — **do this**.
4. **Skull-strip / brain mask**: AIDAmri's built-in mouse strip, or a simple Otsu+morphology, or `antspynet` rodent brain extraction. A mask is needed for both registration and contralateral normalization.

> **Tooling note:** AIDAmri (Aswendt Lab, Python, open source) bundles 2.1–2.4 *and* the Allen registration in one purpose-built mouse-stroke pipeline. Strong candidate to adopt wholesale for the MRI half rather than rebuild.

### 2.3 Lesion segmentation — options

| Option | Method | Effort | Accuracy | Keeps lab in Fiji? |
|---|---|---|---|---|
| **A. Relative threshold** | Mask = voxels > (contralateral-mean + k·SD) of T2 signal, k≈2–3, within ipsilateral hemisphere; connected-components; size filter | Low | Good baseline; sensitive to bias field (→ do N4 first) | Review in Fiji ✔ |
| **B. Pixel classifier** (recommended) | Fiji **Labkit** or **Trainable Weka Segmentation**, or QuPath pixel classifier: paint a few examples, apply to all | Low–med | Better borders; lab can retrain | Native Fiji ✔✔ |
| C. Mirror z-score map | Register hemisphere to its mirror, lesion = where ipsi ≫ contra; threshold the difference map | Med | Robust to global intensity; elegant | Map exported to Fiji ✔ |
| D. Deep learning | nnU-Net / a rodent-stroke U-Net | High (needs ~annotated set) | Best at scale | — |

**Recommendation:** **B as primary** (a Labkit/Weka classifier your collaborators can open, tweak, and re-run), with **A as an automatic first guess** that B corrects, and **C** as a cross-check. **Defer D** until you have ≥1 protocol's worth of curated masks to train on — then it becomes nearly free per-animal. Every mask gets a human QC pass in Fiji (§8).

### 2.4 Edema correction — options
Raw lesion volume overstates infarct because the ipsilateral hemisphere swells. Options:
- **A. Swanson / indirect formula** (you've used this): corrected = contralateral-hemisphere-derived estimate, slice-wise. Manual hemisphere tracing or auto from the brain mask + midline. Transparent, journal-standard.
- **B. Atlas-based correction** (Koch et al. 2019): nonlinear MRI→Allen warp; the Jacobian encodes swelling, giving voxel-wise edema and a deformation-corrected volume. Agreed with manual at r≈0.98 in MCAO. More principled, gives an edema *map*.
- **C. Both, report both** — A as the headline number, B as the swelling map + sensitivity check.

**Recommendation:** **A for the reported corrected volume** (simple, defensible, matches your prior work), **B as an add-on map** once atlas registration (§2.6) is running anyway — it's almost free at that point.

### 2.5 Volume computation
Corrected lesion volume = Σ over slices (corrected lesion area per slice) × slice thickness (0.5 mm). Carry both raw and edema-corrected. Also export per-slice areas (useful for the rostro-caudal lesion profile and for picking IHC-matching planes later).

### 2.6 MRI → Allen registration — options

| Option | Tool | Pros | Cons |
|---|---|---|---|
| **A. AIDAmri** (recommended start) | Python, Allen-targeted, stroke-validated, ships an MRI-resolution label atlas | Purpose-built for exactly this (Bruker mouse T2 → Allen, with lesions); least custom code | Opinionated; some setup; older codebase |
| B. ANTs + mouse template | Register T2 → a dedicated mouse **MRI** template (DSURQE/Turone/AMBMC) that is itself in/linked to Allen; compose transforms | Maximum control; SyN is gold-standard nonlinear; scriptable with `antspyx` | You assemble it; template↔Allen link must be sourced |
| C. Semi-manual per-slice | ITK-SNAP / `brainglobe` tools to label slices against atlas without full warp | Simplest; good enough if you only need region labels, not voxel warps | No edema map; coarser region boundaries |

**Recommendation:** Try **A (AIDAmri)** first — if its registration QCs well on your data, you inherit edema correction (§2.4-B) and Allen labels in one step. Keep **B (ANTs/`antspyx`)** as the fallback for animals where the lesion deformation defeats AIDAmri (large 5d lesions are the stress test). **Note the honest limitation:** with only 18 × 0.5 mm slices, the through-plane warp is coarse — region assignment is reliable; sub-region voxel precision in the slice direction is not. Don't over-claim z-resolution. (BrainGlobe's `brainreg` is built for dense light-sheet/2-photon volumes, not 18-slice thick MRI, so it's a poor fit here — prefer AIDAmri or ANTs.)

### 2.7 Compartments: core / périlésionnel / contralatéral (geometric + mirrored)
Computed in MRI/Allen space, then exported as masks:
- **Core** = the edema-corrected lesion mask.
- **Périlésionnel** = morphological dilation ring around the core, **width set in physical units** (e.g. 0.5–1.0 mm; expose as a parameter), minus the core, clipped to brain.
- **Contralatéral** = the core (and peri) **mirrored across the midline** into the healthy hemisphere — same shape/size, the internal control. Midline from the Allen registration (cleanest) or from a symmetry-plane fit on the brain mask.
- Optionally also tag each voxel with its **Allen region** so "peri in striatum" vs "peri in cortex" is separable.

Output: a per-animal label image (0=other,1=core,2=peri,3=contra-core,4=contra-peri) in Allen space → this is what travels to the IHC side.

### 2.8 Optional adjuncts
- **T2\*** hemorrhage mask (threshold hypointensity) → flag/exclude hemorrhagic voxels from "ischemic" volume and note hemorrhage as a covariate.
- **TOF** → MCA territory mask for "was the expected territory hit" QC.

---

## 3. IHC pipeline (QuPath-centric, Python-orchestrated)

### 3.1 .vsi handling + channel map (first QC gate)
- Each `.vsi`: Olympus VS-series, fluorescence, uint16, 20×, **0.325 µm/px**, pyramidal, ~3 GB, **4 channels (CZT 4×1×1)**, 8 sections/animal/panel + 1 overview. Bio-Formats reads these natively in QuPath and Fiji — no conversion needed; **keep `.vsi` as the working format** so collaborators open them as usual.
- **Required before anything:** a definitive **channel → marker table per panel.**
  - Panel A (10 slides): DAPI, **NeuroTrace** (neurons), **Podocalyxin** (endothelium), **Anti-IgG-FITC** (= LYS241).
  - Panel B (10 slides): DAPI, **IBA1** (microglia), **GFAP** (astrocytes), **Anti-IgG-FITC** (= LYS241).
  - Record each fluorophore's **emission**. NeuroTrace ships in blue/green/red/deep-red variants; if you're running the **green** NeuroTrace it overlaps FITC and will contaminate the LYS241 channel — confirm it's a non-green variant, or plan spectral unmixing.
- **Confounder flag (carry into interpretation):** anti-IgG-FITC detects deposited IgG. Post-stroke BBB breakdown lets **endogenous mouse IgG** leak into parenchyma (worst in the core). Unless the secondary is specific to the LYS241 isotype/species, "% LYS241" partly measures native IgG extravasation. Mitigations: isotype/species-specific secondary, a vehicle/no-drug control to subtract, and/or treating core-region LYS241 cautiously. **Resolve before quantifying.**

### 3.2 Section → Allen registration with ABBA — options

| Option | Flow | Pros | Cons |
|---|---|---|---|
| **A. ABBA + DeepSlice auto-position, manual BigWarp refine** (recommended) | QuPath project → ABBA in Fiji → DeepSlice sets plane/angle → affine+spline auto → refine on DAPI/white-matter landmarks → import back to QuPath | The field-standard for serial sections→Allen; keeps everyone in QuPath/Fiji; importable region annotations | DeepSlice is happiest on brightfield; on fluorescence, drive it off the DAPI channel and expect more manual refine |
| B. ABBA fully manual | Skip DeepSlice; position by hand | Full control on tricky/torn sections | Slower |
| C. QuickNII/VisuAlign (QUINT) | Alternative registrar | Mature | Leaves the QuPath ecosystem; more handoffs |

**Recommendation:** **A.** Register off DAPI (most atlas-like channel). After import, ABBA gives every QuPath detection an Allen region — the join key for the table. **Accuracy honesty:** ABBA alignment should be checked by two people on a subset (field practice); budget manual BigWarp time for the 5d sections where the lesion distorts anatomy.

### 3.3 Cell detection / segmentation — options

| Option | Tool (in QuPath) | Pros | Cons |
|---|---|---|---|
| **A. InstanSeg** (recommended on your M1) | QuPath InstanSeg extension | Fluorescence/multiplex-aware; **Apple-Silicon GPU** (fast on M1); strong benchmarks vs StarDist/Cellpose/Mesmer; one model → nuclei or whole-cell | Newer; whole-slide tiling needs a sanity check (some users report under-detection at WSI scale vs StarDist — verify on a region) |
| B. StarDist (DAPI) | QuPath StarDist extension | Battle-tested for nuclei; tons of docs; `dsb2018_heavy_augment` works well on fluorescence DAPI; ~0.5 µm/px sweet spot (you're at 0.325) | TensorFlow; CPU-slower on Mac; convex-only shapes |
| C. Cellpose | via QuPath extension / Python | Great on irregular cells | Heavier setup; GPU ideally |
| D. Classic watershed | QuPath built-in cell detection | Zero deps, instant | Weakest on crowded/uneven nuclei |

**Recommendation:** **A (InstanSeg) primary, B (StarDist) as the reference/validation method.** Detect nuclei on **DAPI**, then **cell expansion** (a few µm) to capture cytoplasmic/membrane markers. Run both on one representative slide and compare counts before committing (cheap insurance given the WSI under-detection reports).

### 3.4 Classification by cell type
Per detected cell, measure mean/median intensity in each marker channel, then classify:
- **Neuron** = NeuroTrace⁺ · **Endothelial** = Podocalyxin⁺ · **Astrocyte** = GFAP⁺ · **Microglia** = IBA1⁺ (panel-dependent).
- Classifier options: (i) **threshold** per channel (simple, transparent), (ii) **QuPath object classifier** (train on a few annotated cells; handles overlap better), (iii) composite rules for double-positives.
- **Caveat:** GFAP and IBA1 mark *processes*, not nuclei — a nucleus-centered "cell" undercounts their territory. For these two, prefer **area-fraction** (next section) and use object counts only for soma density.

### 3.5 The object-level vs area-level question — best-practice answer

Use **both, matched to the biology of each marker.** This is the current convention in quantitative neuro-IHC:

| Readout | Best-practice measure | Why |
|---|---|---|
| **LYS241 in neurons / endothelium** | **Object-level**: mean LYS241 (FITC) intensity *per classified cell*, + % of that cell type that is LYS241⁺ | You want drug *per cell type* → must attribute signal to segmented cells |
| **LYS241 overall burden** | **Area-level**: **% positive area** (FITC) per region/compartment, intensity-thresholded | Drug is partly extracellular/perivascular; area-fraction captures diffuse deposition cells miss |
| **DAPI** | **Object-level density**: nuclei count ÷ region area (cells/mm²) | "Densité DAPI" is by definition a count density |
| **GFAP, IBA1** | **Area-level** (% positive area) as primary; soma counts secondary; optionally skeleton/branching for microglial activation morphology | Astro/microglia are morphological; area & morphology beat nucleus counts |
| **Podocalyxin / NeuroTrace** | Object-level for classification; area-fraction as QC | Used mainly to define the cell compartments |

So the table holds, per region×compartment: cell-type counts & densities (object), cell-type-specific LYS241 intensity (object), and % positive area for LYS241/GFAP/IBA1 (area). Report **both** — reviewers in this field expect cell-resolved numbers *and* area fractions, and they fail differently (segmentation errors vs threshold sensitivity), so agreement between them is itself a quality signal.

### 3.6 Compartments in IHC
Two sources, combined:
- **Atlas-derived regions** come from ABBA directly (every cell tagged with its Allen region).
- **Lesion-derived core/peri/contra** come from the **MRI** compartment masks (§2.7), pulled into the histology via the shared atlas: for each animal, transform the MRI core/peri/contra label (in Allen space) onto the histology section's atlas plane, giving each cell a compartment tag. Where a given animal has no MRI (1h/3h/6h), fall back to a **histology-intrinsic** core proxy (e.g. the IgG-leakage / tissue-damage footprint, or NeuroTrace loss) + mirrored contralateral — flagged as a different compartment-definition method in the table so the two aren't silently pooled.

### 3.7 Ipsi vs contra
Hemisphere (ipsi/contra) is a top-level split derived from the atlas midline, reported for every readout. Contralateral = within-animal internal control; the headline drug/cellular effects are ipsi-vs-contra contrasts within region×compartment.

---

## 4. Integration: building the joined dataset

For each cell (IHC) and each lesion voxel (MRI), you now have: `animal, timepoint, panel, hemisphere, Allen_region, compartment`. The **join** is on `(animal, Allen_region, compartment, hemisphere)`:
- MRI contributes: lesion volume (raw + corrected), per-region lesion fraction, edema, hemorrhage flag.
- IHC contributes: cell counts/densities by type, LYS241 per cell type, % positive areas, DAPI density.

Because both sides carry the same region+compartment vocabulary, the join is a clean relational merge in pandas — no image-to-image warping needed at this stage.

---

## 5. Output: the tidy long table

One row per `animal × region × compartment × cell_type × measure`:

| column | example | source |
|---|---|---|
| animal_id | M07 | meta |
| timepoint | 24h | meta |
| panel | A (NeuroT/Podo) | meta |
| hemisphere | ipsi | atlas |
| allen_region | Caudoputamen | ABBA |
| compartment | peri | MRI→atlas |
| compartment_method | MRI-geometric / histo-proxy | pipeline |
| cell_type | neuron | classifier |
| measure | LYS241_mean_intensity | readout |
| value | 1234.5 | readout |
| unit | a.u. (16-bit) | — |
| n_cells | 842 | detection |
| area_mm2 | 1.73 | atlas region |
| qc_flag | ok / review | QC |

Keep it **long** (one measure per row) so it drops straight into R/`ggplot`/mixed models. A wide pivot is a downstream convenience, not the storage format. Mixed-effects models (region/animal as random effects) are the natural analysis given the nesting — but that's the stats layer, out of scope here.

---

## 6. Direct MRI↔IHC overlay (figure-only, second pass)
Once §1–4 work, generate overlays *through* the atlas: for an IHC section at Allen plane *p*, resample the MRI lesion mask at plane *p* and composite. This gives publication overlays ("LYS241⁺ endothelium sits in the T2 peri-lesional rim") without a fragile direct registration. For a hero figure needing pixel-tight overlay, do a **one-off manual** BigWarp of that specific section to that specific MRI slice — acceptable for a figure, not for quantification.

---

## 7. Automation architecture (what's scripted vs interactive)

**Automated (Python orchestrates, no clicks):**
- Bruker→NIfTI, preprocessing, N4, masking.
- First-pass lesion segmentation, edema correction, volume, compartment masks.
- MRI→Allen registration runs (AIDAmri/ANTs as a batch).
- QuPath/ABBA driven **headlessly** for project creation, running the saved cell-detection + classifier scripts, exporting measurements (QuPath runs Groovy scripts from CLI; orchestrate from Python via `subprocess`, or use **`paquo`** to touch QuPath projects from Python).
- Final join + table assembly (pandas).

**Interactive (QC gates, humans in QuPath/Fiji):**
- Channel-map confirmation (§3.1).
- Lesion-mask review in Fiji.
- ABBA alignment review (two-person on a subset).
- Classifier spot-check.

This satisfies "automate where accuracy allows, but they still open images in their usual tools": the heavy lifting is scripted, the *judgment* steps happen in the GUIs they know.

> **Practical orchestration:** keep a per-animal config (YAML) listing file paths, panel, timepoint, channel map, and parameters (threshold k, peri-ring width). One Python entry point loops animals, calls each stage, writes intermediate masks/CSVs to a structured tree, and logs QC flags. This is what makes it reusable across future protocols — a new study is a new config, not new code.

---

## 8. QC gates (each must pass before the next stage)
1. **Header/spacing** correct on every NIfTI (voxel = 0.07×0.07×0.5 mm).
2. **Channel map** confirmed per panel; NeuroTrace-vs-FITC overlap ruled out.
3. **Bias correction** visibly flattened before thresholding.
4. **Lesion mask** reviewed in Fiji (over/under-segmentation, hemorrhage exclusion).
5. **MRI→Allen** registration overlay inspected (esp. large 5d lesions).
6. **ABBA** alignment double-checked on a subset.
7. **Cell detection** count sanity (InstanSeg vs StarDist agreement on one slide).
8. **anti-IgG specificity** resolved (control subtraction or specific secondary).
9. **Object vs area** readouts broadly agree per region; large divergence → investigate.

---

## 9. Suggested build order (time-boxed — get value early, don't over-engineer)

**Phase 1 — vertical slice on ONE animal (prove the spine).**
MRI: Bruker→NIfTI → N4 → threshold lesion → volume. IHC: one `.vsi` → QuPath → InstanSeg on DAPI → threshold classify → % positive area + per-cell LYS241. No atlas yet. Output a mini table. *This de-risks the whole thing in days, not weeks.*

**Phase 2 — add the atlas to both halves.**
MRI→Allen via AIDAmri; IHC→Allen via ABBA+DeepSlice. Now regions appear in the table.

**Phase 3 — compartments + edema + the join.**
Core/peri/contra masks, Swanson correction, MRI→IHC compartment transfer, the relational join, the full tidy table.

**Phase 4 — robustness + reuse.**
Swap threshold→trained classifier; batch all animals; YAML configs; QC logging; overlay figures. Only now consider deep-learning lesion seg if scale justifies it.

Stop at the phase where accuracy is "good enough for the biology" — for a 10-animal exploratory drug-distribution study, Phase 3 with solid QC is likely the right stopping point, with Phase 4 reserved for when this becomes a recurring assay.

---

## 10. Tools / dependencies

| Layer | Tool | Role |
|---|---|---|
| Bruker IO | brkraw (you have it) | raw → NIfTI |
| MRI pipeline | **AIDAmri** (primary), **ANTs/`antspyx`**, SimpleITK | preprocess, register to Allen, edema |
| MRI seg/QC | **Fiji** (Labkit / Trainable Weka), nibabel | lesion mask + review |
| Histology IO | **QuPath** + Bio-Formats | open `.vsi`, project mgmt |
| Section→atlas | **ABBA** (Fiji) + **DeepSlice** + BigWarp | IHC → Allen |
| Cell seg | **QuPath InstanSeg** (M1-GPU) + **StarDist** (validation) | detection/classification |
| Atlas API | **BrainGlobe** atlas API / Allen CCFv3 | region labels, common space |
| Orchestration | Python (`pathlib`, pandas, PyYAML), `paquo`, `subprocess`→QuPath/Fiji headless | glue + table |
| Stats (downstream) | R | mixed models, plots |

**Key references:** Drieu et al. 2020 (thrombin model); AIDAmri (Pallast et al., *Front. Neuroinform.* 2019); Koch et al. 2019 (atlas edema-corrected lesion volume); ABBA+BraiAn (Chiaruttini et al., *Cell Reports* 2025); DeepSlice (Carey et al., *Nat. Commun.* 2023); InstanSeg (Goldsborough et al. 2024); StarDist (Schmidt et al. 2018); QuPath (Bankhead et al. 2017).

---

### Open items I need from you to finalize parameters
1. **anti-IgG-FITC specificity** — isotype/species-specific to LYS241, or generic? (Decides whether core LYS241 is trustworthy.)
2. **NeuroTrace variant / emission** — green or non-green? (Decides FITC contamination risk.)
3. **Exact channel order** in the `.vsi` per panel.
4. Whether you want **AIDAmri adopted wholesale** for the MRI half, or a leaner ANTs-only registration you control end-to-end.
5. Péri-lesional **ring width** you consider biologically meaningful (sets the peri compartment).
