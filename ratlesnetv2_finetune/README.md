# RatLesNetV2 Finetuning Branch

This folder is the first independent track for T2w lesion-mask neural-network
work. It does not replace v1. The current v1 threshold/manual-review MRI path
continues to produce the corrected lesion masks that become training labels
here.

## Scope

- Prepare LYS T2w NIfTI volumes and human-corrected lesion masks in the folder
  structure expected by RatLesNetV2.
- Keep local work lightweight on the MacBook Pro M1 8 GB: validation,
  manifesting, and dataset conversion only.
- Run actual training or finetuning on a cloud GPU runtime such as Google
  Colab.
- Keep upstream RatLesNetV2 as an external checkout rather than vendoring model
  code or weights into this repo.

## Source Model Contract

Upstream RatLesNetV2: <https://github.com/jmlipman/RatLesNetv2>

The paper is Valverde et al., RatLesNetv2, arXiv:2001.09138.
The upstream code expects training data under:

```text
<split-root>/
└── <study>/
    └── <timepoint>/
        └── <case_id>/
            ├── scan.nii.gz
            └── scan_lesionIAM.nii.gz
```

Important upstream mismatch: the README describes `scan_lesion.nii.gz`, but
the current `lib/DataWrapper.py` uses `scan_lesionIAM.nii.gz`. The exporter
writes the code-compatible filename by default and also writes
`scan_lesion.nii.gz` as an inspection alias.

## Training Strategy

RatLesNetV2 starts from a rat-trained rodent T2w lesion prior. The current LYS
target is mouse T2w MRI, so the intended adaptation path is staged:

```text
RatLesNetV2 pretrained rat weights
  -> public mouse T2w stroke datasets with manual native-space masks
  -> LYS T2w scans with human-reviewed native lesion masks
  -> held-out LYS validation
  -> draft predictions returned to human review
```

The public mouse datasets are not final validation data. They are useful for
mouse-domain adaptation and for reducing overfitting when the LYS mask count is
small. Final performance must be judged on held-out LYS animals.

Current public dataset notes, overlap rules, geometry observations, and import
decisions live in:

```text
docs/ratlesnetv2_external_datasets.md
```

Important summary:

- `An2022`, `Knab2025`, and `Koch2017` provide native mouse T2w NIfTI scans
  with manual lesion masks that can support public-mouse adaptation.
- `An2022` includes a 19-case `Mulder2017` subset.
- `Knab2025` overlaps with `An2022` Charite cases; do not train on duplicates.
- `Koch2017` appears mostly independent by case ID, but has native and cropped
  copies that must not be treated as separate animals.
- External public geometry is mostly around `256 x 256 x 32` at
  `0.1 x 0.1 x 0.5 mm`; LYS is `256 x 256 x 18` at
  `0.07 x 0.07 x 0.5 mm`, so LYS fine-tuning remains mandatory.

## Download And Clean Public Mouse Data

To create a cleaned external source folder on the external hard drive:

```bash
make ratlesnetv2-download-external \
  RUN_ARGS="--output-root /Volumes/Untitled/external_datasets --include-mulder --delete-archives"
```

The script downloads the public archives, exports only useful native-space
manual T2w/T2-map image and lesion-mask pairs, skips known duplicate public
cases, and writes:

```text
/Volumes/Untitled/external_datasets/
├── ratlesnetv2_clean_source/
│   ├── T2w/
│   └── masks/
├── manifests/
│   ├── external_dataset_manifest.csv
│   └── skipped_cases.csv
└── README.md
```

`ratlesnetv2_clean_source/` is directly compatible with the source-folder
importer below. Keep `--include-mulder` only when you want the lower-resolution
Mulder T2-map/IAM manual-mask pairs included; omit it for the stricter
T2w-native An/Knab/Koch adaptation set.

The 2026-07-01 run on `/Volumes/Untitled/external_datasets` exported 547 pairs:
331 `An2022`, 80 non-duplicate `Knab2025`, 15 recoverable `Koch2017`, and 121
`Mulder2017`. Koch's public zip has malformed large-zip offsets; the script
repairs/falls back where possible and records unrecoverable native pairs in
`manifests/skipped_cases.csv`.

The non-Mulder public NIfTI headers should be normalized before mixing them
with LYS. Visual ITK-SNAP checks on representative `An2022`, `Knab2025`, and
`Koch2017` cases showed that their 32-slice axis is a coronal stack but is
declared as superior/inferior. Create a separate LSP-oriented copy:

```bash
make ratlesnetv2-orient-external-lsp \
  RUN_ARGS="--external-root /Volumes/Untitled/external_datasets --target-axcodes LSP --overwrite"
```

This writes:

```text
/Volumes/Untitled/external_datasets/
├── ratlesnetv2_clean_source_LSP_oriented/
│   ├── T2w/
│   └── masks/
└── manifests/external_dataset_manifest_LSP_oriented.csv
```

This is a header/affine relabel only: voxel arrays are not permuted or
resampled, and each mask receives the same new affine as its scan. `Mulder2017`
is excluded by default because it is lower-resolution T2-map data rather than
normal T2w source data. The 2026-07-01 verification found 426 `LSP` scan/mask
pairs with unchanged shapes, binary masks, and unchanged lesion voxel counts.
All oriented rows remain marked `needs_visual_qc_lsp_orientation`.

## Local Source Folder Import

Use the existing `lys-bbb` conda env. Do not create a new local conda env for
this branch unless you explicitly decide to change that project rule.

Expected source-folder layout:

```text
<source-folder>/
├── <scan-subfolder>/
│   ├── animal_01.nii.gz
│   └── animal_02.nii.gz
└── <mask-subfolder>/
    ├── animal_01_lesion_mask.nii.gz
    └── animal_02_lesion_mask.nii.gz
```

The scan and mask basenames must match exactly. For example,
`BD_01_24h.nii.gz` is paired with `BD_01_24h_lesion_mask.nii.gz`.

## Fiji RoiSet To NIfTI Mask

If lesion masks start as Fiji/ImageJ ROI Manager annotations, convert each
`RoiSet.zip` to a binary NIfTI mask before importing source folders:

```bash
make ratlesnetv2-roiset-to-mask \
  RUN_ARGS="--nifti /path/to/BD_01_24h.nii.gz --roiset /path/to/RoiSet.zip --output /path/to/masks/BD_01_24h_lesion_mask.nii.gz"
```

The converter writes a uint8 mask with the same 3-D shape, affine, and voxel
spacing as the original NIfTI. The output filename must end with
`_lesion_mask.nii.gz`, because the source-folder importer matches
`name.nii.gz` to `name_lesion_mask.nii.gz`.

Supported Fiji ROI area types are polygon/freehand/traced outlines, rectangle,
and oval. Non-area ROIs such as lines, points, and polylines are rejected.
Composite ImageJ shape ROIs are also rejected; save lesion outlines as separate
simple ROIs in Fiji.

Slice mapping defaults to Fiji ROI stack positions. If the `.roi` files do not
carry stack positions, the converter falls back to the last integer in each ROI
filename, interpreted as one-based by default. Useful options:

```bash
# Force filename-based slice mapping, e.g. slice_002.roi -> NIfTI slice index 1.
--slice-source filename --filename-index-base 1

# If a specific Fiji/NIfTI import swaps displayed x/y axes, override the mapping.
--xy-axes 1,0

# If --output is omitted, masks are written under work/ratlesnetv2_finetune/masks/.
--output-dir work/ratlesnetv2_finetune/masks
```

## Bulk LYS RoiSet/Bruker Cleanup

For the scattered LYS hard-drive data, use the bulk cleanup script instead of
manually pairing each RoiSet with each Bruker folder:

```bash
make ratlesnetv2-prepare-lys-roisets \
  RUN_ARGS="--source-drive '/Volumes/Lys T Dec 2021 N1' --output-root ~/Desktop/LYS_RatLesNetV2_clean_source --resume"
```

The source drive is treated as read-only. For each discovered `RoiSet*.zip`,
the script finds the associated Bruker study, selects the native T2w RARE scan
from Bruker metadata, copies only the selected scan folder plus the RoiSet to a
local temporary staging folder, converts that local copy, and deletes staging
by default. It never writes into `/Volumes/...`.

The output is:

```text
~/Desktop/LYS_RatLesNetV2_clean_source/
├── ratlesnetv2_clean_source/
│   ├── T2w/
│   └── masks/
├── roi_archives/
├── manifests/
│   ├── lys_clean_dataset_manifest.csv
│   ├── lys_skipped_or_needs_review.csv
│   └── lys_scan_inventory.csv
└── README.md
```

`ratlesnetv2_clean_source/` can be passed directly to
`ratlesnetv2-add-source`. The manifest records source paths, selected scan ID,
shape, voxel spacing, mask voxel count, lesion volume in mm3, and `qc_flag`.
All generated masks are marked `needs_visual_qc`; they are training-label
candidates, not final reviewed labels.

Use a dry run first when changing roots or path lists:

```bash
make ratlesnetv2-prepare-lys-roisets \
  RUN_ARGS="--source-drive '/Volumes/Lys T Dec 2021 N1' --output-root ~/Desktop/LYS_RatLesNetV2_clean_source --dry-run --resume"
```

Rows in `manifests/lys_skipped_or_needs_review.csv` are intentionally not
guessed. Typical reasons are ambiguous Bruker study matches, duplicate case
IDs, ambiguous T2 scan selection, or a local conversion failure. Resolve those
manually before using them for training.

Start from a local, gitignored dataset plan:

```bash
cp ratlesnetv2_finetune/configs/dataset_from_folders_template.yml \
  ratlesnetv2_finetune/configs/local_dataset.yml
```

Add one source folder to the plan:

```bash
make ratlesnetv2-add-source \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_dataset.yml \
  RUN_ARGS="--source-root /path/to/folder_24h --split train --study LYS --timepoint 24h --scan-subdir T2w --mask-subdir masks"
```

Repeat the command for each local folder. Use `--split validation` or
`--split test` for held-out folders. If a source folder has exactly one
subfolder containing `*.nii.gz` scans and exactly one subfolder containing
`*_lesion_mask.nii.gz` masks, `--scan-subdir` and `--mask-subdir` may be
omitted. If filenames are not globally unique across folders, add a stable
prefix such as `--id-prefix cohort1`.

The updater is idempotent by `case_id`: running it again for the same folder
updates existing entries rather than duplicating them.

Preview without writing:

```bash
make ratlesnetv2-add-source \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_dataset.yml \
  RUN_ARGS="--source-root /path/to/folder_24h --split train --timepoint 24h --dry-run"
```

## Local Harmonized Export

After all source folders have been added to the YAML plan, build the
RatLesNetV2-ready dataset:

```bash
make ratlesnetv2-prepare \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_dataset.yml
```

This writes:

```text
work/ratlesnetv2_finetune/datasets/LYS_T2w_manual_v0/
├── manifest.csv
    └── train/LYS/5d/BD_08_5D/
        ├── scan.nii.gz
        ├── scan_lesionIAM.nii.gz
        └── scan_lesion.nii.gz
```

`manifest.csv` records source case, split, spacing, shape, lesion voxel count,
and lesion volume in mm3.

Use `RUN_ARGS=--overwrite` only when you intentionally want to rebuild an
existing prepared dataset after changing the YAML plan:

```bash
make ratlesnetv2-prepare \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_dataset.yml \
  RUN_ARGS=--overwrite
```

Optional upload package:

```bash
tar -C work/ratlesnetv2_finetune/datasets \
  -czf work/ratlesnetv2_finetune/LYS_T2w_manual_v0.tar.gz \
  LYS_T2w_manual_v0
```

## Cloud Run Plan

For Colab/cloud training, import only:

1. This repo/branch, usually by cloning it in Colab.
2. The prepared dataset folder or tarball:
   `work/ratlesnetv2_finetune/datasets/LYS_T2w_manual_v0`.
3. Optional upstream pretrained weights, for example `RatLesNetv2.model`.

Raw local source folders are not needed on the cloud if local harmonization has
already been run.

Print cloud commands locally:

```bash
make ratlesnetv2-cloud-plan \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_dataset.yml \
  RUN_ARGS="--cloud-dataset-root /content/LYS_T2w_manual_v0 --epochs 1 --max-train-cases 1 --save-every 1"
```

Minimal Colab command sequence:

```bash
git clone --branch dl-ratlesnetv2-finetune <your-repo-url> /content/LYS_PROJ1
cd /content/LYS_PROJ1
pip install -r ratlesnetv2_finetune/requirements-colab.txt
git clone --depth 1 https://github.com/jmlipman/RatLesNetv2.git /content/RatLesNetv2
```

Copy or extract the prepared dataset so this folder exists:

```text
/content/LYS_T2w_manual_v0/
├── manifest.csv
├── train/
├── validation/   # only if configured locally
└── test/         # only if configured locally
```

One-case cloud smoke test:

```bash
python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /content/RatLesNetv2 \
  --input /content/LYS_T2w_manual_v0/train \
  --output /content/ratlesnet_runs \
  --epochs 1 \
  --lr 1e-4 \
  --gpu 0 \
  --loadMemory 0 \
  --max-train-cases 1 \
  --save-every 1
```

If the YAML has validation cases and the prepared dataset contains a validation
folder, add:

```bash
--validation /content/LYS_T2w_manual_v0/validation
```

Finetune from pretrained weights:

```bash
python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /content/RatLesNetv2 \
  --input /content/LYS_T2w_manual_v0/train \
  --validation /content/LYS_T2w_manual_v0/validation \
  --output /content/ratlesnet_runs \
  --pretrained-model /content/pretrained/RatLesNetv2.model \
  --epochs 100 \
  --lr 1e-4 \
  --gpu 0 \
  --loadMemory 0 \
  --save-every 10
```

Omit `--validation` if there is no validation split. Omit
`--pretrained-model` for a smoke-test training run from initialization.

## Data Rules

- Never write generated training data under `data/`.
- Do not commit prepared NIfTI files, model weights, or checkpoints.
- The manual lesion mask is the label source; draft masks are not valid
  training labels.
- Header spacing is checked against the YAML plan. LYS defaults to
  `0.07 x 0.07 x 0.5 mm`; external public mouse datasets have different
  geometry and should be imported through explicit dataset-specific plans or
  after a geometry report.
- Use native-space images/masks for RatLesNetV2 training. Atlas-space files
  such as `x_masklesion.nii` are not native segmentation labels.
- Automated public masks are not ground truth unless an explicit pseudo-label
  experiment is configured.
- The branch currently supports one T2w modality/channel.

## Next Steps

1. Add a geometry/provenance report for all candidate public and LYS cases.
2. Add dataset-specific importers or documented conversion recipes for
   `An2022`, `Knab2025`, `Koch2017`, and optionally `Mulder2017`.
3. Deduplicate public cases before export, especially `An2022` vs `Knab2025`
   and `An2022` vs full `Mulder2017`.
4. Run the one-case cloud smoke test, then public-mouse adaptation, then LYS
   fine-tuning.
5. Bring predictions back through the existing human review gate and compare
   draft-vs-corrected Dice before using any neural-network lesion volume.
