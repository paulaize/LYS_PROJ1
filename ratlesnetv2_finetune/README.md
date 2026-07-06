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

`ratlesnetv2_clean_source/` is the raw cleaned source folder. For external
training, use the orientation-normalized non-Mulder copy described below rather
than importing this folder directly. Keep `--include-mulder` only when you want
the lower-resolution Mulder T2-map/IAM manual-mask pairs included for later
manual review; omit it for the stricter T2w-native An/Knab/Koch adaptation set.

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

Visual review then showed the coronal stack and mask overlays were still
superior/inferior upside down relative to the LYS images. After a three-case
test was confirmed in ITK-SNAP, create the final external source copy by
flipping the LSP voxel arrays along axis 1 while preserving the LSP
affine/header:

```bash
make ratlesnetv2-flip-external-si \
  RUN_ARGS="--external-root /Volumes/Untitled/external_datasets --output-source-root /Volumes/Untitled/external_datasets/ratlesnetv2_clean_source_LSP_SI_flipped --output-manifest /Volumes/Untitled/external_datasets/manifests/external_dataset_manifest_LSP_SI_flipped.csv --all --overwrite"
```

This writes the source folder that should be used for non-Mulder external
training import:

```text
/Volumes/Untitled/external_datasets/
├── ratlesnetv2_clean_source_LSP_SI_flipped/
│   ├── T2w/
│   └── masks/
└── manifests/external_dataset_manifest_LSP_SI_flipped.csv
```

The 2026-07-03 full run exported 426 scan/mask pairs: 331 `An2022`, 80
`Knab2025`, and 15 `Koch2017`. Verification found zero technical problems:
all outputs are `LSP`, scan/mask affines match, shapes are unchanged, masks are
binary, lesion voxel counts are unchanged, and every output is exactly the
axis-1 flip of the corresponding LSP-oriented input.

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

## Review And Edit LYS Masks In ITK-SNAP

The RoiSet-converted LYS masks should be visually checked before they become
training labels. Use the ITK-SNAP review helper to create editable mask copies
in a reviewed source folder and open them with the matching T2w image:

```bash
make ratlesnetv2-review-lys-masks \
  RUN_ARGS="--source-root ~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source --case C1S1 --limit 1 --dry-run"
```

Run without `--dry-run` to create the reviewed folder and launch ITK-SNAP:

```bash
make ratlesnetv2-review-lys-masks \
  RUN_ARGS="--source-root ~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source --case C1S1 --limit 1"
```

By default this writes:

```text
~/Desktop/LYS_RatLesNetV2_clean_source/
├── ratlesnetv2_clean_source/          # original generated source, not edited
└── ratlesnetv2_clean_source_reviewed/
    ├── T2w/                           # scan symlinks by default
    ├── masks/                         # editable mask copies
    └── manifests/mask_review_queue.csv
```

Open masks one case at a time, use the ITK-SNAP segmentation brush/eraser, and
save the mask. Then press Enter in the terminal to open the next case. If the
terminal cannot read Enter through `conda run`, the script waits for you to
close the ITK-SNAP window after saving, then opens the next case. Existing
reviewed masks are preserved by default, so rerunning the command continues
from the current reviewed copy. Useful options:

```bash
# Queue every case after checking the dry run.
make ratlesnetv2-review-lys-masks \
  RUN_ARGS="--source-root ~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source"

# Skip cases that already have a reviewed mask copy.
make ratlesnetv2-review-lys-masks \
  RUN_ARGS="--skip-existing"

# Copy scans instead of symlinking them, useful before moving the reviewed folder.
make ratlesnetv2-review-lys-masks \
  RUN_ARGS="--copy-scans --prepare-only"
```

After review, import
`~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source_reviewed/`
with `ratlesnetv2-add-source`, not the original unreviewed source folder.
Pass `--viewer /Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP` if the
script cannot discover ITK-SNAP automatically.

## First LYS Colab Smoke-Test Dataset

For the first Colab run, keep the target deliberately small: prepare the
reviewed LYS source folder locally, upload one tarball to Google Drive, then
run a one-case/one-epoch cloud smoke test. This confirms that the prepared
folder, upstream RatLesNetV2 checkout, CUDA runtime, and training wrapper all
work together.

This is **not** the final scientific split. The commands below put all reviewed
LYS cases into `train` and then limit the cloud run with `--max-train-cases 1`.
Before a real finetune/evaluation, create explicit train/validation/test
splits and reserve held-out LYS animals for validation/test.

Current local source folder for this smoke test:

```text
~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source_reviewed/
├── T2w/
└── masks/
```

Prepare the local YAML plan:

```bash
cp ratlesnetv2_finetune/configs/dataset_from_folders_template.yml \
  ratlesnetv2_finetune/configs/local_lys_reviewed.yml
```

Add the reviewed LYS source folder:

```bash
make ratlesnetv2-add-source \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_lys_reviewed.yml \
  RUN_ARGS="--source-root ~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source_reviewed --split train --study LYS --timepoint mixed --scan-subdir T2w --mask-subdir masks"
```

Build the RatLesNetV2-ready dataset:

```bash
make ratlesnetv2-prepare \
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_lys_reviewed.yml \
  RUN_ARGS="--overwrite"
```

Package it for upload to Google Drive:

```bash
tar -C work/ratlesnetv2_finetune/datasets \
  -czf ~/Desktop/LYS_T2w_manual_v0.tar.gz \
  LYS_T2w_manual_v0
```

Upload `~/Desktop/LYS_T2w_manual_v0.tar.gz` to Google Drive. The cloud runtime
does not need the original Desktop source folder once this tarball exists.

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

For Google Drive upload, it is usually simpler to write the tarball to the
Desktop as shown in the first smoke-test recipe.

## Cloud Run Plan

Current handoff state, 2026-07-06:

- Prepared Colab upload archives already exist locally and have been uploaded
  to Google Drive:
  - `LYS_T2w_manual_v0.tar.gz`
  - `External_Mouse_T2w_manual_LSP_SI_v0.tar.gz`
- The prepared LYS archive contains 262 train cases:
  - 254 cases shaped `256 x 256 x 18 x 1`
  - 8 cases shaped `256 x 256 x 20 x 1`
- The prepared external archive contains 426 train cases:
  - 391 cases shaped `256 x 256 x 32 x 1`
  - 35 cases shaped `192 x 192 x 32 x 1`
- Both archives are currently all-train smoke-test exports. They are suitable
  for checking that Colab, RatLesNetV2, and the data contract work. They are
  not suitable for final model evaluation until explicit
  train/validation/test splits are created.
- Colab may use modern `nibabel >= 5`. Upstream RatLesNetV2 still calls the
  removed `img.get_data()` API, so `scripts/finetune_ratlesnetv2.py` now
  patches nibabel compatibility at runtime. Do not downgrade Colab nibabel or
  force-reinstall old packages; restart the runtime and use this branch's
  current script instead.
- Local verification of the current wrapper passed:
  - `make lint`
  - `make test`
  - one synthetic one-case RatLesNetV2 CPU smoke train against a fresh upstream
    `jmlipman/RatLesNetv2` clone.

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
  RATLESNET_CONFIG=ratlesnetv2_finetune/configs/local_lys_reviewed.yml \
  RUN_ARGS="--cloud-dataset-root /content/LYS_T2w_manual_v0 --cloud-output /content/drive/MyDrive/ratlesnet_runs --epochs 1 --max-train-cases 1 --save-every 1"
```

Minimal Colab command sequence:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
git clone --branch dl-ratlesnetv2-finetune https://github.com/paulaize/LYS_PROJ1.git /content/LYS_PROJ1
cd /content/LYS_PROJ1
pip install -r ratlesnetv2_finetune/requirements-colab.txt
git clone --depth 1 https://github.com/jmlipman/RatLesNetv2.git /content/RatLesNetv2
```

If the repository is private, configure a GitHub personal access token before
the clone rather than using the interactive username/password prompt:

```python
import getpass
import os
from pathlib import Path

token = getpass.getpass("GitHub personal access token: ").strip()
netrc = Path("/root/.netrc")
netrc.write_text(f"machine github.com\nlogin paulaize\npassword {token}\n")
os.chmod(netrc, 0o600)
```

Extract the prepared dataset from Google Drive:

```bash
tar -xzf /content/drive/MyDrive/LYS_T2w_manual_v0.tar.gz -C /content
```

Then this folder should exist:

```text
/content/LYS_T2w_manual_v0/
├── manifest.csv
├── train/
├── validation/   # only if configured locally
└── test/         # only if configured locally
```

For a real training/evaluation run, do not train from the all-train smoke-test
folder directly. Split the prepared folders first. Use external mouse data for
mouse-domain adaptation, but monitor that adaptation against target-domain LYS
validation. Use LYS train for target-domain fine-tuning and held-out LYS test
for final reporting:

```bash
# LYS target-domain split: train + validation + held-out test.
python -m ratlesnetv2_finetune.scripts.split_prepared_dataset \
  --input /content/LYS_T2w_manual_v0 \
  --output /content/LYS_T2w_manual_v0_split \
  --validation-fraction 0.15 \
  --test-fraction 0.15 \
  --seed 20260706

# External mouse adaptation split: train-only. LYS validation is used below.
tar -xzf /content/drive/MyDrive/External_Mouse_T2w_manual_LSP_SI_v0.tar.gz -C /content
python -m ratlesnetv2_finetune.scripts.split_prepared_dataset \
  --input /content/External_Mouse_T2w_manual_LSP_SI_v0 \
  --output /content/External_Mouse_T2w_manual_LSP_SI_v0_split \
  --validation-fraction 0 \
  --test-fraction 0 \
  --seed 20260706
```

The splitter writes a new `manifest.csv` and `split_summary.json`. It groups
by `animal_id` by default, so repeated rows for one animal cannot leak across
train/validation/test.

One-case cloud smoke test:

```bash
cd /content/LYS_PROJ1
python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /content/RatLesNetv2 \
  --input /content/LYS_T2w_manual_v0/train \
  --output /content/drive/MyDrive/ratlesnet_runs_lys_smoke \
  --epochs 1 \
  --lr 1e-4 \
  --gpu 0 \
  --loadMemory 0 \
  --max-train-cases 1 \
  --save-every 1
```

This is useful for confirming the runtime and data contract only. Do not
interpret the one-case loss as model performance.

If this LYS smoke test succeeds, run the same one-case smoke test on the
external mouse archive:

```bash
cd /content/LYS_PROJ1
python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /content/RatLesNetv2 \
  --input /content/External_Mouse_T2w_manual_LSP_SI_v0/train \
  --output /content/drive/MyDrive/ratlesnet_runs_external_smoke \
  --epochs 1 \
  --lr 1e-4 \
  --gpu 0 \
  --loadMemory 0 \
  --max-train-cases 1 \
  --save-every 1
```

Common Colab failure modes already encountered:

- `ExpiredDeprecationError: get_data() is deprecated`: use the current branch
  script, which patches nibabel at runtime. Restart the runtime after any
  attempted package downgrade, reclone this branch, and rerun. Do not
  `pip install "nibabel<5" --force-reinstall`.
- `Repository not found` while cloning a private repo: the GitHub token lacks
  access to `paulaize/LYS_PROJ1` or the branch has not been pushed.
- Argument/path typos: use normal double hyphens, for example `--gpu`, not a
  typographic dash; use `LYS_T2w_manual_v0`, not `LYS_T2w_manual_vo`; do not
  add spaces inside `/content/...` paths.

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
  --output /content/drive/MyDrive/ratlesnet_runs \
  --pretrained-model /content/pretrained/RatLesNetv2.model \
  --require-pretrained \
  --epochs 100 \
  --lr 1e-4 \
  --gpu 0 \
  --loadMemory 0 \
  --save-every 10
```

Omit `--validation` if there is no validation split. Omit
`--pretrained-model` only for a smoke-test training run from initialization.
For true RatLesNetV2 fine-tuning, upload or otherwise stage the upstream
pretrained `RatLesNetv2.model` in Colab and pass it with
`--pretrained-model`. Use `--require-pretrained` on real runs so the script
fails loudly instead of silently training from scratch.

First full staged Colab run after the smoke tests:

```bash
# 1. External mouse adaptation. Validation is LYS target-domain validation.
#    Do not pass the held-out LYS test split at this stage.
python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /content/RatLesNetv2 \
  --input /content/External_Mouse_T2w_manual_LSP_SI_v0_split/train \
  --validation /content/LYS_T2w_manual_v0_split/validation \
  --output /content/drive/MyDrive/ratlesnet_runs_external_adapt \
  --pretrained-model /content/drive/MyDrive/RatLesNetv2.model \
  --require-pretrained \
  --epochs 50 \
  --lr 1e-4 \
  --gpu 0 \
  --loadMemory 0 \
  --save-every 5 \
  --eval-every 1

# 2. LYS target-domain fine-tune and held-out test evaluation.
#    Adjust the pretrained path if the external adaptation run number is not 1.
python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /content/RatLesNetv2 \
  --input /content/LYS_T2w_manual_v0_split/train \
  --validation /content/LYS_T2w_manual_v0_split/validation \
  --test /content/LYS_T2w_manual_v0_split/test \
  --output /content/drive/MyDrive/ratlesnet_runs_lys_finetune \
  --pretrained-model /content/drive/MyDrive/ratlesnet_runs_external_adapt/1/RatLesNetv2.model \
  --require-pretrained \
  --epochs 100 \
  --lr 1e-4 \
  --gpu 0 \
  --loadMemory 0 \
  --save-every 10 \
  --eval-every 1
```

Each run directory now writes:

```text
training_loss
validation_loss                  # when --validation is provided
metrics_epoch.csv                # split-level loss, Dice, IoU, accuracy, etc.
metrics_cases.csv                # per-case metrics per evaluated epoch/split
final_metrics.json               # final validation/test summary
loss_curves.png                  # train/eval loss over epochs
validation_metric_curves.png     # Dice/IoU/precision/recall over epochs
final_metric_summary.png         # final validation/test metric bars
RatLesNetv2.model
```

Report final model performance from the LYS `test` rows in
`final_metrics.json` or `metrics_epoch.csv`. The LYS validation rows are useful
for monitoring/model selection during external adaptation and LYS fine-tuning,
but they are not the final held-out performance claim.

In Colab, view the latest plots with:

```python
from IPython.display import Image, display
from pathlib import Path

root = Path("/content/drive/MyDrive/ratlesnet_runs_lys_finetune")
run = sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
             key=lambda p: int(p.name))[-1]

for name in ["loss_curves.png", "validation_metric_curves.png", "final_metric_summary.png"]:
    path = run / name
    if path.exists():
        display(Image(filename=str(path)))
```

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
4. Run the one-case cloud smoke test.
5. Create explicit train/validation/test splits with held-out LYS animals.
6. Run public-mouse adaptation, then LYS fine-tuning.
7. Bring predictions back through the existing human review gate and compare
   draft-vs-corrected Dice before using any neural-network lesion volume.
