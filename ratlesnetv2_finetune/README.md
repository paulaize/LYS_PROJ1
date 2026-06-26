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

## Local Preparation

Use the existing `lys-bbb` conda env. Do not create a new local conda env for
this branch unless you explicitly decide to change that project rule.

```bash
make ratlesnetv2-prepare RATLESNET_CONFIG=ratlesnetv2_finetune/configs/dataset_template.yml
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

## Cloud Run Plan

After copying this repo/folder and the prepared dataset to Colab or another
GPU runtime:

```bash
pip install -r ratlesnetv2_finetune/requirements-colab.txt
make ratlesnetv2-cloud-plan RATLESNET_CONFIG=ratlesnetv2_finetune/configs/dataset_template.yml
```

The plan command prints the exact upstream clone and finetuning commands. A
direct command looks like:

```bash
git clone --depth 1 https://github.com/jmlipman/RatLesNetv2.git /content/RatLesNetv2
python -m ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 \
  --ratlesnet-repo /content/RatLesNetv2 \
  --input /content/LYS_T2w_manual_v0/train \
  --validation /content/LYS_T2w_manual_v0/validation \
  --output /content/ratlesnet_runs \
  --pretrained-model /content/pretrained/RatLesNetv2.model \
  --epochs 100 \
  --lr 1e-4 \
  --gpu 0 \
  --loadMemory 0
```

Omit `--pretrained-model` for a smoke-test training run from initialization.
Use `--max-train-cases 1 --epochs 1` first on cloud to prove the data path.

## Data Rules

- Never write generated training data under `data/`.
- Do not commit prepared NIfTI files, model weights, or checkpoints.
- The manual lesion mask is the label source; draft masks are not valid
  training labels.
- Header spacing is checked against `0.07 x 0.07 x 0.5 mm` by default.
- The branch currently supports one T2w modality/channel.

## Next Steps

1. Add additional animals only after their T2w NIfTI and reviewed
   `lesion_corrected.nii.gz` masks exist.
2. Create a real train/validation split once there are at least several
   corrected masks. With one case, only smoke testing is meaningful.
3. Run the one-case cloud smoke test, then a small finetune from available
   RatLesNetV2 weights if the upstream weight file is available.
4. Bring predictions back through the existing human review gate and compare
   draft-vs-corrected Dice before using any neural-network lesion volume.

