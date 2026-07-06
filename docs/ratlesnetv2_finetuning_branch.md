# RatLesNetV2 Finetuning Branch

Branch: `dl-ratlesnetv2-finetune`

This branch starts the independent MRI lesion deep-learning track requested by
Paul. It is deliberately separate from the active v1 spine: v1 still produces
human-reviewed lesion masks, while this branch turns those masks into training
data for a future neural-network draft segmenter.

## Why RatLesNetV2 Here

RatLesNetV2 is a rodent T2w lesion segmentation network with public code at
<https://github.com/jmlipman/RatLesNetv2> and paper record
<https://arxiv.org/abs/2001.09138>. It is not the active v1 segmentation
backend. It is a candidate transfer-learning branch for later MRI lesion
drafts once enough corrected LYS masks exist.

RatLesNetV2 was trained on rat T2w stroke MRI. The LYS target is mouse T2w MRI,
so the branch should treat the upstream weights as a rodent lesion prior, then
adapt through public mouse datasets before final LYS fine-tuning:

```text
RatLesNetV2 rat weights
  -> public mouse native-space manual masks
  -> LYS native-space human-reviewed masks
  -> held-out LYS validation
```

The external mouse dataset inventory, overlap notes, geometry summary, and
recommended inclusion rules are tracked in
`docs/ratlesnetv2_external_datasets.md`.

## Local vs Cloud Responsibilities

Local MacBook Pro M1 8 GB:

- prepare NIfTI folders,
- validate shapes/spacing/labels,
- write manifests,
- run unit tests,
- plan cloud commands.

Cloud GPU runtime:

- clone upstream RatLesNetV2,
- run the finetuning/training script,
- save checkpoints outside git,
- export predictions for later human review.

No local CUDA setup is expected. If PyTorch is used locally, the code selects
MPS when available rather than CUDA, but real training should happen on cloud.

## Added Folder

`ratlesnetv2_finetune/` contains:

- `dataset.py`: deterministic conversion from configured NIfTI scan/manual mask
  pairs to RatLesNetV2 folders.
- `roiset_to_nifti_mask.py`: Fiji/ImageJ `RoiSet.zip` to
  `*_lesion_mask.nii.gz` converter for polygon/freehand lesion outlines.
- `source_folders.py`: source-folder scanner that appends matched
  `case.nii.gz` / `case_lesion_mask.nii.gz` pairs to a dataset YAML plan.
- `scripts/review_lys_masks_itksnap.py`: opens reviewed copies of LYS T2w
  images and masks in ITK-SNAP so masks can be lightly corrected without
  overwriting the original cleaned source folder.
- `scripts/add_source_folder.py`: CLI for running that scanner once per local
  source folder.
- `scripts/prepare_dataset.py`: local dataset-preparation CLI.
- `scripts/plan_cloud_run.py`: prints Colab/cloud commands.
- `scripts/finetune_ratlesnetv2.py`: cloud training loop that imports the
  upstream model, optionally loads pretrained weights, and continues training.
- `configs/dataset_template.yml`: first dataset plan, seeded with `BD_08_5D`.
- `configs/dataset_from_folders_template.yml`: empty plan intended to be copied
  to a gitignored local YAML before importing local source folders.
- `requirements-colab.txt`: cloud runtime dependency list.

Relevant documents:

- `ratlesnetv2_finetune/README.md`: command-oriented local/Colab workflow.
- `docs/ratlesnetv2_external_datasets.md`: public mouse dataset sources,
  overlaps, geometry, and training-set construction rules.
- `docs/MRI_IHC_pipeline_plan_DL.md`: broader DL-first pipeline context.

## Intended Local-To-Colab Flow

Current handoff state, 2026-07-06:

- The LYS reviewed dataset and the external public mouse dataset have already
  been exported into RatLesNetV2 folder format and packaged as Google Drive
  upload tarballs:
  - `LYS_T2w_manual_v0.tar.gz`
  - `External_Mouse_T2w_manual_LSP_SI_v0.tar.gz`
- LYS export: 262 train cases, all under `train/LYS/mixed`; shape counts are
  254 `256 x 256 x 18 x 1` and 8 `256 x 256 x 20 x 1`.
- External export: 426 train cases, all under `train/ExternalMouse/mixed`;
  shape counts are 391 `256 x 256 x 32 x 1` and 35
  `192 x 192 x 32 x 1`.
- These are all-train smoke-test exports. Use them to verify Colab runtime,
  upstream RatLesNetV2 imports, and the NIfTI folder contract. Do not use them
  for final model-performance claims.
- Colab authentication for the private GitHub repository should use a GitHub
  personal access token configured through `/root/.netrc`; do not rely on the
  interactive username/password prompt.
- The first Colab attempt reached the RatLesNetV2 training loop, then failed
  because upstream `DataWrapper.py` calls nibabel's removed `img.get_data()`
  method under modern Colab nibabel. The local wrapper now patches nibabel
  compatibility inside `scripts/finetune_ratlesnetv2.py`; do not downgrade
  Colab packages to solve this.
- Current local verification after that fix:
  - `make lint` passed.
  - `make test` passed with 53 tests.
  - A synthetic one-case end-to-end training smoke test against a fresh
    upstream `jmlipman/RatLesNetv2` clone completed and saved
    `RatLesNetv2.model`.

1. Download external public archives outside git if public mouse adaptation is
   being used.
2. Convert/import only native-space manual public masks into source-folder
   format, following `docs/ratlesnetv2_external_datasets.md`.
3. If LYS manual lesion outlines are Fiji ROI Manager exports, run the bulk
   LYS RoiSet/Bruker cleanup first to create the local source folder:
   `~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source/`.
4. Review/edit LYS masks with `make ratlesnetv2-review-lys-masks`. This writes
   editable copies under
   `~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source_reviewed/`.
   Use this reviewed folder for training import, not the original unreviewed
   source folder.
5. For the first Colab smoke test, copy
   `configs/dataset_from_folders_template.yml` to
   `ratlesnetv2_finetune/configs/local_lys_reviewed.yml`.
6. Add the reviewed LYS folder to that plan as `train`, then run
   `make ratlesnetv2-prepare ... RUN_ARGS="--overwrite"`. This writes the
   harmonized RatLesNetV2 dataset under `work/ratlesnetv2_finetune/datasets/`.
7. Tar `LYS_T2w_manual_v0` and upload the tarball to Google Drive. Raw Desktop
   source folders are not required on the cloud after the local harmonized
   export exists.
8. In Colab, mount Google Drive, clone this branch, install
   `ratlesnetv2_finetune/requirements-colab.txt`, clone upstream RatLesNetV2,
   extract the tarball to `/content`, and run one case for one epoch with
   `--max-train-cases 1 --save-every 1`.
9. Treat that first run as a runtime/data-contract smoke test only. Before a
   real finetune/evaluation, create explicit train/validation/test splits and
   reserve held-out LYS animals for validation/test.

Immediate next actions in a new session:

1. Commit and push the current nibabel compatibility fix if it has not already
   been pushed:

   ```bash
   git add ratlesnetv2_finetune/scripts/finetune_ratlesnetv2.py \
     tests/test_ratlesnetv2_finetune.py \
     ratlesnetv2_finetune/README.md \
     docs/ratlesnetv2_finetuning_branch.md
   git commit -m "Document and patch RatLesNetV2 Colab smoke test"
   git push origin dl-ratlesnetv2-finetune
   ```

2. In Colab, restart the runtime, reclone the branch, clone upstream
   RatLesNetV2, extract the uploaded tarballs, and rerun the one-case,
   one-epoch LYS smoke test.
3. If the LYS smoke test succeeds, run the same one-case smoke test on
   `External_Mouse_T2w_manual_LSP_SI_v0`.
4. After both smoke tests pass, create proper split folders from the prepared
   tarballs with `ratlesnetv2_finetune.scripts.split_prepared_dataset`:
   external `train + validation`, and LYS `train + validation + test`.
5. Run external mouse adaptation first, then LYS fine-tuning with `--test`.
   `scripts/finetune_ratlesnetv2.py` writes `metrics_epoch.csv`,
   `metrics_cases.csv`, and `final_metrics.json`; final performance must be
   reported only from the held-out LYS `test` split.

## Current Limitations

- One case is enough to smoke-test the path, not to estimate model performance.
- Finetuned predictions must still pass through the human mask review gate.
- Dataset export is single-modality T2w only.
- The upstream code expects `scan_lesionIAM.nii.gz`; the exporter also writes
  `scan_lesion.nii.gz` for README compatibility.
- The current tarball exports are all-train smoke-test datasets; split exports
  or split folders are required for real training/evaluation.
- Public dataset import, orientation correction, and deduplication have been
  run for the current external source folder. The documented policy remains
  manual/native labels only, no duplicates, and final validation on held-out
  LYS cases.
- The finetuning script reports Dice, IoU, accuracy, balanced accuracy,
  precision, recall, and specificity for evaluation splits. Overall accuracy is
  background-dominated for sparse lesions, so Dice/recall/precision should be
  read first.

## Acceptance For This Branch Setup

- `main` includes the committed `ihc` history.
- `ihc` remains as a branch for future work.
- `dl-ratlesnetv2-finetune` branches from updated `main`.
- Local tests cover dataset preparation and cloud-command planning without
  requiring PyTorch, model downloads, or network calls.
