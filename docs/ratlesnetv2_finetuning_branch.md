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

1. Download external public archives outside git if public mouse adaptation is
   being used.
2. Convert/import only native-space manual public masks into source-folder
   format, following `docs/ratlesnetv2_external_datasets.md`.
3. If LYS manual lesion outlines are Fiji ROI Manager exports, run
   `make ratlesnetv2-roiset-to-mask` first to create one
   `case_lesion_mask.nii.gz` beside or inside the mask source folder for each
   original `case.nii.gz`.
4. Copy `configs/dataset_from_folders_template.yml` to a local, gitignored
   dataset plan such as `ratlesnetv2_finetune/configs/local_dataset.yml`.
5. Run `make ratlesnetv2-add-source` once per local source folder. Each source
   folder should contain one T2w scan subfolder and one lesion-mask subfolder.
   Filenames are matched as `name.nii.gz` -> `name_lesion_mask.nii.gz`.
6. Run a geometry/provenance report before final export. External data are not
   LYS geometry; duplicates and atlas-space masks must be excluded.
7. Run `make ratlesnetv2-prepare` once after all folders have been added. This
   writes one harmonized RatLesNetV2 dataset under `work/`.
8. Upload or copy the prepared dataset folder/tarball, this repo branch, and
   optional pretrained weights to Colab. Raw source folders are not required on
   the cloud after the local harmonized export exists.
9. Run the one-case cloud smoke test before any longer finetune.

## Current Limitations

- One case is enough to smoke-test the path, not to estimate model performance.
- Finetuned predictions must still pass through the human mask review gate.
- Dataset export is single-modality T2w only.
- The upstream code expects `scan_lesionIAM.nii.gz`; the exporter also writes
  `scan_lesion.nii.gz` for README compatibility.
- Public dataset importers and deduplication are not fully automated yet. The
  documented policy is manual/native labels only, no duplicates, and final
  validation on held-out LYS cases.

## Acceptance For This Branch Setup

- `main` includes the committed `ihc` history.
- `ihc` remains as a branch for future work.
- `dl-ratlesnetv2-finetune` branches from updated `main`.
- Local tests cover dataset preparation and cloud-command planning without
  requiring PyTorch, model downloads, or network calls.
