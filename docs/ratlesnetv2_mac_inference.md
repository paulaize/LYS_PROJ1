# Frozen RatLesNetV2 inference on macOS

This procedure runs the frozen LYS v1 model exactly as selected: five direct
CE+Dice fold checkpoints, unweighted mean lesion probability, OOF threshold
`0.40`, and no postprocessing. Predictions remain draft masks requiring human
review.

## 1. Build the minimal download in the completed Kaggle session

Run this after Cell 19 of `ratlesnetv2_lys_v1_kaggle_workflow.md`, while
`FINAL_MODELS`, `FINAL_THRESHOLD_JSON`, `FROZEN_SPEC`, `PROJECT`, `RAT_REPO`,
and `resolved_ratlesnet_commit` are still defined:

```python
import hashlib
import json
import shutil

MAC_BUNDLE_ROOT = WORK / "LYS_v1_RatLesNetV2_mac_inference"
MAC_BUNDLE_ZIP = WORK / "LYS_v1_RatLesNetV2_mac_inference.zip"
assert not MAC_BUNDLE_ROOT.exists(), MAC_BUNDLE_ROOT
assert not MAC_BUNDLE_ZIP.exists(), MAC_BUNDLE_ZIP
assert (
    PROJECT
    / "ratlesnetv2_finetune/scripts/infer_ratlesnetv2_ensemble.py"
).is_file(), "Synchronize the updated project source before building the Mac bundle"

(MAC_BUNDLE_ROOT / "models").mkdir(parents=True)
portable_models = []
for fold, source in enumerate(FINAL_MODELS):
    destination = MAC_BUNDLE_ROOT / "models" / f"fold_{fold}.model"
    shutil.copy2(source, destination)
    portable_models.append(destination)

shutil.copy2(
    FINAL_THRESHOLD_JSON,
    MAC_BUNDLE_ROOT / "selected_threshold.json",
)
shutil.copy2(
    FROZEN_SPEC,
    MAC_BUNDLE_ROOT / "frozen_spec.json",
)

shutil.copytree(
    PROJECT / "ratlesnetv2_finetune",
    MAC_BUNDLE_ROOT / "ratlesnetv2_finetune",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)
shutil.copytree(
    RAT_REPO / "lib",
    MAC_BUNDLE_ROOT / "RatLesNetv2/lib",
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
)
if (RAT_REPO / "LICENSE").is_file():
    shutil.copy2(RAT_REPO / "LICENSE", MAC_BUNDLE_ROOT / "RatLesNetv2/LICENSE")

(MAC_BUNDLE_ROOT / "RatLesNetv2/UPSTREAM_GIT_COMMIT.txt").write_text(
    resolved_ratlesnet_commit + "\n"
)
(MAC_BUNDLE_ROOT / "requirements-inference.txt").write_text(
    "torch\nnibabel\nnumpy\nscipy\nscikit-image\npandas\npyyaml\n"
)

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

frozen = json.loads(FROZEN_SPEC.read_text())
assert sorted(file_sha256(path) for path in portable_models) == sorted(
    model["sha256"] for model in frozen["fold_models"]
)

bundle_manifest = {
    "models": [
        {
            "fold": fold,
            "file": f"models/fold_{fold}.model",
            "sha256": file_sha256(path),
        }
        for fold, path in enumerate(portable_models)
    ],
    "threshold": json.loads(FINAL_THRESHOLD_JSON.read_text())["selected_threshold"],
    "ensemble": "unweighted mean lesion probability",
    "postprocessing": "none",
    "ratlesnetv2_git_commit": resolved_ratlesnet_commit,
}
(MAC_BUNDLE_ROOT / "bundle_manifest.json").write_text(
    json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n"
)

archive = shutil.make_archive(
    str(MAC_BUNDLE_ZIP.with_suffix("")),
    "zip",
    root_dir=MAC_BUNDLE_ROOT,
)
print("Download from Kaggle outputs:", archive)
```

Download `LYS_v1_RatLesNetV2_mac_inference.zip`. The large final experiment
archive is valuable for long-term reproduction but is not required for routine
inference.

## 2. Prepare an input scan

The scan must come through the same preparation route as the training data. Do
not merely rename an arbitrary image. The inference gate requires:

- filename `scan.nii.gz`;
- shape `X x Y x slices x 1`;
- finite, non-constant intensities;
- spacing `0.07 x 0.07 x 0.5 mm`;
- a valid native-space affine.

Labels are not needed. A one-case input can be arranged as:

```text
mac_inference_input/
└── NewStudy/
    └── mixed/
        └── my_mouse_case/
            └── scan.nii.gz
```

## 3. Run on the existing `lys-bbb` environment

```bash
unzip LYS_v1_RatLesNetV2_mac_inference.zip -d LYS_v1_RatLesNetV2_mac_inference
cd LYS_v1_RatLesNetV2_mac_inference

conda run -n lys-bbb python -m pip install -r requirements-inference.txt

conda run -n lys-bbb python -m \
  ratlesnetv2_finetune.scripts.infer_ratlesnetv2_ensemble \
  --ratlesnet-repo RatLesNetv2 \
  --input /absolute/path/to/mac_inference_input \
  --model models/fold_0.model \
  --model models/fold_1.model \
  --model models/fold_2.model \
  --model models/fold_3.model \
  --model models/fold_4.model \
  --threshold-json selected_threshold.json \
  --frozen-spec frozen_spec.json \
  --output /absolute/path/to/inference_output \
  --device auto
```

`--device auto` prefers Apple MPS, then CUDA, then CPU. The output contains one
native-space `ensemble_probability.nii.gz` and `ensemble_mask.nii.gz` per case,
plus `inference_manifest.csv` and `inference_summary.json`. The command verifies
all five checkpoint hashes against the frozen specification before inference.
