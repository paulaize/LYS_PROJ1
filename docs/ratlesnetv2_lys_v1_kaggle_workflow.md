# LYS v1 RatLesNetV2: controlled Kaggle workflow

This is the active deep-learning workflow for the corrected `LYS_v1` masks.
Use a new Kaggle notebook. Do not reuse the historical `LYS_v0` notebook or
its split.

The experiment answers two questions in this order:

1. On the same five LYS development folds, is Tversky or CE+Dice better?
2. With the selected loss fixed, does an external-only adaptation stage improve
   LYS performance over direct LYS fine-tuning?

“Direct LYS” here means no external dataset is used: the upstream RatLesNetV2
rat checkpoint is fine-tuned directly on LYS. “External-pretrained” starts from
that same upstream checkpoint, adapts it on external train/validation only,
then fine-tunes on the same LYS folds with the same target-stage settings.

The locked test is absent from loss selection, threshold calibration,
checkpoint selection, and initialization selection. It is evaluated once only
after the five models, OOF threshold, and `postprocessing=none` are frozen.

## Required Kaggle inputs

Create or update private Kaggle datasets containing:

- `LYS_T2w_manual_v1.tar.gz` from
  `outputs/ratlesnetv2_finetune/LYS_T2w_manual_v1.tar.gz`;
- `External_Mouse_T2w_manual_LSP_SI_v0.tar.gz` from the Desktop archive;
- this Git repository pushed on branch `dl-ratlesnetv2-finetune`.

Kaggle may expose archive contents instead of the original tarball and may
inconsistently decompress or truncate NIfTI filenames. Cell 2
therefore identifies compression from the file bytes, fully validates every
payload, and writes canonical `.nii.gz` inputs under `/kaggle/working`. Do not
rename the Kaggle input files and do not write under `/kaggle/input`.

The LYS archive contains 258 cases after the four requested exclusions. The
external archive contains 426 prepared records. Its manifest has 426 distinct
`animal_id` values; the source split therefore groups on the provided
`animal_id`. If richer source-animal metadata becomes available later, rebuild
that external split before treating the comparison as definitive.

Enable a Kaggle GPU and Internet before starting. Keep the notebook private
because the datasets are not public deliverables.

## Cell 1 — identity, GPU, and repository setup

```python
from pathlib import Path
import json
import subprocess
import sys

RUN_SEED = 20260715
BRANCH = "dl-ratlesnetv2-finetune"
RATLESNET_COMMIT = "c9dfb7eddf5c3369151d2f7a64add9a42b2d6983"

subprocess.run(["nvidia-smi"], check=True)

WORK = Path("/kaggle/working")
PROJECT = WORK / "LYS_PROJ1"
if not (PROJECT / ".git").is_dir():
    subprocess.run(
        [
            "git", "clone", "--branch", BRANCH,
            "https://github.com/paulaize/LYS_PROJ1.git", str(PROJECT),
        ],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(PROJECT), "pull", "--ff-only"], check=True)

PROJECT_COMMIT = subprocess.check_output(
    ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
).strip()

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "-r",
        str(PROJECT / "ratlesnetv2_finetune/requirements-colab.txt"),
        "matplotlib",
    ],
    check=True,
)

RAT_REPO = WORK / "RatLesNetv2"
if not (RAT_REPO / ".git").is_dir():
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "https://github.com/jmlipman/RatLesNetv2.git", str(RAT_REPO),
        ],
        check=True,
    )

subprocess.run(
    [
        "git", "-C", str(RAT_REPO), "fetch", "--depth", "1",
        "origin", RATLESNET_COMMIT,
    ],
    check=True,
)
subprocess.run(
    ["git", "-C", str(RAT_REPO), "checkout", "--detach", RATLESNET_COMMIT],
    check=True,
)
resolved_ratlesnet_commit = subprocess.check_output(
    ["git", "-C", str(RAT_REPO), "rev-parse", "HEAD"], text=True
).strip()
assert resolved_ratlesnet_commit == RATLESNET_COMMIT

RAT_PRETRAINED = (
    RAT_REPO
    / "trained_models/Table2-3/RatLesNetv2/homogeneous/model-1"
)
assert RAT_PRETRAINED.is_file(), RAT_PRETRAINED
print("Project:", PROJECT)
print("Project commit:", PROJECT_COMMIT)
print("RatLesNetV2 commit:", resolved_ratlesnet_commit)
print("Upstream RatLesNet checkpoint:", RAT_PRETRAINED)
```

## Cell 2 — locate, validate, and normalize both prepared datasets

This reads `/kaggle/input` and extracts only into `/kaggle/working` when
Kaggle has not already exposed the archive contents. The normalizer detects
plain versus gzip data from the payload rather than the suffix, forces every
array to load, checks scan/mask geometry and label aliases, and writes
canonical `.nii.gz` inputs with a provenance report.

```python
import tarfile


def locate_or_extract_prepared_dataset(dataset_name: str) -> Path:
    input_root = Path("/kaggle/input")
    matches = sorted(input_root.rglob(f"{dataset_name}/manifest.csv"))
    if len(matches) == 1:
        return matches[0].parent
    assert not matches, f"Found multiple manifests for {dataset_name}: {matches}"

    archives = sorted(input_root.rglob(f"{dataset_name}.tar.gz"))
    assert len(archives) == 1, (
        f"Expected one {dataset_name}.tar.gz under /kaggle/input; found {archives}"
    )
    extract_root = WORK / "uploaded_archives" / dataset_name
    extract_root.mkdir(parents=True, exist_ok=True)
    root_resolved = extract_root.resolve()
    with tarfile.open(archives[0], "r:gz") as archive:
        for member in archive.getmembers():
            destination = (extract_root / member.name).resolve()
            assert destination == root_resolved or root_resolved in destination.parents
        archive.extractall(extract_root)
    matches = sorted(extract_root.rglob(f"{dataset_name}/manifest.csv"))
    assert len(matches) == 1, matches
    return matches[0].parent


NORMALIZED_BASE = WORK / "normalized_inputs"


def normalize_prepared_dataset(
    source_root: Path,
    *,
    dataset_name: str,
    expected_cases: int,
) -> Path:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ratlesnetv2_finetune.scripts.normalize_prepared_dataset",
            "--input", str(source_root),
            "--output-base", str(NORMALIZED_BASE),
            "--dataset-name", dataset_name,
            "--expected-cases", str(expected_cases),
        ],
        cwd=PROJECT,
        check=True,
    )
    result = NORMALIZED_BASE / dataset_name
    assert (result / "manifest.csv").is_file()
    assert (result / "normalization_report.csv").is_file()
    return result


LYS_SOURCE_ROOT = locate_or_extract_prepared_dataset("LYS_T2w_manual_v1")
EXTERNAL_SOURCE_ROOT = locate_or_extract_prepared_dataset(
    "External_Mouse_T2w_manual_LSP_SI_v0"
)
LYS_ROOT = normalize_prepared_dataset(
    LYS_SOURCE_ROOT,
    dataset_name="LYS_T2w_manual_v1",
    expected_cases=258,
)
EXTERNAL_ROOT = normalize_prepared_dataset(
    EXTERNAL_SOURCE_ROOT,
    dataset_name="External_Mouse_T2w_manual_LSP_SI_v0",
    expected_cases=426,
)

assert len(list(LYS_ROOT.rglob("scan.nii.gz"))) == 258
assert len(list(EXTERNAL_ROOT.rglob("scan.nii.gz"))) == 426
print("LYS_ROOT:", LYS_ROOT)
print("EXTERNAL_ROOT:", EXTERNAL_ROOT)
```

## Cell 3 — automatic LYS geometry and mask QC

```python
QC_ROOT = WORK / "LYS_v1_qc"
subprocess.run(
    [
        sys.executable, "-m",
        "ratlesnetv2_finetune.scripts.audit_prepared_dataset",
        "--input", str(LYS_ROOT),
        "--output", str(QC_ROOT),
        "--expected-spacing", "0.07", "0.07", "0.5",
        "--spacing-tolerance", "0.01",
        "--label-version", "LYS_v1",
        "--overwrite",
    ],
    cwd=PROJECT,
    check=True,
)
```

## Cell 4 — inspect QC before training

```python
import pandas as pd
from IPython.display import HTML, Image, display

qc_summary = json.loads((QC_ROOT / "qc_summary.json").read_text())
qc = pd.read_csv(QC_ROOT / "dataset_qc.csv")
display(qc_summary)
display(qc[qc.qc_flag != "pass"].sort_values(["qc_flag", "lesion_volume_mm3"]))
display(Image(filename=str(QC_ROOT / "flagged_contact_sheet.png")))
display(HTML((QC_ROOT / "overlay_gallery.html").read_text()))
```

Review every flagged case and a random sample from every cohort. If a mask or
scan-mask pairing is wrong, correct the source review folder, rebuild the v1
archive, and restart with a new version. A QC flag alone is not a reason to
delete a case.

## Cell 5 — lock 20% of inferred subjects and create five LYS folds

The conservative inference keeps the clear D1/D7 pairs together and treats all
other case IDs as singletons. Save the resulting assignment CSV permanently;
do not regenerate it between experiments.

```python
GROUPED_ROOT = WORK / "LYS_v1_grouped"
subprocess.run(
    [
        sys.executable, "-m",
        "ratlesnetv2_finetune.scripts.create_grouped_cv",
        "--input", str(LYS_ROOT),
        "--infer-longitudinal-subjects",
        "--label-version", "LYS_v1",
        "--output", str(GROUPED_ROOT),
        "--test-fraction", "0.20",
        "--folds", "5",
        "--seed", str(RUN_SEED),
        "--copy-mode", "symlink",
        "--overwrite",
    ],
    cwd=PROJECT,
    check=True,
)

assignments = pd.read_csv(
    GROUPED_ROOT / "split_assignments.csv", keep_default_na=False
)
assert assignments.groupby("subject_id").outer_split.nunique().max() == 1
development = assignments[assignments.outer_split == "development"]
assert development.groupby("subject_id").cv_fold.nunique().max() == 1
assert set(development.cv_fold.astype(str)) == {"0", "1", "2", "3", "4"}
assert set(assignments[assignments.outer_split == "test"].cv_fold) == {""}
display(json.loads((GROUPED_ROOT / "split_summary.json").read_text()))
display(pd.crosstab([assignments.outer_split, assignments.cv_fold], assignments.cohort))
```

## Cell 6 — create the external-only train/validation split

No LYS case is used in this split or in selection of the external source
checkpoint. There is intentionally no external test partition because this
stage is representation adaptation, not the final scientific evaluation.

```python
EXTERNAL_SPLIT = WORK / "External_pretrain_split"
subprocess.run(
    [
        sys.executable, "-m",
        "ratlesnetv2_finetune.scripts.split_prepared_dataset",
        "--input", str(EXTERNAL_ROOT),
        "--output", str(EXTERNAL_SPLIT),
        "--group-by", "animal_id",
        "--validation-fraction", "0.20",
        "--test-fraction", "0",
        "--seed", str(RUN_SEED),
        "--copy-mode", "symlink",
        "--overwrite",
    ],
    cwd=PROJECT,
    check=True,
)

external_manifest = pd.read_csv(EXTERNAL_SPLIT / "manifest.csv")
external_manifest["source_dataset"] = external_manifest.case_id.str.split("__").str[0]
assert len(external_manifest) == 426
assert not (
    set(external_manifest.loc[external_manifest.split == "train", "animal_id"])
    & set(external_manifest.loc[external_manifest.split == "validation", "animal_id"])
)
assert len(list((EXTERNAL_SPLIT / "train").rglob("scan.nii.gz"))) == 341
assert len(list((EXTERNAL_SPLIT / "validation").rglob("scan.nii.gz"))) == 85
display(pd.crosstab(external_manifest.split, external_manifest.source_dataset))
```

With seed `20260715`, the expected source counts are:

| split | An2022 | Knab2025 | Koch2017 |
|---|---:|---:|---:|
| train | 265 | 66 | 10 |
| validation | 66 | 14 | 5 |

Stop if the table differs unexpectedly or a source family is absent from
validation.

## Cell 7 — reusable training helpers

The target-stage policy is fixed for every comparison: maximum 50 epochs,
validation every epoch, best checkpoint by validation Dice, early stopping 12,
LR plateau patience 4, and fold seed `RUN_SEED + fold`. Final validation
probability maps are exported from the restored best checkpoint.

```python
import numpy as np

RUNS_ROOT = WORK / "lys_v1_runs"
THRESHOLD_ROOT = WORK / "lys_v1_thresholds"

LOSS_CONFIGS = {
    "direct_tversky": {
        "loss": "tversky",
        "lr": 1e-5,
        "extra": ["--tversky-alpha", "0.3", "--tversky-beta", "0.7"],
    },
    "direct_ce_dice": {
        "loss": "ce-dice",
        "lr": 5e-5,
        "extra": [],
    },
}

def successful_run(
    output_root: Path,
    *,
    require_model: bool = True,
) -> Path | None:
    if not output_root.is_dir():
        return None
    candidates = sorted(
        (path for path in output_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for run in reversed(candidates):
        status_path = run / "run_status.json"
        model_path = run / "RatLesNetv2.model"
        if not status_path.is_file():
            continue
        if require_model and not model_path.is_file():
            continue
        status = json.loads(status_path.read_text()).get("status")
        if status in {"completed", "early_stopped", "evaluated"}:
            return run
    return None

def ratlesnet_command(
    *,
    train: Path,
    validation: Path | None,
    output: Path,
    pretrained: Path,
    loss_config: dict,
    seed: int,
    export_validation: bool,
) -> list[str]:
    command = [
        sys.executable, "-m",
        "ratlesnetv2_finetune.scripts.finetune_ratlesnetv2",
        "--ratlesnet-repo", str(RAT_REPO),
        "--input", str(train),
        "--output", str(output),
        "--pretrained-model", str(pretrained),
        "--require-pretrained",
        "--loss", loss_config["loss"],
        "--lr", str(loss_config["lr"]),
        "--epochs", "50",
        "--seed", str(seed),
        "--gpu", "0",
        "--loadMemory", "0",
        "--save-every", "0",
        "--eval-every", "1",
        "--metrics-threshold", "0.5",
        "--early-stop-patience", "12",
        "--early-stop-min-delta", "0.001",
        "--lr-scheduler", "reduce-on-plateau",
        "--lr-scheduler-metric", "validation_dice",
        "--lr-plateau-patience", "4",
        "--lr-plateau-factor", "0.5",
        "--lr-plateau-min-delta", "0.001",
        "--min-lr", "1e-6",
    ] + list(loss_config["extra"])
    if validation is not None:
        command += ["--validation", str(validation)]
    if export_validation:
        command += [
            "--export-predictions", "validation",
            "--export-prediction-limit", "0",
            "--export-prediction-epochs", "final",
        ]
    return command

def train_if_needed(**kwargs) -> Path:
    output = Path(kwargs["output"])
    existing = successful_run(output)
    if existing is not None:
        print("Using completed run:", existing)
        return existing
    command = ratlesnet_command(**kwargs)
    print("Running:", " ".join(map(str, command)))
    subprocess.run(command, cwd=PROJECT, check=True)
    completed = successful_run(output)
    assert completed is not None, output
    return completed
```

## Cell 8 — train both direct losses on folds 0–4

This is 10 target runs. Do not choose a loss using fold 0 alone. If Kaggle time
or quota requires several sessions, change `FOLDS_TO_RUN` to the missing folds,
restore the earlier run folders as notebook inputs, and let `train_if_needed`
skip completed runs. Preserve the entire `lys_v1_runs` output between sessions.

```python
FOLDS_TO_RUN = [0, 1, 2, 3, 4]

for candidate_name, loss_config in LOSS_CONFIGS.items():
    for fold in FOLDS_TO_RUN:
        fold_root = GROUPED_ROOT / "folds" / f"fold_{fold}"
        train_if_needed(
            train=fold_root / "train",
            validation=fold_root / "validation",
            output=RUNS_ROOT / candidate_name / f"fold_{fold}",
            pretrained=RAT_PRETRAINED,
            loss_config=loss_config,
            seed=RUN_SEED + fold,
            export_validation=True,
        )
```

## Cell 9 — verify complete OOF coverage and calibrate each loss separately

```python
def require_five_oof_manifests(candidate_name: str) -> list[Path]:
    from ratlesnetv2_finetune.scripts.calibrate_probability_threshold import (
        _canonical_prediction_case_id,
    )

    manifests = []
    for fold in range(5):
        run = successful_run(RUNS_ROOT / candidate_name / f"fold_{fold}")
        assert run is not None, f"Missing completed {candidate_name} fold {fold}"
        manifest = (
            run
            / "prediction_exports/validation/final/prediction_export_manifest.csv"
        )
        assert manifest.is_file(), manifest
        manifests.append(manifest)
    rows = pd.concat([pd.read_csv(path) for path in manifests], ignore_index=True)
    expected = set(
        assignments.loc[assignments.outer_split == "development", "case_id"]
    )
    rows["case_id"] = [
        _canonical_prediction_case_id(row)
        for row in rows.to_dict(orient="records")
    ]
    assert len(rows) == len(expected)
    assert rows.case_id.is_unique
    assert set(rows.case_id) == expected
    assert set(rows.split) == {"validation"}
    return manifests

for candidate_name in LOSS_CONFIGS:
    require_five_oof_manifests(candidate_name)
    subprocess.run(
        [
            sys.executable, "-m",
            "ratlesnetv2_finetune.scripts.calibrate_probability_threshold",
            "--prediction-root", str(RUNS_ROOT / candidate_name),
            "--metadata", str(GROUPED_ROOT / "split_assignments.csv"),
            "--output", str(THRESHOLD_ROOT / candidate_name),
            "--thresholds", "0.20:0.80:0.05",
            "--surface-tolerance-mm", "0.2",
            "--bootstrap-samples", "2000",
            "--seed", str(RUN_SEED),
            "--overwrite",
        ],
        cwd=PROJECT,
        check=True,
    )

display(pd.DataFrame([
    json.loads((THRESHOLD_ROOT / name / "selected_threshold.json").read_text())
    for name in LOSS_CONFIGS
]))
```

## Cell 10 — paired OOF comparison helper

The comparison uses the same cases and reports the paired Dice difference,
95% paired bootstrap interval, fold wins, failure rates, volume error, and
cohort differences. A positive difference means the candidate named second is
better.

```python
def compare_oof(
    reference_name: str,
    reference_csv: Path,
    candidate_name: str,
    candidate_csv: Path,
    output_root: Path,
) -> dict:
    reference = pd.read_csv(reference_csv).set_index("case_id").sort_index()
    candidate = pd.read_csv(candidate_csv).set_index("case_id").sort_index()
    assert reference.index.is_unique and candidate.index.is_unique
    assert reference.index.equals(candidate.index)

    metadata_columns = [
        column for column in
        ["subject_id", "cv_fold", "cohort", "timepoint", "lesion_volume_bin"]
        if column in reference.columns
    ]
    paired = reference[metadata_columns].copy()
    metrics = [
        "dice", "precision", "recall", "absolute_volume_error_mm3",
        "volume_error_mm3", "hd95_mm", "surface_dice",
    ]
    for metric in metrics:
        paired[f"{reference_name}_{metric}"] = pd.to_numeric(
            reference[metric], errors="coerce"
        )
        paired[f"{candidate_name}_{metric}"] = pd.to_numeric(
            candidate[metric], errors="coerce"
        )
        paired[f"candidate_minus_reference_{metric}"] = (
            paired[f"{candidate_name}_{metric}"]
            - paired[f"{reference_name}_{metric}"]
        )
    paired = paired.reset_index()

    dice_difference = paired["candidate_minus_reference_dice"].to_numpy()
    rng = np.random.default_rng(RUN_SEED)
    indices = rng.integers(
        0, len(dice_difference), size=(2000, len(dice_difference))
    )
    bootstrap = dice_difference[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    fold_summary = (
        paired.groupby("cv_fold", dropna=False)[
            [f"{reference_name}_dice", f"{candidate_name}_dice"]
        ]
        .mean()
        .reset_index()
    )
    fold_summary["candidate_minus_reference_dice"] = (
        fold_summary[f"{candidate_name}_dice"]
        - fold_summary[f"{reference_name}_dice"]
    )
    cohort_summary = (
        paired.groupby("cohort", dropna=False)[
            [f"{reference_name}_dice", f"{candidate_name}_dice"]
        ]
        .mean()
        .reset_index()
    )
    cohort_summary["candidate_minus_reference_dice"] = (
        cohort_summary[f"{candidate_name}_dice"]
        - cohort_summary[f"{reference_name}_dice"]
    )
    summary = {
        "reference": reference_name,
        "candidate": candidate_name,
        "n_cases": int(len(paired)),
        "reference_mean_dice": float(paired[f"{reference_name}_dice"].mean()),
        "candidate_mean_dice": float(paired[f"{candidate_name}_dice"].mean()),
        "candidate_minus_reference_mean_dice": float(dice_difference.mean()),
        "paired_dice_difference_ci95_low": float(ci_low),
        "paired_dice_difference_ci95_high": float(ci_high),
        "reference_zero_dice_cases": int((paired[f"{reference_name}_dice"] == 0).sum()),
        "candidate_zero_dice_cases": int((paired[f"{candidate_name}_dice"] == 0).sum()),
        "reference_mean_absolute_volume_error_mm3": float(
            paired[f"{reference_name}_absolute_volume_error_mm3"].mean()
        ),
        "candidate_mean_absolute_volume_error_mm3": float(
            paired[f"{candidate_name}_absolute_volume_error_mm3"].mean()
        ),
        "candidate_fold_wins": int(
            (fold_summary.candidate_minus_reference_dice > 0).sum()
        ),
        "reference_fold_wins": int(
            (fold_summary.candidate_minus_reference_dice < 0).sum()
        ),
        "fold_ties": int((fold_summary.candidate_minus_reference_dice == 0).sum()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_root / "paired_cases.csv", index=False)
    fold_summary.to_csv(output_root / "paired_folds.csv", index=False)
    cohort_summary.to_csv(output_root / "paired_cohorts.csv", index=False)
    (output_root / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    display(summary)
    display(fold_summary)
    display(cohort_summary)
    return summary
```

## Cell 11 — compare Tversky with CE+Dice and select the loss

```python
loss_comparison = compare_oof(
    "direct_tversky",
    THRESHOLD_ROOT / "direct_tversky/selected_threshold_case_metrics.csv",
    "direct_ce_dice",
    THRESHOLD_ROOT / "direct_ce_dice/selected_threshold_case_metrics.csv",
    WORK / "lys_v1_comparisons/loss",
)
```

Use the OOF mean Dice as the primary endpoint, then check that the direction is
supported in most folds, does not increase complete failures, does not create
unacceptable volume error, and does not cause a major cohort-specific drop. If
these disagree, treat the result as inconclusive rather than selecting from a
single fold. Save the three comparison CSVs and JSON. They are the audit trail.

Set exactly one value after reviewing the output:

```python
# Change only after reviewing Cell 11. Ask for review if the evidence conflicts.
SELECTED_DIRECT = None  # "direct_tversky" or "direct_ce_dice"
assert SELECTED_DIRECT in LOSS_CONFIGS, "Select the better five-fold OOF loss first."
SELECTED_LOSS_CONFIG = LOSS_CONFIGS[SELECTED_DIRECT]
print("Selected LYS loss:", SELECTED_DIRECT, SELECTED_LOSS_CONFIG)
```

## Cell 12 — create one clean external-pretrained checkpoint

The selected LYS loss and LR are now fixed. The source checkpoint is trained
from the same upstream rat checkpoint and selected by external validation Dice
only. LYS data do not enter this cell.

```python
EXTERNAL_SOURCE_OUTPUT = RUNS_ROOT / "external_source" / SELECTED_DIRECT
external_source_run = train_if_needed(
    train=EXTERNAL_SPLIT / "train",
    validation=EXTERNAL_SPLIT / "validation",
    output=EXTERNAL_SOURCE_OUTPUT,
    pretrained=RAT_PRETRAINED,
    loss_config=SELECTED_LOSS_CONFIG,
    seed=RUN_SEED,
    export_validation=False,
)
EXTERNAL_CHECKPOINT = external_source_run / "RatLesNetv2.model"
source_selection = json.loads(
    (external_source_run / "selected_checkpoint.json").read_text()
)
assert EXTERNAL_CHECKPOINT.is_file()
display(source_selection)
print("Frozen external source checkpoint:", EXTERNAL_CHECKPOINT)
```

## Cell 13 — fine-tune the external checkpoint on LYS folds 0–4

Only initialization differs from the selected direct runs. Loss, LR, folds,
epochs, seed policy, scheduler, checkpoint rule, and OOF export are identical.

```python
EXTERNAL_TARGET_NAME = (
    "external_pretrained_" + SELECTED_DIRECT.removeprefix("direct_")
)

for fold in [0, 1, 2, 3, 4]:
    fold_root = GROUPED_ROOT / "folds" / f"fold_{fold}"
    train_if_needed(
        train=fold_root / "train",
        validation=fold_root / "validation",
        output=RUNS_ROOT / EXTERNAL_TARGET_NAME / f"fold_{fold}",
        pretrained=EXTERNAL_CHECKPOINT,
        loss_config=SELECTED_LOSS_CONFIG,
        seed=RUN_SEED + fold,
        export_validation=True,
    )
```

## Cell 14 — calibrate the external-initialized OOF threshold

```python
require_five_oof_manifests(EXTERNAL_TARGET_NAME)
subprocess.run(
    [
        sys.executable, "-m",
        "ratlesnetv2_finetune.scripts.calibrate_probability_threshold",
        "--prediction-root", str(RUNS_ROOT / EXTERNAL_TARGET_NAME),
        "--metadata", str(GROUPED_ROOT / "split_assignments.csv"),
        "--output", str(THRESHOLD_ROOT / EXTERNAL_TARGET_NAME),
        "--thresholds", "0.20:0.80:0.05",
        "--surface-tolerance-mm", "0.2",
        "--bootstrap-samples", "2000",
        "--seed", str(RUN_SEED),
        "--overwrite",
    ],
    cwd=PROJECT,
    check=True,
)
display(json.loads(
    (THRESHOLD_ROOT / EXTERNAL_TARGET_NAME / "selected_threshold.json").read_text()
))
```

## Cell 15 — paired direct-versus-external comparison

```python
initialization_comparison = compare_oof(
    SELECTED_DIRECT,
    THRESHOLD_ROOT / SELECTED_DIRECT / "selected_threshold_case_metrics.csv",
    EXTERNAL_TARGET_NAME,
    THRESHOLD_ROOT / EXTERNAL_TARGET_NAME / "selected_threshold_case_metrics.csv",
    WORK / "lys_v1_comparisons/initialization",
)
```

Keep external pretraining only if its improvement is supported across most
folds and is not driven by one cohort or by a worse complete-failure/volume
profile. A confidence interval crossing zero does not prove “no effect,” but it
does mean the external route is not decisively established by this experiment.

Set exactly one value after reviewing the output:

```python
# Change only after reviewing Cell 15.
SELECTED_INITIALIZATION = None  # "direct" or "external_pretrained"
assert SELECTED_INITIALIZATION in {"direct", "external_pretrained"}

FINAL_CANDIDATE = (
    SELECTED_DIRECT
    if SELECTED_INITIALIZATION == "direct"
    else EXTERNAL_TARGET_NAME
)
print("Final candidate:", FINAL_CANDIDATE)
```

## Cell 16 — freeze the five models, OOF threshold, and postprocessing

No postprocessing has yet been validated on OOF data, so the scientifically
clean frozen rule is `postprocessing=none`. Do not add component removal or
hole filling after seeing the test.

```python
import hashlib
import platform

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

FINAL_MODELS = []
for fold in range(5):
    run = successful_run(RUNS_ROOT / FINAL_CANDIDATE / f"fold_{fold}")
    assert run is not None, f"Missing final fold {fold}"
    model = run / "RatLesNetv2.model"
    FINAL_MODELS.append(model)

FINAL_THRESHOLD_JSON = THRESHOLD_ROOT / FINAL_CANDIDATE / "selected_threshold.json"
threshold_record = json.loads(FINAL_THRESHOLD_JSON.read_text())
assert threshold_record["selection_data"] == "out_of_fold_validation_only"
assert threshold_record["locked_test_used"] is False

frozen_spec = {
    "dataset": "LYS_v1",
    "project_git_commit": PROJECT_COMMIT,
    "ratlesnetv2_git_commit": resolved_ratlesnet_commit,
    "split_assignments": str(GROUPED_ROOT / "split_assignments.csv"),
    "normalized_input_provenance": {
        "lys_manifest_sha256": sha256(LYS_ROOT / "manifest.csv"),
        "lys_report": str(LYS_ROOT / "normalization_report.csv"),
        "external_manifest_sha256": sha256(EXTERNAL_ROOT / "manifest.csv"),
        "external_report": str(EXTERNAL_ROOT / "normalization_report.csv"),
    },
    "architecture": "RatLesNetV2",
    "selected_loss_run": SELECTED_DIRECT,
    "selected_initialization": SELECTED_INITIALIZATION,
    "external_source_checkpoint": (
        str(EXTERNAL_CHECKPOINT)
        if SELECTED_INITIALIZATION == "external_pretrained"
        else None
    ),
    "fold_models": [
        {"fold": fold, "path": str(path), "sha256": sha256(path)}
        for fold, path in enumerate(FINAL_MODELS)
    ],
    "threshold_json": str(FINAL_THRESHOLD_JSON),
    "threshold": threshold_record["selected_threshold"],
    "threshold_source": "pooled OOF validation probabilities",
    "ensemble": "unweighted mean lesion probability",
    "postprocessing": "none",
    "seed_policy": "20260715 + fold",
    "python": platform.python_version(),
}
FROZEN_SPEC = WORK / "lys_v1_final_frozen_spec.json"
FROZEN_SPEC.write_text(json.dumps(frozen_spec, indent=2, sort_keys=True) + "\n")
display(frozen_spec)
```

Save the frozen JSON, both normalization reports, both paired-comparison
folders, OOF threshold folder, all five final checkpoints, split assignments,
and source checkpoint if selected. Only then continue.

## Cell 17 — locked-test gate

Read and deliberately acknowledge this cell. It prevents an accidental early
test run.

```python
RUN_LOCKED_TEST_ONCE = False
assert RUN_LOCKED_TEST_ONCE, (
    "Set RUN_LOCKED_TEST_ONCE=True only after Cell 16 artifacts are saved and "
    "no model, threshold, or postprocessing choice remains."
)
```

## Cell 18 — export one locked-test probability map per frozen fold model

Do not inspect per-model test metrics between folds. Complete all five exports
and the ensemble before opening the final report.

```python
LOCKED_TEST = GROUPED_ROOT / "locked_test" / "test"
TEST_EXPORT_ROOT = WORK / "lys_v1_locked_test_fold_predictions"
FROZEN_THRESHOLD = str(threshold_record["selected_threshold"])

def export_test_if_needed(fold: int, model: Path) -> Path:
    output = TEST_EXPORT_ROOT / f"fold_{fold}"
    existing = successful_run(output, require_model=False)
    if existing is None:
        fold_train = GROUPED_ROOT / "folds" / f"fold_{fold}" / "train"
        command = [
            sys.executable, "-m",
            "ratlesnetv2_finetune.scripts.finetune_ratlesnetv2",
            "--ratlesnet-repo", str(RAT_REPO),
            "--input", str(fold_train),
            "--test", str(LOCKED_TEST),
            "--output", str(output),
            "--pretrained-model", str(model),
            "--require-pretrained",
            "--eval-only",
            "--loss", SELECTED_LOSS_CONFIG["loss"],
            "--lr", str(SELECTED_LOSS_CONFIG["lr"]),
            "--metrics-threshold", FROZEN_THRESHOLD,
            "--export-predictions", "test",
            "--export-prediction-limit", "0",
            "--export-prediction-epochs", "final",
            "--gpu", "0",
            "--loadMemory", "0",
        ] + list(SELECTED_LOSS_CONFIG["extra"])
        subprocess.run(command, cwd=PROJECT, check=True)
        existing = successful_run(output, require_model=False)
    assert existing is not None
    manifest = existing / "prediction_exports/test/final/prediction_export_manifest.csv"
    assert manifest.is_file(), manifest
    return manifest

TEST_MANIFESTS = [
    export_test_if_needed(fold, model)
    for fold, model in enumerate(FINAL_MODELS)
]
assert len(TEST_MANIFESTS) == 5
```

## Cell 19 — average the five probabilities and evaluate the test once

The ensemble evaluator rejects validation manifests, requires five identical
test case sets, requires an OOF-derived threshold JSON, and refuses to replace
an existing final output unless explicitly forced. Do not add `--overwrite`.

```python
FINAL_TEST_OUTPUT = WORK / "LYS_v1_FINAL_LOCKED_TEST"
command = [
    sys.executable, "-m",
    "ratlesnetv2_finetune.scripts.evaluate_probability_ensemble",
    "--threshold-json", str(FINAL_THRESHOLD_JSON),
    "--metadata", str(GROUPED_ROOT / "split_assignments.csv"),
    "--output", str(FINAL_TEST_OUTPUT),
    "--expected-models", "5",
    "--surface-tolerance-mm", "0.2",
    "--bootstrap-samples", "2000",
    "--seed", str(RUN_SEED),
]
for manifest in TEST_MANIFESTS:
    command += ["--prediction-manifest", str(manifest)]
subprocess.run(command, cwd=PROJECT, check=True)

final_summary = json.loads(
    (FINAL_TEST_OUTPUT / "locked_test_summary.json").read_text()
)
display(final_summary)
display(pd.read_csv(FINAL_TEST_OUTPUT / "locked_test_subgroups.csv"))
```

This locked-test result is the final unbiased estimate for the frozen model. Do
not change the model because one test cohort or case performed poorly. Any
future change starts a new version and needs a new untouched test design.

## Cell 20 — end-of-run paired visual QC of all three OOF candidates

Run this only after the one-time locked-test evaluation above. It compares the
three candidates' development OOF masks with their manual masks at each
candidate's selected OOF threshold. It does not read the locked test. Direct
CE+Dice selects four lowest-Dice cases plus four cases spread from the first
quartile through the best result. Tversky and external-pretrained CE+Dice then
use those exact case IDs, anatomical axes, and slices for a paired comparison.

This is a final audit visualization, not a new selection step. Do not use it to
change the completed model, threshold, split, postprocessing, or case
inclusion. Any problem discovered here belongs to a future version with a new
untouched test design.

```python
QC_CANDIDATES = {
    "direct_ce_dice": "Direct CE+Dice",
    "direct_tversky": "Direct Tversky",
    "external_pretrained_ce_dice": "External-pretrained CE+Dice",
}
OOF_QC_PNGS = {}
DIRECT_CE_QC_PNG = WORK / "direct_ce_dice_oof_qc.png"
DIRECT_CE_SELECTION = DIRECT_CE_QC_PNG.with_suffix(".csv")

for candidate_name, candidate_label in QC_CANDIDATES.items():
    output_png = WORK / f"{candidate_name}_oof_qc.png"
    command = [
        sys.executable, "-m",
        "ratlesnetv2_finetune.scripts.create_oof_qc_contact_sheet",
        "--prediction-root", str(RUNS_ROOT / candidate_name),
        "--threshold-json",
        str(THRESHOLD_ROOT / candidate_name / "selected_threshold.json"),
        "--case-metrics",
        str(
            THRESHOLD_ROOT
            / candidate_name
            / "selected_threshold_case_metrics.csv"
        ),
        "--candidate-label", candidate_label,
        "--output", str(output_png),
        "--cases", "8",
        "--overwrite",
    ]
    if candidate_name != "direct_ce_dice":
        command += ["--reference-selection", str(DIRECT_CE_SELECTION)]
    subprocess.run(command, cwd=PROJECT, check=True)
    assert output_png.is_file()
    OOF_QC_PNGS[candidate_name] = output_png
    display(Image(filename=str(output_png)))
    display(pd.read_csv(output_png.with_suffix(".csv")))
    print("Saved Kaggle output PNG:", output_png)
```

Panel colors are:

- manual mask: green;
- candidate prediction: red;
- error panel: true positive yellow, false positive red, false negative cyan.

## Cell 21 — package the final reproducibility artifacts

This deliberately excludes the normalized image copies and the symlinked fold
trees because the source datasets already exist as private Kaggle inputs. It
includes the input manifests and normalization reports, QC, split provenance,
all training runs and checkpoints, OOF outputs, comparisons, frozen rule, test
probability exports, and final ensemble report. Do not overwrite an older
bundle from a previous scientific run.

```python
ARTIFACT_BUNDLE = WORK / "LYS_v1_RatLesNetV2_final_artifacts.tar.gz"
assert not ARTIFACT_BUNDLE.exists(), ARTIFACT_BUNDLE

required_group_metadata = [
    GROUPED_ROOT / "split_assignments.csv",
    GROUPED_ROOT / "inferred_subject_groups.csv",
    GROUPED_ROOT / "split_summary.json",
]
required_external_metadata = [
    EXTERNAL_SPLIT / "manifest.csv",
    EXTERNAL_SPLIT / "split_summary.json",
]
for path in required_group_metadata + required_external_metadata:
    assert path.is_file(), path

bundle_items = [
    (PROJECT / "docs/ratlesnetv2_lys_v1_kaggle_workflow.md", "guide.md"),
    (LYS_ROOT / "manifest.csv", "input_provenance/LYS_manifest.csv"),
    (
        LYS_ROOT / "normalization_report.csv",
        "input_provenance/LYS_normalization_report.csv",
    ),
    (
        EXTERNAL_ROOT / "manifest.csv",
        "input_provenance/external_manifest.csv",
    ),
    (
        EXTERNAL_ROOT / "normalization_report.csv",
        "input_provenance/external_normalization_report.csv",
    ),
    (QC_ROOT, "lys_qc"),
    (RUNS_ROOT, "runs"),
    (THRESHOLD_ROOT, "thresholds"),
    (WORK / "lys_v1_comparisons", "comparisons"),
    (FROZEN_SPEC, "frozen/lys_v1_final_frozen_spec.json"),
    (TEST_EXPORT_ROOT, "locked_test/fold_predictions"),
    (FINAL_TEST_OUTPUT, "locked_test/final_ensemble"),
]
bundle_items += [
    (path, f"comparisons/{candidate_name}_oof_qc.png")
    for candidate_name, path in OOF_QC_PNGS.items()
]
bundle_items += [
    (path.with_suffix(".csv"), f"comparisons/{candidate_name}_oof_qc.csv")
    for candidate_name, path in OOF_QC_PNGS.items()
]
bundle_items += [
    (path, f"split_provenance/LYS/{path.name}")
    for path in required_group_metadata
]
bundle_items += [
    (path, f"split_provenance/external/{path.name}")
    for path in required_external_metadata
]

for source, _ in bundle_items:
    assert source.exists(), source

bundle_record = {
    "project_git_commit": PROJECT_COMMIT,
    "ratlesnetv2_git_commit": resolved_ratlesnet_commit,
    "final_candidate": FINAL_CANDIDATE,
    "selected_initialization": SELECTED_INITIALIZATION,
    "selected_threshold": threshold_record["selected_threshold"],
    "items": [archive_name for _, archive_name in bundle_items],
}
BUNDLE_RECORD = WORK / "LYS_v1_artifact_bundle_contents.json"
BUNDLE_RECORD.write_text(
    json.dumps(bundle_record, indent=2, sort_keys=True) + "\n"
)
bundle_items.append((BUNDLE_RECORD, "bundle_contents.json"))

with tarfile.open(ARTIFACT_BUNDLE, "w:gz") as archive:
    for source, archive_name in bundle_items:
        archive.add(source, arcname=archive_name, recursive=True)

print("Artifact bundle:", ARTIFACT_BUNDLE)
print("Size GiB:", round(ARTIFACT_BUNDLE.stat().st_size / 1024**3, 3))
print("SHA-256:", sha256(ARTIFACT_BUNDLE))
```

Use Kaggle **Save Version** with output saving enabled. Confirm that the
artifact tarball appears in the saved notebook output before ending the
session.

## What is intentionally deferred

- nnU-Net is a valuable later benchmark, but it is not required to complete
  this controlled RatLesNetV2 path.
- Additional losses are deferred because Tversky and CE+Dice already represent
  the most credible direct candidates from the preliminary work.
- Intensity normalization, N4, augmentation, small-lesion sampling, and
  postprocessing
  ablations should be tested one at a time using OOF development predictions
  before another locked-test version; none should be improvised after Cell 17.
- negative-control performance can only be estimated when representative
  lesion-negative LYS cases are available and explicitly labeled.

The experiment is complete only when both normalization reports, the split,
15 target fold checkpoints (10 direct plus 5 external-initialized), one
external source checkpoint, three OOF calibrations, two paired-comparison
reports, frozen specification, five final test probability exports, and final
ensemble report are preserved in the final artifact bundle.
