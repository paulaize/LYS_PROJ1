"""Command builders for cloud RatLesNetV2 runs."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

UPSTREAM_GIT_URL = "https://github.com/jmlipman/RatLesNetv2.git"


@dataclass(frozen=True)
class CloudCommandPlan:
    clone_command: list[str]
    finetune_command: list[str]

    def as_shell(self) -> str:
        return "\n".join(format_command(cmd) for cmd in [self.clone_command, self.finetune_command])


def build_cloud_command_plan(
    *,
    ratlesnet_repo: str | Path,
    train_input: str | Path,
    output_dir: str | Path,
    validation_input: str | Path | None = None,
    test_input: str | Path | None = None,
    pretrained_model: str | Path | None = None,
    require_pretrained: bool = False,
    epochs: int = 100,
    lr: float = 1e-4,
    gpu: int = 0,
    load_memory: int = 0,
    save_every: int | None = None,
    eval_only: bool = False,
    eval_every: int | None = None,
    eval_train: bool = False,
    metrics_threshold: float | None = None,
    loss: str | None = None,
    background_class_weight: float | None = None,
    lesion_class_weight: float | None = None,
    tversky_alpha: float | None = None,
    tversky_beta: float | None = None,
    focal_tversky_gamma: float | None = None,
    no_plots: bool = False,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float | None = None,
    lr_scheduler: str | None = None,
    lr_scheduler_metric: str | None = None,
    lr_plateau_patience: int | None = None,
    lr_plateau_factor: float | None = None,
    lr_plateau_min_delta: float | None = None,
    min_lr: float | None = None,
    export_predictions: str | None = None,
    export_prediction_limit: int | None = None,
    export_prediction_epochs: str | None = None,
    max_train_cases: int | None = None,
    max_validation_cases: int | None = None,
    max_test_cases: int | None = None,
    allow_partial_state_dict: bool = False,
) -> CloudCommandPlan:
    """Build reproducible shell commands for a notebook/cloud GPU runtime."""
    repo_path = str(ratlesnet_repo)
    clone_command = ["git", "clone", "--depth", "1", UPSTREAM_GIT_URL, repo_path]
    finetune_command = [
        "python",
        "-m",
        "ratlesnetv2_finetune.scripts.finetune_ratlesnetv2",
        "--ratlesnet-repo",
        repo_path,
        "--input",
        str(train_input),
        "--output",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--gpu",
        str(gpu),
        "--loadMemory",
        str(load_memory),
    ]
    if validation_input:
        finetune_command.extend(["--validation", str(validation_input)])
    if test_input:
        finetune_command.extend(["--test", str(test_input)])
    if pretrained_model:
        finetune_command.extend(["--pretrained-model", str(pretrained_model)])
    if require_pretrained:
        finetune_command.append("--require-pretrained")
    if save_every is not None:
        finetune_command.extend(["--save-every", str(save_every)])
    if eval_only:
        finetune_command.append("--eval-only")
    if eval_every is not None:
        finetune_command.extend(["--eval-every", str(eval_every)])
    if eval_train:
        finetune_command.append("--eval-train")
    if metrics_threshold is not None:
        finetune_command.extend(["--metrics-threshold", str(metrics_threshold)])
    if loss is not None:
        finetune_command.extend(["--loss", str(loss)])
    if background_class_weight is not None:
        finetune_command.extend(["--background-class-weight", str(background_class_weight)])
    if lesion_class_weight is not None:
        finetune_command.extend(["--lesion-class-weight", str(lesion_class_weight)])
    if tversky_alpha is not None:
        finetune_command.extend(["--tversky-alpha", str(tversky_alpha)])
    if tversky_beta is not None:
        finetune_command.extend(["--tversky-beta", str(tversky_beta)])
    if focal_tversky_gamma is not None:
        finetune_command.extend(["--focal-tversky-gamma", str(focal_tversky_gamma)])
    if no_plots:
        finetune_command.append("--no-plots")
    if early_stop_patience is not None:
        finetune_command.extend(["--early-stop-patience", str(early_stop_patience)])
    if early_stop_min_delta is not None:
        finetune_command.extend(["--early-stop-min-delta", str(early_stop_min_delta)])
    if lr_scheduler:
        finetune_command.extend(["--lr-scheduler", lr_scheduler])
    if lr_scheduler_metric:
        finetune_command.extend(["--lr-scheduler-metric", lr_scheduler_metric])
    if lr_plateau_patience is not None:
        finetune_command.extend(["--lr-plateau-patience", str(lr_plateau_patience)])
    if lr_plateau_factor is not None:
        finetune_command.extend(["--lr-plateau-factor", str(lr_plateau_factor)])
    if lr_plateau_min_delta is not None:
        finetune_command.extend(["--lr-plateau-min-delta", str(lr_plateau_min_delta)])
    if min_lr is not None:
        finetune_command.extend(["--min-lr", str(min_lr)])
    if export_predictions:
        finetune_command.extend(["--export-predictions", export_predictions])
    if export_prediction_limit is not None:
        finetune_command.extend(["--export-prediction-limit", str(export_prediction_limit)])
    if export_prediction_epochs:
        finetune_command.extend(["--export-prediction-epochs", export_prediction_epochs])
    if max_train_cases is not None:
        finetune_command.extend(["--max-train-cases", str(max_train_cases)])
    if max_validation_cases is not None:
        finetune_command.extend(["--max-validation-cases", str(max_validation_cases)])
    if max_test_cases is not None:
        finetune_command.extend(["--max-test-cases", str(max_test_cases)])
    if allow_partial_state_dict:
        finetune_command.append("--allow-partial-state-dict")
    return CloudCommandPlan(clone_command=clone_command, finetune_command=finetune_command)


def format_command(cmd: list[str]) -> str:
    """Return a shell-safe one-line command."""
    return shlex.join(cmd)
