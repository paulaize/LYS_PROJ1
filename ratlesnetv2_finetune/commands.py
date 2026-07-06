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
    epochs: int = 100,
    lr: float = 1e-4,
    gpu: int = 0,
    load_memory: int = 0,
    save_every: int | None = None,
    eval_every: int | None = None,
    eval_train: bool = False,
    metrics_threshold: float | None = None,
    max_train_cases: int | None = None,
    max_validation_cases: int | None = None,
    max_test_cases: int | None = None,
    allow_partial_state_dict: bool = False,
) -> CloudCommandPlan:
    """Build reproducible shell commands for a Colab/cloud GPU runtime."""
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
    if save_every is not None:
        finetune_command.extend(["--save-every", str(save_every)])
    if eval_every is not None:
        finetune_command.extend(["--eval-every", str(eval_every)])
    if eval_train:
        finetune_command.append("--eval-train")
    if metrics_threshold is not None:
        finetune_command.extend(["--metrics-threshold", str(metrics_threshold)])
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
