"""Run a configured grid of lesion-segmentation fine-tuning experiments.

The grid is intended for Kaggle. Each experiment writes to:

```
<output_root>/<experiment_name>/<numeric_run_dir>
```

After the grid finishes, a comparison report can be generated automatically by
``summarize_training_grid.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ratlesnetv2_finetune.commands import format_command
from ratlesnetv2_finetune.scripts.summarize_training_grid import (
    discover_run_dirs,
    summarize_runs,
)


@dataclass(frozen=True)
class ExperimentCommand:
    name: str
    kind: str
    output_dir: Path
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training grid YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated experiment names to run. Default: all enabled experiments.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have at least one numeric run with metrics.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not generate the comparison report after running experiments.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the grid at the first failed experiment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = _load_config(config_path)
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    commands = build_experiment_commands(config, only=only)
    if not commands:
        raise ValueError("Training grid contains no enabled experiments")

    if args.dry_run:
        for command in commands:
            print(f"\n# {command.name}")
            print(format_command(command.command))
        return 0

    output_root = Path(_required(config, "output_root"))
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "command_plan.sh"
    _write_command_plan(plan_path, commands)
    print(f"Wrote command plan: {plan_path}")

    records = []
    failures = 0
    for command in commands:
        if args.skip_existing and _existing_completed_run(command.output_dir):
            run_dir = _latest_numeric_run(command.output_dir)
            print(f"Skipping {command.name}; existing run: {run_dir}")
            records.append(_record(command, status="skipped", run_dir=run_dir, returncode=0))
            continue

        command.output_dir.mkdir(parents=True, exist_ok=True)
        before = set(_numeric_run_dirs(command.output_dir))
        print(f"\n# Running {command.name}")
        print(format_command(command.command))
        completed = subprocess.run(command.command, check=False)
        after = set(_numeric_run_dirs(command.output_dir))
        new_runs = sorted(after - before, key=lambda path: int(path.name))
        run_dir = new_runs[-1] if new_runs else _latest_numeric_run(command.output_dir)
        if run_dir is not None:
            _write_experiment_metadata(run_dir, command)
        status = "completed" if completed.returncode == 0 else "failed"
        records.append(
            _record(command, status=status, run_dir=run_dir, returncode=completed.returncode)
        )
        if completed.returncode != 0:
            failures += 1
            if args.fail_fast:
                break

    _write_records(output_root / "grid_run_records.json", records)
    if not args.no_report:
        report_root = Path(config.get("report_root", output_root / "comparison_report"))
        run_dirs = discover_run_dirs(grid_root=output_root, runs=[])
        if run_dirs:
            summarize_runs(
                run_dirs=run_dirs,
                output_dir=report_root,
                max_overlays=int(config.get("max_report_overlays", 12)),
            )
            print(f"Wrote comparison report: {report_root}")
    return 1 if failures else 0


def build_experiment_commands(
    config: dict[str, Any],
    *,
    only: set[str] | None = None,
) -> list[ExperimentCommand]:
    output_root = Path(_required(config, "output_root"))
    defaults = config.get("defaults", {})
    experiments = config.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("Grid config key 'experiments' must be a list")
    commands = []
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each experiment entry must be a mapping")
        if experiment.get("enabled", True) is False:
            continue
        name = _required(experiment, "name")
        if only and name not in only:
            continue
        kind = _required(experiment, "kind")
        output_dir = output_root / _safe_name(name)
        if kind == "ratlesnetv2":
            command = _ratlesnet_command(config, defaults, experiment, output_dir)
        elif kind == "an2023":
            command = _an2023_command(config, defaults, experiment, output_dir)
        else:
            raise ValueError(f"Unknown experiment kind {kind!r}; expected ratlesnetv2 or an2023")
        commands.append(
            ExperimentCommand(name=name, kind=kind, output_dir=output_dir, command=command)
        )
    return commands


def _ratlesnet_command(
    config: dict[str, Any],
    defaults: dict[str, Any],
    experiment: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    command = [
        _python(config),
        "-m",
        "ratlesnetv2_finetune.scripts.finetune_ratlesnetv2",
        "--ratlesnet-repo",
        _resolve_path(config, _setting(experiment, defaults, "ratlesnet_repo")),
        "--input",
        _resolve_path(config, _setting(experiment, defaults, "input")),
        "--output",
        str(output_dir),
        "--epochs",
        str(_setting(experiment, defaults, "epochs", 10)),
        "--lr",
        str(_setting(experiment, defaults, "lr", 1e-4)),
        "--gpu",
        str(_setting(experiment, defaults, "gpu", 0)),
        "--loadMemory",
        str(
            _setting(
                experiment,
                defaults,
                "loadMemory",
                _setting(experiment, defaults, "load_memory", 0),
            )
        ),
    ]
    _append_optional_path(command, config, experiment, defaults, "validation", "--validation")
    _append_optional_path(command, config, experiment, defaults, "test", "--test")
    _append_optional_path(
        command,
        config,
        experiment,
        defaults,
        "pretrained_model",
        "--pretrained-model",
    )
    _append_common_options(command, experiment, defaults, include_an_options=False)
    _append_ratlesnet_loss_options(command, experiment, defaults)
    _append_flag(command, experiment, defaults, "require_pretrained", "--require-pretrained")
    _append_flag(
        command,
        experiment,
        defaults,
        "allow_partial_state_dict",
        "--allow-partial-state-dict",
    )
    _append_extra_args(command, experiment)
    return command


def _an2023_command(
    config: dict[str, Any],
    defaults: dict[str, Any],
    experiment: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    command = [
        _python(config),
        "-m",
        "ratlesnetv2_finetune.scripts.finetune_an2023",
        "--input",
        _resolve_path(config, _setting(experiment, defaults, "input")),
        "--output",
        str(output_dir),
        "--model-path",
        _resolve_path(config, _setting(experiment, defaults, "model_path")),
        "--epochs",
        str(_setting(experiment, defaults, "epochs", 10)),
        "--lr",
        str(_setting(experiment, defaults, "lr", 1e-5)),
        "--gpu",
        str(_setting(experiment, defaults, "gpu", 0)),
    ]
    _append_optional_path(command, config, experiment, defaults, "validation", "--validation")
    _append_optional_path(command, config, experiment, defaults, "test", "--test")
    _append_common_options(command, experiment, defaults, include_an_options=True)
    _append_flag(command, experiment, defaults, "require_pretrained", "--require-pretrained")
    _append_extra_args(command, experiment)
    return command


def _append_common_options(
    command: list[str],
    experiment: dict[str, Any],
    defaults: dict[str, Any],
    *,
    include_an_options: bool,
) -> None:
    option_map = {
        "save_every": "--save-every",
        "eval_every": "--eval-every",
        "metrics_threshold": "--metrics-threshold",
        "early_stop_patience": "--early-stop-patience",
        "early_stop_min_delta": "--early-stop-min-delta",
        "lr_scheduler": "--lr-scheduler",
        "lr_scheduler_metric": "--lr-scheduler-metric",
        "lr_plateau_patience": "--lr-plateau-patience",
        "lr_plateau_factor": "--lr-plateau-factor",
        "lr_plateau_min_delta": "--lr-plateau-min-delta",
        "min_lr": "--min-lr",
        "export_predictions": "--export-predictions",
        "export_prediction_limit": "--export-prediction-limit",
        "export_prediction_epochs": "--export-prediction-epochs",
        "max_train_cases": "--max-train-cases",
        "max_validation_cases": "--max-validation-cases",
        "max_test_cases": "--max-test-cases",
    }
    if include_an_options:
        option_map.update(
            {
                "output_key": "--output-key",
                "output_index": "--output-index",
                "crop_x": "--crop-x",
                "crop_y": "--crop-y",
                "crop_z": "--crop-z",
                "positive_class_weight": "--positive-class-weight",
            }
        )
    for key, option in option_map.items():
        value = _setting(experiment, defaults, key, None)
        if value is not None:
            command.extend([option, str(value)])
    _append_flag(command, experiment, defaults, "eval_train", "--eval-train")
    _append_flag(command, experiment, defaults, "eval_only", "--eval-only")
    _append_flag(command, experiment, defaults, "no_plots", "--no-plots")


def _append_ratlesnet_loss_options(
    command: list[str],
    experiment: dict[str, Any],
    defaults: dict[str, Any],
) -> None:
    option_map = {
        "loss": "--loss",
        "background_class_weight": "--background-class-weight",
        "lesion_class_weight": "--lesion-class-weight",
        "tversky_alpha": "--tversky-alpha",
        "tversky_beta": "--tversky-beta",
        "focal_tversky_gamma": "--focal-tversky-gamma",
    }
    for key, option in option_map.items():
        value = _setting(experiment, defaults, key, None)
        if value is not None:
            command.extend([option, str(value)])


def _append_optional_path(
    command: list[str],
    config: dict[str, Any],
    experiment: dict[str, Any],
    defaults: dict[str, Any],
    key: str,
    option: str,
) -> None:
    value = _setting(experiment, defaults, key, None)
    if value is not None and value != "":
        command.extend([option, _resolve_path(config, value)])


def _append_flag(
    command: list[str],
    experiment: dict[str, Any],
    defaults: dict[str, Any],
    key: str,
    option: str,
) -> None:
    if bool(_setting(experiment, defaults, key, False)):
        command.append(option)


def _append_extra_args(command: list[str], experiment: dict[str, Any]) -> None:
    extra = experiment.get("extra_args", [])
    if isinstance(extra, str):
        raise ValueError("extra_args must be a list of command tokens, not one string")
    if extra:
        command.extend(str(item) for item in extra)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Grid config did not parse to a mapping: {path}")
    return loaded


def _python(config: dict[str, Any]) -> str:
    return str(config.get("python", sys.executable))


def _setting(
    experiment: dict[str, Any],
    defaults: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    if key in experiment:
        return experiment[key]
    return defaults.get(key, default)


def _resolve_path(config: dict[str, Any], value: Any) -> str:
    if value is None:
        raise ValueError("Missing required path value")
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    for section in ["splits", "paths"]:
        mapping = config.get(section, {})
        if isinstance(mapping, dict) and text in mapping:
            return str(mapping[text])
    return text


def _required(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if value in {None, ""}:
        raise ValueError(f"Missing required grid config key: {key}")
    return str(value)


def _write_command_plan(path: Path, commands: list[ExperimentCommand]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for command in commands:
        lines.append(f"# {command.name}")
        lines.append(format_command(command.command))
        lines.append("")
    path.write_text("\n".join(lines))


def _numeric_run_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return sorted(
        [child for child in output_dir.iterdir() if child.is_dir() and child.name.isdigit()],
        key=lambda child: int(child.name),
    )


def _latest_numeric_run(output_dir: Path) -> Path | None:
    runs = _numeric_run_dirs(output_dir)
    return runs[-1] if runs else None


def _existing_completed_run(output_dir: Path) -> bool:
    run_dir = _latest_numeric_run(output_dir)
    return run_dir is not None and (run_dir / "metrics_epoch.csv").is_file()


def _write_experiment_metadata(run_dir: Path, command: ExperimentCommand) -> None:
    payload = {
        "name": command.name,
        "kind": command.kind,
        "output_dir": str(command.output_dir),
        "command": command.command,
    }
    with (run_dir / "experiment_metadata.json").open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _record(
    command: ExperimentCommand,
    *,
    status: str,
    run_dir: Path | None,
    returncode: int,
) -> dict[str, Any]:
    return {
        "name": command.name,
        "kind": command.kind,
        "status": status,
        "returncode": returncode,
        "output_dir": str(command.output_dir),
        "run_dir": "" if run_dir is None else str(run_dir),
        "command": command.command,
    }


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        json.dump(records, fh, indent=2, sort_keys=True)


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "experiment"


if __name__ == "__main__":
    raise SystemExit(main())
