"""Summarize lesion-model training runs into comparison reports.

The script accepts explicit run directories or a grid root containing
experiment folders. It reads the common files written by the RatLesNetV2 and
An2023 wrappers: ``metrics_epoch.csv``, ``metrics_cases.csv``,
``best_checkpoints.json``, ``run_config.json``, and QC overlay metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

COMPARISON_COLUMNS = [
    "rank",
    "experiment",
    "model",
    "run_dir",
    "status",
    "best_epoch",
    "best_validation_dice",
    "validation_loss_at_best_dice",
    "precision_mean",
    "recall_mean",
    "target_voxels",
    "pred_voxels",
    "pred_to_target_voxel_ratio",
    "best_checkpoint",
    "latest_overlay",
    "qc_flag",
]


@dataclass(frozen=True)
class RunSummary:
    experiment: str
    model: str
    run_dir: Path
    status: str
    best_epoch: int | None
    best_validation_dice: float | None
    validation_loss_at_best_dice: float | None
    precision_mean: float | None
    recall_mean: float | None
    target_voxels: int | None
    pred_voxels: int | None
    pred_to_target_voxel_ratio: float | None
    best_checkpoint: Path | None
    latest_overlay: Path | None
    qc_flag: str

    def to_row(self, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "experiment": self.experiment,
            "model": self.model,
            "run_dir": str(self.run_dir),
            "status": self.status,
            "best_epoch": self.best_epoch,
            "best_validation_dice": self.best_validation_dice,
            "validation_loss_at_best_dice": self.validation_loss_at_best_dice,
            "precision_mean": self.precision_mean,
            "recall_mean": self.recall_mean,
            "target_voxels": self.target_voxels,
            "pred_voxels": self.pred_voxels,
            "pred_to_target_voxel_ratio": self.pred_to_target_voxel_ratio,
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else "",
            "latest_overlay": str(self.latest_overlay) if self.latest_overlay else "",
            "qc_flag": self.qc_flag,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-root",
        default=None,
        help="Root containing experiment output folders.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Explicit run directories. Each may be a final numeric run dir or an output root.",
    )
    parser.add_argument("--output", required=True, help="Report output directory")
    parser.add_argument("--max-overlays", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dirs = discover_run_dirs(
        grid_root=Path(args.grid_root) if args.grid_root else None,
        runs=[Path(value) for value in args.runs or []],
    )
    if not run_dirs:
        raise ValueError("No run directories containing metrics_epoch.csv were found")
    summarize_runs(run_dirs=run_dirs, output_dir=Path(args.output), max_overlays=args.max_overlays)
    return 0


def discover_run_dirs(*, grid_root: Path | None, runs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in runs:
        found.extend(_run_dirs_from_path(path))
    if grid_root is not None:
        found.extend(_run_dirs_from_path(grid_root))
    return sorted(dict.fromkeys(path.resolve() for path in found), key=lambda path: str(path))


def _run_dirs_from_path(path: Path) -> list[Path]:
    if (path / "metrics_epoch.csv").is_file():
        return [path]
    direct = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.name.isdigit() and (child / "metrics_epoch.csv").is_file()
    ] if path.is_dir() else []
    if direct:
        return sorted(direct, key=lambda child: int(child.name))
    if not path.is_dir():
        return []
    return sorted(
        {metrics.parent for metrics in path.rglob("metrics_epoch.csv")},
        key=lambda child: str(child),
    )


def summarize_runs(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    max_overlays: int = 12,
) -> list[RunSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_summarize_run(run_dir) for run_dir in run_dirs]
    summaries = sorted(
        summaries,
        key=lambda item: (
            item.best_validation_dice is not None,
            item.best_validation_dice if item.best_validation_dice is not None else -math.inf,
        ),
        reverse=True,
    )
    rows = [summary.to_row(rank=index + 1) for index, summary in enumerate(summaries)]
    _write_csv(output_dir / "comparison.csv", rows, COMPARISON_COLUMNS)
    _write_markdown(output_dir / "comparison.md", rows)
    _write_html_report(output_dir / "report.html", summaries, rows)
    _write_recommendation(output_dir / "selected_recommendation.json", summaries)
    _copy_overlays(output_dir, summaries, max_overlays=max_overlays)
    _write_contact_sheet(output_dir / "qc_contact_sheet.png", summaries, max_overlays=max_overlays)
    return summaries


def _summarize_run(run_dir: Path) -> RunSummary:
    metrics = _read_csv_dicts(run_dir / "metrics_epoch.csv")
    metadata = _read_json(run_dir / "experiment_metadata.json")
    run_config = _read_json(run_dir / "run_config.json")
    status = _read_json(run_dir / "run_status.json").get("status", "unknown")
    best = _best_validation_row(run_dir, metrics)
    experiment = str(metadata.get("name") or _experiment_from_path(run_dir))
    model = str(metadata.get("kind") or _infer_model(run_dir, run_config))
    checkpoint = _best_checkpoint_path(run_dir)
    overlay = _latest_overlay_path(run_dir)
    target_voxels = _to_int(best.get("target_voxels")) if best else None
    pred_voxels = _to_int(best.get("pred_voxels")) if best else None
    ratio = None
    if target_voxels is not None and target_voxels > 0 and pred_voxels is not None:
        ratio = pred_voxels / target_voxels
    return RunSummary(
        experiment=experiment,
        model=model,
        run_dir=run_dir,
        status=status,
        best_epoch=_to_int(best.get("epoch")) if best else None,
        best_validation_dice=_to_float(best.get("dice_mean")) if best else None,
        validation_loss_at_best_dice=_to_float(best.get("loss")) if best else None,
        precision_mean=_to_float(best.get("precision_mean")) if best else None,
        recall_mean=_to_float(best.get("recall_mean")) if best else None,
        target_voxels=target_voxels,
        pred_voxels=pred_voxels,
        pred_to_target_voxel_ratio=ratio,
        best_checkpoint=checkpoint,
        latest_overlay=overlay,
        qc_flag="needs_overlay_review" if overlay else "missing_overlay",
    )


def _best_validation_row(run_dir: Path, metrics: list[dict[str, str]]) -> dict[str, str] | None:
    best_json = _read_json(run_dir / "best_checkpoints.json").get("validation_dice", {})
    best_epoch = _to_int(best_json.get("epoch"))
    validation_rows = [
        row
        for row in metrics
        if row.get("split") == "validation" and row.get("dice_mean") not in {"", None}
    ]
    if best_epoch is not None:
        for row in validation_rows:
            if _to_int(row.get("epoch")) == best_epoch:
                return row
    if not validation_rows:
        return None
    return max(validation_rows, key=lambda row: _to_float(row.get("dice_mean")) or -math.inf)


def _best_checkpoint_path(run_dir: Path) -> Path | None:
    best_json = _read_json(run_dir / "best_checkpoints.json").get("validation_dice", {})
    filename = best_json.get("filename")
    if filename:
        path = run_dir / str(filename)
        if path.exists():
            return path
    for name in ["best_by_validation_dice.model", "best_by_validation_dice.pt"]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def _latest_overlay_path(run_dir: Path) -> Path | None:
    metadata = _read_json(run_dir / "latest_validation_qc_overlay.json")
    for key in ["latest_overlay", "source_overlay"]:
        value = metadata.get(key)
        if value and Path(str(value)).exists():
            return Path(str(value))
    for name in ["latest_validation_qc_overlay.png", "latest_qc_overlay.png"]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def _experiment_from_path(run_dir: Path) -> str:
    parent = run_dir.parent.name
    return parent if parent else run_dir.name


def _infer_model(run_dir: Path, run_config: dict[str, Any]) -> str:
    if "ratlesnet_repo" in run_config:
        return "ratlesnetv2"
    if "model_path" in run_config:
        return "an2023"
    if (run_dir / "an2023_finetuned.pt").exists():
        return "an2023"
    return "unknown"


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fh:
        loaded = json.load(fh)
    return loaded if isinstance(loaded, dict) else {}


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [{column: _csv_value(row.get(column)) for column in columns} for row in rows]
        )


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Model Comparison",
        "",
        "Do not use this report as final evidence until QC overlays are reviewed.",
        "",
    ]
    if rows:
        columns = [
            "rank",
            "experiment",
            "model",
            "best_epoch",
            "best_validation_dice",
            "precision_mean",
            "recall_mean",
            "pred_to_target_voxel_ratio",
            "qc_flag",
        ]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            values = [_format_md(row.get(column)) for column in columns]
            lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def _write_html_report(path: Path, summaries: list[RunSummary], rows: list[dict[str, Any]]) -> None:
    table_columns = [
        "rank",
        "experiment",
        "model",
        "best_epoch",
        "best_validation_dice",
        "precision_mean",
        "recall_mean",
        "pred_to_target_voxel_ratio",
        "qc_flag",
    ]
    html = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>LYS Model Comparison</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;color:#1f2933}",
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}",
        "th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left}",
        "th{background:#f6f8fa}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}",
        ".card{border:1px solid #d0d7de;padding:10px}",
        "img{max-width:100%;height:auto}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>LYS Model Comparison</h1>",
        (
            "<p>Selection is based on LYS validation metrics only. "
            "QC overlays still require review.</p>"
        ),
        "<table>",
        "<thead><tr>" + "".join(f"<th>{column}</th>" for column in table_columns) + "</tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        html.append(
            "<tr>"
            + "".join(
                f"<td>{_html_escape(_format_md(row.get(column)))}</td>"
                for column in table_columns
            )
            + "</tr>"
        )
    html.extend(["</tbody>", "</table>", "<h2>Validation QC Overlays</h2>", '<div class="grid">'])
    for summary in summaries:
        if not summary.latest_overlay:
            continue
        rel_overlay = _relative_to(path.parent, summary.latest_overlay)
        html.extend(
            [
                '<div class="card">',
                f"<h3>{_html_escape(summary.experiment)}</h3>",
                f"<p>Dice: {_format_md(summary.best_validation_dice)}; "
                f"Precision: {_format_md(summary.precision_mean)}; "
                f"Recall: {_format_md(summary.recall_mean)}</p>",
                (
                    f'<img src="{_html_escape(rel_overlay)}" '
                    f'alt="{_html_escape(summary.experiment)} overlay">'
                ),
                "</div>",
            ]
        )
    html.extend(["</div>", "</body>", "</html>"])
    path.write_text("\n".join(html) + "\n")


def _write_recommendation(path: Path, summaries: list[RunSummary]) -> None:
    candidate = next(
        (summary for summary in summaries if summary.best_validation_dice is not None),
        None,
    )
    payload: dict[str, Any] = {
        "selection_rule": (
            "Highest LYS validation Dice among runs with acceptable QC overlays; "
            "do not use the held-out test split for strategy selection."
        ),
        "qc_required": True,
    }
    if candidate is not None:
        payload.update(
            {
                "candidate_experiment": candidate.experiment,
                "candidate_model": candidate.model,
                "candidate_run_dir": str(candidate.run_dir),
                "best_validation_dice": candidate.best_validation_dice,
                "best_epoch": candidate.best_epoch,
                "best_checkpoint": (
                    str(candidate.best_checkpoint) if candidate.best_checkpoint else ""
                ),
                "latest_overlay": str(candidate.latest_overlay) if candidate.latest_overlay else "",
            }
        )
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _copy_overlays(output_dir: Path, summaries: list[RunSummary], *, max_overlays: int) -> None:
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    for index, summary in enumerate(summaries[:max_overlays], start=1):
        if not summary.latest_overlay:
            continue
        suffix = summary.latest_overlay.suffix or ".png"
        dest = overlay_dir / f"{index:02d}_{_safe_name(summary.experiment)}{suffix}"
        shutil.copyfile(summary.latest_overlay, dest)


def _write_contact_sheet(path: Path, summaries: list[RunSummary], *, max_overlays: int) -> None:
    overlays = [summary for summary in summaries if summary.latest_overlay][:max_overlays]
    if not overlays:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cols = min(3, len(overlays))
    rows = math.ceil(len(overlays) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes_array = np.asarray(axes).reshape(-1)
    for ax, summary in zip(axes_array, overlays, strict=False):
        try:
            image = plt.imread(summary.latest_overlay)
        except Exception:
            ax.text(0.5, 0.5, "Overlay unreadable", ha="center", va="center")
        else:
            ax.imshow(image)
        ax.set_title(
            f"{summary.experiment}\nDice={_format_md(summary.best_validation_dice)}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes_array[len(overlays):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _to_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(float(value))


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def _format_md(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._")


if __name__ == "__main__":
    raise SystemExit(main())
