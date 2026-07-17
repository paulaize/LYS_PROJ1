"""Calibrate a lesion-probability threshold from out-of-fold validation maps.

Prediction manifests must contain validation cases only. The command rejects
duplicate case IDs so that a case cannot silently contribute more than once.
The locked test set is not accepted for threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        action="append",
        default=[],
        help="Root searched recursively for prediction_export_manifest.csv; repeatable",
    )
    parser.add_argument(
        "--prediction-manifest",
        action="append",
        default=[],
        help="Explicit prediction_export_manifest.csv; repeatable",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="0.20:0.80:0.05")
    parser.add_argument("--metadata", default=None, help="Optional split_assignments.csv")
    parser.add_argument("--surface-tolerance-mm", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--min-lesion-detection-sensitivity", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = [Path(value) for value in args.prediction_manifest]
    for root_value in args.prediction_root:
        manifests.extend(Path(root_value).rglob("prediction_export_manifest.csv"))
    manifests = sorted(dict.fromkeys(path.resolve() for path in manifests))
    if not manifests:
        raise ValueError("Provide --prediction-root and/or --prediction-manifest")
    selection = calibrate_probability_threshold(
        prediction_manifests=manifests,
        output_root=Path(args.output),
        thresholds=_parse_thresholds(args.thresholds),
        metadata_path=Path(args.metadata) if args.metadata else None,
        surface_tolerance_mm=args.surface_tolerance_mm,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        min_detection_sensitivity=args.min_lesion_detection_sensitivity,
        overwrite=args.overwrite,
    )
    print(f"Selected threshold: {selection['selected_threshold']:.3f}")
    print(f"Selection mean Dice: {selection['mean_dice']:.4f}")
    print(f"Calibration report: {Path(args.output) / 'threshold_sweep.csv'}")
    return 0


def calibrate_probability_threshold(
    *,
    prediction_manifests: list[Path],
    output_root: Path,
    thresholds: list[float],
    metadata_path: Path | None = None,
    surface_tolerance_mm: float = 0.2,
    bootstrap_samples: int = 2000,
    seed: int = 20260715,
    min_detection_sensitivity: float = 0.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not thresholds or any(not 0 < value < 1 for value in thresholds):
        raise ValueError("All thresholds must be between 0 and 1")
    if surface_tolerance_mm <= 0:
        raise ValueError("surface_tolerance_mm must be > 0")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be >= 0")
    if not 0 <= min_detection_sensitivity <= 1:
        raise ValueError("min_detection_sensitivity must be between 0 and 1")
    _prepare_output(output_root, overwrite=overwrite)

    records = _read_prediction_records(prediction_manifests)
    metadata = _read_metadata(metadata_path) if metadata_path is not None else {}
    cases = [_load_case(record) for record in records]
    sweep_rows = []
    case_metrics_by_threshold: dict[float, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        case_rows = [_case_metrics(case, threshold=threshold) for case in cases]
        case_metrics_by_threshold[threshold] = case_rows
        sweep_rows.append(
            _aggregate_threshold(
                case_rows,
                threshold=threshold,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    eligible = [
        row
        for row in sweep_rows
        if float(row["lesion_detection_sensitivity"]) >= min_detection_sensitivity
    ]
    if not eligible:
        raise ValueError(
            "No threshold satisfies min lesion-detection sensitivity "
            f"{min_detection_sensitivity:g}"
        )
    selected = max(
        eligible,
        key=lambda row: (
            float(row["mean_dice"]),
            -abs(float(row["mean_volume_bias_mm3"])),
            -abs(float(row["threshold"]) - 0.5),
        ),
    )
    selected_threshold = float(selected["threshold"])
    selected_cases = []
    for case, row in zip(cases, case_metrics_by_threshold[selected_threshold], strict=True):
        detailed = dict(row)
        detailed.update(
            _surface_metrics(
                case.target,
                case.probability >= selected_threshold,
                spacing=case.spacing,
                tolerance_mm=surface_tolerance_mm,
            )
        )
        detailed.update(metadata.get(case.case_id, {}))
        selected_cases.append(detailed)

    _write_csv(output_root / "threshold_sweep.csv", sweep_rows)
    _write_csv(output_root / "selected_threshold_case_metrics.csv", selected_cases)
    subgroup_rows = _subgroup_metrics(selected_cases)
    if subgroup_rows:
        _write_csv(output_root / "selected_threshold_subgroups.csv", subgroup_rows)
    selection = {
        **selected,
        "selected_threshold": selected_threshold,
        "n_cases": len(cases),
        "n_prediction_manifests": len(prediction_manifests),
        "surface_tolerance_mm": surface_tolerance_mm,
        "selection_data": "out_of_fold_validation_only",
        "locked_test_used": False,
        "selection_rule": (
            "Maximize per-case mean Dice subject to the configured lesion-detection "
            "sensitivity floor; ties minimize absolute mean volume bias, then distance from 0.5."
        ),
    }
    (output_root / "selected_threshold.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    _write_plots(output_root, sweep_rows, selected_cases, selected_threshold=selected_threshold)
    return selection


class LoadedPrediction:
    def __init__(
        self,
        *,
        case_id: str,
        probability: np.ndarray,
        target: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        self.case_id = case_id
        self.probability = probability
        self.target = target
        self.spacing = spacing


def _read_prediction_records(paths: list[Path]) -> list[dict[str, str]]:
    records = []
    seen: dict[str, Path] = {}
    for path in paths:
        rows = _read_csv(path)
        for row in rows:
            split = _required(row, "split")
            if split != "validation":
                raise ValueError(
                    "Threshold calibration accepts split='validation' only, "
                    f"got {split!r} in {path}"
                )
            case_id = _required(row, "case_id")
            if case_id in seen:
                raise ValueError(
                    f"Duplicate out-of-fold case_id={case_id!r} in {seen[case_id]} and {path}"
                )
            seen[case_id] = path
            records.append(row)
    if not records:
        raise ValueError("Prediction manifests contain no rows")
    return sorted(records, key=lambda row: row["case_id"])


def _load_case(record: dict[str, str]) -> LoadedPrediction:
    probability_path = Path(_required(record, "lesion_probability"))
    target_path = Path(_required(record, "target_mask"))
    if not probability_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(
            f"Missing probability or target for case_id={record.get('case_id')!r}: "
            f"{probability_path}, {target_path}"
        )
    probability_img = nib.load(str(probability_path))
    target_img = nib.load(str(target_path))
    probability = np.squeeze(np.asanyarray(probability_img.dataobj)).astype(np.float32)
    target = np.squeeze(np.asanyarray(target_img.dataobj)) > 0.5
    if probability.shape != target.shape:
        raise ValueError(
            f"Probability/target shape mismatch for {record['case_id']}: "
            f"{probability.shape} vs {target.shape}"
        )
    if not np.allclose(probability_img.affine, target_img.affine, rtol=0, atol=1e-4):
        raise ValueError(f"Probability/target affine mismatch for {record['case_id']}")
    finite = probability[np.isfinite(probability)]
    if not finite.size or float(finite.min()) < 0 or float(finite.max()) > 1:
        raise ValueError(f"Probability map is not bounded in [0, 1] for {record['case_id']}")
    spacing = tuple(float(value) for value in target_img.header.get_zooms()[:3])
    return LoadedPrediction(
        case_id=_required(record, "case_id"),
        probability=probability,
        target=target,
        spacing=spacing,
    )


def _case_metrics(case: LoadedPrediction, *, threshold: float) -> dict[str, Any]:
    prediction = case.probability >= threshold
    target = case.target
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    target_voxels = int(target.sum())
    pred_voxels = int(prediction.sum())
    voxel_volume = float(np.prod(case.spacing))
    target_volume = target_voxels * voxel_volume
    pred_volume = pred_voxels * voxel_volume
    return {
        "case_id": case.case_id,
        "threshold": threshold,
        "dice": _overlap_score(2 * tp, 2 * tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "target_positive": target_voxels > 0,
        "lesion_detected": tp > 0,
        "target_voxels": target_voxels,
        "pred_voxels": pred_voxels,
        "target_volume_mm3": target_volume,
        "pred_volume_mm3": pred_volume,
        "volume_error_mm3": pred_volume - target_volume,
        "absolute_volume_error_mm3": abs(pred_volume - target_volume),
        "relative_volume_error": (
            (pred_volume - target_volume) / target_volume if target_volume > 0 else ""
        ),
        "false_positive_volume_mm3": fp * voxel_volume,
    }


def _aggregate_threshold(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    positive = [row for row in rows if row["target_positive"]]
    negative = [row for row in rows if not row["target_positive"]]
    dice = np.asarray([float(row["dice"]) for row in rows])
    positive_dice = np.asarray([float(row["dice"]) for row in positive])
    ci_low, ci_high = _bootstrap_mean_ci(
        dice,
        samples=bootstrap_samples,
        seed=seed + round(threshold * 1000),
    )
    return {
        "threshold": threshold,
        "n_cases": len(rows),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "mean_dice": float(dice.mean()),
        "median_dice": float(np.median(dice)),
        "mean_positive_dice": float(positive_dice.mean()) if positive_dice.size else "",
        "dice_mean_ci95_low": ci_low,
        "dice_mean_ci95_high": ci_high,
        "zero_dice_rate": float(np.mean(dice == 0)),
        "lesion_detection_sensitivity": (
            float(np.mean([row["lesion_detected"] for row in positive])) if positive else 1.0
        ),
        "mean_precision": _mean_optional(row["precision"] for row in rows),
        "mean_recall": _mean_optional(row["recall"] for row in rows),
        "mean_volume_bias_mm3": float(np.mean([row["volume_error_mm3"] for row in rows])),
        "mean_absolute_volume_error_mm3": float(
            np.mean([row["absolute_volume_error_mm3"] for row in rows])
        ),
        "mean_false_positive_volume_mm3": float(
            np.mean([row["false_positive_volume_mm3"] for row in rows])
        ),
        "negative_any_false_positive_rate": (
            float(np.mean([row["pred_voxels"] > 0 for row in negative])) if negative else ""
        ),
    }


def _surface_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing: tuple[float, float, float],
    tolerance_mm: float,
) -> dict[str, Any]:
    if not target.any() and not prediction.any():
        return {"hd95_mm": 0.0, "surface_dice": 1.0}
    if not target.any() or not prediction.any():
        return {"hd95_mm": "", "surface_dice": 0.0}
    structure = ndimage.generate_binary_structure(3, 1)
    target_surface = np.logical_xor(target, ndimage.binary_erosion(target, structure=structure))
    pred_surface = np.logical_xor(
        prediction,
        ndimage.binary_erosion(prediction, structure=structure),
    )
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    pred_to_target = distance_to_target[pred_surface]
    target_to_pred = distance_to_prediction[target_surface]
    distances = np.concatenate([pred_to_target, target_to_pred])
    surface_dice = (
        int((pred_to_target <= tolerance_mm).sum())
        + int((target_to_pred <= tolerance_mm).sum())
    ) / max(pred_to_target.size + target_to_pred.size, 1)
    return {
        "hd95_mm": float(np.percentile(distances, 95)),
        "surface_dice": float(surface_dice),
    }


def _subgroup_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions = [
        key
        for key in ["cohort", "timepoint", "lesion_volume_bin"]
        if rows and key in rows[0]
    ]
    result = []
    for dimension in dimensions:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(dimension, ""))].append(row)
        for value, group_rows in sorted(grouped.items()):
            result.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "n_cases": len(group_rows),
                    "mean_dice": float(np.mean([row["dice"] for row in group_rows])),
                    "median_dice": float(np.median([row["dice"] for row in group_rows])),
                    "zero_dice_rate": float(np.mean([row["dice"] == 0 for row in group_rows])),
                    "mean_precision": _mean_optional(row["precision"] for row in group_rows),
                    "mean_recall": _mean_optional(row["recall"] for row in group_rows),
                    "mean_absolute_volume_error_mm3": float(
                        np.mean([row["absolute_volume_error_mm3"] for row in group_rows])
                    ),
                    "mean_hd95_mm": _mean_optional(row["hd95_mm"] for row in group_rows),
                    "mean_surface_dice": _mean_optional(row["surface_dice"] for row in group_rows),
                }
            )
    return result


def _write_plots(
    output_root: Path,
    sweep_rows: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    *,
    selected_threshold: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    thresholds = [float(row["threshold"]) for row in sweep_rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(thresholds, [row["mean_dice"] for row in sweep_rows], marker="o")
    axes[0].set_ylabel("Mean Dice")
    axes[1].plot(
        thresholds,
        [row["lesion_detection_sensitivity"] for row in sweep_rows],
        marker="o",
    )
    axes[1].set_ylabel("Lesion detection sensitivity")
    axes[2].plot(
        thresholds,
        [row["mean_volume_bias_mm3"] for row in sweep_rows],
        marker="o",
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Mean volume bias (mm³)")
    for ax in axes:
        ax.axvline(selected_threshold, color="red", linestyle="--")
        ax.set_xlabel("Probability threshold")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "threshold_curves.png", dpi=160)
    plt.close(fig)

    means = np.asarray(
        [(row["target_volume_mm3"] + row["pred_volume_mm3"]) / 2 for row in selected_cases]
    )
    differences = np.asarray([row["volume_error_mm3"] for row in selected_cases])
    bias = float(differences.mean())
    sd = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(means, differences, s=18, alpha=0.7)
    ax.axhline(bias, color="red", label=f"bias={bias:.3g}")
    ax.axhline(bias + 1.96 * sd, color="gray", linestyle="--")
    ax.axhline(bias - 1.96 * sd, color="gray", linestyle="--")
    ax.set_xlabel("Mean target/predicted lesion volume (mm³)")
    ax.set_ylabel("Predicted − target volume (mm³)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "bland_altman_volume.png", dpi=160)
    plt.close(fig)


def _parse_thresholds(value: str) -> list[float]:
    text = str(value).strip()
    if ":" in text:
        start, stop, step = (float(item) for item in text.split(":"))
        if step <= 0 or stop < start:
            raise ValueError("Threshold range must be start:stop:positive_step")
        count = int(math.floor((stop - start) / step + 1e-9)) + 1
        return [round(start + index * step, 10) for index in range(count)]
    return [float(item) for item in text.split(",") if item.strip()]


def _read_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    by_case = {}
    for row in rows:
        case_id = _required(row, "case_id")
        if row.get("outer_split") == "test":
            continue
        by_case[case_id] = {
            key: row[key]
            for key in [
                "subject_id",
                "outer_split",
                "cv_fold",
                "cohort",
                "timepoint",
                "lesion_volume_bin",
            ]
            if key in row
        }
    return by_case


def _bootstrap_mean_ci(values: np.ndarray, *, samples: int, seed: int) -> tuple[Any, Any]:
    if samples == 0 or not values.size:
        return "", ""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _overlap_score(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean_optional(values: Any) -> Any:
    cleaned = [float(value) for value in values if value not in {None, ""}]
    return float(np.mean(cleaned)) if cleaned else ""


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite to rebuild it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value in {None, "", "TODO"}:
        raise ValueError(f"case_id={row.get('case_id')!r} is missing required value: {key}")
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
