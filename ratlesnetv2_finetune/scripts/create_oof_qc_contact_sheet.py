"""Render a development-only OOF scan/manual/prediction QC contact sheet."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from ratlesnetv2_finetune.scripts.calibrate_probability_threshold import (
    _canonical_prediction_case_id,
)


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
    parser.add_argument("--threshold-json")
    parser.add_argument("--case-metrics")
    parser.add_argument(
        "--artifact-bundle",
        help="Prior LYS_v1_RatLesNetV2_final_artifacts.tar.gz to extract direct CE+Dice OOF files",
    )
    parser.add_argument("--staging-root", help="Writable extraction root for --artifact-bundle")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests = [Path(value) for value in args.prediction_manifest]
    threshold_json = Path(args.threshold_json) if args.threshold_json else None
    case_metrics = Path(args.case_metrics) if args.case_metrics else None
    if args.artifact_bundle:
        if args.prediction_root or args.prediction_manifest or threshold_json or case_metrics:
            raise ValueError(
                "--artifact-bundle cannot be combined with explicit prediction/threshold inputs"
            )
        if not args.staging_root:
            raise ValueError("--staging-root is required with --artifact-bundle")
        prediction_root, threshold_json, case_metrics = extract_direct_ce_oof_artifact(
            artifact_bundle=Path(args.artifact_bundle),
            staging_root=Path(args.staging_root),
        )
        manifests.extend(prediction_root.rglob("prediction_export_manifest.csv"))
    else:
        if threshold_json is None or case_metrics is None:
            raise ValueError("--threshold-json and --case-metrics are required")
        for root in args.prediction_root:
            manifests.extend(Path(root).rglob("prediction_export_manifest.csv"))
    manifests = sorted(dict.fromkeys(path.resolve() for path in manifests))
    assert threshold_json is not None and case_metrics is not None
    result = create_oof_qc_contact_sheet(
        prediction_manifests=manifests,
        threshold_json=threshold_json,
        case_metrics_csv=case_metrics,
        output_png=Path(args.output),
        case_count=args.cases,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def extract_direct_ce_oof_artifact(
    *,
    artifact_bundle: Path,
    staging_root: Path,
) -> tuple[Path, Path, Path]:
    """Extract only direct CE+Dice OOF inputs needed for visual QC."""
    if not artifact_bundle.is_file():
        raise FileNotFoundError(artifact_bundle)
    staging_root.mkdir(parents=True, exist_ok=True)
    resolved_root = staging_root.resolve()
    required_threshold = "thresholds/direct_ce_dice/selected_threshold.json"
    required_metrics = (
        "thresholds/direct_ce_dice/selected_threshold_case_metrics.csv"
    )
    extracted_names = set()
    with tarfile.open(artifact_bundle, "r:*") as archive:
        for member in archive.getmembers():
            name = member.name.lstrip("./")
            prediction_file = (
                member.isfile()
                and name.startswith("runs/direct_ce_dice/")
                and "/prediction_exports/validation/final/" in name
                and (
                    name.endswith("prediction_export_manifest.csv")
                    or name.endswith("/scan.nii.gz")
                    or name.endswith("/target_mask.nii.gz")
                    or name.endswith("/lesion_probability.nii.gz")
                )
            )
            report_file = member.isfile() and name in {required_threshold, required_metrics}
            if not prediction_file and not report_file:
                continue
            destination = (staging_root / name).resolve()
            if resolved_root not in destination.parents:
                raise ValueError(f"Unsafe artifact member path: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot extract artifact member: {member.name}")
            with destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted_names.add(name)
    for required in (required_threshold, required_metrics):
        if required not in extracted_names:
            raise FileNotFoundError(f"Artifact lacks {required}")
    prediction_root = staging_root / "runs/direct_ce_dice"
    manifests = list(prediction_root.rglob("prediction_export_manifest.csv"))
    if len(manifests) != 5:
        raise ValueError(f"Expected five direct CE+Dice OOF manifests; found {len(manifests)}")
    return (
        prediction_root,
        staging_root / required_threshold,
        staging_root / required_metrics,
    )


def create_oof_qc_contact_sheet(
    *,
    prediction_manifests: list[Path],
    threshold_json: Path,
    case_metrics_csv: Path,
    output_png: Path,
    case_count: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a stratified OOF contact sheet and its selected-case CSV."""
    if not prediction_manifests:
        raise ValueError("No OOF prediction manifests supplied")
    if case_count < 1:
        raise ValueError("case_count must be >= 1")
    output_csv = output_png.with_suffix(".csv")
    for path in (output_png, output_csv):
        if path.exists() and not overwrite:
            raise FileExistsError(path)

    threshold_record = json.loads(threshold_json.read_text())
    threshold = float(threshold_record.get("selected_threshold", 0))
    if not 0 < threshold < 1:
        raise ValueError(f"Invalid selected threshold {threshold} in {threshold_json}")
    metrics = pd.read_csv(case_metrics_csv)
    required_metrics = {"case_id", "dice"}
    if not required_metrics <= set(metrics.columns):
        raise ValueError(f"Case metrics lack {sorted(required_metrics - set(metrics.columns))}")
    if metrics.empty or not metrics.case_id.is_unique:
        raise ValueError("Case metrics must contain unique cases")
    metrics["dice"] = pd.to_numeric(metrics.dice, errors="raise")

    records = _read_prediction_records(prediction_manifests)
    records_by_case = {record["case_id"]: record for record in records}
    if set(metrics.case_id) != set(records_by_case):
        raise ValueError("OOF prediction and selected-threshold case sets differ")
    selected = _select_cases(metrics, case_count=case_count)
    loaded = [
        _load_case(records_by_case[row.case_id], threshold=threshold, expected_dice=row.dice)
        for row in selected.itertuples(index=False)
    ]

    output_png.parent.mkdir(parents=True, exist_ok=True)
    _render_contact_sheet(output_png, loaded, threshold=threshold)
    selected_rows = []
    for selection, case in zip(selected.to_dict("records"), loaded, strict=True):
        selected_rows.append(
            {
                **selection,
                "selected_threshold": threshold,
                "representative_axis": case["axis"],
                "representative_slice": case["slice_index"],
                "true_positive_voxels": int(
                    np.logical_and(case["target"], case["prediction"]).sum()
                ),
                "false_positive_voxels": int(
                    np.logical_and(~case["target"], case["prediction"]).sum()
                ),
                "false_negative_voxels": int(
                    np.logical_and(case["target"], ~case["prediction"]).sum()
                ),
            }
        )
    _write_csv(output_csv, selected_rows)
    return {
        "output_png": str(output_png),
        "output_csv": str(output_csv),
        "n_oof_cases": len(records),
        "n_displayed_cases": len(loaded),
        "selected_threshold": threshold,
        "selection": "four lowest-Dice cases plus evenly spaced cases from Q1 to maximum Dice",
        "locked_test_used": False,
    }


def _read_prediction_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for manifest in paths:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != "validation":
                    raise ValueError(f"QC accepts validation OOF predictions only: {manifest}")
                case_id = _canonical_prediction_case_id(row)
                if case_id in seen:
                    raise ValueError(f"Missing or duplicate OOF case ID {case_id!r}")
                seen.add(case_id)
                record = dict(row)
                record["case_id"] = case_id
                record["manifest"] = manifest
                records.append(record)
    if not records:
        raise ValueError("OOF manifests contain no cases")
    return sorted(records, key=lambda row: row["case_id"])


def _select_cases(metrics: pd.DataFrame, *, case_count: int) -> pd.DataFrame:
    ordered = metrics.sort_values(["dice", "case_id"], kind="stable").reset_index(drop=True)
    count = min(case_count, len(ordered))
    worst_count = min(4, count)
    indices = list(range(worst_count))
    remaining = count - worst_count
    if remaining:
        start = max(worst_count, int(round(0.25 * (len(ordered) - 1))))
        indices.extend(np.linspace(start, len(ordered) - 1, remaining, dtype=int).tolist())
    indices = list(dict.fromkeys(indices))
    for index in range(len(ordered)):
        if len(indices) >= count:
            break
        if index not in indices:
            indices.append(index)
    selected = ordered.iloc[indices[:count]].copy()
    selected["qc_selection"] = [
        "lowest_dice" if index < worst_count else "distribution_reference"
        for index in range(len(selected))
    ]
    return selected


def _load_case(
    record: dict[str, Any],
    *,
    threshold: float,
    expected_dice: float,
) -> dict[str, Any]:
    manifest = Path(record["manifest"])
    paths = {
        role: _resolve_artifact_path(record, role, manifest)
        for role in ("scan", "target_mask", "lesion_probability")
    }
    images = {role: nib.load(str(path)) for role, path in paths.items()}
    scan = np.squeeze(np.asarray(images["scan"].dataobj)).astype(np.float32)
    target = np.squeeze(np.asarray(images["target_mask"].dataobj)) > 0.5
    probability = np.squeeze(np.asarray(images["lesion_probability"].dataobj)).astype(np.float32)
    if scan.shape != target.shape or probability.shape != target.shape or target.ndim != 3:
        raise ValueError(f"{record['case_id']}: scan/target/probability shape mismatch")
    if not np.allclose(images["target_mask"].affine, images["lesion_probability"].affine):
        raise ValueError(f"{record['case_id']}: target/probability affine mismatch")
    finite = probability[np.isfinite(probability)]
    if not finite.size or float(finite.min()) < 0 or float(finite.max()) > 1:
        raise ValueError(f"{record['case_id']}: invalid probability range")
    prediction = probability >= threshold
    dice = _dice(target, prediction)
    if not np.isclose(dice, float(expected_dice), rtol=0, atol=1e-6):
        raise RuntimeError(
            f"{record['case_id']}: rendered Dice {dice} differs from report {expected_dice}"
        )
    axis = _slice_axis(target)
    slice_index = _representative_slice(target, prediction, axis=axis)
    return {
        "case_id": record["case_id"],
        "scan": scan,
        "target": target,
        "prediction": prediction,
        "dice": dice,
        "axis": axis,
        "slice_index": slice_index,
    }


def _resolve_artifact_path(record: dict[str, Any], role: str, manifest: Path) -> Path:
    raw = record.get(role, "")
    if not raw:
        raise ValueError(f"Missing {role!r} path for case {record.get('case_id')!r}")
    path = Path(raw)
    if path.is_file():
        return path
    export_name = Path(record.get("export_dir", "")).name
    candidates = [manifest.parent / export_name / path.name, manifest.parent / path.name]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        attempted = [str(candidate) for candidate in candidates]
        raise FileNotFoundError(
            f"Cannot resolve {role} for {record.get('case_id')!r}; tried {attempted}"
        )
    return existing[0]


def _render_contact_sheet(path: Path, cases: list[dict[str, Any]], *, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(cases), 4, figsize=(13, 3.1 * len(cases)), squeeze=False)
    headers = ("T2 scan", "Manual mask (green)", "CE+Dice prediction (red)", "Errors")
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontsize=11, fontweight="bold")
    for row_index, case in enumerate(cases):
        axis = int(case["axis"])
        index = int(case["slice_index"])
        scan = _normalize(np.take(case["scan"], index, axis=axis))
        target = np.take(case["target"], index, axis=axis)
        prediction = np.take(case["prediction"], index, axis=axis)
        panels = [
            np.stack((scan, scan, scan), axis=-1),
            _mask_overlay(scan, target, color=(0.0, 1.0, 0.0)),
            _mask_overlay(scan, prediction, color=(1.0, 0.0, 0.0)),
            _error_overlay(scan, target, prediction),
        ]
        for column, panel in enumerate(panels):
            axes[row_index, column].imshow(np.rot90(panel), interpolation="nearest")
            axes[row_index, column].axis("off")
        label = f"{case['case_id']}\nDice={case['dice']:.3f}; axis={axis}; slice={index}"
        axes[row_index, 0].text(
            -0.05,
            0.5,
            label,
            transform=axes[row_index, 0].transAxes,
            fontsize=8,
            ha="right",
            va="center",
        )
    figure.suptitle(
        f"direct_ce_dice OOF development QC at selected threshold {threshold:.2f}\n"
        "Error colors: TP=yellow, FP=red, FN=cyan",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _mask_overlay(scan: np.ndarray, mask: np.ndarray, *, color: tuple[float, ...]) -> np.ndarray:
    rgb = np.stack((scan, scan, scan), axis=-1)
    color_array = np.asarray(color, dtype=np.float32)
    rgb[mask] = 0.35 * rgb[mask] + 0.65 * color_array
    return np.clip(rgb, 0, 1)


def _error_overlay(scan: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    rgb = np.stack((scan, scan, scan), axis=-1)
    true_positive = np.logical_and(target, prediction)
    false_positive = np.logical_and(~target, prediction)
    false_negative = np.logical_and(target, ~prediction)
    for mask, color in (
        (true_positive, (1.0, 1.0, 0.0)),
        (false_positive, (1.0, 0.0, 0.0)),
        (false_negative, (0.0, 1.0, 1.0)),
    ):
        rgb[mask] = 0.2 * rgb[mask] + 0.8 * np.asarray(color)
    return np.clip(rgb, 0, 1)


def _slice_axis(mask: np.ndarray) -> int:
    return int(np.argmin(mask.shape))


def _representative_slice(target: np.ndarray, prediction: np.ndarray, *, axis: int) -> int:
    other_axes = tuple(index for index in range(3) if index != axis)
    counts = np.logical_or(target, prediction).sum(axis=other_axes)
    return int(np.argmax(counts)) if counts.size else 0


def _normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if not finite.size:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0, 1)


def _dice(target: np.ndarray, prediction: np.ndarray) -> float:
    intersection = int(np.logical_and(target, prediction).sum())
    denominator = int(target.sum()) + int(prediction.sum())
    return 1.0 if denominator == 0 else 2 * intersection / denominator


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
