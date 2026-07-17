"""Average frozen fold probability maps and evaluate the locked test once.

The probability threshold must come from an out-of-fold calibration JSON. The
script accepts test prediction manifests only, requires identical cases from
every model, averages probabilities, applies no additional postprocessing, and
writes case-level plus aggregate test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from ratlesnetv2_finetune.scripts.calibrate_probability_threshold import (
    LoadedPrediction,
    _aggregate_threshold,
    _case_metrics,
    _load_case,
    _subgroup_metrics,
    _surface_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-manifest",
        action="append",
        required=True,
        help="Test prediction_export_manifest.csv; repeat once per frozen fold model",
    )
    parser.add_argument(
        "--threshold-json",
        required=True,
        help="selected_threshold.json created from OOF validation predictions",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", default=None, help="Optional split_assignments.csv")
    parser.add_argument("--expected-models", type=int, default=5)
    parser.add_argument("--surface-tolerance-mm", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output. Do not use this for the real locked-test run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate_probability_ensemble(
        prediction_manifests=[Path(value) for value in args.prediction_manifest],
        threshold_json=Path(args.threshold_json),
        output_root=Path(args.output),
        metadata_path=Path(args.metadata) if args.metadata else None,
        expected_models=args.expected_models,
        surface_tolerance_mm=args.surface_tolerance_mm,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Locked-test cases: {summary['n_cases']}")
    print(f"Ensemble mean Dice: {summary['mean_dice']:.4f}")
    print(f"Results: {Path(args.output) / 'locked_test_summary.json'}")
    return 0


def evaluate_probability_ensemble(
    *,
    prediction_manifests: list[Path],
    threshold_json: Path,
    output_root: Path,
    metadata_path: Path | None = None,
    expected_models: int = 5,
    surface_tolerance_mm: float = 0.2,
    bootstrap_samples: int = 2000,
    seed: int = 20260715,
    overwrite: bool = False,
) -> dict[str, Any]:
    if expected_models < 2:
        raise ValueError("expected_models must be >= 2")
    manifests = [path.resolve() for path in prediction_manifests]
    if len(manifests) != expected_models:
        raise ValueError(
            f"Expected {expected_models} fold manifests, received {len(manifests)}"
        )
    if len(set(manifests)) != len(manifests):
        raise ValueError("Each fold prediction manifest must be different")
    if surface_tolerance_mm <= 0:
        raise ValueError("surface_tolerance_mm must be > 0")
    _prepare_output(output_root, overwrite=overwrite)

    threshold_record = _read_threshold_record(threshold_json)
    threshold = float(threshold_record["selected_threshold"])
    record_sets = [_read_test_records(path) for path in manifests]
    expected_cases = set(record_sets[0])
    for manifest, records in zip(manifests[1:], record_sets[1:], strict=True):
        if set(records) != expected_cases:
            missing = sorted(expected_cases - set(records))
            extra = sorted(set(records) - expected_cases)
            raise ValueError(
                f"Fold manifests do not contain identical test cases: {manifest}; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    metadata = _read_test_metadata(metadata_path) if metadata_path else {}
    case_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    for case_id in sorted(expected_cases):
        source_records = [records[case_id] for records in record_sets]
        loaded = [_load_case(record) for record in source_records]
        reference = loaded[0]
        for model_index, item in enumerate(loaded[1:], start=2):
            if item.probability.shape != reference.probability.shape:
                raise ValueError(
                    f"Probability shape mismatch for case_id={case_id!r}, model={model_index}"
                )
            if item.spacing != reference.spacing or not np.array_equal(
                item.target,
                reference.target,
            ):
                raise ValueError(
                    f"Target geometry/content mismatch for case_id={case_id!r}, model={model_index}"
                )
        probability = np.mean(
            np.stack([item.probability for item in loaded], axis=0),
            axis=0,
            dtype=np.float32,
        )
        ensemble_case = LoadedPrediction(
            case_id=case_id,
            probability=probability,
            target=reference.target,
            spacing=reference.spacing,
        )
        prediction = probability >= threshold
        row = _case_metrics(ensemble_case, threshold=threshold)
        row.update(
            _surface_metrics(
                reference.target,
                prediction,
                spacing=reference.spacing,
                tolerance_mm=surface_tolerance_mm,
            )
        )
        row.update(metadata.get(case_id, {}))
        case_rows.append(row)

        case_dir = output_root / "cases" / _safe_path_part(case_id)
        case_dir.mkdir(parents=True)
        target_image = nib.load(str(Path(source_records[0]["target_mask"])))
        probability_path = case_dir / "ensemble_probability.nii.gz"
        mask_path = case_dir / "ensemble_mask.nii.gz"
        _save_nifti_like(probability_path, probability.astype(np.float32), target_image)
        _save_nifti_like(mask_path, prediction.astype(np.uint8), target_image)
        ensemble_rows.append(
            {
                "case_id": case_id,
                "split": "test",
                "n_models": expected_models,
                "threshold": threshold,
                "postprocessing": "none",
                "target_mask": source_records[0]["target_mask"],
                "ensemble_probability": str(probability_path),
                "ensemble_mask": str(mask_path),
            }
        )

    aggregate = _aggregate_threshold(
        case_rows,
        threshold=threshold,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    summary = {
        **aggregate,
        "n_cases": len(case_rows),
        "n_models": expected_models,
        "prediction_manifests": [str(path) for path in manifests],
        "threshold_json": str(threshold_json.resolve()),
        "threshold_selection_data": threshold_record["selection_data"],
        "evaluation_data": "locked_test_once",
        "probability_ensemble": "unweighted_mean_of_five_fold_models",
        "postprocessing": "none",
        "surface_tolerance_mm": surface_tolerance_mm,
        "mean_hd95_mm": _mean_optional(row["hd95_mm"] for row in case_rows),
        "mean_surface_dice": _mean_optional(row["surface_dice"] for row in case_rows),
    }
    _write_csv(output_root / "locked_test_case_metrics.csv", case_rows)
    subgroups = _subgroup_metrics(case_rows)
    if subgroups:
        _write_csv(output_root / "locked_test_subgroups.csv", subgroups)
    _write_csv(output_root / "ensemble_prediction_manifest.csv", ensemble_rows)
    (output_root / "locked_test_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _read_threshold_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"OOF threshold JSON not found: {path}")
    record = json.loads(path.read_text())
    threshold = float(record.get("selected_threshold", 0))
    if not 0 < threshold < 1:
        raise ValueError(f"Invalid selected_threshold in {path}: {threshold}")
    if record.get("selection_data") != "out_of_fold_validation_only":
        raise ValueError("Threshold must be selected from out-of-fold validation only")
    if record.get("locked_test_used") is not False:
        raise ValueError("Threshold JSON does not explicitly confirm locked_test_used=false")
    return record


def _read_test_records(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("split") != "test":
            raise ValueError(f"Locked-test ensemble accepts split='test' only: {path}")
        case_id = _required(row, "case_id")
        if case_id in result:
            raise ValueError(f"Duplicate case_id={case_id!r} in {path}")
        result[case_id] = row
    return result


def _read_test_metadata(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    for row in _read_csv(path):
        if row.get("outer_split") != "test":
            continue
        case_id = _required(row, "case_id")
        result[case_id] = {
            key: row[key]
            for key in ["subject_id", "cohort", "timepoint", "lesion_volume_bin"]
            if key in row
        }
    return result


def _save_nifti_like(path: Path, data: np.ndarray, reference: Any) -> None:
    header = reference.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(data.dtype)
    nib.save(nib.Nifti1Image(data, reference.affine, header), path)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Locked-test output already exists: {path}. Do not rerun or overwrite it."
            )
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


def _safe_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "case"


def _mean_optional(values: Any) -> Any:
    cleaned = [float(value) for value in values if value not in {None, ""}]
    return float(np.mean(cleaned)) if cleaned else ""


if __name__ == "__main__":
    raise SystemExit(main())
