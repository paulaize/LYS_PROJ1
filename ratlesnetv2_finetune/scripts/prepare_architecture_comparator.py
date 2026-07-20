"""Prepare nnU-Net inputs and export native-space validation probabilities.

This module supports the versioned architecture-comparator protocol. It never
creates a new target split: LYS development membership and folds must come from
the preserved ``split_assignments.csv`` produced by the RatLesNetV2 workflow.
The locked-test cases are deliberately omitted from the nnU-Net raw dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    target = subparsers.add_parser(
        "prepare-target",
        help="Convert preserved LYS development cases and grouped folds to nnU-Net format",
    )
    target.add_argument("--input", required=True, help="Normalized LYS prepared-dataset root")
    target.add_argument("--split-assignments", required=True)
    target.add_argument("--output", required=True, help="DatasetXXX_Name directory")
    target.add_argument("--dataset-id", type=int, default=701)
    target.add_argument("--dataset-name", default="LYSDevelopmentV1")

    source = subparsers.add_parser(
        "prepare-external",
        help="Convert all normalized external-mouse cases to an nnU-Net source dataset",
    )
    source.add_argument("--input", required=True, help="Normalized external prepared dataset")
    source.add_argument("--output", required=True, help="DatasetXXX_Name directory")
    source.add_argument("--dataset-id", type=int, default=702)
    source.add_argument("--dataset-name", default="ExternalMouseV0")

    paper = subparsers.add_parser(
        "prepare-paper-folds",
        help="Stage the preserved LYS development folds as symlink-only prepared datasets",
    )
    paper.add_argument("--input", required=True, help="Normalized LYS prepared-dataset root")
    paper.add_argument("--split-assignments", required=True)
    paper.add_argument("--output", required=True)

    export = subparsers.add_parser(
        "export-validation",
        help="Convert one nnU-Net validation folder to the common OOF probability contract",
    )
    export.add_argument("--validation", required=True)
    export.add_argument("--case-mapping", required=True)
    export.add_argument("--fold", required=True, type=int)
    export.add_argument("--output", required=True)
    export.add_argument("--candidate", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare-target":
        result = prepare_target_dataset(
            input_root=Path(args.input),
            split_assignments=Path(args.split_assignments),
            output_root=Path(args.output),
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
        )
    elif args.command == "prepare-external":
        result = prepare_external_dataset(
            input_root=Path(args.input),
            output_root=Path(args.output),
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
        )
    elif args.command == "prepare-paper-folds":
        result = prepare_paper_folds(
            input_root=Path(args.input),
            split_assignments=Path(args.split_assignments),
            output_root=Path(args.output),
        )
    else:
        result = export_nnunet_validation(
            validation_root=Path(args.validation),
            case_mapping=Path(args.case_mapping),
            fold=args.fold,
            output_root=Path(args.output),
            candidate=args.candidate,
        )
    print(result)
    return 0


def prepare_target_dataset(
    *,
    input_root: Path,
    split_assignments: Path,
    output_root: Path,
    dataset_id: int = 701,
    dataset_name: str = "LYSDevelopmentV1",
) -> dict[str, Any]:
    """Create a target dataset containing development cases only.

    The function intentionally has no overwrite option. Existing content is
    accepted only when its recorded input hashes match the requested inputs.
    """
    _validate_dataset_identity(output_root, dataset_id=dataset_id, dataset_name=dataset_name)
    manifest_path = input_root / "manifest.csv"
    manifest = _rows_by_case(manifest_path)
    assignments = _read_csv(split_assignments)
    if not assignments:
        raise ValueError(f"No split assignments in {split_assignments}")
    assignment_ids = [_required(row, "case_id") for row in assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("Duplicate case IDs in split assignments")
    missing = sorted(set(assignment_ids) - set(manifest))
    extra = sorted(set(manifest) - set(assignment_ids))
    if missing or extra:
        raise ValueError(
            "Manifest/split case mismatch: "
            f"missing_from_manifest={missing[:10]}, missing_from_splits={extra[:10]}"
        )

    development = [row for row in assignments if _required(row, "outer_split") == "development"]
    locked = [row for row in assignments if _required(row, "outer_split") == "test"]
    unknown = sorted(
        {
            _required(row, "outer_split")
            for row in assignments
            if _required(row, "outer_split") not in {"development", "test"}
        }
    )
    if unknown:
        raise ValueError(f"Unknown outer split values: {unknown}")
    if not development or not locked:
        raise ValueError("Both development and locked-test assignments are required")
    folds = sorted({_required(row, "cv_fold") for row in development})
    if folds != ["0", "1", "2", "3", "4"]:
        raise ValueError(f"Expected preserved folds 0-4, found {folds}")
    if any(row.get("cv_fold", "") for row in locked):
        raise ValueError("Locked-test cases must not have a CV fold")

    input_hashes = {
        "manifest_sha256": _sha256(manifest_path),
        "split_assignments_sha256": _sha256(split_assignments),
    }
    existing = _existing_matching_report(output_root, input_hashes)
    if existing is not None:
        return existing

    _require_absent(output_root)
    images = output_root / "imagesTr"
    labels = output_root / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir()

    safe_ids = {
        case_id: f"LYS_{index:04d}"
        for index, case_id in enumerate(sorted(assignment_ids))
    }
    mapping_rows = []
    try:
        for assignment in sorted(development, key=lambda row: row["case_id"]):
            case_id = _required(assignment, "case_id")
            safe_id = safe_ids[case_id]
            scan_path, label_path = _case_paths(input_root, manifest[case_id])
            _write_nnunet_case(
                scan_path=scan_path,
                label_path=label_path,
                image_output=images / f"{safe_id}_0000.nii.gz",
                label_output=labels / f"{safe_id}.nii.gz",
                case_id=case_id,
            )
            mapping_rows.append(
                {
                    "nnunet_case_id": safe_id,
                    "case_id": case_id,
                    "outer_split": "development",
                    "cv_fold": _required(assignment, "cv_fold"),
                    "source_scan": str(scan_path),
                    "source_label": str(label_path),
                }
            )

        splits = []
        development_ids = {row["nnunet_case_id"] for row in mapping_rows}
        for fold in range(5):
            validation_ids = {
                row["nnunet_case_id"]
                for row in mapping_rows
                if row["cv_fold"] == str(fold)
            }
            train_ids = development_ids - validation_ids
            if not validation_ids or not train_ids or validation_ids & train_ids:
                raise RuntimeError(f"Invalid fold {fold} after nnU-Net conversion")
            splits.append({"train": sorted(train_ids), "val": sorted(validation_ids)})

        _write_csv(output_root / "case_mapping.csv", mapping_rows)
        _write_json(output_root / "splits_final.json", splits)
        _write_dataset_json(
            output_root,
            num_training=len(mapping_rows),
            dataset_name=dataset_name,
        )
        report = {
            "mode": "lys_development_only",
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "n_training": len(mapping_rows),
            "n_locked_test_omitted": len(locked),
            "fold_validation_counts": [len(split["val"]) for split in splits],
            "locked_test_materialized": False,
            **input_hashes,
        }
        _write_json(output_root / "conversion_report.json", report)
        _validate_target_output(output_root, report=report)
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    return report


def prepare_external_dataset(
    *,
    input_root: Path,
    output_root: Path,
    dataset_id: int = 702,
    dataset_name: str = "ExternalMouseV0",
) -> dict[str, Any]:
    """Create a supervised source dataset, preserving an explicit source split if present."""
    _validate_dataset_identity(output_root, dataset_id=dataset_id, dataset_name=dataset_name)
    manifest_path = input_root / "manifest.csv"
    manifest = _rows_by_case(manifest_path)
    if not manifest:
        raise ValueError(f"No manifest cases in {manifest_path}")
    input_hashes = {"manifest_sha256": _sha256(manifest_path)}
    existing = _existing_matching_report(output_root, input_hashes)
    if existing is not None:
        return existing

    _require_absent(output_root)
    images = output_root / "imagesTr"
    labels = output_root / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir()
    mapping_rows = []
    try:
        split_ids: dict[str, list[str]] = {}
        for index, case_id in enumerate(sorted(manifest)):
            safe_id = f"EXT_{index:04d}"
            scan_path, label_path = _case_paths(input_root, manifest[case_id])
            _write_nnunet_case(
                scan_path=scan_path,
                label_path=label_path,
                image_output=images / f"{safe_id}_0000.nii.gz",
                label_output=labels / f"{safe_id}.nii.gz",
                case_id=case_id,
            )
            mapping_rows.append(
                {
                    "nnunet_case_id": safe_id,
                    "case_id": case_id,
                    "outer_split": "source_pretraining",
                    "cv_fold": "",
                    "source_split": _required(manifest[case_id], "split"),
                    "source_scan": str(scan_path),
                    "source_label": str(label_path),
                }
            )
            split_ids.setdefault(_required(manifest[case_id], "split"), []).append(safe_id)
        _write_csv(output_root / "case_mapping.csv", mapping_rows)
        source_splits = sorted(split_ids)
        if source_splits == ["train", "validation"]:
            _write_json(
                output_root / "splits_final.json",
                [{"train": sorted(split_ids["train"]), "val": sorted(split_ids["validation"])}],
            )
            mode = "external_preserved_train_validation_source_pretraining"
        elif len(source_splits) == 1:
            mode = "external_all_cases_source_pretraining"
        else:
            raise ValueError(
                "External manifest must be all one split or exactly train/validation; "
                f"found {source_splits}"
            )
        _write_dataset_json(
            output_root,
            num_training=len(mapping_rows),
            dataset_name=dataset_name,
        )
        report = {
            "mode": mode,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "n_training": len(mapping_rows),
            "target_lys_used": False,
            "source_split_counts": {
                split: len(ids) for split, ids in sorted(split_ids.items())
            },
            **input_hashes,
        }
        _write_json(output_root / "conversion_report.json", report)
        _validate_external_output(output_root, report=report)
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    return report


def prepare_paper_folds(
    *,
    input_root: Path,
    split_assignments: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create symlink-only train/validation trees from preserved development folds."""
    manifest_path = input_root / "manifest.csv"
    manifest = _rows_by_case(manifest_path)
    assignments = _read_csv(split_assignments)
    assignment_ids = [_required(row, "case_id") for row in assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("Duplicate case IDs in split assignments")
    missing = sorted(set(assignment_ids) - set(manifest))
    extra = sorted(set(manifest) - set(assignment_ids))
    if missing or extra:
        raise ValueError(
            "Manifest/split case mismatch: "
            f"missing_from_manifest={missing[:10]}, missing_from_splits={extra[:10]}"
        )
    development = [
        row for row in assignments if _required(row, "outer_split") == "development"
    ]
    locked = [row for row in assignments if _required(row, "outer_split") == "test"]
    folds = sorted({_required(row, "cv_fold") for row in development})
    if folds != ["0", "1", "2", "3", "4"]:
        raise ValueError(f"Expected preserved folds 0-4, found {folds}")
    if not locked or any(row.get("cv_fold", "") for row in locked):
        raise ValueError("Locked-test assignments are missing or have a CV fold")

    expected_hashes = {
        "manifest_sha256": _sha256(manifest_path),
        "split_assignments_sha256": _sha256(split_assignments),
    }
    if output_root.exists():
        report_path = output_root / "conversion_report.json"
        if not report_path.is_file():
            raise FileExistsError(f"Unverified existing paper-fold tree: {output_root}")
        report = json.loads(report_path.read_text())
        observed = {key: report.get(key) for key in expected_hashes}
        if observed != expected_hashes:
            raise RuntimeError(
                f"Existing paper-fold provenance differs: {observed} vs {expected_hashes}"
            )
        _validate_paper_folds(output_root, development=development)
        return report

    output_root.mkdir(parents=True)
    try:
        for fold in range(5):
            for assignment in development:
                case_id = _required(assignment, "case_id")
                row = manifest[case_id]
                split = "validation" if assignment["cv_fold"] == str(fold) else "train"
                source_case = _case_paths(input_root, row)[0].parent
                destination = (
                    output_root
                    / f"fold_{fold}"
                    / split
                    / _required(row, "study")
                    / _required(row, "timepoint")
                    / case_id
                )
                destination.mkdir(parents=True)
                for name in (
                    "scan.nii.gz",
                    "scan_lesionIAM.nii.gz",
                    "scan_lesion.nii.gz",
                ):
                    source = source_case / name
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    (destination / name).symlink_to(source.resolve())
        report = {
            "mode": "lys_development_folds_symlink_only",
            "n_development": len(development),
            "n_locked_test_omitted": len(locked),
            "locked_test_materialized": False,
            "fold_validation_counts": [
                sum(row["cv_fold"] == str(fold) for row in development)
                for fold in range(5)
            ],
            **expected_hashes,
        }
        _write_json(output_root / "conversion_report.json", report)
        _validate_paper_folds(output_root, development=development)
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    return report


def export_nnunet_validation(
    *,
    validation_root: Path,
    case_mapping: Path,
    fold: int,
    output_root: Path,
    candidate: str,
) -> dict[str, Any]:
    """Export class-1 nnU-Net probabilities in native NIfTI orientation."""
    if fold not in range(5):
        raise ValueError("fold must be between 0 and 4")
    if not validation_root.is_dir():
        raise FileNotFoundError(f"nnU-Net validation folder not found: {validation_root}")
    if not candidate or Path(candidate).name != candidate:
        raise ValueError("candidate must be one safe path component")
    mappings = _read_csv(case_mapping)
    expected = [row for row in mappings if _required(row, "cv_fold") == str(fold)]
    if not expected:
        raise ValueError(f"No mapping rows for fold {fold}")
    expected_ids = {_required(row, "nnunet_case_id") for row in expected}
    observed_ids = {path.stem for path in validation_root.glob("*.npz")}
    if observed_ids != expected_ids:
        raise ValueError(
            f"Fold {fold} validation probability IDs differ: "
            f"missing={sorted(expected_ids - observed_ids)[:10]}, "
            f"unexpected={sorted(observed_ids - expected_ids)[:10]}"
        )
    if output_root.exists():
        raise FileExistsError(f"OOF export already exists: {output_root}")
    output_root.mkdir(parents=True)

    records = []
    try:
        for row in sorted(expected, key=lambda item: item["case_id"]):
            safe_id = _required(row, "nnunet_case_id")
            case_id = _required(row, "case_id")
            npz_path = validation_root / f"{safe_id}.npz"
            segmentation_path = validation_root / f"{safe_id}.nii.gz"
            label_path = Path(_required(row, "source_label"))
            if not segmentation_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing segmentation/label for {case_id}: "
                    f"{segmentation_path}, {label_path}"
                )
            with np.load(npz_path) as data:
                probabilities = np.asarray(data["probabilities"], dtype=np.float32)
            if probabilities.ndim != 4 or probabilities.shape[0] != 2:
                raise ValueError(
                    f"Expected two-class probabilities for {case_id}; found {probabilities.shape}"
                )
            # NibabelIO stores nnU-Net arrays as Z,Y,X and writes them back as X,Y,Z.
            lesion_probability = probabilities[1].transpose((2, 1, 0))
            argmax_native = probabilities.argmax(axis=0).transpose((2, 1, 0)).astype(np.uint8)
            exported_segmentation = np.asarray(nib.load(str(segmentation_path)).dataobj)
            if not np.array_equal(argmax_native, exported_segmentation):
                raise RuntimeError(
                    f"Probability orientation check failed for {case_id}; refusing OOF export"
                )
            target_image = nib.load(str(label_path))
            target = np.asarray(target_image.dataobj)
            if lesion_probability.shape != target.shape:
                raise ValueError(
                    f"Probability/target shape mismatch for {case_id}: "
                    f"{lesion_probability.shape} vs {target.shape}"
                )
            finite = lesion_probability[np.isfinite(lesion_probability)]
            if not finite.size or float(finite.min()) < 0 or float(finite.max()) > 1:
                raise ValueError(f"Invalid probability range for {case_id}")

            case_output = output_root / safe_id
            case_output.mkdir()
            probability_path = case_output / "lesion_probability.nii.gz"
            target_output = case_output / "target_mask.nii.gz"
            _save_nifti_like(probability_path, lesion_probability, target_image, np.float32)
            _save_nifti_like(target_output, (target > 0.5).astype(np.uint8), target_image, np.uint8)
            records.append(
                {
                    "split": "validation",
                    "case_id": case_id,
                    "candidate": candidate,
                    "fold": fold,
                    "lesion_probability": str(probability_path),
                    "target_mask": str(target_output),
                    "nnunet_probability_npz": str(npz_path),
                }
            )
        manifest_path = output_root / "prediction_export_manifest.csv"
        _write_csv(manifest_path, records)
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    return {
        "candidate": candidate,
        "fold": fold,
        "n_cases": len(records),
        "manifest": str(output_root / "prediction_export_manifest.csv"),
    }


def _write_nnunet_case(
    *,
    scan_path: Path,
    label_path: Path,
    image_output: Path,
    label_output: Path,
    case_id: str,
) -> None:
    scan_image = nib.load(str(scan_path))
    label_image = nib.load(str(label_path))
    scan = np.asarray(scan_image.dataobj)
    label = np.asarray(label_image.dataobj)
    if scan.ndim != 4 or scan.shape[-1] != 1:
        raise ValueError(f"{case_id}: expected X,Y,Z,1 scan; found {scan.shape}")
    scan = scan[..., 0]
    if label.shape != scan.shape:
        raise ValueError(f"{case_id}: scan/label shape mismatch {scan.shape} vs {label.shape}")
    if not np.allclose(scan_image.affine, label_image.affine, rtol=0, atol=1e-5):
        raise ValueError(f"{case_id}: scan/label affine mismatch")
    unique = np.unique(label)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{case_id}: non-binary label values {unique[:20]}")
    if not np.isfinite(scan).all():
        raise ValueError(f"{case_id}: non-finite scan values")
    _save_nifti_like(image_output, scan.astype(np.float32), scan_image, np.float32)
    _save_nifti_like(label_output, label.astype(np.uint8), label_image, np.uint8)


def _save_nifti_like(
    path: Path,
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    dtype: Any,
) -> None:
    header = reference.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(dtype)
    header.set_zooms(reference.header.get_zooms()[: data.ndim])
    image = nib.Nifti1Image(data.astype(dtype, copy=False), reference.affine, header)
    nib.save(image, str(path))


def _write_dataset_json(root: Path, *, num_training: int, dataset_name: str) -> None:
    _write_json(
        root / "dataset.json",
        {
            "channel_names": {"0": "T2"},
            "labels": {"background": 0, "lesion": 1},
            "numTraining": num_training,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "NibabelIO",
            "name": dataset_name,
        },
    )


def _validate_target_output(root: Path, *, report: dict[str, Any]) -> None:
    expected = int(report["n_training"])
    if len(list((root / "imagesTr").glob("*_0000.nii.gz"))) != expected:
        raise RuntimeError("Incomplete target imagesTr")
    if len(list((root / "labelsTr").glob("*.nii.gz"))) != expected:
        raise RuntimeError("Incomplete target labelsTr")
    mapping = _read_csv(root / "case_mapping.csv")
    splits = json.loads((root / "splits_final.json").read_text())
    if len(mapping) != expected or len(splits) != 5:
        raise RuntimeError("Incomplete target mapping/splits")
    if (root / "imagesTs").exists():
        raise RuntimeError("Locked-test images must not exist in comparator target dataset")


def _validate_external_output(root: Path, *, report: dict[str, Any]) -> None:
    expected = int(report["n_training"])
    if len(list((root / "imagesTr").glob("*_0000.nii.gz"))) != expected:
        raise RuntimeError("Incomplete external imagesTr")
    if len(list((root / "labelsTr").glob("*.nii.gz"))) != expected:
        raise RuntimeError("Incomplete external labelsTr")
    if len(_read_csv(root / "case_mapping.csv")) != expected:
        raise RuntimeError("Incomplete external mapping")
    split_counts = report.get("source_split_counts", {})
    if set(split_counts) == {"train", "validation"}:
        split_path = root / "splits_final.json"
        if not split_path.is_file():
            raise RuntimeError("External train/validation split was not preserved")
        splits = json.loads(split_path.read_text())
        if len(splits) != 1:
            raise RuntimeError("Expected exactly one external source split")
        if len(splits[0]["train"]) != int(split_counts["train"]):
            raise RuntimeError("External source train count differs")
        if len(splits[0]["val"]) != int(split_counts["validation"]):
            raise RuntimeError("External source validation count differs")


def _validate_paper_folds(
    root: Path,
    *,
    development: list[dict[str, str]],
) -> None:
    expected_ids = {_required(row, "case_id") for row in development}
    for fold in range(5):
        train_ids = {
            path.parent.name for path in (root / f"fold_{fold}/train").rglob("scan.nii.gz")
        }
        validation_ids = {
            path.parent.name
            for path in (root / f"fold_{fold}/validation").rglob("scan.nii.gz")
        }
        expected_validation = {
            _required(row, "case_id")
            for row in development
            if _required(row, "cv_fold") == str(fold)
        }
        if train_ids & validation_ids:
            raise RuntimeError(f"Paper fold {fold} has train/validation leakage")
        if validation_ids != expected_validation or train_ids | validation_ids != expected_ids:
            raise RuntimeError(f"Paper fold {fold} differs from preserved assignments")
        for path in (root / f"fold_{fold}").rglob("*.nii.gz"):
            if not path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Broken or copied paper-fold input: {path}")


def _existing_matching_report(
    output_root: Path,
    expected_hashes: dict[str, str],
) -> dict[str, Any] | None:
    if not output_root.exists():
        return None
    report_path = output_root / "conversion_report.json"
    if not report_path.is_file():
        raise FileExistsError(f"Unverified existing comparator dataset: {output_root}")
    report = json.loads(report_path.read_text())
    observed = {key: report.get(key) for key in expected_hashes}
    if observed != expected_hashes:
        raise RuntimeError(
            f"Existing comparator dataset provenance differs: {observed} vs {expected_hashes}"
        )
    if report.get("mode") == "lys_development_only":
        _validate_target_output(output_root, report=report)
    else:
        _validate_external_output(output_root, report=report)
    return report


def _validate_dataset_identity(root: Path, *, dataset_id: int, dataset_name: str) -> None:
    if not 1 <= dataset_id <= 999:
        raise ValueError("dataset_id must be between 1 and 999")
    if not dataset_name or not dataset_name.replace("_", "").isalnum():
        raise ValueError("dataset_name must contain only letters, numbers, and underscores")
    expected = f"Dataset{dataset_id:03d}_{dataset_name}"
    if root.name != expected:
        raise ValueError(f"Output directory must be named {expected}; got {root.name}")


def _case_paths(input_root: Path, row: dict[str, str]) -> tuple[Path, Path]:
    relative = (
        Path(_required(row, "split"))
        / _required(row, "study")
        / _required(row, "timepoint")
        / _required(row, "case_id")
    )
    case_root = input_root / relative
    scan = case_root / "scan.nii.gz"
    label = case_root / "scan_lesionIAM.nii.gz"
    if not scan.is_file() or not label.is_file():
        raise FileNotFoundError(f"Missing canonical case files under {case_root}")
    return scan, label


def _rows_by_case(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    result = {}
    for row in rows:
        case_id = _required(row, "case_id")
        if case_id in result:
            raise ValueError(f"Duplicate case_id={case_id!r} in {path}")
        result[case_id] = row
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing required value {key!r} in row {row}")
    return value


def _require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)


if __name__ == "__main__":
    raise SystemExit(main())
