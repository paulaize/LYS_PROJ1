"""Normalize Kaggle-altered prepared NIfTI datasets to canonical filenames.

Kaggle may expose ``.nii.gz`` inputs as uncompressed ``.nii`` files or leave
gzip payloads under truncated suffixes. This command detects compression from
the bytes, fully validates every scan/mask pair, and writes a canonical
``scan.nii.gz`` / ``scan_lesionIAM.nii.gz`` / ``scan_lesion.nii.gz`` tree.
The source dataset is read-only.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Exposed prepared dataset root")
    parser.add_argument("--output-base", required=True, help="Writable parent for normalized data")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--expected-cases", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = normalize_prepared_dataset(
        source_root=Path(args.input),
        output_base=Path(args.output_base),
        dataset_name=args.dataset_name,
        expected_cases=args.expected_cases,
    )
    print(f"Normalized dataset: {result}")
    return 0


def normalize_prepared_dataset(
    *,
    source_root: Path,
    output_base: Path,
    dataset_name: str,
    expected_cases: int,
) -> Path:
    """Write a fully validated canonical copy and return its final root."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Prepared dataset root not found: {source_root}")
    if (
        not dataset_name
        or dataset_name in {".", ".."}
        or Path(dataset_name).name != dataset_name
    ):
        raise ValueError("dataset_name must be one non-empty path component")
    if expected_cases <= 0:
        raise ValueError("expected_cases must be > 0")

    output_base.mkdir(parents=True, exist_ok=True)
    final_root = output_base / dataset_name
    building_root = output_base / f".{dataset_name}.building"
    temporary_root = output_base / ".normalization_tmp"
    temporary_root.mkdir(exist_ok=True)

    if final_root.exists():
        _validate_complete_root(final_root, expected_cases=expected_cases)
        return final_root

    # This path is reserved for an incomplete build and is safe to replace.
    if building_root.exists():
        shutil.rmtree(building_root)
    building_root.mkdir()

    rows, fieldnames = _read_manifest(source_root / "manifest.csv")
    if len(rows) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} manifest rows for {dataset_name}; found {len(rows)}"
        )
    for column in ("case_dir", "scan_path", "label_path"):
        if column not in fieldnames:
            fieldnames.append(column)

    output_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, object]] = []
    try:
        for index, row in enumerate(rows, start=1):
            output_row, report_row = _normalize_case(
                source_root=source_root,
                building_root=building_root,
                final_root=final_root,
                temporary_root=temporary_root,
                row=row,
                index=index,
            )
            output_rows.append(output_row)
            report_rows.append(report_row)
            if index % 25 == 0 or index == expected_cases:
                print(f"{dataset_name}: normalized {index}/{expected_cases}")

        _write_csv(building_root / "manifest.csv", output_rows, fieldnames)
        _write_csv(
            building_root / "normalization_report.csv",
            report_rows,
            list(report_rows[0]),
        )
        _validate_complete_root(building_root, expected_cases=expected_cases)
        building_root.rename(final_root)
    except Exception:
        # Keep the incomplete tree for diagnosis; the next run replaces it.
        raise
    return final_root


def _normalize_case(
    *,
    source_root: Path,
    building_root: Path,
    final_root: Path,
    temporary_root: Path,
    row: dict[str, str],
    index: int,
) -> tuple[dict[str, str], dict[str, object]]:
    case_id = _required_component(row, "case_id")
    relative_case = (
        Path(_required_component(row, "split"))
        / _required_component(row, "study")
        / _required_component(row, "timepoint")
        / case_id
    )
    source_case = source_root / relative_case
    destination_case = building_root / relative_case
    final_case = final_root / relative_case
    if not source_case.is_dir():
        raise FileNotFoundError(f"{case_id}: missing source case directory {source_case}")

    files = [path for path in source_case.iterdir() if path.is_file()]
    source_scan = _require_one(
        (path for path in files if path.name in {"scan.nii", "scan.nii.gz"}),
        role="scan",
        case_id=case_id,
    )
    source_iam = _require_one(
        (path for path in files if path.name.startswith("scan_lesionIAM.")),
        role="IAM lesion mask",
        case_id=case_id,
    )
    source_alias = _require_one(
        (path for path in files if path.name.startswith("scan_lesion.")),
        role="lesion-mask alias",
        case_id=case_id,
    )
    destination_case.mkdir(parents=True)

    with tempfile.TemporaryDirectory(
        dir=temporary_root,
        prefix=f"{index:04d}_",
    ) as temporary:
        temporary_path = Path(temporary)
        staged_scan, scan_transport = _stage_with_correct_extension(
            source_scan, temporary_path, "source_scan"
        )
        staged_iam, iam_transport = _stage_with_correct_extension(
            source_iam, temporary_path, "source_iam"
        )
        staged_alias, alias_transport = _stage_with_correct_extension(
            source_alias, temporary_path, "source_alias"
        )

        scan_image = nib.load(str(staged_scan))
        iam_image = nib.load(str(staged_iam))
        alias_image = nib.load(str(staged_alias))
        scan_data = _full_array(scan_image, source=source_scan)
        iam_data = _full_array(iam_image, source=source_iam)
        alias_data = _full_array(alias_image, source=source_alias)
        _validate_case(
            case_id=case_id,
            scan_image=scan_image,
            scan_data=scan_data,
            iam_image=iam_image,
            iam_data=iam_data,
            alias_image=alias_image,
            alias_data=alias_data,
        )

        outputs = {
            "scan.nii.gz": (scan_image, scan_data),
            "scan_lesionIAM.nii.gz": (iam_image, iam_data),
            "scan_lesion.nii.gz": (alias_image, alias_data),
        }
        for filename, (image, expected_data) in outputs.items():
            temporary_output = temporary_path / filename
            nib.save(image, str(temporary_output))
            reloaded = nib.load(str(temporary_output))
            if not np.array_equal(np.asarray(reloaded.dataobj), expected_data):
                raise RuntimeError(f"{case_id}: data changed while writing {filename}")
            if not np.allclose(reloaded.affine, image.affine, rtol=0, atol=1e-7):
                raise RuntimeError(f"{case_id}: affine changed while writing {filename}")
            os.replace(temporary_output, destination_case / filename)

    output_row = dict(row)
    output_row["case_dir"] = str(final_case)
    output_row["scan_path"] = str(final_case / "scan.nii.gz")
    output_row["label_path"] = str(final_case / "scan_lesionIAM.nii.gz")
    report_row: dict[str, object] = {
        "case_id": case_id,
        "source_scan_name": source_scan.name,
        "source_scan_transport": scan_transport,
        "source_iam_name": source_iam.name,
        "source_iam_transport": iam_transport,
        "source_alias_name": source_alias.name,
        "source_alias_transport": alias_transport,
        "scan_shape": str(tuple(scan_data.shape)),
        "mask_shape": str(tuple(iam_data.shape)),
        "iam_alias_identical": True,
    }
    return output_row, report_row


def _validate_case(
    *,
    case_id: str,
    scan_image: nib.spatialimages.SpatialImage,
    scan_data: np.ndarray,
    iam_image: nib.spatialimages.SpatialImage,
    iam_data: np.ndarray,
    alias_image: nib.spatialimages.SpatialImage,
    alias_data: np.ndarray,
) -> None:
    if scan_data.ndim != 4 or scan_data.shape[-1] != 1:
        raise RuntimeError(
            f"{case_id}: expected scan shape (X,Y,Z,1); found {scan_data.shape}"
        )
    if iam_data.shape != scan_data.shape[:3]:
        raise RuntimeError(
            f"{case_id}: scan/mask shape mismatch: {scan_data.shape} versus {iam_data.shape}"
        )
    if alias_data.shape != iam_data.shape:
        raise RuntimeError(
            f"{case_id}: IAM/alias shape mismatch: {iam_data.shape} versus {alias_data.shape}"
        )
    if not np.array_equal(iam_data, alias_data):
        raise RuntimeError(f"{case_id}: IAM and alias lesion masks are not identical")
    unique_values = np.unique(iam_data)
    if not np.all(np.isin(unique_values, (0, 1))):
        raise RuntimeError(f"{case_id}: non-binary lesion mask values: {unique_values[:20]}")
    for role, image in (("IAM mask", iam_image), ("alias mask", alias_image)):
        if not np.allclose(scan_image.affine, image.affine, rtol=0, atol=1e-5):
            raise RuntimeError(f"{case_id}: scan/{role} affine mismatch")


def _stage_with_correct_extension(
    source: Path,
    temporary_dir: Path,
    staged_name: str,
) -> tuple[Path, str]:
    transport = _payload_transport(source)
    suffix = ".nii.gz" if transport == "gzip" else ".nii"
    staged = temporary_dir / f"{staged_name}{suffix}"
    shutil.copyfile(source, staged)
    return staged, transport


def _payload_transport(path: Path) -> str:
    with path.open("rb") as handle:
        magic = handle.read(2)
    return "gzip" if magic == b"\x1f\x8b" else "plain"


def _full_array(image: nib.spatialimages.SpatialImage, *, source: Path) -> np.ndarray:
    data = np.asarray(image.dataobj)
    if not np.isfinite(data).all():
        raise RuntimeError(f"Non-finite values found in {source}")
    return data


def _require_one(paths, *, role: str, case_id: str) -> Path:
    matches = list(paths)
    if len(matches) != 1:
        raise RuntimeError(
            f"{case_id}: expected exactly one {role}; found {[path.name for path in matches]}"
        )
    return matches[0]


def _validate_complete_root(root: Path, *, expected_cases: int) -> None:
    expected = {
        "scan.nii.gz": len(list(root.rglob("scan.nii.gz"))),
        "scan_lesionIAM.nii.gz": len(list(root.rglob("scan_lesionIAM.nii.gz"))),
        "scan_lesion.nii.gz": len(list(root.rglob("scan_lesion.nii.gz"))),
    }
    wrong = {name: count for name, count in expected.items() if count != expected_cases}
    if wrong:
        raise RuntimeError(
            f"Normalized root {root} does not contain {expected_cases} files per role: {wrong}"
        )
    for filename in ("manifest.csv", "normalization_report.csv"):
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Missing normalized dataset file: {root / filename}")


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows, fieldnames


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _required(row: dict[str, str], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"Manifest row is missing required value {key!r}: {row}")
    return value


def _required_component(row: dict[str, str], key: str) -> str:
    value = _required(row, key)
    if value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Manifest value {key!r} must be one safe path component: {value!r}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
