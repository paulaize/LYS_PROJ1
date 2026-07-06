"""Create a superior/inferior flipped copy of oriented external cases.

This is a voxel-array operation, unlike the header-only orientation relabel in
``orient_external_dataset_lsp.py``. It flips the scan and matching lesion mask
along array axis 1 and keeps the existing affine/header orientation. Use this
only on a separate copy; never rewrite the original external source files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

DEFAULT_EXTERNAL_ROOT = Path("/Volumes/Untitled/external_datasets")
DEFAULT_INPUT_SOURCE_NAME = "ratlesnetv2_clean_source_LSP_oriented"
DEFAULT_OUTPUT_SOURCE_NAME = "ratlesnetv2_clean_source_LSP_SI_flipped_test"
DEFAULT_INPUT_MANIFEST_NAME = "external_dataset_manifest_LSP_oriented.csv"
DEFAULT_OUTPUT_MANIFEST_NAME = "external_dataset_manifest_LSP_SI_flipped_test.csv"
DEFAULT_REFERENCE_CASE_IDS = (
    "An2022__20190320CH_Exp4_M20",
    "Koch2017__20170428AR_TTC_M06",
    "Knab2025__20170207CH_SC01",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root",
        default=str(DEFAULT_EXTERNAL_ROOT),
        help="Root containing external dataset manifests and source folders",
    )
    parser.add_argument(
        "--input-source-root",
        help="Input source root. Defaults to <external-root>/ratlesnetv2_clean_source_LSP_oriented",
    )
    parser.add_argument(
        "--input-manifest",
        help=(
            "Input manifest CSV. Defaults to "
            "<external-root>/manifests/external_dataset_manifest_LSP_oriented.csv"
        ),
    )
    parser.add_argument(
        "--output-source-root",
        help=(
            "Output source root. Defaults to "
            "<external-root>/ratlesnetv2_clean_source_LSP_SI_flipped_test"
        ),
    )
    parser.add_argument(
        "--output-manifest",
        help=(
            "Output manifest CSV. Defaults to "
            "<external-root>/manifests/external_dataset_manifest_LSP_SI_flipped_test.csv"
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help=(
            "Case ID to flip. May be repeated. Defaults to the three visually "
            "checked reference cases."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Flip all rows in the input manifest instead of the reference cases only",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=1,
        help="Array axis to flip. Default: 1, the S/I axis for the current LSP copy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs in the output source root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rows without writing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    external_root = Path(args.external_root)
    input_source_root = Path(args.input_source_root) if args.input_source_root else (
        external_root / DEFAULT_INPUT_SOURCE_NAME
    )
    input_manifest = Path(args.input_manifest) if args.input_manifest else (
        external_root / "manifests" / DEFAULT_INPUT_MANIFEST_NAME
    )
    output_source_root = Path(args.output_source_root) if args.output_source_root else (
        external_root / DEFAULT_OUTPUT_SOURCE_NAME
    )
    output_manifest = Path(args.output_manifest) if args.output_manifest else (
        external_root / "manifests" / DEFAULT_OUTPUT_MANIFEST_NAME
    )
    rows = _read_manifest(input_manifest)
    selected = _select_rows(
        rows,
        case_ids=set(args.case_id or DEFAULT_REFERENCE_CASE_IDS),
        include_all=args.all,
    )
    print(f"Input manifest rows: {len(rows)}")
    print(f"Selected rows: {len(selected)}")
    print(f"Input source root: {input_source_root}")
    print(f"Output source root: {output_source_root}")
    print(f"Flip axis: {args.axis}")
    if args.dry_run:
        for row in selected:
            print(f"  {row['global_case_id']} {row['source_dataset']} {row['shape']}")
        return 0

    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        scan_path = Path(row["scan_path"])
        mask_path = Path(row["mask_path"])
        out_scan = output_source_root / "T2w" / scan_path.name
        out_mask = output_source_root / "masks" / mask_path.name
        print(f"[{index}/{len(selected)}] {row['global_case_id']}")
        records.append(
            flip_pair_axis(
                row=row,
                scan_path=scan_path,
                mask_path=mask_path,
                out_scan=out_scan,
                out_mask=out_mask,
                axis=args.axis,
                overwrite=args.overwrite,
            )
        )

    _write_manifest(output_manifest, records)
    _write_readme(
        output_source_root / "README.md",
        input_source_root=input_source_root,
        input_manifest=input_manifest,
        output_source_root=output_source_root,
        output_manifest=output_manifest,
        row_count=len(records),
        axis=args.axis,
    )
    print(f"Wrote S/I flipped source root: {output_source_root}")
    print(f"Wrote manifest: {output_manifest}")
    return 0


def flip_pair_axis(
    *,
    row: dict[str, Any],
    scan_path: Path,
    mask_path: Path,
    out_scan: Path,
    out_mask: Path,
    axis: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Flip scan and mask along one array axis while preserving the affine."""
    if out_scan.exists() and out_mask.exists() and not overwrite:
        return _record_from_output(row, scan_path, mask_path, out_scan, out_mask, axis=axis)

    scan = nib.load(str(scan_path))
    mask = nib.load(str(mask_path))
    if len(scan.shape) != 3:
        raise ValueError(f"Expected a 3-D scan for {scan_path}, got shape {scan.shape}")
    if tuple(scan.shape) != tuple(mask.shape):
        raise ValueError(f"Scan shape {scan.shape} does not match mask shape {mask.shape}")
    if axis < 0 or axis >= len(scan.shape):
        raise ValueError(f"Flip axis {axis} is out of range for shape {scan.shape}")

    scan_data = np.flip(np.asanyarray(scan.dataobj), axis=axis).copy()
    mask_before = np.asanyarray(mask.dataobj) > 0
    mask_data = np.flip(mask_before, axis=axis).astype(np.uint8)
    spacing = tuple(float(v) for v in scan.header.get_zooms()[:3])

    out_scan.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(_new_like(scan_data, scan, spacing=spacing), str(out_scan))
    nib.save(_new_like(mask_data, scan, spacing=spacing, dtype=np.uint8), str(out_mask))

    written_scan = nib.load(str(out_scan))
    written_mask = nib.load(str(out_mask))
    if tuple(written_scan.shape) != tuple(scan.shape):
        raise ValueError(f"Output scan shape changed for {row['global_case_id']}")
    if tuple(written_mask.shape) != tuple(mask.shape):
        raise ValueError(f"Output mask shape changed for {row['global_case_id']}")
    if not np.array_equal(np.asanyarray(written_scan.dataobj), scan_data):
        raise ValueError(f"Output scan data mismatch after flip for {row['global_case_id']}")
    if int((np.asanyarray(written_mask.dataobj) > 0).sum()) != int(mask_before.sum()):
        raise ValueError(f"Mask voxel count changed for {row['global_case_id']}")
    if nib.aff2axcodes(written_scan.affine) != nib.aff2axcodes(scan.affine):
        raise ValueError(f"Output scan orientation changed for {row['global_case_id']}")
    if not np.allclose(written_scan.affine, written_mask.affine):
        raise ValueError(f"Output scan/mask affine mismatch for {row['global_case_id']}")

    return _record_from_output(row, scan_path, mask_path, out_scan, out_mask, axis=axis)


def _new_like(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    *,
    spacing: tuple[float, float, float],
    dtype: np.dtype | type | None = None,
) -> nib.Nifti1Image:
    header = reference.header.copy()
    header.set_data_shape(data.shape)
    header.set_zooms(spacing)
    if dtype is not None:
        header.set_data_dtype(dtype)
    else:
        header.set_data_dtype(data.dtype)
    image = nib.Nifti1Image(data, reference.affine, header)
    image.set_qform(reference.affine, code=int(reference.header["qform_code"]) or 1)
    image.set_sform(reference.affine, code=int(reference.header["sform_code"]) or 1)
    return image


def _record_from_output(
    row: dict[str, Any],
    source_scan_path: Path,
    source_mask_path: Path,
    out_scan: Path,
    out_mask: Path,
    *,
    axis: int,
) -> dict[str, Any]:
    scan = nib.load(str(out_scan))
    mask = nib.load(str(out_mask))
    mask_voxels = int((np.asanyarray(mask.dataobj) > 0).sum())
    spacing = tuple(float(v) for v in scan.header.get_zooms()[:3])
    record = dict(row)
    record.update(
        {
            "scan_path": str(out_scan),
            "mask_path": str(out_mask),
            "shape": "x".join(str(int(v)) for v in scan.shape[:3]),
            "spacing_x_mm": spacing[0],
            "spacing_y_mm": spacing[1],
            "spacing_z_mm": spacing[2],
            "mask_voxels": mask_voxels,
            "lesion_volume_mm3": f"{mask_voxels * float(np.prod(spacing)):.10g}",
            "qc_flag": "needs_visual_qc_lsp_si_flip",
            "source_scan_path_before_si_flip": str(source_scan_path),
            "source_mask_path_before_si_flip": str(source_mask_path),
            "axcodes_after_si_flip": "".join(nib.aff2axcodes(scan.affine)),
            "si_flip_axis": axis,
            "si_flip_transform": f"voxel_array_flip_axis_{axis}_keep_affine",
        }
    )
    return record


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _select_rows(
    rows: list[dict[str, str]],
    *,
    case_ids: set[str],
    include_all: bool,
) -> list[dict[str, str]]:
    if include_all:
        return rows
    selected = [row for row in rows if row["global_case_id"] in case_ids]
    missing = sorted(case_ids - {row["global_case_id"] for row in selected})
    if missing:
        raise ValueError(f"Case IDs not found in manifest: {', '.join(missing)}")
    return selected


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    path: Path,
    *,
    input_source_root: Path,
    input_manifest: Path,
    output_source_root: Path,
    output_manifest: Path,
    row_count: int,
    axis: int,
) -> None:
    lines = [
        "# LSP S/I Flipped External Dataset",
        "",
        "Generated by `python -m ratlesnetv2_finetune.scripts.flip_external_si_axis`.",
        "",
        f"Input source root: `{input_source_root}`",
        f"Input manifest: `{input_manifest}`",
        f"Output source root: `{output_source_root}`",
        f"Output manifest: `{output_manifest}`",
        f"Rows written: {row_count}",
        f"Array flip axis: `{axis}`",
        "",
        "This copy flips the voxel arrays along the selected axis while",
        "preserving the current affine/header orientation. The same flip is",
        "applied to each scan and its matching binary lesion mask. Masks",
        "preserve voxel counts.",
        "",
        "Do not use this folder for training until visual QC confirms that the",
        "displayed anatomy and mask overlays are correct in ITK-SNAP.",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
