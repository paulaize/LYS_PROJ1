"""Create an orientation-relabeled copy of external mouse T2w scans and masks.

This fixes the header/affine convention for non-Mulder public datasets whose
third array axis is a coronal slice stack but is declared as superior/inferior
in the source NIfTI header. Voxel arrays are not permuted or resampled. The
same new affine is written to the scan and the corresponding binary mask. The
default target is LSP, but ``--target-axcodes`` can be used for a visually
confirmed target such as RSA.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

DEFAULT_EXTERNAL_ROOT = Path("/Volumes/Untitled/external_datasets")
DEFAULT_INPUT_SOURCE_NAME = "ratlesnetv2_clean_source"
DEFAULT_OUTPUT_SOURCE_NAME = "ratlesnetv2_clean_source_LSP_oriented"
DEFAULT_TARGET_AXCODES = ("L", "S", "P")
DEFAULT_EXCLUDED_DATASETS = ("Mulder2017",)

_AXIS_CODE_TO_WORLD = {
    "R": (0, 1.0),
    "L": (0, -1.0),
    "A": (1, 1.0),
    "P": (1, -1.0),
    "S": (2, 1.0),
    "I": (2, -1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root",
        default=str(DEFAULT_EXTERNAL_ROOT),
        help="Root created by download_external_datasets.py",
    )
    parser.add_argument(
        "--input-source-root",
        help="Existing clean source root. Defaults to <external-root>/ratlesnetv2_clean_source",
    )
    parser.add_argument(
        "--input-manifest",
        help=(
            "Existing external manifest CSV. Defaults to "
            "<external-root>/manifests/external_dataset_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-source-root",
        help=(
            "New oriented source root to write. Defaults to "
            "<external-root>/ratlesnetv2_clean_source_LSP_oriented"
        ),
    )
    parser.add_argument(
        "--output-manifest",
        help=(
            "Manifest for the oriented copy. Defaults to "
            "<external-root>/manifests/external_dataset_manifest_LSP_oriented.csv"
        ),
    )
    parser.add_argument(
        "--target-axcodes",
        default="".join(DEFAULT_TARGET_AXCODES),
        help="Three-letter NIfTI orientation code to write. Default: LSP",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=list(DEFAULT_EXCLUDED_DATASETS),
        help="Source dataset to omit. Defaults to Mulder2017. May be repeated.",
    )
    parser.add_argument(
        "--include-dataset",
        action="append",
        help="If provided, only include these source datasets. May be repeated.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output source root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected cases without writing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    external_root = Path(args.external_root)
    input_source_root = Path(args.input_source_root) if args.input_source_root else (
        external_root / DEFAULT_INPUT_SOURCE_NAME
    )
    input_manifest = Path(args.input_manifest) if args.input_manifest else (
        external_root / "manifests" / "external_dataset_manifest.csv"
    )
    output_source_root = Path(args.output_source_root) if args.output_source_root else (
        external_root / DEFAULT_OUTPUT_SOURCE_NAME
    )
    output_manifest = Path(args.output_manifest) if args.output_manifest else (
        external_root / "manifests" / "external_dataset_manifest_LSP_oriented.csv"
    )
    target_axcodes = _parse_axcodes(args.target_axcodes)

    rows = _read_manifest(input_manifest)
    selected = _select_rows(
        rows,
        include_datasets=set(args.include_dataset or []),
        exclude_datasets=set(args.exclude_dataset or []),
    )
    print(f"Input manifest rows: {len(rows)}")
    print(f"Selected rows: {len(selected)}")
    print(f"Target orientation: {''.join(target_axcodes)}")
    print(f"Input source root: {input_source_root}")
    print(f"Output source root: {output_source_root}")
    if args.dry_run:
        for row in selected[:20]:
            print(f"  {row['global_case_id']} {row['source_dataset']} {row['shape']}")
        if len(selected) > 20:
            print(f"  ... {len(selected) - 20} more")
        return 0

    output_records: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        scan_path = Path(row["scan_path"])
        mask_path = Path(row["mask_path"])
        out_scan = output_source_root / "T2w" / scan_path.name
        out_mask = output_source_root / "masks" / mask_path.name
        print(f"[{index}/{len(selected)}] {row['global_case_id']}")
        record = relabel_pair_to_axcodes(
            row=row,
            scan_path=scan_path,
            mask_path=mask_path,
            out_scan=out_scan,
            out_mask=out_mask,
            target_axcodes=target_axcodes,
            overwrite=args.overwrite,
        )
        output_records.append(record)

    _write_manifest(output_manifest, output_records)
    _write_readme(
        output_source_root.parent / "README_LSP_oriented_external_dataset.md",
        input_source_root=input_source_root,
        input_manifest=input_manifest,
        output_source_root=output_source_root,
        output_manifest=output_manifest,
        target_axcodes=target_axcodes,
        row_count=len(output_records),
        excluded=args.exclude_dataset,
    )
    print(f"Wrote oriented source root: {output_source_root}")
    print(f"Wrote manifest: {output_manifest}")
    return 0


def relabel_pair_to_axcodes(
    *,
    row: dict[str, Any],
    scan_path: Path,
    mask_path: Path,
    out_scan: Path,
    out_mask: Path,
    target_axcodes: tuple[str, str, str] = DEFAULT_TARGET_AXCODES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write scan/mask copies with a new affine and unchanged voxel arrays."""
    if out_scan.exists() and out_mask.exists() and not overwrite:
        return _record_from_output(row, scan_path, mask_path, out_scan, out_mask, target_axcodes)

    scan = nib.load(str(scan_path))
    mask = nib.load(str(mask_path))
    source_axcodes = nib.aff2axcodes(scan.affine)
    if len(scan.shape) != 3:
        raise ValueError(f"Expected a 3-D scan for {scan_path}, got shape {scan.shape}")
    if tuple(scan.shape) != tuple(mask.shape):
        raise ValueError(f"Scan shape {scan.shape} does not match mask shape {mask.shape}")

    spacing = tuple(float(v) for v in scan.header.get_zooms()[:3])
    affine = affine_for_axcodes(scan.shape, spacing, target_axcodes)
    scan_data = np.asanyarray(scan.dataobj).copy()
    mask_bool = np.asanyarray(mask.dataobj) > 0
    mask_data = mask_bool.astype(np.uint8)
    mask_voxels_before = int(mask_bool.sum())

    out_scan.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(_new_image(scan_data, scan.header, affine, spacing), str(out_scan))
    nib.save(_new_image(mask_data, scan.header, affine, spacing, dtype=np.uint8), str(out_mask))

    written_scan = nib.load(str(out_scan))
    written_mask = nib.load(str(out_mask))
    if nib.aff2axcodes(written_scan.affine) != target_axcodes:
        raise ValueError(
            f"Output scan {out_scan} has orientation {nib.aff2axcodes(written_scan.affine)}, "
            f"expected {target_axcodes}"
        )
    if nib.aff2axcodes(written_mask.affine) != target_axcodes:
        raise ValueError(
            f"Output mask {out_mask} has orientation {nib.aff2axcodes(written_mask.affine)}, "
            f"expected {target_axcodes}"
        )
    shape_changed = tuple(written_scan.shape) != tuple(scan.shape) or tuple(
        written_mask.shape
    ) != tuple(scan.shape)
    if shape_changed:
        raise ValueError(f"Output shape changed unexpectedly for {row['global_case_id']}")
    mask_voxels_after = int((np.asanyarray(written_mask.dataobj) > 0).sum())
    if mask_voxels_after != mask_voxels_before:
        raise ValueError(f"Mask voxel count changed for {row['global_case_id']}")

    return _record_from_output(
        row,
        scan_path,
        mask_path,
        out_scan,
        out_mask,
        target_axcodes,
        source_axcodes_before=source_axcodes,
    )


def affine_for_axcodes(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    axcodes: tuple[str, str, str],
) -> np.ndarray:
    """Build a centered affine with the requested NIfTI axis codes."""
    _validate_axcodes(axcodes)
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = 0.0
    for voxel_axis, code in enumerate(axcodes):
        world_axis, sign = _AXIS_CODE_TO_WORLD[code]
        affine[world_axis, voxel_axis] = sign * spacing[voxel_axis]
    center_voxel = (np.asarray(shape, dtype=float) - 1.0) / 2.0
    affine[:3, 3] = -(affine[:3, :3] @ center_voxel)
    return affine


def _new_image(
    data: np.ndarray,
    source_header: nib.nifti1.Nifti1Header,
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    dtype: np.dtype | type | None = None,
) -> nib.Nifti1Image:
    header = source_header.copy()
    header.set_data_shape(data.shape)
    if dtype is not None:
        header.set_data_dtype(dtype)
    else:
        header.set_data_dtype(data.dtype)
    header.set_zooms(spacing)
    image = nib.Nifti1Image(data, affine, header)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    return image


def _record_from_output(
    row: dict[str, Any],
    source_scan_path: Path,
    source_mask_path: Path,
    out_scan: Path,
    out_mask: Path,
    target_axcodes: tuple[str, str, str],
    *,
    source_axcodes_before: tuple[str, str, str] | None = None,
) -> dict[str, Any]:
    if source_axcodes_before is None:
        source_scan = nib.load(str(source_scan_path))
        source_axcodes_before = nib.aff2axcodes(source_scan.affine)
    output_scan = nib.load(str(out_scan))
    output_mask = nib.load(str(out_mask))
    mask_voxels = int((np.asanyarray(output_mask.dataobj) > 0).sum())
    spacing = tuple(float(v) for v in output_scan.header.get_zooms()[:3])
    target = "".join(target_axcodes)
    record = dict(row)
    record.update(
        {
            "scan_path": str(out_scan),
            "mask_path": str(out_mask),
            "shape": "x".join(str(int(v)) for v in output_scan.shape[:3]),
            "spacing_x_mm": spacing[0],
            "spacing_y_mm": spacing[1],
            "spacing_z_mm": spacing[2],
            "mask_voxels": mask_voxels,
            "lesion_volume_mm3": f"{mask_voxels * float(np.prod(spacing)):.10g}",
            "qc_flag": f"needs_visual_qc_{target.lower()}_orientation",
            "source_scan_path_before_orientation": str(source_scan_path),
            "source_mask_path_before_orientation": str(source_mask_path),
            "source_axcodes_before_orientation": "".join(source_axcodes_before),
            "target_axcodes_after_orientation": target,
            "orientation_transform": f"header_relabel_to_{target}_no_data_permutation",
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
    include_datasets: set[str],
    exclude_datasets: set[str],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        dataset = row["source_dataset"]
        if include_datasets and dataset not in include_datasets:
            continue
        if dataset in exclude_datasets:
            continue
        selected.append(row)
    return selected


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No oriented rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=base_fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    path: Path,
    *,
    input_source_root: Path,
    input_manifest: Path,
    output_source_root: Path,
    output_manifest: Path,
    target_axcodes: tuple[str, str, str],
    row_count: int,
    excluded: list[str],
) -> None:
    target = "".join(target_axcodes)
    lines = [
        f"# {target}-Oriented External Dataset Copy",
        "",
        "Generated by `python -m ratlesnetv2_finetune.scripts.orient_external_dataset_lsp`.",
        "",
        f"Input source root: `{input_source_root}`",
        f"Input manifest: `{input_manifest}`",
        f"Output source root: `{output_source_root}`",
        f"Output manifest: `{output_manifest}`",
        f"Rows written: {row_count}",
        f"Target NIfTI axis codes: `{target}`",
        f"Excluded datasets: `{', '.join(excluded) if excluded else 'none'}`",
        "",
        "This is a header/affine relabel only. Voxel arrays are not permuted,",
        "resampled, or intensity-normalized. The same affine is written to each",
        "scan and its matching binary lesion mask. Masks preserve voxel counts.",
        "",
        f"All rows remain marked `needs_visual_qc_{target.lower()}_orientation` until the",
        "orientation has been inspected in ITK-SNAP or an equivalent viewer.",
        "",
    ]
    path.write_text("\n".join(lines))


def _parse_axcodes(raw: str) -> tuple[str, str, str]:
    axcodes = tuple(raw.strip().upper())
    if len(axcodes) != 3:
        raise ValueError(f"Expected three orientation letters, got {raw!r}")
    return _validate_axcodes(axcodes)


def _validate_axcodes(axcodes: tuple[str, ...]) -> tuple[str, str, str]:
    if len(axcodes) != 3:
        raise ValueError(f"Expected three orientation letters, got {axcodes!r}")
    normalized = tuple(str(code).upper() for code in axcodes)
    used_world_axes = []
    for code in normalized:
        if code not in _AXIS_CODE_TO_WORLD:
            raise ValueError(f"Unsupported orientation code {code!r}")
        used_world_axes.append(_AXIS_CODE_TO_WORLD[code][0])
    if len(set(used_world_axes)) != 3:
        raise ValueError(f"Orientation codes must use each world axis once: {axcodes!r}")
    return normalized  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
