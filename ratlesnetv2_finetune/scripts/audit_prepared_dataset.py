"""Audit prepared scan/mask pairs and create review-first QC artifacts.

This command never edits the prepared dataset. It writes a per-case CSV,
summary JSON, overlay gallery, and a metadata template that must be completed
before subject-disjoint splitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

METADATA_COLUMNS = [
    "case_id",
    "subject_id",
    "cohort",
    "timepoint",
    "acquisition_protocol",
    "old_mask_version",
    "new_mask_version",
    "label_version",
    "reviewer",
    "correction_reason",
    "model_prediction_viewed_during_review",
    "qc_status",
    "exclude",
    "exclude_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared dataset root with manifest.csv")
    parser.add_argument("--output", required=True, help="QC report output directory")
    parser.add_argument(
        "--expected-spacing",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional expected xyz spacing in mm",
    )
    parser.add_argument("--spacing-tolerance", type=float, default=0.02)
    parser.add_argument("--outside-nonzero-warning-fraction", type=float, default=0.01)
    parser.add_argument("--many-components", type=int, default=5)
    parser.add_argument("--max-contact-sheet", type=int, default=36)
    parser.add_argument("--label-version", default="LYS_v1")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = audit_prepared_dataset(
        input_root=Path(args.input),
        output_root=Path(args.output),
        expected_spacing=(
            tuple(float(value) for value in args.expected_spacing)
            if args.expected_spacing is not None
            else None
        ),
        spacing_tolerance=args.spacing_tolerance,
        outside_warning_fraction=args.outside_nonzero_warning_fraction,
        many_components=args.many_components,
        max_contact_sheet=args.max_contact_sheet,
        label_version=args.label_version,
        write_overlays=not args.no_overlays,
        overwrite=args.overwrite,
    )
    flagged = sum(row["qc_flag"] != "pass" for row in rows)
    print(f"Audited cases: {len(rows)}")
    print(f"Flagged for review: {flagged}")
    print(f"QC report: {Path(args.output) / 'dataset_qc.csv'}")
    print(f"Metadata template: {Path(args.output) / 'metadata_template.csv'}")
    return 0


def audit_prepared_dataset(
    *,
    input_root: Path,
    output_root: Path,
    expected_spacing: tuple[float, float, float] | None = None,
    spacing_tolerance: float = 0.02,
    outside_warning_fraction: float = 0.01,
    many_components: int = 5,
    max_contact_sheet: int = 36,
    label_version: str = "LYS_v1",
    write_overlays: bool = True,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Prepared dataset root not found: {input_root}")
    if spacing_tolerance < 0:
        raise ValueError("spacing_tolerance must be >= 0")
    if not 0 <= outside_warning_fraction <= 1:
        raise ValueError("outside_warning_fraction must be between 0 and 1")
    _prepare_output(output_root, overwrite=overwrite)

    manifest = _read_csv(input_root / "manifest.csv")
    overlay_root = output_root / "overlays"
    if write_overlays:
        overlay_root.mkdir()

    qc_rows: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(manifest, start=1):
        case_id = _required(manifest_row, "case_id")
        case_dir = _case_dir(input_root, manifest_row)
        scan_path = _first_existing(case_dir, ("scan.nii.gz", "scan.nii"))
        mask_path = _first_existing(
            case_dir,
            ("scan_lesionIAM.nii.gz", "scan_lesionIAM.nii", "scan_lesion.nii.gz"),
        )
        if scan_path is None or mask_path is None:
            missing = "scan" if scan_path is None else "mask"
            raise FileNotFoundError(f"Missing {missing} for {case_id} under {case_dir}")
        overlay_path = overlay_root / f"{_safe_name(case_id)}.png" if write_overlays else None
        row = _audit_case(
            case_id=case_id,
            scan_path=scan_path,
            mask_path=mask_path,
            expected_spacing=expected_spacing,
            spacing_tolerance=spacing_tolerance,
            outside_warning_fraction=outside_warning_fraction,
            many_components=many_components,
            overlay_path=overlay_path,
        )
        row["manifest_index"] = index
        qc_rows.append(row)

    _add_dataset_level_flags(qc_rows)
    _write_csv(output_root / "dataset_qc.csv", qc_rows)
    _write_metadata_template(
        output_root / "metadata_template.csv",
        manifest,
        label_version=label_version,
    )
    _write_summary(output_root / "qc_summary.json", qc_rows, input_root=input_root)
    if write_overlays:
        _write_gallery(output_root / "overlay_gallery.html", qc_rows)
        _write_contact_sheet(
            output_root / "flagged_contact_sheet.png",
            qc_rows,
            limit=max_contact_sheet,
        )
    return qc_rows


def _audit_case(
    *,
    case_id: str,
    scan_path: Path,
    mask_path: Path,
    expected_spacing: tuple[float, float, float] | None,
    spacing_tolerance: float,
    outside_warning_fraction: float,
    many_components: int,
    overlay_path: Path | None,
) -> dict[str, Any]:
    scan_img = nib.load(str(scan_path))
    mask_img = nib.load(str(mask_path))
    scan = _scan_array(scan_img)
    label_raw = np.asanyarray(mask_img.dataobj)
    label = np.squeeze(label_raw)
    flags: list[str] = []

    if scan.shape != label.shape:
        flags.append("shape_mismatch")
    affine_match = bool(np.allclose(scan_img.affine, mask_img.affine, rtol=0, atol=1e-4))
    if not affine_match:
        flags.append("affine_mismatch")
    spacing = tuple(float(value) for value in scan_img.header.get_zooms()[:3])
    mask_spacing = tuple(float(value) for value in mask_img.header.get_zooms()[:3])
    spacing_match = all(
        abs(left - right) <= 1e-5
        for left, right in zip(spacing, mask_spacing, strict=True)
    )
    if not spacing_match:
        flags.append("scan_mask_spacing_mismatch")
    if expected_spacing is not None and any(
        abs(got - expected) > spacing_tolerance
        for got, expected in zip(spacing, expected_spacing, strict=True)
    ):
        flags.append("unexpected_spacing")

    finite_label = label[np.isfinite(label)]
    values = np.unique(finite_label) if finite_label.size else np.asarray([])
    is_binary = bool(values.size and set(values.tolist()).issubset({0, 1}))
    if not is_binary:
        flags.append("non_binary_label")
    mask = np.isfinite(label) & (label > 0.5)
    lesion_voxels = int(mask.sum())
    if lesion_voxels == 0:
        flags.append("empty_label")

    row: dict[str, Any] = {
        "case_id": case_id,
        "scan_path": str(scan_path),
        "mask_path": str(mask_path),
        "scan_shape": "x".join(str(value) for value in scan.shape),
        "mask_shape": "x".join(str(value) for value in label.shape),
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "scan_axcodes": "".join(nib.aff2axcodes(scan_img.affine)),
        "mask_axcodes": "".join(nib.aff2axcodes(mask_img.affine)),
        "affine_match": affine_match,
        "spacing_match": spacing_match,
        "label_values": ";".join(f"{float(value):g}" for value in values[:16]),
        "lesion_voxels": lesion_voxels,
        "lesion_volume_mm3": float(lesion_voxels * np.prod(spacing)),
        "scan_hash": _array_hash(scan),
        "label_hash": _array_hash(mask.astype(np.uint8)),
        "overlay": str(overlay_path) if overlay_path is not None else "",
    }

    if scan.shape == label.shape:
        _add_spatial_metrics(
            row,
            scan=scan,
            mask=mask,
            affine=scan_img.affine,
            spacing=spacing,
            flags=flags,
            outside_warning_fraction=outside_warning_fraction,
            many_components=many_components,
        )
        if overlay_path is not None:
            _write_case_overlay(overlay_path, scan, mask)
    row["qc_flag"] = ";".join(sorted(set(flags))) if flags else "pass"
    return row


def _add_spatial_metrics(
    row: dict[str, Any],
    *,
    scan: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    flags: list[str],
    outside_warning_fraction: float,
    many_components: int,
) -> None:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    components, component_count = ndimage.label(mask, structure=structure)
    component_sizes = np.bincount(components.ravel())[1:]
    largest_component = int(component_sizes.max()) if component_sizes.size else 0
    small_component_voxels = int(component_sizes[component_sizes < 10].sum())
    nonzero = np.isfinite(scan) & (np.abs(scan) > 0)
    outside_voxels = int(np.logical_and(mask, ~nonzero).sum())
    outside_fraction = outside_voxels / max(int(mask.sum()), 1)
    border_touch = _touches_border(mask)
    slice_axis = int(np.argmin(mask.shape))
    occupied_slices = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != slice_axis)))

    if component_count > many_components:
        flags.append("many_components")
    if mask.any() and largest_component / int(mask.sum()) < 0.9:
        flags.append("fragmented_label")
    if outside_fraction > outside_warning_fraction:
        flags.append("mask_outside_nonzero_image")
    if border_touch:
        flags.append("mask_touches_image_border")

    row.update(
        {
            "component_count": int(component_count),
            "largest_component_voxels": largest_component,
            "small_component_voxels_lt10": small_component_voxels,
            "outside_nonzero_voxels": outside_voxels,
            "outside_nonzero_fraction": outside_fraction,
            "touches_image_border": border_touch,
            "slice_axis": slice_axis,
            "first_lesion_slice": int(occupied_slices[0]) if occupied_slices.size else "",
            "last_lesion_slice": int(occupied_slices[-1]) if occupied_slices.size else "",
        }
    )
    if mask.any():
        centroid_voxel = np.argwhere(mask).mean(axis=0)
        centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)
        for axis, value in zip("xyz", centroid_voxel, strict=True):
            row[f"centroid_{axis}_voxel"] = float(value)
        for axis, value in zip("xyz", centroid_mm, strict=True):
            row[f"centroid_{axis}_mm"] = float(value)
    else:
        for axis in "xyz":
            row[f"centroid_{axis}_voxel"] = ""
            row[f"centroid_{axis}_mm"] = ""
    row["voxel_volume_mm3"] = float(np.prod(spacing))


def _add_dataset_level_flags(rows: list[dict[str, Any]]) -> None:
    positive_volumes = np.asarray(
        [float(row["lesion_volume_mm3"]) for row in rows if row["lesion_voxels"] > 0],
        dtype=float,
    )
    low = high = None
    if positive_volumes.size >= 8:
        q1, q3 = np.quantile(positive_volumes, [0.25, 0.75])
        spread = q3 - q1
        low = max(0.0, float(q1 - 3 * spread))
        high = float(q3 + 3 * spread)

    by_scan_hash: dict[str, list[str]] = defaultdict(list)
    by_label_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_scan_hash[str(row["scan_hash"])].append(str(row["case_id"]))
        by_label_hash[str(row["label_hash"])].append(str(row["case_id"]))

    for row in rows:
        flags = [] if row["qc_flag"] == "pass" else str(row["qc_flag"]).split(";")
        volume = float(row["lesion_volume_mm3"])
        if low is not None and row["lesion_voxels"] > 0 and volume < low:
            flags.append("volume_low_outlier")
        if high is not None and volume > high:
            flags.append("volume_high_outlier")
        scan_duplicates = by_scan_hash[str(row["scan_hash"])]
        label_duplicates = by_label_hash[str(row["label_hash"])]
        if len(scan_duplicates) > 1:
            flags.append("duplicate_scan_content")
        if len(label_duplicates) > 1:
            flags.append("duplicate_label_content")
        row["duplicate_scan_cases"] = ";".join(scan_duplicates) if len(scan_duplicates) > 1 else ""
        row["duplicate_label_cases"] = (
            ";".join(label_duplicates) if len(label_duplicates) > 1 else ""
        )
        row["qc_flag"] = ";".join(sorted(set(flags))) if flags else "pass"


def _write_case_overlay(path: Path, scan: np.ndarray, mask: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    axis = int(np.argmin(mask.shape))
    indices = _overlay_indices(mask, axis=axis)
    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
    axes_array = np.atleast_1d(axes)
    for ax, index in zip(axes_array, indices, strict=True):
        image = _normalize(np.take(scan, index, axis=axis))
        lesion = np.take(mask, index, axis=axis)
        ax.imshow(np.rot90(image), cmap="gray", vmin=0, vmax=1)
        if lesion.any():
            ax.contour(np.rot90(lesion.astype(float)), levels=[0.5], colors=["red"], linewidths=1)
        ax.set_title(f"axis {axis}, slice {index}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _overlay_indices(mask: np.ndarray, *, axis: int) -> list[int]:
    other_axes = tuple(index for index in range(mask.ndim) if index != axis)
    counts = mask.sum(axis=other_axes)
    occupied = np.flatnonzero(counts)
    if occupied.size:
        candidates = [int(occupied[0]), int(np.argmax(counts)), int(occupied[-1])]
    else:
        candidates = [0, mask.shape[axis] // 2, mask.shape[axis] - 1]
    return list(dict.fromkeys(candidates))


def _write_contact_sheet(path: Path, rows: list[dict[str, Any]], *, limit: int) -> None:
    flagged = [row for row in rows if row["qc_flag"] != "pass" and Path(row["overlay"]).is_file()]
    flagged = flagged[:limit]
    if not flagged:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    columns = min(4, len(flagged))
    grid_rows = math.ceil(len(flagged) / columns)
    fig, axes = plt.subplots(grid_rows, columns, figsize=(columns * 4, grid_rows * 3.5))
    axes_array = np.asarray(axes).reshape(-1)
    for ax, row in zip(axes_array, flagged, strict=False):
        ax.imshow(plt.imread(row["overlay"]))
        ax.set_title(f"{row['case_id']}\n{row['qc_flag']}", fontsize=7)
        ax.axis("off")
    for ax in axes_array[len(flagged) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_gallery(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in sorted(rows, key=lambda item: (item["qc_flag"] == "pass", item["case_id"])):
        overlay = Path(str(row["overlay"]))
        relative = overlay.relative_to(path.parent) if overlay.is_file() else None
        image = f'<img src="{html.escape(str(relative))}">' if relative is not None else ""
        cards.append(
            "<article>"
            f"<h3>{html.escape(str(row['case_id']))}</h3>"
            f"<p>{html.escape(str(row['qc_flag']))}</p>{image}</article>"
        )
    document = """<!doctype html><html><head><meta charset="utf-8">
<style>body{font-family:Arial;margin:20px}.grid{display:grid;grid-template-columns:
repeat(auto-fit,minmax(320px,1fr));gap:12px}article{border:1px solid #ccc;padding:8px}
img{max-width:100%}h3{font-size:12px;overflow-wrap:anywhere}p{font-size:11px}</style>
</head><body><h1>Prepared dataset QC overlays</h1><div class="grid">"""
    path.write_text(document + "".join(cards) + "</div></body></html>\n")


def _write_metadata_template(
    path: Path,
    manifest: list[dict[str, str]],
    *,
    label_version: str,
) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for row in manifest:
            writer.writerow(
                {
                    "case_id": _required(row, "case_id"),
                    "subject_id": "TODO",
                    "cohort": "TODO",
                    "timepoint": "TODO",
                    "acquisition_protocol": "TODO",
                    "old_mask_version": "LYS_v0",
                    "new_mask_version": label_version,
                    "label_version": label_version,
                    "reviewer": "TODO",
                    "correction_reason": "TODO",
                    "model_prediction_viewed_during_review": "TODO",
                    "qc_status": "needs_review",
                    "exclude": "false",
                    "exclude_reason": "",
                }
            )


def _write_summary(path: Path, rows: list[dict[str, Any]], *, input_root: Path) -> None:
    flag_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["qc_flag"] == "pass":
            flag_counts["pass"] += 1
        else:
            for flag in str(row["qc_flag"]).split(";"):
                flag_counts[flag] += 1
    payload = {
        "input_root": str(input_root),
        "n_cases": len(rows),
        "n_flagged_cases": sum(row["qc_flag"] != "pass" for row in rows),
        "flag_counts": dict(sorted(flag_counts.items())),
        "review_required": True,
        "note": "Automatic flags are review priorities, not automatic exclusion decisions.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    return rows


def _case_dir(root: Path, row: dict[str, str]) -> Path:
    canonical = (
        root
        / _required(row, "split")
        / _required(row, "study")
        / _required(row, "timepoint")
        / _required(row, "case_id")
    )
    if canonical.is_dir():
        return canonical
    recorded = Path(str(row.get("case_dir", "")))
    if recorded.is_dir():
        return recorded
    raise FileNotFoundError(f"Prepared case directory not found for {_required(row, 'case_id')}")


def _scan_array(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected one 3-D scan channel, got shape {data.shape}")
    return np.asarray(data)


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        or mask[-1].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
        or mask[:, :, 0].any()
        or mask[:, :, -1].any()
    )


def _normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros_like(values)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"QC output already exists: {path}; pass --overwrite to replace it"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    return next((root / name for name in names if (root / name).is_file()), None)


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value in {None, "", "TODO"}:
        raise ValueError(f"Manifest row is missing required value: {key}")
    return str(value)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
