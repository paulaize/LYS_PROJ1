"""Validated adapters between RatLesNetV2 outputs and subject-space atlas labels."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

PREPARED_MANIFEST_COLUMNS = (
    "case_id",
    "native_t2",
    "lesion_mask",
    "orientation_review_json",
    "native_atlas_labels",
    "structures_csv",
    "atlas_id",
    "atlas_version",
    "coordinate_space",
    "annotation_variant",
    "registration_backend",
    "registration_version",
    "registration_revision",
    "label_interpolation",
    "registration_qc_json",
    "lesion_mask_status",
    "lesion_mask_review_json",
    "lesion_mask_reviewer",
)

REQUIRED_SUMMARY_FIELDS = (
    "case_id",
    "native_t2",
    "lesion_mask",
    "native_atlas_labels",
    "structures_csv",
    "atlas_id",
    "atlas_version",
    "coordinate_space",
    "annotation_variant",
    "registration_backend",
    "registration_version",
    "registration_revision",
    "label_interpolation",
    "lesion_mask_status",
)

ALLOWED_MASK_STATUSES = {"model_draft", "human_reviewed"}
ALLOWED_REVIEW_STATUSES = {"pending", "approved", "rejected"}


@dataclass(frozen=True)
class SpatialVolume:
    """Three-dimensional data plus its native NIfTI geometry."""

    path: Path
    image: nib.spatialimages.SpatialImage
    data: np.ndarray
    spacing_mm: tuple[float, float, float]
    orientation: str


def prepare_aidamri_inputs(
    *,
    inference_manifest: Path,
    output_root: Path,
    expected_header_orientation: str = "LIP",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stage validated 3-D T2w scans and draft masks for AIDAmri v3.

    This function verifies only the orientation encoded in the NIfTI affine. It
    deliberately emits a pending review record because anatomical orientation
    cannot be established from the header alone.
    """
    inference_manifest = inference_manifest.resolve()
    output_root = output_root.resolve()
    rows = _read_csv(inference_manifest)
    required = {"case_id", "input_scan", "ensemble_mask"}
    _require_columns(rows, required, inference_manifest)
    _require_unique_case_ids(rows, inference_manifest)

    expected_header_orientation = expected_header_orientation.upper()
    if len(expected_header_orientation) != 3:
        raise ValueError("Expected header orientation must be a valid three-axis code")
    try:
        expected_ornt = nib.orientations.axcodes2ornt(tuple(expected_header_orientation))
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Expected header orientation must be a valid three-axis code"
        ) from error
    if np.isnan(expected_ornt).any():
        raise ValueError("Expected header orientation must be a valid three-axis code")

    validated: list[tuple[dict[str, str], SpatialVolume, SpatialVolume]] = []
    for row in rows:
        case_id = _validate_case_id(row["case_id"])
        t2_path = _resolve_path(row["input_scan"], inference_manifest.parent)
        lesion_path = _resolve_path(row["ensemble_mask"], inference_manifest.parent)
        t2 = load_t2(t2_path)
        lesion = load_binary_mask(lesion_path)
        require_same_grid(t2, lesion, names=("T2w scan", "lesion mask"))
        if t2.orientation != expected_header_orientation:
            raise ValueError(
                f"Case {case_id!r}: header orientation is {t2.orientation}, but AIDAmri "
                f"expects {expected_header_orientation}. Do not relabel the header. Use "
                "AIDAmri's reviewed reorientation workflow and apply the identical voxel "
                "transform to the lesion mask."
            )
        validated.append(({**row, "case_id": case_id}, t2, lesion))

    _prepare_output_root(output_root, overwrite=overwrite)
    prepared_rows: list[dict[str, Any]] = []
    orientation_records = []
    for source_row, t2, lesion in validated:
        case_id = source_row["case_id"]
        case_dir = output_root / "cases" / case_id
        case_dir.mkdir(parents=True)
        staged_t2 = case_dir / "t2_for_aidamri.nii.gz"
        staged_lesion = case_dir / "lesion_model_draft_native.nii.gz"
        save_like(staged_t2, t2.data.astype(np.float32), t2.image)
        save_like(staged_lesion, lesion.data.astype(np.uint8), lesion.image)

        orientation_review = case_dir / "orientation_review.json"
        orientation_record = {
            "schema_version": 1,
            "case_id": case_id,
            "orientation_status": "pending",
            "observed_header_orientation": t2.orientation,
            "required_aidamri_orientation": expected_header_orientation,
            "anatomical_orientation_matches_header": None,
            "reviewer": None,
            "review_date": None,
            "notes": "",
            "native_t2_sha256": sha256(staged_t2),
            "lesion_mask_sha256": sha256(staged_lesion),
        }
        orientation_review.write_text(
            json.dumps(orientation_record, indent=2, sort_keys=True) + "\n"
        )
        orientation_records.append(orientation_record)

        prepared_rows.append(
            {
                "case_id": case_id,
                "native_t2": _portable_path(staged_t2, output_root),
                "lesion_mask": _portable_path(staged_lesion, output_root),
                "orientation_review_json": _portable_path(orientation_review, output_root),
                "native_atlas_labels": "",
                "structures_csv": "",
                "atlas_id": "",
                "atlas_version": "",
                "coordinate_space": "",
                "annotation_variant": "",
                "registration_backend": "AIDAmri",
                "registration_version": "",
                "registration_revision": "",
                "label_interpolation": "nearest_neighbor",
                "registration_qc_json": "",
                "lesion_mask_status": "model_draft",
                "lesion_mask_review_json": "",
                "lesion_mask_reviewer": "",
            }
        )

    prepared_manifest = output_root / "atlas_mapping_manifest.csv"
    _write_csv(prepared_manifest, prepared_rows, fieldnames=PREPARED_MANIFEST_COLUMNS)
    summary = {
        "schema_version": 1,
        "n_cases": len(prepared_rows),
        "source_inference_manifest": str(inference_manifest),
        "source_inference_manifest_sha256": sha256(inference_manifest),
        "expected_aidamri_header_orientation": expected_header_orientation,
        "orientation_review_status": "pending",
        "predictions_are_drafts": True,
        "prepared_manifest": str(prepared_manifest),
        "cases": [
            {
                "case_id": record["case_id"],
                "observed_header_orientation": record["observed_header_orientation"],
                "orientation_review_json": prepared_rows[index]["orientation_review_json"],
            }
            for index, record in enumerate(orientation_records)
        ],
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def summarize_atlas_mappings(
    *,
    manifest: Path,
    output_root: Path,
    qc_slice_axis: int = 2,
    require_approved_registration: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Summarize lesion overlap with explicit subject-space atlas annotations."""
    manifest = manifest.resolve()
    output_root = output_root.resolve()
    rows = _read_csv(manifest)
    _require_columns(rows, set(REQUIRED_SUMMARY_FIELDS), manifest)
    _require_unique_case_ids(rows, manifest)
    if qc_slice_axis not in (0, 1, 2):
        raise ValueError("QC slice axis must be 0, 1, or 2")

    loaded = [_load_mapping_case(row, manifest) for row in rows]
    _require_homogeneous_mapping_metadata(loaded, manifest)
    if require_approved_registration:
        problems = [
            case["case_id"]
            for case in loaded
            if case["combined_qc_status"] != "approved"
        ]
        if problems:
            raise ValueError(
                "Approved orientation and registration reviews are required for: "
                + ", ".join(problems)
            )

    _prepare_output_root(output_root, overwrite=overwrite)
    pooled_affected: list[dict[str, Any]] = []
    case_manifest_rows: list[dict[str, Any]] = []
    for case in loaded:
        case_id = case["case_id"]
        case_dir = output_root / "cases" / case_id
        case_dir.mkdir(parents=True)
        metrics = region_overlap_metrics(
            case_id=case_id,
            labels=case["labels"].data,
            lesion=case["lesion"].data,
            structures=case["structures"],
            voxel_volume_mm3=float(np.prod(case["t2"].spacing_mm)),
            metadata={
                **case["metadata"],
                "lesion_mask_status": case["lesion_mask_status"],
                "lesion_mask_reviewer": case["lesion_mask_reviewer"],
            },
            combined_qc_status=case["combined_qc_status"],
        )
        metrics_path = case_dir / "region_metrics.csv"
        affected_path = case_dir / "affected_regions.csv"
        _write_csv(metrics_path, metrics)
        affected = [row for row in metrics if int(row["lesion_overlap_voxels"]) > 0]
        if affected:
            _write_csv(affected_path, affected)
        else:
            _write_empty_csv(affected_path, list(metrics[0]))
        pooled_affected.extend(affected)

        qc_path = case_dir / "atlas_mapping_qc.png"
        write_qc_overlay(
            t2=case["t2"].data,
            labels=case["labels"].data,
            lesion=case["lesion"].data,
            output_path=qc_path,
            slice_axis=qc_slice_axis,
            title=(
                f"{case_id}: atlas boundaries and {case['lesion_mask_status']} lesion "
                "(registration review required)"
            ),
        )

        labels_hash = sha256(case["labels"].path)
        review_template_path = case_dir / "registration_review_template.json"
        review_template = {
            "schema_version": 1,
            "case_id": case_id,
            "registration_qc_status": "pending",
            "atlas_id": case["metadata"]["atlas_id"],
            "atlas_version": case["metadata"]["atlas_version"],
            "coordinate_space": case["metadata"]["coordinate_space"],
            "annotation_variant": case["metadata"]["annotation_variant"],
            "registration_backend": case["metadata"]["registration_backend"],
            "registration_version": case["metadata"]["registration_version"],
            "registration_revision": case["metadata"]["registration_revision"],
            "native_atlas_labels_sha256": labels_hash,
            "reviewer": None,
            "review_date": None,
            "notes": "",
        }
        review_template_path.write_text(
            json.dumps(review_template, indent=2, sort_keys=True) + "\n"
        )

        lesion_voxels = int(np.count_nonzero(case["lesion"].data))
        mapped_lesion_voxels = int(
            np.count_nonzero((case["lesion"].data != 0) & (case["labels"].data != 0))
        )
        coverage = mapped_lesion_voxels / lesion_voxels if lesion_voxels else None
        case_summary = {
            "schema_version": 1,
            "case_id": case_id,
            **case["metadata"],
            "orientation_review_status": case["orientation_review_status"],
            "registration_qc_status": case["registration_qc_status"],
            "combined_qc_status": case["combined_qc_status"],
            "lesion_mask_status": case["lesion_mask_status"],
            "lesion_mask_reviewer": case["lesion_mask_reviewer"],
            "predictions_are_drafts": case["lesion_mask_status"] == "model_draft",
            "native_t2": str(case["t2"].path),
            "native_t2_sha256": sha256(case["t2"].path),
            "lesion_mask": str(case["lesion"].path),
            "lesion_mask_sha256": sha256(case["lesion"].path),
            "native_atlas_labels": str(case["labels"].path),
            "native_atlas_labels_sha256": labels_hash,
            "structures_csv": str(case["structures_path"]),
            "structures_csv_sha256": sha256(case["structures_path"]),
            "native_shape": list(case["t2"].data.shape),
            "native_spacing_mm": list(case["t2"].spacing_mm),
            "native_header_orientation": case["t2"].orientation,
            "lesion_voxels": lesion_voxels,
            "lesion_volume_mm3": lesion_voxels * float(np.prod(case["t2"].spacing_mm)),
            "atlas_mapped_lesion_voxels": mapped_lesion_voxels,
            "atlas_coverage_fraction_of_lesion": coverage,
            "hemisphere_metadata_available": case["hemisphere_available"],
            "region_metrics": str(metrics_path),
            "affected_regions": str(affected_path),
            "qc_overlay": str(qc_path),
            "registration_review_template": str(review_template_path),
        }
        case_summary_path = case_dir / "mapping_summary.json"
        case_summary_path.write_text(json.dumps(case_summary, indent=2, sort_keys=True) + "\n")
        case_manifest_rows.append(
            {
                "case_id": case_id,
                "atlas_id": case["metadata"]["atlas_id"],
                "atlas_version": case["metadata"]["atlas_version"],
                "coordinate_space": case["metadata"]["coordinate_space"],
                "annotation_variant": case["metadata"]["annotation_variant"],
                "combined_qc_status": case["combined_qc_status"],
                "lesion_mask_status": case["lesion_mask_status"],
                "atlas_coverage_fraction_of_lesion": "" if coverage is None else coverage,
                "mapping_summary": str(case_summary_path),
            }
        )

    cases_manifest = output_root / "atlas_mapping_cases.csv"
    pooled_path = output_root / "affected_regions_all_cases.csv"
    _write_csv(cases_manifest, case_manifest_rows)
    if pooled_affected:
        _write_csv(pooled_path, pooled_affected)
    else:
        _write_empty_csv(pooled_path, list(metrics[0]))
    summary = {
        "schema_version": 1,
        "n_cases": len(case_manifest_rows),
        "atlas_id": loaded[0]["metadata"]["atlas_id"],
        "atlas_version": loaded[0]["metadata"]["atlas_version"],
        "coordinate_space": loaded[0]["metadata"]["coordinate_space"],
        "annotation_variant": loaded[0]["metadata"]["annotation_variant"],
        "registration_backend": loaded[0]["metadata"]["registration_backend"],
        "registration_version": loaded[0]["metadata"]["registration_version"],
        "registration_revision": loaded[0]["metadata"]["registration_revision"],
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "require_approved_registration": require_approved_registration,
        "n_approved": sum(
            row["combined_qc_status"] == "approved" for row in case_manifest_rows
        ),
        "n_needing_review": sum(
            row["combined_qc_status"] != "approved" for row in case_manifest_rows
        ),
        "n_rejected": sum(
            row["combined_qc_status"] == "rejected" for row in case_manifest_rows
        ),
        "predictions_remain_drafts": any(
            row["lesion_mask_status"] == "model_draft" for row in case_manifest_rows
        ),
        "case_manifest": str(cases_manifest),
        "affected_regions_all_cases": str(pooled_path),
    }
    (output_root / "atlas_mapping_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def load_t2(path: Path) -> SpatialVolume:
    """Load a finite non-constant 3-D T2 image or a singleton-channel 4-D image."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"T2w NIfTI not found: {path}")
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected 3-D or X x Y x Z x 1 T2w image, got {image.shape}")
    _validate_spatial_image(path, image, data)
    finite = data[np.isfinite(data)]
    if finite.size != data.size:
        raise ValueError(f"{path}: T2w image contains non-finite voxels")
    if not finite.size or float(finite.std()) == 0.0:
        raise ValueError(f"{path}: T2w image has zero intensity variance")
    return _spatial_volume(path, image, np.asarray(data))


def load_binary_mask(path: Path) -> SpatialVolume:
    """Load a finite binary 3-D lesion mask."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Lesion mask not found: {path}")
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D binary lesion mask, got {image.shape}")
    _validate_spatial_image(path, image, data)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: lesion mask contains non-finite voxels")
    unique = np.unique(data)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{path}: lesion mask must be binary 0/1, found {unique.tolist()}")
    return _spatial_volume(path, image, np.asarray(data, dtype=np.uint8))


def load_label_image(path: Path) -> SpatialVolume:
    """Load a non-negative integer atlas label image in a native T2 grid."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Native-space atlas labels not found: {path}")
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D atlas label image, got {image.shape}")
    _validate_spatial_image(path, image, data)
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: atlas labels contain non-finite voxels")
    rounded = np.rint(data)
    if not np.allclose(data, rounded, rtol=0, atol=1e-6):
        raise ValueError(
            f"{path}: atlas labels are not integer-valued; nearest-neighbour label "
            "interpolation is required"
        )
    if np.any(rounded < 0):
        raise ValueError(f"{path}: atlas labels must be non-negative")
    return _spatial_volume(path, image, rounded.astype(np.int64))


def require_same_grid(
    first: SpatialVolume,
    second: SpatialVolume,
    *,
    names: tuple[str, str],
) -> None:
    """Require exact array shape and matching native-world geometry."""
    if first.data.shape != second.data.shape:
        raise ValueError(
            f"{names[0]} shape {first.data.shape} does not match "
            f"{names[1]} shape {second.data.shape}"
        )
    if not np.allclose(first.image.affine, second.image.affine, rtol=0, atol=1e-5):
        raise ValueError(f"{names[0]} and {names[1]} affines do not match")
    if not np.allclose(first.spacing_mm, second.spacing_mm, rtol=0, atol=1e-6):
        raise ValueError(f"{names[0]} and {names[1]} voxel spacings do not match")


def load_structures(path: Path, observed_label_ids: set[int]) -> tuple[dict[int, dict], bool]:
    """Load a normalized label lookup and require all observed labels to be defined."""
    path = path.resolve()
    rows = _read_csv(path)
    _require_columns(rows, {"label_id", "acronym", "name"}, path)
    structures: dict[int, dict] = {}
    for line_number, row in enumerate(rows, start=2):
        try:
            label_id = int(row["label_id"])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: label_id must be an integer") from error
        if label_id < 0:
            raise ValueError(f"{path}:{line_number}: label_id must be non-negative")
        if label_id in structures:
            raise ValueError(f"{path}: duplicate label_id {label_id}")
        acronym = row["acronym"].strip()
        name = row["name"].strip()
        if not acronym or not name:
            raise ValueError(f"{path}:{line_number}: acronym and name are required")
        structures[label_id] = {
            "label_id": label_id,
            "acronym": acronym,
            "name": name,
            "hemisphere": row.get("hemisphere", "").strip(),
        }
    structures.setdefault(
        0,
        {
            "label_id": 0,
            "acronym": "unmapped",
            "name": "outside atlas or unmapped",
            "hemisphere": "",
        },
    )
    unknown = sorted(observed_label_ids - set(structures))
    if unknown:
        raise ValueError(f"{path}: missing lookup rows for observed label IDs {unknown}")
    nonzero_ids = sorted(label_id for label_id in observed_label_ids if label_id != 0)
    hemisphere_available = bool(nonzero_ids) and all(
        structures[label_id]["hemisphere"] for label_id in nonzero_ids
    )
    return structures, hemisphere_available


def region_overlap_metrics(
    *,
    case_id: str,
    labels: np.ndarray,
    lesion: np.ndarray,
    structures: dict[int, dict],
    voxel_volume_mm3: float,
    metadata: dict[str, str],
    combined_qc_status: str,
) -> list[dict[str, Any]]:
    """Compute native-grid atlas and lesion overlap volumes for every observed label."""
    if labels.shape != lesion.shape:
        raise ValueError("Atlas labels and lesion mask must have the same shape")
    total_lesion = int(np.count_nonzero(lesion))
    label_ids, label_counts = np.unique(labels, return_counts=True)
    rows = []
    for label_id_value, region_voxels_value in zip(label_ids, label_counts, strict=True):
        label_id = int(label_id_value)
        region_voxels = int(region_voxels_value)
        if label_id not in structures:
            raise ValueError(f"No structure metadata for atlas label {label_id}")
        overlap = int(np.count_nonzero((labels == label_id) & (lesion != 0)))
        info = structures[label_id]
        rows.append(
            {
                "case_id": case_id,
                **metadata,
                "combined_qc_status": combined_qc_status,
                "label_id": label_id,
                "acronym": info["acronym"],
                "region_name": info["name"],
                "hemisphere": info["hemisphere"],
                "atlas_region_voxels_native_grid": region_voxels,
                "atlas_region_volume_native_grid_mm3": region_voxels * voxel_volume_mm3,
                "lesion_overlap_voxels": overlap,
                "lesion_overlap_volume_mm3": overlap * voxel_volume_mm3,
                "fraction_of_region_native_grid": overlap / region_voxels,
                "fraction_of_total_lesion": overlap / total_lesion if total_lesion else 0.0,
            }
        )
    rows.sort(key=lambda row: (-int(row["lesion_overlap_voxels"]), int(row["label_id"])))
    return rows


def write_qc_overlay(
    *,
    t2: np.ndarray,
    labels: np.ndarray,
    lesion: np.ndarray,
    output_path: Path,
    slice_axis: int,
    title: str,
) -> None:
    """Write evenly sampled native-slice overlays for human registration review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage

    n_slices = t2.shape[slice_axis]
    indices = np.unique(
        np.linspace(0, n_slices - 1, min(6, n_slices), dtype=int)
    ).tolist()
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
    flat_axes = axes.ravel()
    finite = t2[np.isfinite(t2)]
    vmin, vmax = np.percentile(finite, (1, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0
    for axis, index in zip(flat_axes, indices, strict=False):
        t2_slice = np.take(t2, index, axis=slice_axis)
        label_slice = np.take(labels, index, axis=slice_axis)
        lesion_slice = np.take(lesion, index, axis=slice_axis).astype(bool)
        boundary = ndimage.maximum_filter(label_slice, size=3) != ndimage.minimum_filter(
            label_slice, size=3
        )
        boundary &= label_slice != 0
        axis.imshow(t2_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axis.imshow(
            np.ma.masked_where(~boundary.T, boundary.T),
            cmap="autumn",
            origin="lower",
            alpha=0.9,
        )
        axis.imshow(
            np.ma.masked_where(~lesion_slice.T, lesion_slice.T),
            cmap="cool",
            origin="lower",
            alpha=0.35,
        )
        axis.set_title(f"voxel slice {index}, axis {slice_axis}")
        axis.axis("off")
    for axis in flat_axes[len(indices) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def save_like(
    path: Path,
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
) -> None:
    """Save a 3-D array with the first three dimensions of a NIfTI reference."""
    header = reference.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(data.dtype)
    header.set_zooms(tuple(float(value) for value in reference.header.get_zooms()[:3]))
    nib.save(nib.Nifti1Image(data, reference.affine, header), str(path))


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping_case(row: dict[str, str], manifest: Path) -> dict[str, Any]:
    case_id = _validate_case_id(row["case_id"])
    for field in REQUIRED_SUMMARY_FIELDS:
        if row[field].strip() in {"", "TODO", "null"}:
            raise ValueError(f"Case {case_id!r}: manifest field {field!r} is unset")
    if row["label_interpolation"].strip().lower() != "nearest_neighbor":
        raise ValueError(
            f"Case {case_id!r}: label_interpolation must be nearest_neighbor"
        )
    lesion_mask_status = row["lesion_mask_status"].strip()
    if lesion_mask_status not in ALLOWED_MASK_STATUSES:
        raise ValueError(
            f"Case {case_id!r}: lesion_mask_status must be one of "
            f"{sorted(ALLOWED_MASK_STATUSES)}"
        )
    t2 = load_t2(_resolve_path(row["native_t2"], manifest.parent))
    lesion = load_binary_mask(_resolve_path(row["lesion_mask"], manifest.parent))
    labels = load_label_image(_resolve_path(row["native_atlas_labels"], manifest.parent))
    require_same_grid(t2, lesion, names=("T2w scan", "lesion mask"))
    require_same_grid(t2, labels, names=("T2w scan", "native atlas labels"))
    structures_path = _resolve_path(row["structures_csv"], manifest.parent)
    structures, hemisphere_available = load_structures(
        structures_path, set(int(value) for value in np.unique(labels.data))
    )
    lesion_mask_reviewer = _load_lesion_mask_review(
        row.get("lesion_mask_review_json", ""),
        manifest=manifest,
        case_id=case_id,
        lesion_path=lesion.path,
        lesion_mask_status=lesion_mask_status,
        manifest_reviewer=row.get("lesion_mask_reviewer", "").strip(),
    )

    orientation_review_status = _load_orientation_review(
        row.get("orientation_review_json", ""),
        manifest=manifest,
        case_id=case_id,
        t2_path=t2.path,
        lesion_path=lesion.path,
        t2_orientation=t2.orientation,
    )
    metadata = {
        "atlas_id": row["atlas_id"].strip(),
        "atlas_version": row["atlas_version"].strip(),
        "coordinate_space": row["coordinate_space"].strip(),
        "annotation_variant": row["annotation_variant"].strip(),
        "registration_backend": row["registration_backend"].strip(),
        "registration_version": row["registration_version"].strip(),
        "registration_revision": row["registration_revision"].strip(),
        "label_interpolation": "nearest_neighbor",
    }
    registration_qc_status = _load_registration_review(
        row.get("registration_qc_json", ""),
        manifest=manifest,
        case_id=case_id,
        labels_path=labels.path,
        metadata=metadata,
    )
    combined_qc_status = _combined_qc_status(
        orientation_review_status, registration_qc_status
    )
    return {
        "case_id": case_id,
        "t2": t2,
        "lesion": lesion,
        "labels": labels,
        "structures": structures,
        "structures_path": structures_path,
        "hemisphere_available": hemisphere_available,
        "metadata": metadata,
        "lesion_mask_status": lesion_mask_status,
        "lesion_mask_reviewer": lesion_mask_reviewer,
        "orientation_review_status": orientation_review_status,
        "registration_qc_status": registration_qc_status,
        "combined_qc_status": combined_qc_status,
    }


def _load_orientation_review(
    value: str,
    *,
    manifest: Path,
    case_id: str,
    t2_path: Path,
    lesion_path: Path,
    t2_orientation: str,
) -> str:
    if not value.strip():
        return "pending"
    path = _resolve_path(value, manifest.parent)
    record = _read_json(path)
    status = str(record.get("orientation_status", "pending"))
    _validate_review_status(status, path)
    if record.get("case_id") != case_id:
        raise ValueError(f"{path}: orientation review case_id does not match {case_id!r}")
    if record.get("native_t2_sha256") != sha256(t2_path):
        raise ValueError(f"{path}: orientation review T2 hash is stale or incorrect")
    if record.get("lesion_mask_sha256") != sha256(lesion_path):
        raise ValueError(f"{path}: orientation review lesion-mask hash is stale or incorrect")
    if record.get("observed_header_orientation") != t2_orientation:
        raise ValueError(f"{path}: orientation review header orientation is stale or incorrect")
    if status == "approved":
        if record.get("anatomical_orientation_matches_header") is not True:
            raise ValueError(
                f"{path}: approved orientation review must confirm anatomical orientation"
            )
        _require_reviewer_fields(record, path)
    return status


def _load_lesion_mask_review(
    value: str,
    *,
    manifest: Path,
    case_id: str,
    lesion_path: Path,
    lesion_mask_status: str,
    manifest_reviewer: str,
) -> str:
    if lesion_mask_status == "model_draft":
        if value.strip() or manifest_reviewer:
            raise ValueError(
                f"Case {case_id!r}: model_draft masks cannot carry human-review approval"
            )
        return ""
    if not value.strip():
        raise ValueError(
            f"Case {case_id!r}: human_reviewed masks require lesion_mask_review_json"
        )
    path = _resolve_path(value, manifest.parent)
    record = _read_json(path)
    required_matches = {
        "case_id": case_id,
        "lesion_mask_status": "human_reviewed",
        "lesion_mask_sha256": sha256(lesion_path),
    }
    for field, expected in required_matches.items():
        if record.get(field) != expected:
            raise ValueError(f"{path}: lesion-mask review {field} is stale or incorrect")
    if record.get("lesion_mask_qc_status") != "approved":
        raise ValueError(f"{path}: human-reviewed lesion mask is not approved")
    _require_reviewer_fields(record, path)
    reviewer = str(record["reviewer"])
    if manifest_reviewer and manifest_reviewer != reviewer:
        raise ValueError(f"{path}: lesion-mask reviewer disagrees with the manifest")
    return reviewer


def _load_registration_review(
    value: str,
    *,
    manifest: Path,
    case_id: str,
    labels_path: Path,
    metadata: dict[str, str],
) -> str:
    if not value.strip():
        return "pending"
    path = _resolve_path(value, manifest.parent)
    record = _read_json(path)
    status = str(record.get("registration_qc_status", "pending"))
    _validate_review_status(status, path)
    required_matches = {
        "case_id": case_id,
        "atlas_id": metadata["atlas_id"],
        "atlas_version": metadata["atlas_version"],
        "coordinate_space": metadata["coordinate_space"],
        "annotation_variant": metadata["annotation_variant"],
        "registration_backend": metadata["registration_backend"],
        "registration_version": metadata["registration_version"],
        "registration_revision": metadata["registration_revision"],
        "native_atlas_labels_sha256": sha256(labels_path),
    }
    for field, expected in required_matches.items():
        if record.get(field) != expected:
            raise ValueError(f"{path}: registration review {field} is stale or incorrect")
    if status == "approved":
        _require_reviewer_fields(record, path)
    return status


def _combined_qc_status(orientation_status: str, registration_status: str) -> str:
    if "rejected" in (orientation_status, registration_status):
        return "rejected"
    if orientation_status == registration_status == "approved":
        return "approved"
    return "needs_human_review"


def _validate_review_status(status: str, path: Path) -> None:
    if status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"{path}: review status must be one of {sorted(ALLOWED_REVIEW_STATUSES)}")


def _require_reviewer_fields(record: dict[str, Any], path: Path) -> None:
    if not record.get("reviewer") or not record.get("review_date"):
        raise ValueError(f"{path}: approved review requires reviewer and review_date")
    try:
        date.fromisoformat(str(record["review_date"]))
    except ValueError as error:
        raise ValueError(f"{path}: review_date must use YYYY-MM-DD") from error


def _require_homogeneous_mapping_metadata(
    cases: list[dict[str, Any]], path: Path
) -> None:
    fields = tuple(cases[0]["metadata"])
    reference = tuple(cases[0]["metadata"][field] for field in fields)
    inconsistent = [
        case["case_id"]
        for case in cases[1:]
        if tuple(case["metadata"][field] for field in fields) != reference
    ]
    if inconsistent:
        raise ValueError(
            f"{path}: pooled atlas reports require one atlas, annotation variant, and "
            f"registration version; inconsistent cases: {inconsistent}"
        )


def _spatial_volume(
    path: Path,
    image: nib.spatialimages.SpatialImage,
    data: np.ndarray,
) -> SpatialVolume:
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    orientation = "".join(str(value) for value in nib.aff2axcodes(image.affine))
    return SpatialVolume(path, image, data, spacing, orientation)


def _validate_spatial_image(
    path: Path,
    image: nib.spatialimages.SpatialImage,
    data: np.ndarray,
) -> None:
    if data.ndim != 3:
        raise ValueError(f"{path}: expected three spatial dimensions")
    affine = image.affine
    if not np.isfinite(affine).all() or np.isclose(np.linalg.det(affine[:3, :3]), 0.0):
        raise ValueError(f"{path}: invalid or singular NIfTI affine")
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"{path}: invalid NIfTI voxel spacing {spacing.tolist()}")
    if any(code is None for code in nib.aff2axcodes(affine)):
        raise ValueError(f"{path}: NIfTI orientation cannot be determined from the affine")


def _prepare_output_root(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return record


def _require_columns(rows: list[dict[str, str]], required: set[str], path: Path) -> None:
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")


def _require_unique_case_ids(rows: list[dict[str, str]], path: Path) -> None:
    case_ids = [_validate_case_id(row.get("case_id", "")) for row in rows]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate case_id values {duplicates}")


def _validate_case_id(value: str) -> str:
    case_id = value.strip()
    if (
        not case_id
        or case_id in {".", ".."}
        or "/" in case_id
        or "\\" in case_id
        or Path(case_id).name != case_id
    ):
        raise ValueError(f"Unsafe or empty case_id: {value!r}")
    return case_id


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value.strip()).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _portable_path(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...] | list[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    columns = list(fieldnames) if fieldnames is not None else list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_empty_csv(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
