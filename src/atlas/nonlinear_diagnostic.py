"""Validate bounded nonlinear partial-slab registration diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.atlas.constrained_registration import (
    _ap_axis,
    _validated_registration_input_paths,
    _write_all_slice_atlas_review,
    _write_registration_diagnostic,
)
from src.atlas.t2w_mapping import (
    load_binary_mask,
    load_label_image,
    load_structures,
    load_t2,
    require_same_grid,
    sha256,
)

DIAGNOSTICS = {
    "standard_f3d": {
        "warped_template": "template_slab_nonlinear_in_subject.nii.gz",
        "labels": "template_slab_atlas_labels_nonlinear_in_subject.nii.gz",
        "support": "template_slab_mask_nonlinear_in_subject.nii.gz",
        "transform": "nonlinear_cpp.nii.gz",
        "jacobian": "nonlinear_jacobian.nii.gz",
        "registration_backend": "NiftyReg reg_f3d",
        "registration_version": "AIDAmri_v3_partial_slab_f3d_diagnostic_v1",
        "registration_command": (
            "reg_f3d -ref subject_t2.nii.gz -flo template_slab_t2.nii.gz "
            "-rmask subject_cost_mask_brain_minus_lesion.nii.gz "
            "-aff rigid_transform.txt -sx 3 -sy 3 -sz 3 -be 0.005 -jl 0.3 "
            "-ln 3 -lp 3 -maxit 300 -res "
            "template_slab_nonlinear_in_subject.nii.gz -cpp nonlinear_cpp.nii.gz "
            "-omp 4"
        ),
        "floating_mask_effective": False,
    },
    "f3d2_bimask": {
        "warped_template": "template_slab_f3d2_bimask_in_subject.nii.gz",
        "labels": "template_slab_atlas_labels_f3d2_bimask_in_subject.nii.gz",
        "support": "template_slab_mask_f3d2_bimask_in_subject.nii.gz",
        "transform": "f3d2_bimask_cpp.nii.gz",
        "jacobian": "f3d2_bimask_jacobian.nii.gz",
        "registration_backend": "NiftyReg reg_f3d F3D2 velocity mode",
        "registration_version": "AIDAmri_v3_partial_slab_f3d2_bimask_diagnostic_v1",
        "registration_command": (
            "reg_f3d -ref subject_t2.nii.gz -flo template_slab_t2.nii.gz "
            "-rmask subject_cost_mask_brain_minus_lesion.nii.gz "
            "-fmask template_slab_mask.nii.gz -aff rigid_transform.txt -vel "
            "-sx 3 -sy 3 -sz 3 -be 0.005 -jl 0.3 -ln 3 -lp 3 -maxit 300 "
            "-res template_slab_f3d2_bimask_in_subject.nii.gz "
            "-cpp f3d2_bimask_cpp.nii.gz -omp 4"
        ),
        "floating_mask_effective": True,
    },
}


def validate_nonlinear_diagnostic(
    *,
    prepared_root: Path,
    structures_csv: Path,
    diagnostic_id: str,
    aidamri_revision: str,
    niftyreg_revision: str,
    container_image_id: str,
    registration_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Validate one named nonlinear experiment without accepting its anatomy."""
    if diagnostic_id not in DIAGNOSTICS:
        raise ValueError(
            f"Unknown nonlinear diagnostic {diagnostic_id!r}; "
            f"expected one of {sorted(DIAGNOSTICS)}"
        )
    spec = DIAGNOSTICS[diagnostic_id]
    prepared_root = prepared_root.resolve()
    input_summary_path = prepared_root / "registration_input_summary.json"
    rigid_validation_path = prepared_root / "rigid_registration_validation.json"
    if not input_summary_path.is_file():
        raise FileNotFoundError(f"Registration input summary not found: {input_summary_path}")
    if not rigid_validation_path.is_file():
        raise FileNotFoundError(f"Rigid validation not found: {rigid_validation_path}")
    input_summary = json.loads(input_summary_path.read_text())
    rigid_validation = json.loads(rigid_validation_path.read_text())
    input_paths = _validated_registration_input_paths(
        prepared_root=prepared_root,
        input_summary=input_summary,
    )

    prefix = diagnostic_id
    validation_path = prepared_root / f"{prefix}_registration_validation.json"
    review_path = prepared_root / f"{prefix}_registration_review.json"
    if review_path.exists():
        existing_review = json.loads(review_path.read_text())
        if (
            existing_review.get("registration_qc_status") != "pending"
            or existing_review.get("reviewer")
            or existing_review.get("review_date")
        ):
            raise FileExistsError(
                "Refusing to overwrite an existing reviewed nonlinear decision: "
                f"{review_path}"
            )

    rigid_transform_path = prepared_root / "rigid_transform.txt"
    paths = {
        key: prepared_root / str(spec[key])
        for key in ("warped_template", "labels", "support", "transform", "jacobian")
    }
    for path in (rigid_transform_path, *paths.values()):
        if not path.is_file():
            raise FileNotFoundError(f"Nonlinear registration input or output not found: {path}")
    if rigid_validation.get("rigid_transform_sha256") != sha256(rigid_transform_path):
        raise ValueError("Nonlinear initialization rigid transform is stale or changed")

    subject = load_t2(input_paths["subject_t2"])
    brain = load_binary_mask(input_paths["subject_brain_mask"])
    lesion = load_binary_mask(input_paths["lesion_mask"])
    labels = load_label_image(paths["labels"])
    support = load_binary_mask(paths["support"])
    for volume, name in (
        (brain, "subject brain mask"),
        (lesion, "lesion mask"),
        (labels, "nonlinearly resampled atlas labels"),
        (support, "nonlinearly resampled template support"),
    ):
        require_same_grid(subject, volume, names=("subject T2w", name))
    warped_template_image = nib.load(str(paths["warped_template"]))
    warped_template = np.asanyarray(warped_template_image.dataobj)
    if warped_template.ndim != 3 or warped_template.shape != subject.data.shape:
        raise ValueError("Nonlinearly warped template does not match the subject shape")
    if not np.allclose(
        warped_template_image.affine, subject.image.affine, rtol=0, atol=1e-5
    ):
        raise ValueError("Nonlinearly warped template affine does not match the subject T2w")
    warped_template_nonfinite = ~np.isfinite(warped_template)
    warped_template_nonfinite_within_support = int(
        np.count_nonzero(warped_template_nonfinite & (support.data != 0))
    )
    supported_finite_values = warped_template[
        (support.data != 0) & ~warped_template_nonfinite
    ]
    if not supported_finite_values.size:
        raise ValueError("Nonlinearly warped template has no finite supported voxels")
    if float(np.std(supported_finite_values)) == 0.0:
        raise ValueError("Nonlinearly warped template has zero variance within its support")
    warped_template = np.nan_to_num(warped_template, copy=True)
    observed_labels = set(int(value) for value in np.unique(labels.data))
    load_structures(structures_csv.resolve(), observed_labels)
    nonzero_labels_outside_support = int(
        np.count_nonzero((labels.data != 0) & (support.data == 0))
    )

    jacobian_image = nib.load(str(paths["jacobian"]))
    jacobian = np.asanyarray(jacobian_image.dataobj)
    if jacobian.ndim != 3 or jacobian.shape != subject.data.shape:
        raise ValueError("Jacobian image does not match the subject shape")
    if not np.allclose(jacobian_image.affine, subject.image.affine, rtol=0, atol=1e-5):
        raise ValueError("Jacobian image affine does not match the subject T2w")
    if not np.isfinite(jacobian).all():
        raise ValueError("Jacobian image contains non-finite values")

    brain_data = brain.data != 0
    lesion_data = lesion.data != 0
    support_data = support.data != 0
    label_support = labels.data != 0
    brain_voxels = int(np.count_nonzero(brain_data))
    lesion_voxels = int(np.count_nonzero(lesion_data))
    support_voxels = int(np.count_nonzero(support_data))
    intersection_voxels = int(np.count_nonzero(brain_data & support_data))
    ap_axis = _ap_axis(subject.orientation)
    per_slice_coverage = []
    for index in range(subject.data.shape[ap_axis]):
        brain_slice = np.take(brain_data, index, axis=ap_axis)
        support_slice = np.take(support_data, index, axis=ap_axis)
        slice_brain_voxels = int(np.count_nonzero(brain_slice))
        if not slice_brain_voxels:
            raise ValueError(f"Subject brain mask is empty on AP slice {index}")
        covered_voxels = int(np.count_nonzero(brain_slice & support_slice))
        per_slice_coverage.append(
            {
                "slice_index": index,
                "brain_voxels": slice_brain_voxels,
                "covered_brain_voxels": covered_voxels,
                "brain_coverage_fraction": covered_voxels / slice_brain_voxels,
            }
        )

    nonpositive_jacobian_voxels = int(np.count_nonzero(jacobian <= 0))
    zero_support_slice_indices = [
        int(row["slice_index"])
        for row in per_slice_coverage
        if row["brain_coverage_fraction"] == 0.0
    ]
    if nonpositive_jacobian_voxels:
        technical_status = "rejected_nonpositive_jacobian"
    elif zero_support_slice_indices:
        technical_status = "rejected_zero_template_support_on_nonempty_brain_slice"
    else:
        technical_status = (
            "passed_file_geometry_positive_jacobian_and_nonzero_slice_support"
        )
    technical_rejection = technical_status.startswith("rejected_")

    registration_qc_path = prepared_root / f"{prefix}_registration_qc.png"
    all_slices_qc_path = prepared_root / f"{prefix}_atlas_all_slices_qc.png"
    lesion_coverage_qc_path = prepared_root / f"{prefix}_lesion_coverage_qc.png"
    _write_registration_diagnostic(
        subject_t2=subject.data,
        subject_brain=brain_data,
        warped_template=warped_template,
        template_support=support_data,
        output_path=registration_qc_path,
        slice_axis=ap_axis,
        title=f"{input_summary['case_id']}: {diagnostic_id} nonlinear diagnostic",
    )
    _write_all_slice_atlas_review(
        subject_t2=subject.data,
        subject_brain=brain_data,
        labels=labels.data,
        lesion=lesion.data,
        template_support=support_data,
        output_path=all_slices_qc_path,
        slice_axis=ap_axis,
        title=(
            f"{input_summary['case_id']}: {diagnostic_id} all AP slices; "
            "atlas red, template extent green, draft lesion cyan"
        ),
    )
    _write_lesion_coverage_diagnostic(
        subject_t2=subject.data,
        labels=labels.data,
        lesion=lesion_data,
        template_support=support_data,
        output_path=lesion_coverage_qc_path,
        slice_axis=ap_axis,
        title=f"{input_summary['case_id']}: {diagnostic_id} lesion coverage",
    )

    brain_jacobian = jacobian[brain_data]
    mapped_lesion_voxels = int(np.count_nonzero(lesion_data & label_support))
    supported_zero_label_lesion_voxels = int(
        np.count_nonzero(lesion_data & support_data & ~label_support)
    )
    labelled_outside_support_lesion_voxels = int(
        np.count_nonzero(lesion_data & ~support_data & label_support)
    )
    outside_support_zero_label_lesion_voxels = int(
        np.count_nonzero(lesion_data & ~support_data & ~label_support)
    )
    result = {
        "schema_version": 1,
        "scientific_status": (
            "rejected_by_automatic_technical_gate"
            if technical_rejection
            else "requires_human_nonlinear_diagnostic_review"
        ),
        "technical_validation_status": technical_status,
        "automatic_anatomical_acceptance_claimed": False,
        "automatic_jacobian_acceptance_threshold_claimed": False,
        "case_id": input_summary["case_id"],
        "selected_candidate_id": input_summary["selected_candidate_id"],
        "diagnostic_id": diagnostic_id,
        "registration_backend": spec["registration_backend"],
        "registration_version": spec["registration_version"],
        "aidamri_revision": aidamri_revision.strip(),
        "niftyreg_revision": niftyreg_revision.strip(),
        "container_image_id": container_image_id.strip(),
        "registration_runtime_seconds": registration_runtime_seconds,
        "registration_command": spec["registration_command"],
        "floating_mask_effective": spec["floating_mask_effective"],
        "input_summary": str(input_summary_path),
        "input_summary_sha256": sha256(input_summary_path),
        "rigid_initialization": str(rigid_transform_path),
        "rigid_initialization_sha256": sha256(rigid_transform_path),
        "rigid_validation": str(rigid_validation_path),
        "rigid_validation_sha256": sha256(rigid_validation_path),
        "deformation_transform": str(paths["transform"]),
        "deformation_transform_sha256": sha256(paths["transform"]),
        "jacobian_image": str(paths["jacobian"]),
        "jacobian_image_sha256": sha256(paths["jacobian"]),
        "jacobian_metrics": {
            "all_voxels_minimum": float(np.min(jacobian)),
            "all_voxels_maximum": float(np.max(jacobian)),
            "nonpositive_voxels": nonpositive_jacobian_voxels,
            "brain_minimum": float(np.min(brain_jacobian)),
            "brain_maximum": float(np.max(brain_jacobian)),
            "brain_percentiles_1_5_50_95_99": [
                float(value)
                for value in np.percentile(brain_jacobian, (1, 5, 50, 95, 99))
            ],
        },
        "native_atlas_labels": str(paths["labels"]),
        "native_atlas_labels_sha256": sha256(paths["labels"]),
        "native_template_support": str(paths["support"]),
        "native_template_support_sha256": sha256(paths["support"]),
        "warped_template_t2": str(paths["warped_template"]),
        "warped_template_t2_sha256": sha256(paths["warped_template"]),
        "warped_template_nonfinite_voxels": int(
            np.count_nonzero(warped_template_nonfinite)
        ),
        "warped_template_nonfinite_voxels_within_nearest_neighbor_support": (
            warped_template_nonfinite_within_support
        ),
        "structures_csv": str(structures_csv.resolve()),
        "structures_csv_sha256": sha256(structures_csv.resolve()),
        "observed_nonzero_label_count": len(observed_labels - {0}),
        "nonzero_atlas_label_voxels_outside_template_support": (
            nonzero_labels_outside_support
        ),
        "brain_voxels": brain_voxels,
        "brain_coverage_fraction": intersection_voxels / brain_voxels,
        "brain_coverage_change_from_rigid": (
            intersection_voxels / brain_voxels
            - float(rigid_validation["brain_coverage_fraction"])
        ),
        "template_support_fraction_inside_brain": intersection_voxels
        / support_voxels,
        "template_support_voxels_outside_brain": int(
            np.count_nonzero(support_data & ~brain_data)
        ),
        "lesion_voxels": lesion_voxels,
        "lesion_coverage_partition": {
            "mapped_nonzero_label_voxels": mapped_lesion_voxels,
            "mapped_nonzero_label_fraction": (
                mapped_lesion_voxels / lesion_voxels if lesion_voxels else None
            ),
            "inside_template_support_zero_label_voxels": (
                supported_zero_label_lesion_voxels
            ),
            "inside_template_support_zero_label_fraction": (
                supported_zero_label_lesion_voxels / lesion_voxels
                if lesion_voxels
                else None
            ),
            "nonzero_label_outside_template_support_voxels": (
                labelled_outside_support_lesion_voxels
            ),
            "nonzero_label_outside_template_support_fraction": (
                labelled_outside_support_lesion_voxels / lesion_voxels
                if lesion_voxels
                else None
            ),
            "zero_label_outside_template_support_voxels": (
                outside_support_zero_label_lesion_voxels
            ),
            "zero_label_outside_template_support_fraction": (
                outside_support_zero_label_lesion_voxels / lesion_voxels
                if lesion_voxels
                else None
            ),
        },
        "lesion_coverage_fraction_by_template_support": (
            np.count_nonzero(lesion_data & support_data) / lesion_voxels
            if lesion_voxels
            else None
        ),
        "lesion_coverage_fraction_by_nonzero_atlas_label": (
            np.count_nonzero(lesion_data & label_support) / lesion_voxels
            if lesion_voxels
            else None
        ),
        "zero_template_support_slice_indices": zero_support_slice_indices,
        "per_ap_slice_brain_coverage": per_slice_coverage,
        "registration_qc_png": str(registration_qc_path),
        "registration_qc_png_sha256": sha256(registration_qc_path),
        "all_slices_qc_png": str(all_slices_qc_path),
        "all_slices_qc_png_sha256": sha256(all_slices_qc_path),
        "lesion_coverage_qc_png": str(lesion_coverage_qc_path),
        "lesion_coverage_qc_png_sha256": sha256(lesion_coverage_qc_path),
    }
    validation_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    review = {
        "schema_version": 1,
        "case_id": input_summary["case_id"],
        "registration_qc_status": "rejected" if technical_rejection else "pending",
        "atlas_id": "AIDAmri_ARA",
        "atlas_version": "CCFv3_50um",
        "coordinate_space": "Allen Mouse Brain CCFv3",
        "annotation_variant": "split_parental",
        "registration_backend": spec["registration_backend"],
        "registration_version": spec["registration_version"],
        "registration_revision": aidamri_revision.strip(),
        "native_atlas_labels_sha256": result["native_atlas_labels_sha256"],
        "deformation_transform_sha256": result["deformation_transform_sha256"],
        "registration_validation_sha256": sha256(validation_path),
        "full_rostrocaudal_coverage_reviewed": None,
        "anatomical_landmarks_acceptable": None,
        "jacobian_deformation_acceptable": None,
        "reviewer": None,
        "review_date": None,
        "notes": (
            f"Automatic technical status: {technical_status}. Inspect every panel "
            f"in {all_slices_qc_path.name}, {lesion_coverage_qc_path.name}, and the "
            "Jacobian metrics. No automatic anatomical or Jacobian acceptance "
            "threshold is claimed."
        ),
    }
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    result["registration_review_json"] = str(review_path)
    return result


def _write_lesion_coverage_diagnostic(
    *,
    subject_t2: np.ndarray,
    labels: np.ndarray,
    lesion: np.ndarray,
    template_support: np.ndarray,
    output_path: Path,
    slice_axis: int,
    title: str,
) -> None:
    """Show whether each lesion voxel is labelled, zero-labelled, or unsupported."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from scipy import ndimage

    lesion = lesion.astype(bool)
    support = template_support.astype(bool)
    other_axes = tuple(axis for axis in range(3) if axis != slice_axis)
    lesion_slice_indices = np.flatnonzero(np.any(lesion, axis=other_axes)).tolist()
    if lesion_slice_indices:
        indices = lesion_slice_indices
    else:
        indices = np.unique(
            np.linspace(0, subject_t2.shape[slice_axis] - 1, 6, dtype=int)
        ).tolist()
    n_columns = min(4, len(indices))
    n_rows = int(np.ceil(len(indices) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4 * n_columns, 4 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    finite = subject_t2[np.isfinite(subject_t2)]
    vmin, vmax = np.percentile(finite, (1, 99))
    atlas_cmap = ListedColormap(["red"])
    mapped_cmap = ListedColormap(["lime"])
    mapped_outside_support_cmap = ListedColormap(["cyan"])
    zero_label_cmap = ListedColormap(["gold"])
    unsupported_cmap = ListedColormap(["magenta"])
    for axis, index in zip(flat_axes, indices, strict=False):
        t2_slice = np.take(subject_t2, index, axis=slice_axis)
        labels_slice = np.take(labels, index, axis=slice_axis)
        lesion_slice = np.take(lesion, index, axis=slice_axis)
        support_slice = np.take(support, index, axis=slice_axis)
        label_boundary = ndimage.maximum_filter(
            labels_slice, size=3
        ) != ndimage.minimum_filter(labels_slice, size=3)
        label_boundary &= labels_slice != 0
        mapped = lesion_slice & (labels_slice != 0) & support_slice
        mapped_outside_support = lesion_slice & (labels_slice != 0) & ~support_slice
        supported_zero = lesion_slice & support_slice & (labels_slice == 0)
        unsupported = lesion_slice & ~support_slice & (labels_slice == 0)
        total = int(np.count_nonzero(lesion_slice))

        axis.imshow(t2_slice.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        for mask, cmap, alpha in (
            (label_boundary, atlas_cmap, 0.7),
            (mapped, mapped_cmap, 0.7),
            (mapped_outside_support, mapped_outside_support_cmap, 0.8),
            (supported_zero, zero_label_cmap, 0.8),
            (unsupported, unsupported_cmap, 0.8),
        ):
            axis.imshow(
                np.ma.masked_where(~mask.T, mask.T),
                cmap=cmap,
                origin="lower",
                alpha=alpha,
            )
        axis.set_title(
            f"AP {index}: region "
            f"{np.count_nonzero(mapped | mapped_outside_support)}/{total}; "
            f"support {np.count_nonzero(lesion_slice & support_slice)}/{total}"
        )
        axis.axis("off")
    for axis in flat_axes[len(indices) :]:
        axis.axis("off")
    figure.legend(
        handles=[
            Patch(color="lime", label="lesion with nonzero atlas region"),
            Patch(color="cyan", label="labelled lesion outside intensity support"),
            Patch(color="gold", label="lesion in template support, atlas label 0"),
            Patch(color="magenta", label="lesion outside template support"),
            Patch(color="red", label="atlas boundary"),
        ],
        loc="lower center",
        ncol=2,
    )
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
