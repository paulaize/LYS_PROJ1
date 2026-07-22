"""Prepare inputs for constrained registration of partial-volume T2w MRI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.atlas.partial_volume import load_approved_slab_selection
from src.atlas.t2w_mapping import (
    load_binary_mask,
    load_label_image,
    load_structures,
    load_t2,
    require_same_grid,
    save_like,
    sha256,
    write_qc_overlay,
)

RIGID_TRANSFORM_ATOL = 1e-4


def prepare_constrained_registration_inputs(
    *,
    selection_json: Path,
    candidates_csv: Path,
    subject_t2: Path,
    subject_brain_mask: Path,
    lesion_mask: Path,
    template_t2: Path,
    template_atlas_labels: Path,
    output_root: Path,
    slab_edge_padding_mm: float = 0.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a selected template slab and lesion-excluded registration masks."""
    subject = load_t2(subject_t2.resolve())
    brain = load_binary_mask(subject_brain_mask.resolve())
    lesion = load_binary_mask(lesion_mask.resolve())
    template = load_t2(template_t2.resolve())
    labels = load_label_image(template_atlas_labels.resolve())
    require_same_grid(subject, brain, names=("subject T2w", "subject brain mask"))
    require_same_grid(subject, lesion, names=("subject T2w", "lesion mask"))
    require_same_grid(template, labels, names=("template T2w", "template atlas labels"))
    if subject.orientation != template.orientation:
        raise ValueError("Subject and template orientations disagree")

    candidate = load_approved_slab_selection(
        selection_json=selection_json,
        candidates_csv=candidates_csv,
        subject_t2=subject.path,
        subject_brain_mask=brain.path,
        lesion_mask=lesion.path,
        template_t2=template.path,
    )
    ap_axis = _ap_axis(subject.orientation)
    if (
        candidate.template_start_index < 0
        or candidate.template_end_index >= template.data.shape[ap_axis]
    ):
        raise ValueError("Approved template slab is outside the template volume")
    if candidate.template_end_index < candidate.template_start_index:
        raise ValueError("Approved template slab is empty")
    if not np.isfinite(slab_edge_padding_mm) or slab_edge_padding_mm < 0:
        raise ValueError("Slab edge padding must be a finite non-negative distance")
    template_ap_spacing_mm = template.spacing_mm[ap_axis]
    padding_voxels = int(np.ceil(slab_edge_padding_mm / template_ap_spacing_mm))
    crop_start = candidate.template_start_index - padding_voxels
    crop_end = candidate.template_end_index + padding_voxels
    if crop_start < 0 or crop_end >= template.data.shape[ap_axis]:
        raise ValueError(
            "Requested slab edge padding is unavailable around the approved "
            f"candidate: selected {candidate.template_start_index}.."
            f"{candidate.template_end_index}, padded {crop_start}..{crop_end}, "
            f"template axis length {template.data.shape[ap_axis]}"
        )

    cost_mask = (brain.data != 0) & (lesion.data == 0)
    if not np.any(cost_mask):
        raise ValueError("Lesion exclusion removed the entire subject brain mask")
    lesion_outside_brain = int(np.count_nonzero((lesion.data != 0) & (brain.data == 0)))
    if lesion_outside_brain:
        raise ValueError(
            f"Lesion mask has {lesion_outside_brain} voxels outside the subject brain mask"
        )

    output_root = output_root.resolve()
    _prepare_output_root(output_root, overwrite=overwrite)
    subject_copy = output_root / "subject_t2.nii.gz"
    brain_copy = output_root / "subject_brain_mask.nii.gz"
    lesion_copy = output_root / "lesion_model_draft_native.nii.gz"
    shutil.copy2(subject.path, subject_copy)
    shutil.copy2(brain.path, brain_copy)
    shutil.copy2(lesion.path, lesion_copy)
    cost_mask_path = output_root / "subject_cost_mask_brain_minus_lesion.nii.gz"
    save_like(cost_mask_path, cost_mask.astype(np.uint8), subject.image)

    slab_t2_path = output_root / "template_slab_t2.nii.gz"
    slab_mask_path = output_root / "template_slab_mask.nii.gz"
    slab_labels_path = output_root / "template_slab_atlas_labels.nii.gz"
    slab_t2_image = crop_nifti(
        image=template.image,
        start=crop_start,
        stop=crop_end + 1,
        axis=ap_axis,
        data=template.data.astype(np.float32),
    )
    template_support = np.isfinite(template.data) & (template.data != 0)
    slab_mask_image = crop_nifti(
        image=template.image,
        start=crop_start,
        stop=crop_end + 1,
        axis=ap_axis,
        data=template_support.astype(np.uint8),
    )
    slab_labels_image = crop_nifti(
        image=labels.image,
        start=crop_start,
        stop=crop_end + 1,
        axis=ap_axis,
        data=labels.data.astype(np.int32),
    )
    nib.save(slab_t2_image, slab_t2_path)
    nib.save(slab_mask_image, slab_mask_path)
    nib.save(slab_labels_image, slab_labels_path)

    summary = {
        "schema_version": 1,
        "scientific_status": "prepared_for_masked_rigid_registration",
        "case_id": json.loads(selection_json.resolve().read_text())["case_id"],
        "selection_json": str(selection_json.resolve()),
        "selection_json_sha256": sha256(selection_json.resolve()),
        "candidates_csv": str(candidates_csv.resolve()),
        "candidates_csv_sha256": sha256(candidates_csv.resolve()),
        "selected_candidate_id": candidate.candidate_id,
        "template_start_index": candidate.template_start_index,
        "template_end_index": candidate.template_end_index,
        "template_crop_start_index": crop_start,
        "template_crop_end_index": crop_end,
        "slab_edge_padding_requested_mm": slab_edge_padding_mm,
        "slab_edge_padding_template_voxels": padding_voxels,
        "slab_edge_padding_actual_mm": padding_voxels * template_ap_spacing_mm,
        "ap_axis": ap_axis,
        "subject_t2": str(subject_copy),
        "subject_t2_sha256": sha256(subject_copy),
        "subject_brain_mask": str(brain_copy),
        "subject_brain_mask_sha256": sha256(brain_copy),
        "lesion_mask": str(lesion_copy),
        "lesion_mask_sha256": sha256(lesion_copy),
        "subject_cost_mask": str(cost_mask_path),
        "subject_cost_mask_sha256": sha256(cost_mask_path),
        "subject_brain_voxels": int(np.count_nonzero(brain.data)),
        "lesion_excluded_voxels": int(np.count_nonzero(lesion.data)),
        "registration_cost_voxels": int(np.count_nonzero(cost_mask)),
        "lesion_mask_dilation_mm": 0.0,
        "template_slab_t2": str(slab_t2_path),
        "template_slab_t2_sha256": sha256(slab_t2_path),
        "template_slab_mask": str(slab_mask_path),
        "template_slab_mask_sha256": sha256(slab_mask_path),
        "template_slab_atlas_labels": str(slab_labels_path),
        "template_slab_atlas_labels_sha256": sha256(slab_labels_path),
        "template_slab_shape": list(slab_t2_image.shape),
        "template_slab_spacing_mm": [
            float(value) for value in slab_t2_image.header.get_zooms()[:3]
        ],
        "planned_registration": {
            "backend": "NiftyReg reg_aladin",
            "transformation": "rigid_only",
            "initialization": "mask_centres_of_gravity",
            "reference_mask": "subject brain minus undilated draft lesion",
            "floating_mask": "finite nonzero selected template slab",
            "selected_slice_centres_have_edge_padding": padding_voxels > 0,
            "affine_scale_or_shear_allowed": False,
            "nonlinear_deformation_allowed": False,
        },
    }
    summary_path = output_root / "registration_input_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def validate_constrained_rigid_registration(
    *,
    prepared_root: Path,
    structures_csv: Path,
    aidamri_revision: str,
    niftyreg_revision: str,
    container_image_id: str,
    registration_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Validate and render a completed masked rigid partial-slab registration.

    The numerical checks establish file integrity and verify that the transform
    is rigid. They do not establish anatomical correctness; the emitted review
    record therefore remains pending until a human inspects the overlays.
    """
    prepared_root = prepared_root.resolve()
    input_summary_path = prepared_root / "registration_input_summary.json"
    if not input_summary_path.is_file():
        raise FileNotFoundError(f"Registration input summary not found: {input_summary_path}")
    input_summary = json.loads(input_summary_path.read_text())
    if input_summary.get("planned_registration", {}).get("transformation") != "rigid_only":
        raise ValueError("Prepared registration was not declared rigid-only")
    review_path = prepared_root / "rigid_registration_review.json"
    if review_path.exists():
        existing_review = json.loads(review_path.read_text())
        if (
            existing_review.get("registration_qc_status") != "pending"
            or existing_review.get("reviewer")
            or existing_review.get("review_date")
        ):
            raise FileExistsError(
                "Refusing to overwrite an existing reviewed registration decision: "
                f"{review_path}"
            )

    input_paths = {
        "subject_t2": prepared_root / "subject_t2.nii.gz",
        "subject_brain_mask": prepared_root / "subject_brain_mask.nii.gz",
        "lesion_mask": prepared_root / "lesion_model_draft_native.nii.gz",
        "subject_cost_mask": prepared_root
        / "subject_cost_mask_brain_minus_lesion.nii.gz",
        "template_slab_t2": prepared_root / "template_slab_t2.nii.gz",
        "template_slab_mask": prepared_root / "template_slab_mask.nii.gz",
        "template_slab_atlas_labels": prepared_root
        / "template_slab_atlas_labels.nii.gz",
    }
    for field, path in input_paths.items():
        expected_hash = input_summary.get(f"{field}_sha256")
        if not path.is_file():
            raise FileNotFoundError(f"Prepared registration input not found: {path}")
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f"Prepared registration input is stale or changed: {field}")

    transform_path = prepared_root / "rigid_transform.txt"
    warped_template_path = prepared_root / "template_slab_rigid_in_subject.nii.gz"
    labels_path = prepared_root / "template_slab_atlas_labels_rigid_in_subject.nii.gz"
    support_path = prepared_root / "template_slab_mask_rigid_in_subject.nii.gz"
    for path in (transform_path, warped_template_path, labels_path, support_path):
        if not path.is_file():
            raise FileNotFoundError(f"Rigid registration output not found: {path}")

    subject = load_t2(input_paths["subject_t2"])
    brain = load_binary_mask(input_paths["subject_brain_mask"])
    lesion = load_binary_mask(input_paths["lesion_mask"])
    warped_template = load_t2(warped_template_path)
    labels = load_label_image(labels_path)
    support = load_binary_mask(support_path)
    for volume, name in (
        (brain, "subject brain mask"),
        (lesion, "lesion mask"),
        (warped_template, "rigidly warped template"),
        (labels, "rigidly resampled atlas labels"),
        (support, "rigidly resampled template support"),
    ):
        require_same_grid(subject, volume, names=("subject T2w", name))

    observed_labels = set(int(value) for value in np.unique(labels.data))
    load_structures(structures_csv.resolve(), observed_labels)
    transform_metrics = rigid_transform_metrics(transform_path)

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
        per_slice_coverage.append(
            {
                "slice_index": index,
                "brain_voxels": slice_brain_voxels,
                "covered_brain_voxels": int(
                    np.count_nonzero(brain_slice & support_slice)
                ),
                "brain_coverage_fraction": float(
                    np.count_nonzero(brain_slice & support_slice) / slice_brain_voxels
                ),
            }
        )

    registration_qc_path = prepared_root / "rigid_registration_qc.png"
    atlas_qc_path = prepared_root / "rigid_atlas_mapping_qc.png"
    all_slices_qc_path = prepared_root / "rigid_atlas_all_slices_qc.png"
    _write_registration_diagnostic(
        subject_t2=subject.data,
        subject_brain=brain_data,
        warped_template=warped_template.data,
        template_support=support_data,
        output_path=registration_qc_path,
        slice_axis=ap_axis,
        title=(
            f"{input_summary['case_id']}: {input_summary['selected_candidate_id']} "
            "masked rigid registration"
        ),
    )
    write_qc_overlay(
        t2=subject.data,
        labels=labels.data,
        lesion=lesion.data,
        output_path=atlas_qc_path,
        slice_axis=ap_axis,
        title=(
            f"{input_summary['case_id']}: rigid split-parental atlas boundaries "
            "(red), draft lesion (cyan)"
        ),
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
            f"{input_summary['case_id']}: all AP slices; atlas red, "
            "template extent green, draft lesion cyan"
        ),
    )

    commands = {
        "registration": (
            "reg_aladin -ref subject_t2.nii.gz -flo template_slab_t2.nii.gz "
            "-rmask subject_cost_mask_brain_minus_lesion.nii.gz "
            "-fmask template_slab_mask.nii.gz -rigOnly -cog "
            "-res template_slab_rigid_in_subject.nii.gz "
            "-aff rigid_transform.txt -omp 4"
        ),
        "label_resampling": (
            "reg_resample -ref subject_t2.nii.gz "
            "-flo template_slab_atlas_labels.nii.gz -trans rigid_transform.txt "
            "-inter 0 -res template_slab_atlas_labels_rigid_in_subject.nii.gz"
        ),
        "support_resampling": (
            "reg_resample -ref subject_t2.nii.gz -flo template_slab_mask.nii.gz "
            "-trans rigid_transform.txt -inter 0 "
            "-res template_slab_mask_rigid_in_subject.nii.gz"
        ),
    }
    result = {
        "schema_version": 1,
        "scientific_status": "requires_human_registration_review",
        "automatic_anatomical_acceptance_claimed": False,
        "case_id": input_summary["case_id"],
        "selected_candidate_id": input_summary["selected_candidate_id"],
        "selection_json_sha256": input_summary["selection_json_sha256"],
        "aidamri_revision": aidamri_revision.strip(),
        "niftyreg_revision": niftyreg_revision.strip(),
        "container_image_id": container_image_id.strip(),
        "registration_runtime_seconds": registration_runtime_seconds,
        "commands": commands,
        "input_summary": str(input_summary_path),
        "input_summary_sha256": sha256(input_summary_path),
        "rigid_transform": str(transform_path),
        "rigid_transform_sha256": sha256(transform_path),
        "transform_metrics": transform_metrics,
        "native_atlas_labels": str(labels_path),
        "native_atlas_labels_sha256": sha256(labels_path),
        "native_template_support": str(support_path),
        "native_template_support_sha256": sha256(support_path),
        "warped_template_t2": str(warped_template_path),
        "warped_template_t2_sha256": sha256(warped_template_path),
        "structures_csv": str(structures_csv.resolve()),
        "structures_csv_sha256": sha256(structures_csv.resolve()),
        "subject_shape": list(subject.data.shape),
        "subject_spacing_mm": list(subject.spacing_mm),
        "subject_orientation": subject.orientation,
        "observed_label_count_including_zero": len(observed_labels),
        "observed_nonzero_label_count": len(observed_labels - {0}),
        "brain_voxels": brain_voxels,
        "template_support_voxels": support_voxels,
        "brain_template_intersection_voxels": intersection_voxels,
        "brain_coverage_fraction": intersection_voxels / brain_voxels,
        "template_support_fraction_inside_brain": intersection_voxels
        / support_voxels,
        "template_support_voxels_outside_brain": int(
            np.count_nonzero(support_data & ~brain_data)
        ),
        "lesion_voxels": lesion_voxels,
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
        "per_ap_slice_brain_coverage": per_slice_coverage,
        "registration_qc_png": str(registration_qc_path),
        "registration_qc_png_sha256": sha256(registration_qc_path),
        "atlas_qc_png": str(atlas_qc_path),
        "atlas_qc_png_sha256": sha256(atlas_qc_path),
        "all_slices_qc_png": str(all_slices_qc_path),
        "all_slices_qc_png_sha256": sha256(all_slices_qc_path),
    }
    result_path = prepared_root / "rigid_registration_validation.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    review = {
        "schema_version": 1,
        "case_id": input_summary["case_id"],
        "registration_qc_status": "pending",
        "atlas_id": "AIDAmri_ARA",
        "atlas_version": "CCFv3_50um",
        "coordinate_space": "Allen Mouse Brain CCFv3",
        "annotation_variant": "split_parental",
        "registration_backend": "AIDAmri masked partial-slab NiftyReg rigid",
        "registration_version": "AIDAmri_v3_partial_slab_rigid_v1",
        "registration_revision": aidamri_revision.strip(),
        "native_atlas_labels_sha256": result["native_atlas_labels_sha256"],
        "rigid_transform_sha256": result["rigid_transform_sha256"],
        "registration_validation_sha256": sha256(result_path),
        "full_rostrocaudal_coverage_reviewed": None,
        "anatomical_landmarks_acceptable": None,
        "reviewer": None,
        "review_date": None,
        "notes": (
            "Inspect rigid_registration_qc.png, rigid_atlas_mapping_qc.png, and "
            "all 18 panels in rigid_atlas_all_slices_qc.png. Automatic integrity "
            "checks do not approve anatomical registration."
        ),
    }
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    result["registration_review_json"] = str(review_path)
    return result


def validate_constrained_affine_registration(
    *,
    prepared_root: Path,
    structures_csv: Path,
    aidamri_revision: str,
    niftyreg_revision: str,
    container_image_id: str,
    registration_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Validate an affine diagnostic initialized from the masked rigid result."""
    prepared_root = prepared_root.resolve()
    input_summary_path = prepared_root / "registration_input_summary.json"
    rigid_validation_path = prepared_root / "rigid_registration_validation.json"
    if not input_summary_path.is_file():
        raise FileNotFoundError(f"Registration input summary not found: {input_summary_path}")
    if not rigid_validation_path.is_file():
        raise FileNotFoundError(f"Rigid validation not found: {rigid_validation_path}")
    input_summary = json.loads(input_summary_path.read_text())
    rigid_validation = json.loads(rigid_validation_path.read_text())
    if input_summary.get("planned_registration", {}).get("transformation") != "rigid_only":
        raise ValueError("Prepared registration was not declared rigid-only")

    review_path = prepared_root / "affine_registration_review.json"
    if review_path.exists():
        existing_review = json.loads(review_path.read_text())
        if (
            existing_review.get("registration_qc_status") != "pending"
            or existing_review.get("reviewer")
            or existing_review.get("review_date")
        ):
            raise FileExistsError(
                "Refusing to overwrite an existing reviewed affine decision: "
                f"{review_path}"
            )

    input_paths = _validated_registration_input_paths(
        prepared_root=prepared_root,
        input_summary=input_summary,
    )
    rigid_transform_path = prepared_root / "rigid_transform.txt"
    affine_transform_path = prepared_root / "affine_transform.txt"
    warped_template_path = prepared_root / "template_slab_affine_in_subject.nii.gz"
    labels_path = prepared_root / "template_slab_atlas_labels_affine_in_subject.nii.gz"
    support_path = prepared_root / "template_slab_mask_affine_in_subject.nii.gz"
    for path in (
        rigid_transform_path,
        affine_transform_path,
        warped_template_path,
        labels_path,
        support_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Affine registration input or output not found: {path}")
    if rigid_validation.get("rigid_transform_sha256") != sha256(rigid_transform_path):
        raise ValueError("Affine initialization rigid transform is stale or changed")

    subject = load_t2(input_paths["subject_t2"])
    brain = load_binary_mask(input_paths["subject_brain_mask"])
    lesion = load_binary_mask(input_paths["lesion_mask"])
    warped_template = load_t2(warped_template_path)
    labels = load_label_image(labels_path)
    support = load_binary_mask(support_path)
    for volume, name in (
        (brain, "subject brain mask"),
        (lesion, "lesion mask"),
        (warped_template, "affinely warped template"),
        (labels, "affinely resampled atlas labels"),
        (support, "affinely resampled template support"),
    ):
        require_same_grid(subject, volume, names=("subject T2w", name))

    observed_labels = set(int(value) for value in np.unique(labels.data))
    load_structures(structures_csv.resolve(), observed_labels)
    transform_metrics = affine_transform_metrics(affine_transform_path)
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
        per_slice_coverage.append(
            {
                "slice_index": index,
                "brain_voxels": slice_brain_voxels,
                "covered_brain_voxels": int(
                    np.count_nonzero(brain_slice & support_slice)
                ),
                "brain_coverage_fraction": float(
                    np.count_nonzero(brain_slice & support_slice) / slice_brain_voxels
                ),
            }
        )

    registration_qc_path = prepared_root / "affine_registration_qc.png"
    all_slices_qc_path = prepared_root / "affine_atlas_all_slices_qc.png"
    _write_registration_diagnostic(
        subject_t2=subject.data,
        subject_brain=brain_data,
        warped_template=warped_template.data,
        template_support=support_data,
        output_path=registration_qc_path,
        slice_axis=ap_axis,
        title=(
            f"{input_summary['case_id']}: {input_summary['selected_candidate_id']} "
            "masked affine diagnostic"
        ),
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
            f"{input_summary['case_id']}: affine all AP slices; atlas red, "
            "template extent green, draft lesion cyan"
        ),
    )

    rigid_slice_coverage = {
        int(row["slice_index"]): float(row["brain_coverage_fraction"])
        for row in rigid_validation["per_ap_slice_brain_coverage"]
    }
    comparison_rows = []
    for row in per_slice_coverage:
        rigid_coverage = rigid_slice_coverage[row["slice_index"]]
        comparison_rows.append(
            {
                **row,
                "rigid_brain_coverage_fraction": rigid_coverage,
                "affine_minus_rigid_coverage_fraction": (
                    row["brain_coverage_fraction"] - rigid_coverage
                ),
            }
        )
    zero_support_slice_indices = [
        int(row["slice_index"])
        for row in per_slice_coverage
        if row["brain_coverage_fraction"] == 0.0
    ]
    technical_status = (
        "rejected_zero_template_support_on_nonempty_brain_slice"
        if zero_support_slice_indices
        else "passed_file_geometry_and_nonzero_slice_support"
    )

    commands = {
        "registration": (
            "reg_aladin -ref subject_t2.nii.gz -flo template_slab_t2.nii.gz "
            "-rmask subject_cost_mask_brain_minus_lesion.nii.gz "
            "-fmask template_slab_mask.nii.gz -inaff rigid_transform.txt "
            "-res template_slab_affine_in_subject.nii.gz "
            "-aff affine_transform.txt -omp 4"
        ),
        "label_resampling": (
            "reg_resample -ref subject_t2.nii.gz "
            "-flo template_slab_atlas_labels.nii.gz -trans affine_transform.txt "
            "-inter 0 -res template_slab_atlas_labels_affine_in_subject.nii.gz"
        ),
        "support_resampling": (
            "reg_resample -ref subject_t2.nii.gz -flo template_slab_mask.nii.gz "
            "-trans affine_transform.txt -inter 0 "
            "-res template_slab_mask_affine_in_subject.nii.gz"
        ),
    }
    result = {
        "schema_version": 1,
        "scientific_status": (
            "rejected_by_automatic_technical_gate"
            if zero_support_slice_indices
            else "requires_human_affine_diagnostic_review"
        ),
        "technical_validation_status": technical_status,
        "zero_template_support_slice_indices": zero_support_slice_indices,
        "automatic_anatomical_acceptance_claimed": False,
        "automatic_affine_distortion_threshold_claimed": False,
        "case_id": input_summary["case_id"],
        "selected_candidate_id": input_summary["selected_candidate_id"],
        "selection_json_sha256": input_summary["selection_json_sha256"],
        "aidamri_revision": aidamri_revision.strip(),
        "niftyreg_revision": niftyreg_revision.strip(),
        "container_image_id": container_image_id.strip(),
        "registration_runtime_seconds": registration_runtime_seconds,
        "commands": commands,
        "input_summary": str(input_summary_path),
        "input_summary_sha256": sha256(input_summary_path),
        "rigid_initialization": str(rigid_transform_path),
        "rigid_initialization_sha256": sha256(rigid_transform_path),
        "rigid_validation": str(rigid_validation_path),
        "rigid_validation_sha256": sha256(rigid_validation_path),
        "affine_transform": str(affine_transform_path),
        "affine_transform_sha256": sha256(affine_transform_path),
        "transform_metrics": transform_metrics,
        "native_atlas_labels": str(labels_path),
        "native_atlas_labels_sha256": sha256(labels_path),
        "native_template_support": str(support_path),
        "native_template_support_sha256": sha256(support_path),
        "warped_template_t2": str(warped_template_path),
        "warped_template_t2_sha256": sha256(warped_template_path),
        "structures_csv": str(structures_csv.resolve()),
        "structures_csv_sha256": sha256(structures_csv.resolve()),
        "subject_shape": list(subject.data.shape),
        "subject_spacing_mm": list(subject.spacing_mm),
        "subject_orientation": subject.orientation,
        "observed_label_count_including_zero": len(observed_labels),
        "observed_nonzero_label_count": len(observed_labels - {0}),
        "brain_voxels": brain_voxels,
        "template_support_voxels": support_voxels,
        "brain_template_intersection_voxels": intersection_voxels,
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
        "per_ap_slice_brain_coverage_comparison": comparison_rows,
        "registration_qc_png": str(registration_qc_path),
        "registration_qc_png_sha256": sha256(registration_qc_path),
        "all_slices_qc_png": str(all_slices_qc_path),
        "all_slices_qc_png_sha256": sha256(all_slices_qc_path),
    }
    result_path = prepared_root / "affine_registration_validation.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    review = {
        "schema_version": 1,
        "case_id": input_summary["case_id"],
        "registration_qc_status": (
            "rejected" if zero_support_slice_indices else "pending"
        ),
        "atlas_id": "AIDAmri_ARA",
        "atlas_version": "CCFv3_50um",
        "coordinate_space": "Allen Mouse Brain CCFv3",
        "annotation_variant": "split_parental",
        "registration_backend": "AIDAmri masked partial-slab NiftyReg affine",
        "registration_version": "AIDAmri_v3_partial_slab_affine_v1",
        "registration_revision": aidamri_revision.strip(),
        "native_atlas_labels_sha256": result["native_atlas_labels_sha256"],
        "affine_transform_sha256": result["affine_transform_sha256"],
        "registration_validation_sha256": sha256(result_path),
        "full_rostrocaudal_coverage_reviewed": None,
        "anatomical_landmarks_acceptable": None,
        "affine_distortion_acceptable": None,
        "reviewer": None,
        "review_date": None,
        "notes": (
            (
                "Automatically rejected because the transformed template has zero "
                "support on nonempty subject brain slice(s) "
                f"{zero_support_slice_indices}. "
            )
            if zero_support_slice_indices
            else ""
        )
        + (
            "Inspect affine_registration_qc.png, all 18 panels in "
            "affine_atlas_all_slices_qc.png, and the reported principal stretches. "
            "No automatic affine-distortion acceptance threshold is claimed."
        ),
    }
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    result["registration_review_json"] = str(review_path)
    return result


def affine_transform_metrics(transform_path: Path) -> dict[str, Any]:
    """Describe a finite orientation-preserving affine without approving it."""
    transform_path = transform_path.resolve()
    try:
        transform = np.loadtxt(transform_path, dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"Could not parse affine transform: {transform_path}") from error
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Affine transform must be a finite 4 x 4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError("Affine transform has an invalid homogeneous final row")
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    singular_values = np.linalg.svd(linear, compute_uv=False)
    if determinant <= 0 or float(np.min(singular_values)) <= 1e-6:
        raise ValueError("Affine transform is singular or contains a reflection")
    left, _, right = np.linalg.svd(linear)
    rotation = left @ right
    stretch = right.T @ np.diag(singular_values) @ right
    off_diagonal_stretch = stretch - np.diag(np.diag(stretch))
    rotation_angle = float(
        np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    )
    return {
        "determinant": determinant,
        "volume_scale_percent_change": (determinant - 1.0) * 100.0,
        "principal_stretches": [float(value) for value in singular_values],
        "principal_stretch_percent_changes": [
            float((value - 1.0) * 100.0) for value in singular_values
        ],
        "condition_number": float(np.max(singular_values) / np.min(singular_values)),
        "polar_stretch_matrix": stretch.tolist(),
        "maximum_off_diagonal_polar_stretch": float(
            np.max(np.abs(off_diagonal_stretch))
        ),
        "polar_rotation_angle_degrees": rotation_angle,
        "translation_mm": [float(value) for value in transform[:3, 3]],
        "automatic_distortion_acceptance": "not_claimed",
    }


def rigid_transform_metrics(transform_path: Path) -> dict[str, Any]:
    """Load a NiftyReg text transform and require a proper rigid matrix."""
    transform_path = transform_path.resolve()
    try:
        transform = np.loadtxt(transform_path, dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"Could not parse rigid transform: {transform_path}") from error
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Rigid transform must be a finite 4 x 4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError("Rigid transform has an invalid homogeneous final row")
    linear = transform[:3, :3]
    determinant = float(np.linalg.det(linear))
    singular_values = np.linalg.svd(linear, compute_uv=False)
    orthogonality_error = float(np.max(np.abs(linear.T @ linear - np.eye(3))))
    if determinant <= 0:
        raise ValueError("Rigid transform contains a reflection")
    if (
        not np.allclose(singular_values, 1.0, rtol=0, atol=RIGID_TRANSFORM_ATOL)
        or abs(determinant - 1.0) > RIGID_TRANSFORM_ATOL
        or orthogonality_error > RIGID_TRANSFORM_ATOL
    ):
        raise ValueError("Transform contains scale or shear and is not rigid")
    rotation_angle = float(
        np.degrees(np.arccos(np.clip((np.trace(linear) - 1.0) / 2.0, -1.0, 1.0)))
    )
    return {
        "determinant": determinant,
        "singular_values": [float(value) for value in singular_values],
        "maximum_orthogonality_error": orthogonality_error,
        "rotation_angle_degrees": rotation_angle,
        "translation_mm": [float(value) for value in transform[:3, 3]],
        "rigid_integrity_check": "passed",
        "rigid_tolerance": RIGID_TRANSFORM_ATOL,
    }


def _write_registration_diagnostic(
    *,
    subject_t2: np.ndarray,
    subject_brain: np.ndarray,
    warped_template: np.ndarray,
    template_support: np.ndarray,
    output_path: Path,
    slice_axis: int,
    title: str,
) -> None:
    """Render subject, warped template, and support-boundary comparison rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scipy import ndimage

    n_slices = subject_t2.shape[slice_axis]
    indices = np.unique(
        np.linspace(0, n_slices - 1, min(6, n_slices), dtype=int)
    ).tolist()
    figure, axes = plt.subplots(3, len(indices), figsize=(3 * len(indices), 8))
    axes = np.asarray(axes).reshape(3, len(indices))
    subject_finite = subject_t2[np.isfinite(subject_t2)]
    template_finite = warped_template[
        np.isfinite(warped_template) & template_support.astype(bool)
    ]
    subject_limits = np.percentile(subject_finite, (1, 99))
    template_limits = np.percentile(template_finite, (1, 99))
    brain_boundary_cmap = ListedColormap(["cyan"])
    template_boundary_cmap = ListedColormap(["orange"])
    for column, index in enumerate(indices):
        subject_slice = np.take(subject_t2, index, axis=slice_axis)
        brain_slice = np.take(subject_brain, index, axis=slice_axis).astype(bool)
        template_slice = np.take(warped_template, index, axis=slice_axis)
        support_slice = np.take(template_support, index, axis=slice_axis).astype(bool)
        brain_boundary = brain_slice & ~ndimage.binary_erosion(brain_slice)
        support_boundary = support_slice & ~ndimage.binary_erosion(support_slice)

        axes[0, column].imshow(
            subject_slice.T,
            cmap="gray",
            origin="lower",
            vmin=subject_limits[0],
            vmax=subject_limits[1],
        )
        axes[1, column].imshow(
            template_slice.T,
            cmap="gray",
            origin="lower",
            vmin=template_limits[0],
            vmax=template_limits[1],
        )
        axes[2, column].imshow(
            subject_slice.T,
            cmap="gray",
            origin="lower",
            vmin=subject_limits[0],
            vmax=subject_limits[1],
        )
        axes[2, column].imshow(
            np.ma.masked_where(~brain_boundary.T, brain_boundary.T),
            cmap=brain_boundary_cmap,
            origin="lower",
            alpha=0.95,
        )
        axes[2, column].imshow(
            np.ma.masked_where(~support_boundary.T, support_boundary.T),
            cmap=template_boundary_cmap,
            origin="lower",
            alpha=0.95,
        )
        axes[0, column].set_title(f"AP slice {index}")
        for row in range(3):
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("subject T2")
    axes[1, 0].set_ylabel("warped template")
    axes[2, 0].set_ylabel("brain cyan / template orange")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def _write_all_slice_atlas_review(
    *,
    subject_t2: np.ndarray,
    subject_brain: np.ndarray,
    labels: np.ndarray,
    lesion: np.ndarray,
    template_support: np.ndarray,
    output_path: Path,
    slice_axis: int,
    title: str,
) -> None:
    """Render every acquired AP slice for the human registration decision."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scipy import ndimage

    n_slices = subject_t2.shape[slice_axis]
    n_columns = min(6, n_slices)
    n_rows = int(np.ceil(n_slices / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3 * n_columns, 3 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    finite = subject_t2[np.isfinite(subject_t2)]
    vmin, vmax = np.percentile(finite, (1, 99))
    support_cmap = ListedColormap(["lime"])
    atlas_cmap = ListedColormap(["red"])
    lesion_cmap = ListedColormap(["deepskyblue"])
    for index in range(n_slices):
        axis = flat_axes[index]
        t2_slice = np.take(subject_t2, index, axis=slice_axis)
        brain_slice = np.take(subject_brain, index, axis=slice_axis).astype(bool)
        label_slice = np.take(labels, index, axis=slice_axis)
        lesion_slice = np.take(lesion, index, axis=slice_axis).astype(bool)
        support_slice = np.take(template_support, index, axis=slice_axis).astype(bool)
        label_boundary = ndimage.maximum_filter(
            label_slice, size=3
        ) != ndimage.minimum_filter(label_slice, size=3)
        label_boundary &= label_slice != 0
        support_boundary = support_slice & ~ndimage.binary_erosion(support_slice)
        brain_voxels = np.count_nonzero(brain_slice)
        coverage = np.count_nonzero(brain_slice & support_slice) / brain_voxels

        axis.imshow(
            t2_slice.T,
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        axis.imshow(
            np.ma.masked_where(~support_boundary.T, support_boundary.T),
            cmap=support_cmap,
            origin="lower",
            alpha=0.9,
        )
        axis.imshow(
            np.ma.masked_where(~label_boundary.T, label_boundary.T),
            cmap=atlas_cmap,
            origin="lower",
            alpha=0.9,
        )
        axis.imshow(
            np.ma.masked_where(~lesion_slice.T, lesion_slice.T),
            cmap=lesion_cmap,
            origin="lower",
            alpha=0.35,
        )
        axis.set_title(f"AP {index}; brain support {coverage:.1%}")
        axis.axis("off")
    for axis in flat_axes[n_slices:]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def crop_nifti(
    *,
    image: nib.spatialimages.SpatialImage,
    data: np.ndarray,
    start: int,
    stop: int,
    axis: int,
) -> nib.Nifti1Image:
    """Crop a NIfTI while preserving the cropped voxels' world coordinates."""
    if data.ndim != 3 or axis not in {0, 1, 2}:
        raise ValueError("crop_nifti expects a three-dimensional image and spatial axis")
    if data.shape != image.shape[:3]:
        raise ValueError("crop data shape does not match the source image")
    if start < 0 or stop <= start or stop > data.shape[axis]:
        raise ValueError("invalid crop bounds")
    slicer = [slice(None)] * 3
    slicer[axis] = slice(start, stop)
    cropped = data[tuple(slicer)]
    voxel_translation = np.eye(4)
    voxel_translation[axis, 3] = start
    affine = image.affine @ voxel_translation
    header = image.header.copy()
    header.set_data_shape(cropped.shape)
    header.set_data_dtype(cropped.dtype)
    header.set_zooms(tuple(float(value) for value in image.header.get_zooms()[:3]))
    output = nib.Nifti1Image(cropped, affine, header)
    output.set_qform(affine, code=1)
    output.set_sform(affine, code=1)
    return output


def _validated_registration_input_paths(
    *,
    prepared_root: Path,
    input_summary: dict[str, Any],
) -> dict[str, Path]:
    """Return standard prepared inputs after checking their recorded hashes."""
    input_paths = {
        "subject_t2": prepared_root / "subject_t2.nii.gz",
        "subject_brain_mask": prepared_root / "subject_brain_mask.nii.gz",
        "lesion_mask": prepared_root / "lesion_model_draft_native.nii.gz",
        "subject_cost_mask": prepared_root
        / "subject_cost_mask_brain_minus_lesion.nii.gz",
        "template_slab_t2": prepared_root / "template_slab_t2.nii.gz",
        "template_slab_mask": prepared_root / "template_slab_mask.nii.gz",
        "template_slab_atlas_labels": prepared_root
        / "template_slab_atlas_labels.nii.gz",
    }
    for field, path in input_paths.items():
        expected_hash = input_summary.get(f"{field}_sha256")
        if not path.is_file():
            raise FileNotFoundError(f"Prepared registration input not found: {path}")
        if not expected_hash or sha256(path) != expected_hash:
            raise ValueError(f"Prepared registration input is stale or changed: {field}")
    return input_paths


def _ap_axis(orientation: str) -> int:
    axes = [axis for axis, code in enumerate(orientation) if code in {"A", "P"}]
    if len(axes) != 1:
        raise ValueError(f"Expected exactly one AP axis in orientation {orientation!r}")
    return axes[0]


def _prepare_output_root(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)
