"""Synthetic tests for the T2w-to-Allen mapping adapters."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.atlas import t2w_mapping
from src.atlas.aidamri_lookup import normalize_parental_structure_lookup
from src.atlas.constrained_registration import (
    affine_transform_metrics,
    crop_nifti,
    prepare_constrained_registration_inputs,
    rigid_transform_metrics,
    validate_constrained_affine_registration,
    validate_constrained_rigid_registration,
)
from src.atlas.nonlinear_diagnostic import validate_nonlinear_diagnostic
from src.atlas.partial_volume import (
    load_approved_slab_selection,
    prepare_partial_volume_slab_review,
    rank_slab_candidates,
)
from src.atlas.t2w_mapping import (
    load_label_image,
    prepare_aidamri_inputs,
    sha256,
    summarize_atlas_mappings,
    write_qc_overlay,
)

LIP_AFFINE = np.array(
    [
        [-0.07, 0.0, 0.0, 10.0],
        [0.0, 0.0, -0.5, 5.0],
        [0.0, -0.07, 0.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray = LIP_AFFINE) -> Path:
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _source_inference_manifest(tmp_path: Path, *, affine: np.ndarray = LIP_AFFINE) -> Path:
    scan = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3, 1)
    lesion = np.zeros((4, 5, 3), dtype=np.uint8)
    lesion[1:3, 2:4, 1] = 1
    scan_path = _write_nifti(tmp_path / "scan.nii.gz", scan, affine)
    lesion_path = _write_nifti(tmp_path / "ensemble_mask.nii.gz", lesion, affine)
    return _write_csv(
        tmp_path / "inference_manifest.csv",
        [
            {
                "case_id": "case_01",
                "input_scan": str(scan_path),
                "ensemble_mask": str(lesion_path),
            }
        ],
    )


def _mapping_inputs(tmp_path: Path) -> dict[str, Path]:
    t2 = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    lesion = np.zeros_like(t2, dtype=np.uint8)
    lesion[1, 1, 1] = 1
    lesion[2, 2, 1] = 1
    lesion[3, 3, 1] = 1
    labels = np.zeros_like(t2, dtype=np.int16)
    labels[:2] = 1
    labels[2:3] = 2
    t2_path = _write_nifti(tmp_path / "native_t2.nii.gz", t2)
    lesion_path = _write_nifti(tmp_path / "lesion.nii.gz", lesion)
    labels_path = _write_nifti(tmp_path / "labels.nii.gz", labels)
    structures_path = _write_csv(
        tmp_path / "structures.csv",
        [
            {"label_id": 1, "acronym": "R1", "name": "Region one", "hemisphere": "left"},
            {"label_id": 2, "acronym": "R2", "name": "Region two", "hemisphere": "right"},
        ],
    )
    orientation_path = tmp_path / "orientation_review.json"
    orientation_path.write_text(
        json.dumps(
            {
                "case_id": "case_01",
                "orientation_status": "approved",
                "anatomical_orientation_matches_header": True,
                "observed_header_orientation": "LIP",
                "native_t2_sha256": sha256(t2_path),
                "lesion_mask_sha256": sha256(lesion_path),
                "reviewer": "reviewer",
                "review_date": "2026-07-21",
            }
        )
    )
    return {
        "t2": t2_path,
        "lesion": lesion_path,
        "labels": labels_path,
        "structures": structures_path,
        "orientation": orientation_path,
    }


def _mapping_manifest(
    tmp_path: Path,
    inputs: dict[str, Path],
    *,
    registration_review: Path | None = None,
    interpolation: str = "nearest_neighbor",
) -> Path:
    return _write_csv(
        tmp_path / "mapping_manifest.csv",
        [
            {
                "case_id": "case_01",
                "native_t2": str(inputs["t2"]),
                "lesion_mask": str(inputs["lesion"]),
                "orientation_review_json": str(inputs["orientation"]),
                "native_atlas_labels": str(inputs["labels"]),
                "structures_csv": str(inputs["structures"]),
                "atlas_id": "AIDAmri_ARA",
                "atlas_version": "CCFv3_50um",
                "coordinate_space": "Allen Mouse Brain CCFv3",
                "annotation_variant": "split_parental",
                "registration_backend": "AIDAmri",
                "registration_version": "3.0",
                "registration_revision": "example-revision",
                "label_interpolation": interpolation,
                "registration_qc_json": ""
                if registration_review is None
                else str(registration_review),
                "lesion_mask_status": "model_draft",
                "lesion_mask_review_json": "",
                "lesion_mask_reviewer": "",
            }
        ],
    )


def _stub_qc(monkeypatch: pytest.MonkeyPatch) -> None:
    def write_stub(**kwargs):
        kwargs["output_path"].write_bytes(b"synthetic-qc")

    monkeypatch.setattr(t2w_mapping, "write_qc_overlay", write_stub)


def test_prepare_aidamri_inputs_squeezes_channel_and_keeps_pending_review(tmp_path: Path):
    manifest = _source_inference_manifest(tmp_path)
    output = tmp_path / "prepared"

    summary = prepare_aidamri_inputs(inference_manifest=manifest, output_root=output)

    assert summary["n_cases"] == 1
    staged = nib.load(str(output / "cases/case_01/t2_for_aidamri.nii.gz"))
    assert staged.shape == (4, 5, 3)
    assert nib.aff2axcodes(staged.affine) == ("L", "I", "P")
    review = json.loads((output / "cases/case_01/orientation_review.json").read_text())
    assert review["orientation_status"] == "pending"
    assert review["anatomical_orientation_matches_header"] is None
    rows = list(csv.DictReader((output / "atlas_mapping_manifest.csv").open()))
    assert rows[0]["lesion_mask_status"] == "model_draft"
    assert rows[0]["native_atlas_labels"] == ""


def test_prepare_aidamri_inputs_rejects_non_lip_header(tmp_path: Path):
    ras_affine = np.diag([0.07, 0.07, 0.5, 1.0])
    manifest = _source_inference_manifest(tmp_path, affine=ras_affine)

    with pytest.raises(ValueError, match="Do not relabel the header"):
        prepare_aidamri_inputs(
            inference_manifest=manifest,
            output_root=tmp_path / "prepared",
        )


def test_summarize_outputs_native_overlap_and_pending_registration_qc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _stub_qc(monkeypatch)
    inputs = _mapping_inputs(tmp_path)
    manifest = _mapping_manifest(tmp_path, inputs)
    output = tmp_path / "summary"

    summary = summarize_atlas_mappings(manifest=manifest, output_root=output)

    assert summary["n_approved"] == 0
    assert summary["n_needing_review"] == 1
    case_summary = json.loads((output / "cases/case_01/mapping_summary.json").read_text())
    assert case_summary["combined_qc_status"] == "needs_human_review"
    assert case_summary["atlas_coverage_fraction_of_lesion"] == pytest.approx(2 / 3)
    assert case_summary["predictions_are_drafts"] is True
    affected = list(csv.DictReader((output / "cases/case_01/affected_regions.csv").open()))
    assert {row["acronym"] for row in affected} == {"R1", "R2", "unmapped"}
    by_acronym = {row["acronym"]: row for row in affected}
    assert float(by_acronym["R1"]["lesion_overlap_volume_mm3"]) == pytest.approx(
        0.07 * 0.07 * 0.5
    )
    assert (output / "cases/case_01/atlas_mapping_qc.png").read_bytes() == b"synthetic-qc"


def test_require_approved_registration_validates_hash_bound_signoffs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _stub_qc(monkeypatch)
    inputs = _mapping_inputs(tmp_path)
    review = tmp_path / "registration_review.json"
    review.write_text(
        json.dumps(
            {
                "case_id": "case_01",
                "registration_qc_status": "approved",
                "atlas_id": "AIDAmri_ARA",
                "atlas_version": "CCFv3_50um",
                "coordinate_space": "Allen Mouse Brain CCFv3",
                "annotation_variant": "split_parental",
                "registration_backend": "AIDAmri",
                "registration_version": "3.0",
                "registration_revision": "example-revision",
                "native_atlas_labels_sha256": sha256(inputs["labels"]),
                "reviewer": "reviewer",
                "review_date": "2026-07-21",
            }
        )
    )
    manifest = _mapping_manifest(tmp_path, inputs, registration_review=review)

    summary = summarize_atlas_mappings(
        manifest=manifest,
        output_root=tmp_path / "approved",
        require_approved_registration=True,
    )

    assert summary["n_approved"] == 1
    review_record = json.loads(review.read_text())
    review_record["native_atlas_labels_sha256"] = "stale"
    review.write_text(json.dumps(review_record))
    with pytest.raises(ValueError, match="stale or incorrect"):
        summarize_atlas_mappings(
            manifest=manifest,
            output_root=tmp_path / "stale",
            require_approved_registration=True,
        )


def test_summarize_requires_nearest_neighbour_labels(tmp_path: Path):
    inputs = _mapping_inputs(tmp_path)
    manifest = _mapping_manifest(tmp_path, inputs, interpolation="linear")

    with pytest.raises(ValueError, match="must be nearest_neighbor"):
        summarize_atlas_mappings(manifest=manifest, output_root=tmp_path / "summary")


def test_lookup_must_cover_every_observed_nonzero_label(tmp_path: Path):
    inputs = _mapping_inputs(tmp_path)
    _write_csv(
        inputs["structures"],
        [{"label_id": 1, "acronym": "R1", "name": "Region one"}],
    )
    manifest = _mapping_manifest(tmp_path, inputs)

    with pytest.raises(ValueError, match="missing lookup rows.*2"):
        summarize_atlas_mappings(manifest=manifest, output_root=tmp_path / "summary")


def test_human_reviewed_mask_requires_hash_bound_review(tmp_path: Path):
    inputs = _mapping_inputs(tmp_path)
    manifest = _mapping_manifest(tmp_path, inputs)
    rows = list(csv.DictReader(manifest.open()))
    rows[0]["lesion_mask_status"] = "human_reviewed"
    rows[0]["lesion_mask_reviewer"] = "reviewer"
    _write_csv(manifest, rows)

    with pytest.raises(ValueError, match="require lesion_mask_review_json"):
        summarize_atlas_mappings(manifest=manifest, output_root=tmp_path / "summary")


def test_non_integer_atlas_labels_are_rejected(tmp_path: Path):
    labels = np.zeros((4, 5, 3), dtype=np.float32)
    labels[1, 1, 1] = 1.25
    path = _write_nifti(tmp_path / "labels.nii.gz", labels)

    with pytest.raises(ValueError, match="nearest-neighbour"):
        load_label_image(path)


def test_qc_overlay_renders_a_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    t2 = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    labels = np.zeros_like(t2, dtype=np.int16)
    labels[:2] = 1
    lesion = np.zeros_like(t2, dtype=np.uint8)
    lesion[1, 1, 1] = 1
    output = tmp_path / "qc.png"

    write_qc_overlay(
        t2=t2,
        labels=labels,
        lesion=lesion,
        output_path=output,
        slice_axis=2,
        title="synthetic atlas QC",
    )

    assert output.read_bytes().startswith(b"\x89PNG")


def test_normalize_parental_structure_lookup_pairs_hemispheres(tmp_path: Path):
    names = tmp_path / "names.txt"
    names.write_text(
        "22\tL_Posterior_parietal_association_areas\n"
        "31\tL_Anterior_cingulate_area\n"
        "2022\tR_Posterior_parietal_association_areas\n"
        "2031\tR_Anterior_cingulate_area\n"
    )
    acronyms = tmp_path / "acronyms.txt"
    acronyms.write_text("22\t\tPTLp\n31\t\tACA\n")
    output = tmp_path / "structures.csv"

    summary = normalize_parental_structure_lookup(
        names_table=names,
        acronyms_table=acronyms,
        output_csv=output,
        aidamri_revision="abc123",
    )

    rows = list(csv.DictReader(output.open()))
    assert summary["n_base_regions"] == 2
    assert summary["n_hemisphere_split_regions"] == 4
    assert rows[0] == {
        "label_id": "22",
        "acronym": "PTLp",
        "name": "Posterior parietal association areas",
        "hemisphere": "left",
    }
    assert rows[-1]["label_id"] == "2031"
    assert rows[-1]["hemisphere"] == "right"
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    assert provenance["aidamri_revision"] == "abc123"
    assert provenance["output_csv_sha256"] == sha256(output)


def test_normalize_parental_structure_lookup_rejects_unpaired_labels(tmp_path: Path):
    names = tmp_path / "names.txt"
    names.write_text("22\tL_Region\n2023\tR_Other_region\n")
    acronyms = tmp_path / "acronyms.txt"
    acronyms.write_text("22\t\tREG\n")

    with pytest.raises(ValueError, match="left/right parental label pairs"):
        normalize_parental_structure_lookup(
            names_table=names,
            acronyms_table=acronyms,
            output_csv=tmp_path / "structures.csv",
            aidamri_revision="abc123",
        )


def test_rank_partial_slab_candidates_recovers_known_start():
    template_profile = np.array(
        [1, 2, 3, 5, 8, 12, 15, 14, 10, 7, 4, 2, 1], dtype=np.float64
    )
    known_start = 3
    subject_indices = [known_start, known_start + 2, known_start + 4, known_start + 6]
    subject_profile = template_profile[subject_indices]

    candidates = rank_slab_candidates(
        subject_profile_mm2=subject_profile,
        template_profile_mm2=template_profile,
        subject_slice_spacing_mm=1.0,
        template_slice_spacing_mm=0.5,
    )

    assert candidates[0].template_start_index == known_start
    assert candidates[0].template_slice_indices == (3, 5, 7, 9)
    assert candidates[0].area_profile_rmse == pytest.approx(0.0)


def test_partial_slab_review_requires_hash_bound_human_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    subject_affine = LIP_AFFINE.copy()
    subject = np.zeros((8, 7, 4), dtype=np.float32)
    subject[2:6, 2:5, :] = 1
    brain = (subject != 0).astype(np.uint8)
    lesion = np.zeros_like(brain)
    lesion[2:3, 2:3, 1] = 1
    template_affine = LIP_AFFINE.copy()
    template_affine[:3, :3] *= 0.5
    template = np.zeros((8, 7, 12), dtype=np.float32)
    widths = [1, 2, 3, 4, 4, 3, 2, 1, 1, 1, 1, 1]
    for index, width in enumerate(widths):
        template[1 : 1 + width, 2:5, index] = 1
    subject_path = _write_nifti(tmp_path / "subject.nii.gz", subject, subject_affine)
    brain_path = _write_nifti(tmp_path / "brain.nii.gz", brain, subject_affine)
    lesion_path = _write_nifti(tmp_path / "lesion.nii.gz", lesion, subject_affine)
    template_path = _write_nifti(tmp_path / "template.nii.gz", template, template_affine)
    output = tmp_path / "slab_review"

    summary = prepare_partial_volume_slab_review(
        case_id="case_01",
        subject_t2=subject_path,
        subject_brain_mask=brain_path,
        lesion_mask=lesion_path,
        template_t2=template_path,
        output_root=output,
        top_candidates=2,
        minimum_candidate_separation_mm=0.5,
    )

    assert summary["scientific_status"] == "requires_human_ap_slab_selection"
    assert summary["ranking_is_anatomical_approval"] is False
    selection_path = output / "slab_selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["selection_status"] == "pending"
    with pytest.raises(ValueError, match="not approved"):
        load_approved_slab_selection(
            selection_json=selection_path,
            candidates_csv=output / "slab_candidates.csv",
            subject_t2=subject_path,
            subject_brain_mask=brain_path,
            lesion_mask=lesion_path,
            template_t2=template_path,
        )

    selected = summary["review_candidates"][0]
    selection.update(
        {
            "selection_status": "approved",
            "selected_candidate_id": selected["candidate_id"],
            "selected_template_start_index": selected["template_start_index"],
            "selected_template_end_index": selected["template_end_index"],
            "anatomical_landmarks_confirmed": True,
            "reviewer": "reviewer",
            "review_date": "2026-07-22",
        }
    )
    selection_path.write_text(json.dumps(selection))
    loaded = load_approved_slab_selection(
        selection_json=selection_path,
        candidates_csv=output / "slab_candidates.csv",
        subject_t2=subject_path,
        subject_brain_mask=brain_path,
        lesion_mask=lesion_path,
        template_t2=template_path,
    )
    assert loaded.candidate_id == selected["candidate_id"]


def test_crop_nifti_preserves_world_coordinates(tmp_path: Path):
    data = np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8)
    image = nib.Nifti1Image(data, LIP_AFFINE)

    cropped = crop_nifti(image=image, data=data, start=2, stop=7, axis=2)

    assert cropped.shape == (4, 5, 5)
    assert np.array_equal(np.asanyarray(cropped.dataobj), data[:, :, 2:7])
    source_world = nib.affines.apply_affine(image.affine, [1, 2, 2])
    cropped_world = nib.affines.apply_affine(cropped.affine, [1, 2, 0])
    assert np.allclose(source_world, cropped_world)


def test_constrained_registration_padding_keeps_selected_centres_away_from_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    subject = np.arange(6 * 7 * 3, dtype=np.float32).reshape(6, 7, 3) + 1
    brain = np.ones_like(subject, dtype=np.uint8)
    lesion = np.zeros_like(brain)
    template_affine = LIP_AFFINE.copy()
    template_affine[:3, :3] *= 0.5
    template = np.arange(6 * 7 * 16, dtype=np.float32).reshape(6, 7, 16) + 1
    labels = np.ones_like(template, dtype=np.int16)
    subject_path = _write_nifti(tmp_path / "subject.nii.gz", subject)
    brain_path = _write_nifti(tmp_path / "brain.nii.gz", brain)
    lesion_path = _write_nifti(tmp_path / "lesion.nii.gz", lesion)
    template_path = _write_nifti(tmp_path / "template.nii.gz", template, template_affine)
    labels_path = _write_nifti(tmp_path / "labels.nii.gz", labels, template_affine)
    review = tmp_path / "review"
    prepare_partial_volume_slab_review(
        case_id="case_01",
        subject_t2=subject_path,
        subject_brain_mask=brain_path,
        lesion_mask=lesion_path,
        template_t2=template_path,
        output_root=review,
        top_candidates=1,
    )
    candidates = list(csv.DictReader((review / "slab_candidates.csv").open()))
    selected_row = next(
        row
        for row in candidates
        if int(row["template_start_index"]) >= 1
        and int(row["template_end_index"]) <= template.shape[2] - 2
    )
    selected = {
        "candidate_id": selected_row["candidate_id"],
        "template_start_index": int(selected_row["template_start_index"]),
        "template_end_index": int(selected_row["template_end_index"]),
    }
    selection_path = review / "slab_selection.json"
    selection = json.loads(selection_path.read_text())
    selection.update(
        {
            "selection_status": "approved",
            "selected_candidate_id": selected["candidate_id"],
            "selected_template_start_index": selected["template_start_index"],
            "selected_template_end_index": selected["template_end_index"],
            "anatomical_landmarks_confirmed": True,
            "reviewer": "reviewer",
            "review_date": "2026-07-22",
        }
    )
    selection_path.write_text(json.dumps(selection))

    prepared = prepare_constrained_registration_inputs(
        selection_json=selection_path,
        candidates_csv=review / "slab_candidates.csv",
        subject_t2=subject_path,
        subject_brain_mask=brain_path,
        lesion_mask=lesion_path,
        template_t2=template_path,
        template_atlas_labels=labels_path,
        output_root=tmp_path / "prepared",
        slab_edge_padding_mm=0.25,
    )

    assert prepared["slab_edge_padding_template_voxels"] == 1
    assert prepared["template_crop_start_index"] == (
        prepared["template_start_index"] - 1
    )
    assert prepared["template_crop_end_index"] == prepared["template_end_index"] + 1


def test_prepare_constrained_registration_excludes_lesion_from_cost_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    subject = np.arange(6 * 7 * 4, dtype=np.float32).reshape(6, 7, 4) + 1
    brain = np.ones_like(subject, dtype=np.uint8)
    lesion = np.zeros_like(brain)
    lesion[2:4, 3:5, 1:3] = 1
    template_affine = LIP_AFFINE.copy()
    template_affine[:3, :3] *= 0.5
    template = np.arange(6 * 7 * 12, dtype=np.float32).reshape(6, 7, 12) + 1
    labels = np.ones_like(template, dtype=np.int16)
    subject_path = _write_nifti(tmp_path / "subject.nii.gz", subject)
    brain_path = _write_nifti(tmp_path / "brain.nii.gz", brain)
    lesion_path = _write_nifti(tmp_path / "lesion.nii.gz", lesion)
    template_path = _write_nifti(tmp_path / "template.nii.gz", template, template_affine)
    labels_path = _write_nifti(tmp_path / "labels.nii.gz", labels, template_affine)
    review = tmp_path / "review"
    summary = prepare_partial_volume_slab_review(
        case_id="case_01",
        subject_t2=subject_path,
        subject_brain_mask=brain_path,
        lesion_mask=lesion_path,
        template_t2=template_path,
        output_root=review,
        top_candidates=1,
    )
    selected = summary["review_candidates"][0]
    selection_path = review / "slab_selection.json"
    selection = json.loads(selection_path.read_text())
    selection.update(
        {
            "selection_status": "approved",
            "selected_candidate_id": selected["candidate_id"],
            "selected_template_start_index": selected["template_start_index"],
            "selected_template_end_index": selected["template_end_index"],
            "anatomical_landmarks_confirmed": True,
            "reviewer": "reviewer",
            "review_date": "2026-07-22",
        }
    )
    selection_path.write_text(json.dumps(selection))

    prepared = prepare_constrained_registration_inputs(
        selection_json=selection_path,
        candidates_csv=review / "slab_candidates.csv",
        subject_t2=subject_path,
        subject_brain_mask=brain_path,
        lesion_mask=lesion_path,
        template_t2=template_path,
        template_atlas_labels=labels_path,
        output_root=tmp_path / "prepared",
    )

    cost = nib.load(prepared["subject_cost_mask"]).get_fdata()
    assert np.count_nonzero(cost) == np.count_nonzero(brain) - np.count_nonzero(lesion)
    assert prepared["planned_registration"]["transformation"] == "rigid_only"


def test_rigid_transform_metrics_rejects_scale(tmp_path: Path):
    rigid = np.eye(4)
    angle = np.deg2rad(7.0)
    rigid[:2, :2] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    rigid[:3, 3] = [0.1, -0.2, 0.3]
    rigid_path = tmp_path / "rigid.txt"
    np.savetxt(rigid_path, rigid)

    metrics = rigid_transform_metrics(rigid_path)

    assert metrics["rigid_integrity_check"] == "passed"
    assert metrics["rotation_angle_degrees"] == pytest.approx(7.0)
    scaled = rigid.copy()
    scaled[0, 0] *= 1.1
    scaled_path = tmp_path / "scaled.txt"
    np.savetxt(scaled_path, scaled)
    with pytest.raises(ValueError, match="scale or shear"):
        rigid_transform_metrics(scaled_path)


def test_affine_transform_metrics_reports_stretch_and_rejects_reflection(
    tmp_path: Path,
):
    affine = np.eye(4)
    affine[:3, :3] = np.diag([1.1, 1.0, 0.9])
    affine_path = tmp_path / "affine.txt"
    np.savetxt(affine_path, affine)

    metrics = affine_transform_metrics(affine_path)

    assert metrics["determinant"] == pytest.approx(0.99)
    assert metrics["principal_stretches"] == pytest.approx([1.1, 1.0, 0.9])
    assert metrics["automatic_distortion_acceptance"] == "not_claimed"
    reflected = affine.copy()
    reflected[0, 0] *= -1
    reflected_path = tmp_path / "reflected.txt"
    np.savetxt(reflected_path, reflected)
    with pytest.raises(ValueError, match="reflection"):
        affine_transform_metrics(reflected_path)


def test_validate_constrained_registration_emits_pending_hash_bound_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    subject = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3) + 1
    brain = np.ones_like(subject, dtype=np.uint8)
    lesion = np.zeros_like(brain)
    lesion[1, 2, 1] = 1
    support = np.ones_like(brain)
    labels = np.ones_like(brain, dtype=np.int16)
    paths_and_data = {
        "subject_t2.nii.gz": subject,
        "subject_brain_mask.nii.gz": brain,
        "lesion_model_draft_native.nii.gz": lesion,
        "subject_cost_mask_brain_minus_lesion.nii.gz": brain & ~lesion,
        "template_slab_t2.nii.gz": subject,
        "template_slab_mask.nii.gz": support,
        "template_slab_atlas_labels.nii.gz": labels,
        "template_slab_rigid_in_subject.nii.gz": subject,
        "template_slab_mask_rigid_in_subject.nii.gz": support,
        "template_slab_atlas_labels_rigid_in_subject.nii.gz": labels,
    }
    for name, data in paths_and_data.items():
        _write_nifti(prepared / name, data)
    input_fields = {
        "subject_t2": "subject_t2.nii.gz",
        "subject_brain_mask": "subject_brain_mask.nii.gz",
        "lesion_mask": "lesion_model_draft_native.nii.gz",
        "subject_cost_mask": "subject_cost_mask_brain_minus_lesion.nii.gz",
        "template_slab_t2": "template_slab_t2.nii.gz",
        "template_slab_mask": "template_slab_mask.nii.gz",
        "template_slab_atlas_labels": "template_slab_atlas_labels.nii.gz",
    }
    input_summary = {
        "case_id": "case_01",
        "selected_candidate_id": "template_start_0001",
        "selection_json_sha256": "selection-hash",
        "planned_registration": {"transformation": "rigid_only"},
    }
    for field, name in input_fields.items():
        input_summary[f"{field}_sha256"] = sha256(prepared / name)
    (prepared / "registration_input_summary.json").write_text(
        json.dumps(input_summary)
    )
    np.savetxt(prepared / "rigid_transform.txt", np.eye(4))
    structures = _write_csv(
        tmp_path / "structures.csv",
        [{"label_id": 1, "acronym": "REG", "name": "Region"}],
    )

    result = validate_constrained_rigid_registration(
        prepared_root=prepared,
        structures_csv=structures,
        aidamri_revision="aidamri-revision",
        niftyreg_revision="niftyreg-revision",
        container_image_id="sha256:image",
        registration_runtime_seconds=12.0,
    )

    assert result["scientific_status"] == "requires_human_registration_review"
    assert result["brain_coverage_fraction"] == pytest.approx(1.0)
    assert result["transform_metrics"]["rigid_integrity_check"] == "passed"
    assert Path(result["registration_qc_png"]).read_bytes().startswith(b"\x89PNG")
    assert Path(result["all_slices_qc_png"]).read_bytes().startswith(b"\x89PNG")
    review = json.loads((prepared / "rigid_registration_review.json").read_text())
    assert review["registration_qc_status"] == "pending"
    assert review["native_atlas_labels_sha256"] == sha256(
        prepared / "template_slab_atlas_labels_rigid_in_subject.nii.gz"
    )

    _write_nifti(prepared / "template_slab_affine_in_subject.nii.gz", subject)
    _write_nifti(
        prepared / "template_slab_mask_affine_in_subject.nii.gz", support
    )
    _write_nifti(
        prepared / "template_slab_atlas_labels_affine_in_subject.nii.gz", labels
    )
    affine = np.eye(4)
    affine[0, 0] = 1.05
    np.savetxt(prepared / "affine_transform.txt", affine)
    affine_result = validate_constrained_affine_registration(
        prepared_root=prepared,
        structures_csv=structures,
        aidamri_revision="aidamri-revision",
        niftyreg_revision="niftyreg-revision",
        container_image_id="sha256:image",
        registration_runtime_seconds=15.0,
    )
    assert affine_result["scientific_status"] == (
        "requires_human_affine_diagnostic_review"
    )
    assert affine_result["technical_validation_status"] == (
        "passed_file_geometry_and_nonzero_slice_support"
    )
    assert affine_result["brain_coverage_change_from_rigid"] == pytest.approx(0.0)
    assert affine_result["transform_metrics"]["principal_stretches"] == pytest.approx(
        [1.05, 1.0, 1.0]
    )
    assert Path(affine_result["all_slices_qc_png"]).read_bytes().startswith(b"\x89PNG")

    _write_nifti(prepared / "template_slab_nonlinear_in_subject.nii.gz", subject)
    _write_nifti(
        prepared / "template_slab_mask_nonlinear_in_subject.nii.gz", support
    )
    _write_nifti(
        prepared / "template_slab_atlas_labels_nonlinear_in_subject.nii.gz", labels
    )
    _write_nifti(prepared / "nonlinear_cpp.nii.gz", np.ones_like(subject))
    _write_nifti(prepared / "nonlinear_jacobian.nii.gz", np.ones_like(subject))
    nonlinear_result = validate_nonlinear_diagnostic(
        prepared_root=prepared,
        structures_csv=structures,
        diagnostic_id="standard_f3d",
        aidamri_revision="aidamri-revision",
        niftyreg_revision="niftyreg-revision",
        container_image_id="sha256:image",
        registration_runtime_seconds=18.0,
    )
    assert nonlinear_result["scientific_status"] == (
        "requires_human_nonlinear_diagnostic_review"
    )
    assert nonlinear_result["jacobian_metrics"]["nonpositive_voxels"] == 0
    assert nonlinear_result["jacobian_metrics"]["brain_minimum"] == pytest.approx(1.0)
    assert nonlinear_result["brain_coverage_change_from_rigid"] == pytest.approx(0.0)
    assert nonlinear_result["lesion_coverage_partition"] == {
        "mapped_nonzero_label_voxels": 1,
        "mapped_nonzero_label_fraction": 1.0,
        "inside_template_support_zero_label_voxels": 0,
        "inside_template_support_zero_label_fraction": 0.0,
        "nonzero_label_outside_template_support_voxels": 0,
        "nonzero_label_outside_template_support_fraction": 0.0,
        "zero_label_outside_template_support_voxels": 0,
        "zero_label_outside_template_support_fraction": 0.0,
    }
    assert Path(nonlinear_result["lesion_coverage_qc_png"]).read_bytes().startswith(
        b"\x89PNG"
    )
    nonlinear_review = json.loads(
        (prepared / "standard_f3d_registration_review.json").read_text()
    )
    assert nonlinear_review["registration_qc_status"] == "pending"

    review["registration_qc_status"] = "approved"
    review["reviewer"] = "reviewer"
    review["review_date"] = "2026-07-22"
    (prepared / "rigid_registration_review.json").write_text(json.dumps(review))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        validate_constrained_rigid_registration(
            prepared_root=prepared,
            structures_csv=structures,
            aidamri_revision="aidamri-revision",
            niftyreg_revision="niftyreg-revision",
            container_image_id="sha256:image",
            registration_runtime_seconds=12.0,
        )
