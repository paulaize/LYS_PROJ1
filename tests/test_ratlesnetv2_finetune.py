import csv
import struct
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from ratlesnetv2_finetune.commands import build_cloud_command_plan, format_command
from ratlesnetv2_finetune.dataset import prepare_dataset
from ratlesnetv2_finetune.roiset_to_nifti_mask import convert_roiset_to_nifti_mask
from ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 import _patch_nibabel_get_data_compat
from ratlesnetv2_finetune.scripts.flip_external_si_axis import flip_pair_axis
from ratlesnetv2_finetune.scripts.orient_external_dataset_lsp import (
    affine_for_axcodes,
    relabel_pair_to_axcodes,
)
from ratlesnetv2_finetune.scripts.review_lys_masks_itksnap import (
    case_from_mask,
    default_output_root,
    find_cases,
    prepare_case,
    validate_grid,
    wait_for_next_case,
)
from ratlesnetv2_finetune.source_folders import add_source_folder_to_plan

SPACING = (0.07, 0.07, 0.5)


def _write_nifti(path: Path, data: np.ndarray, spacing=SPACING) -> None:
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms(spacing[: data.ndim])
    nib.save(img, path)


def _imagej_polygon_roi_bytes(
    *,
    points: list[tuple[int, int]],
    position: int = 0,
    roi_type: int = 7,
) -> bytes:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = min(xs)
    right = max(xs) + 1
    top = min(ys)
    bottom = max(ys) + 1
    n = len(points)
    data = bytearray(64 + 4 * n)
    data[:4] = b"Iout"
    struct.pack_into(">H", data, 4, 227)
    data[6] = roi_type
    struct.pack_into(">h", data, 8, top)
    struct.pack_into(">h", data, 10, left)
    struct.pack_into(">h", data, 12, bottom)
    struct.pack_into(">h", data, 14, right)
    struct.pack_into(">H", data, 16, n)
    struct.pack_into(">i", data, 56, position)
    for index, (x, _y) in enumerate(points):
        struct.pack_into(">H", data, 64 + 2 * index, x - left)
    y_base = 64 + 2 * n
    for index, (_x, y) in enumerate(points):
        struct.pack_into(">H", data, y_base + 2 * index, y - top)
    return bytes(data)


def _write_roiset(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)


def _write_plan(tmp_path: Path, scan: Path, mask: Path) -> Path:
    plan = {
        "dataset_name": "unit_dataset",
        "output_root": str(tmp_path / "prepared"),
        "preprocessing": {
            "expected_spacing_mm": list(SPACING),
            "spacing_tolerance_mm": 0.001,
            "label_threshold": 0.5,
        },
        "ratlesnetv2": {
            "scan_filename": "scan.nii.gz",
            "label_filename": "scan_lesionIAM.nii.gz",
            "write_readme_label_alias": True,
        },
        "splits": {
            "train": [
                {
                    "animal_id": "A1",
                    "study": "LYS",
                    "timepoint": "5d",
                    "scan_nifti": str(scan),
                    "lesion_mask": str(mask),
                }
            ],
            "validation": [],
            "test": [],
        },
    }
    path = tmp_path / "ratlesnet_plan.yml"
    path.write_text(yaml.safe_dump(plan))
    return path


def _write_empty_plan(tmp_path: Path) -> Path:
    plan = {
        "dataset_name": "unit_dataset",
        "output_root": str(tmp_path / "prepared"),
        "preprocessing": {
            "expected_spacing_mm": list(SPACING),
            "spacing_tolerance_mm": 0.001,
            "label_threshold": 0.5,
        },
        "ratlesnetv2": {
            "scan_filename": "scan.nii.gz",
            "label_filename": "scan_lesionIAM.nii.gz",
            "write_readme_label_alias": True,
        },
        "splits": {"train": [], "validation": [], "test": []},
    }
    path = tmp_path / "ratlesnet_plan.yml"
    path.write_text(yaml.safe_dump(plan))
    return path


def test_prepare_dataset_writes_4d_scan_label_and_manifest(tmp_path):
    scan = tmp_path / "scan.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    _write_nifti(scan, np.ones((8, 9, 3), dtype=np.float32))
    label = np.zeros((8, 9, 3), dtype=np.uint8)
    label[2:4, 3:6, 1] = 1
    _write_nifti(mask, label)
    plan = _write_plan(tmp_path, scan, mask)

    dataset_root, prepared = prepare_dataset(plan, repo_root=tmp_path)

    assert len(prepared) == 1
    case = prepared[0]
    assert case.image_shape == (8, 9, 3, 1)
    assert case.label_voxels == 6
    assert case.lesion_volume_mm3 == pytest.approx(6 * 0.07 * 0.07 * 0.5)
    assert (case.case_dir / "scan.nii.gz").exists()
    assert (case.case_dir / "scan_lesionIAM.nii.gz").exists()
    assert (case.case_dir / "scan_lesion.nii.gz").exists()

    saved_scan = nib.load(case.scan_path)
    saved_label = nib.load(case.label_path)
    assert saved_scan.shape == (8, 9, 3, 1)
    assert saved_label.shape == (8, 9, 3)
    assert set(np.unique(np.asanyarray(saved_label.dataobj))) == {0, 1}

    with (dataset_root / "manifest.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["animal_id"] == "A1"
    assert rows[0]["image_shape"] == "8x9x3x1"


def test_nibabel_get_data_compat_patch_restores_upstream_loader_call():
    img = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.float32), np.eye(4))

    _patch_nibabel_get_data_compat()

    loaded = img.get_data()
    assert loaded.shape == (2, 3, 4)
    assert np.allclose(loaded, 1)


def test_prepare_dataset_rejects_spacing_mismatch(tmp_path):
    scan = tmp_path / "scan_bad_spacing.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    _write_nifti(scan, np.ones((8, 9, 3), dtype=np.float32), spacing=(0.1, 0.1, 0.5))
    _write_nifti(mask, np.zeros((8, 9, 3), dtype=np.uint8), spacing=(0.1, 0.1, 0.5))
    plan = _write_plan(tmp_path, scan, mask)

    with pytest.raises(ValueError, match="spacing check failed"):
        prepare_dataset(plan, repo_root=tmp_path)


def test_prepare_dataset_rejects_label_shape_mismatch(tmp_path):
    scan = tmp_path / "scan.nii.gz"
    mask = tmp_path / "mask_bad_shape.nii.gz"
    _write_nifti(scan, np.ones((8, 9, 3), dtype=np.float32))
    _write_nifti(mask, np.zeros((8, 9, 4), dtype=np.uint8))
    plan = _write_plan(tmp_path, scan, mask)

    with pytest.raises(ValueError, match="does not match scan shape"):
        prepare_dataset(plan, repo_root=tmp_path)


def test_add_source_folder_updates_plan_and_prepares_dataset(tmp_path):
    source_root = tmp_path / "incoming_24h"
    scan_dir = source_root / "T2w"
    mask_dir = source_root / "masks"
    scan_dir.mkdir(parents=True)
    mask_dir.mkdir()

    for case_id in ["BD_01_24h", "BD_02_24h"]:
        _write_nifti(scan_dir / f"{case_id}.nii.gz", np.ones((8, 9, 3), dtype=np.float32))
        label = np.zeros((8, 9, 3), dtype=np.uint8)
        label[1:3, 2:4, 1] = 1
        _write_nifti(mask_dir / f"{case_id}_lesion_mask.nii.gz", label)

    plan = _write_empty_plan(tmp_path)
    result = add_source_folder_to_plan(
        plan_path=plan,
        source_root=source_root,
        split="train",
        study="LYS",
        timepoint="24h",
        repo_root=tmp_path,
    )

    assert result.added == 2
    assert result.updated == 0
    assert result.unchanged == 0
    loaded = yaml.safe_load(plan.read_text())
    assert [case["case_id"] for case in loaded["splits"]["train"]] == ["BD_01_24h", "BD_02_24h"]
    assert loaded["splits"]["train"][0]["scan_nifti"] == "incoming_24h/T2w/BD_01_24h.nii.gz"
    assert (
        loaded["splits"]["train"][0]["lesion_mask"]
        == "incoming_24h/masks/BD_01_24h_lesion_mask.nii.gz"
    )

    rerun = add_source_folder_to_plan(
        plan_path=plan,
        source_root=source_root,
        split="train",
        study="LYS",
        timepoint="24h",
        repo_root=tmp_path,
    )
    loaded_after_rerun = yaml.safe_load(plan.read_text())
    assert rerun.unchanged == 2
    assert len(loaded_after_rerun["splits"]["train"]) == 2

    dataset_root, prepared = prepare_dataset(plan, repo_root=tmp_path)
    assert len(prepared) == 2
    assert (dataset_root / "train" / "LYS" / "24h" / "BD_01_24h" / "scan.nii.gz").exists()
    assert (
        dataset_root / "train" / "LYS" / "24h" / "BD_01_24h" / "scan_lesionIAM.nii.gz"
    ).exists()


def test_add_source_folder_requires_matching_lesion_masks(tmp_path):
    source_root = tmp_path / "incoming"
    scan_dir = source_root / "mri"
    mask_dir = source_root / "lesion_masks"
    scan_dir.mkdir(parents=True)
    mask_dir.mkdir()
    _write_nifti(scan_dir / "BD_01.nii.gz", np.ones((8, 9, 3), dtype=np.float32))
    plan = _write_empty_plan(tmp_path)

    with pytest.raises(FileNotFoundError, match="No lesion mask files"):
        add_source_folder_to_plan(
            plan_path=plan,
            source_root=source_root,
            split="train",
            scan_subdir="mri",
            mask_subdir="lesion_masks",
            repo_root=tmp_path,
        )


def test_review_lys_masks_prepares_editable_copy_without_touching_source(tmp_path):
    source_root = tmp_path / "LYS_RatLesNetV2_clean_source" / "ratlesnetv2_clean_source"
    scan_dir = source_root / "T2w"
    mask_dir = source_root / "masks"
    scan_dir.mkdir(parents=True)
    mask_dir.mkdir()
    scan = scan_dir / "BD_01_24h.nii.gz"
    mask = mask_dir / "BD_01_24h_lesion_mask.nii.gz"
    _write_nifti(scan, np.ones((8, 9, 3), dtype=np.float32))
    label = np.zeros((8, 9, 3), dtype=np.uint8)
    label[2:4, 3:5, 1] = 1
    _write_nifti(mask, label)

    output_root = default_output_root(source_root)
    cases = find_cases(source_root=source_root, output_root=output_root, filters=["BD_01"])

    assert len(cases) == 1
    assert case_from_mask(mask) == "BD_01_24h"
    record = prepare_case(cases[0], copy_scans=True)

    reviewed_scan = output_root / "T2w" / "BD_01_24h.nii.gz"
    reviewed_mask = output_root / "masks" / "BD_01_24h_lesion_mask.nii.gz"
    assert record["status"] == "copied_source_mask"
    assert record["scan_status"] == "copied_scan"
    assert reviewed_scan.exists()
    assert reviewed_mask.exists()
    validate_grid(reviewed_scan, reviewed_mask)
    assert np.array_equal(np.asanyarray(nib.load(mask).dataobj), label)
    edited = np.zeros_like(label)
    _write_nifti(reviewed_mask, edited)
    assert int((np.asanyarray(nib.load(mask).dataobj) > 0).sum()) == int(label.sum())


def test_review_lys_masks_rejects_shape_mismatch(tmp_path):
    source_root = tmp_path / "ratlesnetv2_clean_source"
    scan_dir = source_root / "T2w"
    mask_dir = source_root / "masks"
    scan_dir.mkdir(parents=True)
    mask_dir.mkdir()
    _write_nifti(scan_dir / "BD_01.nii.gz", np.ones((8, 9, 3), dtype=np.float32))
    _write_nifti(mask_dir / "BD_01_lesion_mask.nii.gz", np.zeros((8, 9, 4), dtype=np.uint8))
    cases = find_cases(source_root=source_root, output_root=default_output_root(source_root))

    with pytest.raises(ValueError, match="Shape mismatch"):
        prepare_case(cases[0], copy_scans=True)


def test_review_lys_masks_waits_for_viewer_when_stdin_is_unavailable(monkeypatch):
    class FakeProcess:
        waited = False

        def wait(self):
            self.waited = True
            return 0

    def raise_eof(_prompt):
        raise EOFError

    process = FakeProcess()
    monkeypatch.setattr("builtins.input", raise_eof)

    wait_for_next_case(process, "prompt")

    assert process.waited


def test_relabel_pair_to_lsp_preserves_voxels_and_updates_affine(tmp_path):
    scan = tmp_path / "scan.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    data = np.arange(4 * 5 * 3, dtype=np.float32).reshape((4, 5, 3))
    label = np.zeros((4, 5, 3), dtype=np.uint8)
    label[1:3, 2:4, 1] = 1
    _write_nifti(scan, data, spacing=(0.1, 0.1, 0.5))
    _write_nifti(mask, label, spacing=(0.1, 0.1, 0.5))
    row = {
        "global_case_id": "External__case1",
        "source_dataset": "An2022",
        "shape": "4x5x3",
        "scan_path": str(scan),
        "mask_path": str(mask),
        "qc_flag": "needs_visual_qc",
    }

    out_scan = tmp_path / "out" / "T2w" / "scan.nii.gz"
    out_mask = tmp_path / "out" / "masks" / "mask.nii.gz"
    record = relabel_pair_to_axcodes(
        row=row,
        scan_path=scan,
        mask_path=mask,
        out_scan=out_scan,
        out_mask=out_mask,
        target_axcodes=("L", "S", "P"),
    )

    saved_scan = nib.load(out_scan)
    saved_mask = nib.load(out_mask)
    assert nib.aff2axcodes(saved_scan.affine) == ("L", "S", "P")
    assert nib.aff2axcodes(saved_mask.affine) == ("L", "S", "P")
    assert saved_scan.shape == data.shape
    assert saved_mask.shape == label.shape
    assert np.array_equal(np.asanyarray(saved_scan.dataobj), data)
    assert np.array_equal(np.asanyarray(saved_mask.dataobj), label)
    assert record["orientation_transform"] == "header_relabel_to_LSP_no_data_permutation"
    assert record["source_axcodes_before_orientation"] == "RAS"
    assert record["target_axcodes_after_orientation"] == "LSP"


def test_affine_for_axcodes_rejects_repeated_world_axis():
    with pytest.raises(ValueError, match="each world axis once"):
        affine_for_axcodes((4, 5, 3), (0.1, 0.1, 0.5), ("L", "R", "P"))


def test_flip_pair_axis_preserves_affine_and_flips_scan_mask(tmp_path):
    scan = tmp_path / "scan_lsp.nii.gz"
    mask = tmp_path / "mask_lsp.nii.gz"
    data = np.arange(4 * 5 * 3, dtype=np.float32).reshape((4, 5, 3))
    label = np.zeros((4, 5, 3), dtype=np.uint8)
    label[1:3, 1:4, 2] = 1
    affine = affine_for_axcodes(data.shape, (0.1, 0.1, 0.5), ("L", "S", "P"))
    scan_img = nib.Nifti1Image(data, affine)
    scan_img.header.set_zooms((0.1, 0.1, 0.5))
    mask_img = nib.Nifti1Image(label, affine)
    mask_img.header.set_zooms((0.1, 0.1, 0.5))
    nib.save(scan_img, scan)
    nib.save(mask_img, mask)
    row = {
        "global_case_id": "External__case1",
        "source_dataset": "An2022",
        "shape": "4x5x3",
        "scan_path": str(scan),
        "mask_path": str(mask),
        "qc_flag": "needs_visual_qc_lsp_orientation",
    }

    out_scan = tmp_path / "out" / "T2w" / "scan_lsp.nii.gz"
    out_mask = tmp_path / "out" / "masks" / "mask_lsp.nii.gz"
    record = flip_pair_axis(
        row=row,
        scan_path=scan,
        mask_path=mask,
        out_scan=out_scan,
        out_mask=out_mask,
        axis=1,
    )

    saved_scan = nib.load(out_scan)
    saved_mask = nib.load(out_mask)
    assert nib.aff2axcodes(saved_scan.affine) == ("L", "S", "P")
    assert nib.aff2axcodes(saved_mask.affine) == ("L", "S", "P")
    assert np.allclose(saved_scan.affine, affine)
    assert np.allclose(saved_mask.affine, affine)
    assert np.array_equal(np.asanyarray(saved_scan.dataobj), data[:, ::-1, :])
    assert np.array_equal(np.asanyarray(saved_mask.dataobj), label[:, ::-1, :])
    assert int(record["mask_voxels"]) == int(label.sum())
    assert record["qc_flag"] == "needs_visual_qc_lsp_si_flip"
    assert record["si_flip_transform"] == "voxel_array_flip_axis_1_keep_affine"


def test_roiset_to_nifti_mask_uses_roi_positions_and_preserves_reference(tmp_path):
    scan = tmp_path / "BD_01.nii.gz"
    affine = np.diag([SPACING[0], SPACING[1], SPACING[2], 1.0])
    img = nib.Nifti1Image(np.zeros((20, 30, 4), dtype=np.float32), affine)
    img.header.set_zooms(SPACING)
    nib.save(img, scan)
    roiset = tmp_path / "RoiSet.zip"
    _write_roiset(
        roiset,
        {
            "lesion_slice_1.roi": _imagej_polygon_roi_bytes(
                points=[(2, 3), (5, 3), (5, 7), (2, 7)],
                position=1,
            ),
            "lesion_slice_3.roi": _imagej_polygon_roi_bytes(
                points=[(10, 12), (15, 12), (15, 18), (10, 18)],
                position=3,
            ),
        },
    )
    output = tmp_path / "BD_01_lesion_mask.nii.gz"

    result = convert_roiset_to_nifti_mask(
        nifti_path=scan,
        roiset_zip=roiset,
        output_path=output,
    )

    saved = nib.load(output)
    mask = np.asanyarray(saved.dataobj)
    assert saved.shape == (20, 30, 4)
    assert np.allclose(saved.affine, affine)
    assert saved.header.get_zooms()[:3] == pytest.approx(SPACING)
    assert set(np.unique(mask)) == {0, 1}
    assert result.total_voxels == int(mask.sum())
    assert [roi.slice_index for roi in result.converted_rois] == [0, 2]

    assert mask[3, 5, 0] == 1
    assert mask[12, 14, 2] == 1
    assert mask[:, :, 1].sum() == 0
    assert mask[:, :, 3].sum() == 0


def test_roiset_to_nifti_mask_can_use_filename_slice_index(tmp_path):
    scan = tmp_path / "BD_02.nii.gz"
    _write_nifti(scan, np.zeros((10, 12, 3), dtype=np.float32))
    roiset = tmp_path / "RoiSet.zip"
    _write_roiset(
        roiset,
        {
            "slice_002.roi": _imagej_polygon_roi_bytes(
                points=[(1, 1), (4, 1), (4, 4), (1, 4)],
                position=0,
            )
        },
    )
    output = tmp_path / "BD_02_lesion_mask.nii.gz"

    convert_roiset_to_nifti_mask(
        nifti_path=scan,
        roiset_zip=roiset,
        output_path=output,
        slice_source="filename",
    )

    mask = np.asanyarray(nib.load(output).dataobj)
    assert mask[:, :, 0].sum() == 0
    assert mask[:, :, 1].sum() > 0
    assert mask[:, :, 2].sum() == 0


def test_roiset_to_nifti_mask_requires_slice_metadata(tmp_path):
    scan = tmp_path / "BD_03.nii.gz"
    _write_nifti(scan, np.zeros((10, 12, 3), dtype=np.float32))
    roiset = tmp_path / "RoiSet.zip"
    _write_roiset(
        roiset,
        {
            "lesion.roi": _imagej_polygon_roi_bytes(
                points=[(1, 1), (4, 1), (4, 4), (1, 4)],
                position=0,
            )
        },
    )
    output = tmp_path / "BD_03_lesion_mask.nii.gz"

    with pytest.raises(ValueError, match="Could not determine stack slice"):
        convert_roiset_to_nifti_mask(
            nifti_path=scan,
            roiset_zip=roiset,
            output_path=output,
        )


def test_cloud_command_plan_includes_pretrained_model():
    plan = build_cloud_command_plan(
        ratlesnet_repo="/content/RatLesNetv2",
        train_input="/content/dataset/train",
        validation_input="/content/dataset/validation",
        output_dir="/content/runs",
        pretrained_model="/content/pretrained/RatLesNetv2.model",
        epochs=3,
        lr=5e-5,
        gpu=0,
        load_memory=0,
        save_every=1,
        max_train_cases=1,
    )

    command = format_command(plan.finetune_command)

    assert "git" in plan.clone_command[0]
    assert "--pretrained-model /content/pretrained/RatLesNetv2.model" in command
    assert "--epochs 3" in command
    assert "--lr 5e-05" in command
    assert "--save-every 1" in command
    assert "--max-train-cases 1" in command
