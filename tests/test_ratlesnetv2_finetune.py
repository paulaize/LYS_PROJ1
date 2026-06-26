import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from ratlesnetv2_finetune.commands import build_cloud_command_plan, format_command
from ratlesnetv2_finetune.dataset import prepare_dataset

SPACING = (0.07, 0.07, 0.5)


def _write_nifti(path: Path, data: np.ndarray, spacing=SPACING) -> None:
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms(spacing[: data.ndim])
    nib.save(img, path)


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
    )

    command = format_command(plan.finetune_command)

    assert "git" in plan.clone_command[0]
    assert "--pretrained-model /content/pretrained/RatLesNetv2.model" in command
    assert "--epochs 3" in command
    assert "--lr 5e-05" in command

