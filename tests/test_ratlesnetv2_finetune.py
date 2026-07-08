import csv
import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import yaml

from ratlesnetv2_finetune.commands import build_cloud_command_plan, format_command
from ratlesnetv2_finetune.dataset import prepare_dataset
from ratlesnetv2_finetune.roiset_to_nifti_mask import convert_roiset_to_nifti_mask
from ratlesnetv2_finetune.scripts.finetune_an2023 import (
    _an2023_normalize,
    _center_crop_or_pad,
    _find_cases,
    _restore_crop_or_pad,
)
from ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 import (
    _build_loss_fn,
    _format_epoch_metrics,
    _parse_export_epochs,
    _parse_export_splits,
    _patch_nibabel_get_data_compat,
    _publish_latest_qc_overlay,
    _save_prediction_artifacts,
    _scheduler_metric_value,
    _segmentation_metrics,
    _should_export_epoch,
    _transpose_to_shape,
    _update_best_checkpoints,
    _write_metric_plots,
    _write_run_status,
)
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
from ratlesnetv2_finetune.scripts.run_training_grid import build_experiment_commands
from ratlesnetv2_finetune.scripts.split_prepared_dataset import split_prepared_dataset
from ratlesnetv2_finetune.scripts.summarize_training_grid import summarize_runs
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


def test_an2023_crop_pad_roundtrip_restores_source_grid():
    source = np.zeros((8, 10, 3), dtype=np.uint8)
    source[3:5, 4:7, 1] = 1

    cropped, plan = _center_crop_or_pad(source, (6, 12, 5), fill=0)
    restored = _restore_crop_or_pad(cropped, plan, fill=0)

    assert cropped.shape == (6, 12, 5)
    assert restored.shape == source.shape
    assert np.array_equal(restored, source)


def test_an2023_normalization_matches_notebook_scaling():
    scan = np.array([0.0, 5.0, 10.0], dtype=np.float32).reshape((3, 1, 1))

    normalized = _an2023_normalize(scan)

    assert normalized[0, 0, 0] == pytest.approx(-1.0)
    assert normalized[1, 0, 0] == pytest.approx(0.0)
    assert normalized[2, 0, 0] == pytest.approx(1.0)


def test_an2023_find_cases_uses_prepared_ratlesnet_layout(tmp_path):
    case_dir = tmp_path / "train" / "LYS" / "5d" / "BD_01_5d"
    case_dir.mkdir(parents=True)
    _write_nifti(case_dir / "scan.nii.gz", np.ones((8, 9, 3), dtype=np.float32))
    _write_nifti(case_dir / "scan_lesionIAM.nii.gz", np.zeros((8, 9, 3), dtype=np.uint8))

    cases = _find_cases(tmp_path / "train")

    assert len(cases) == 1
    assert cases[0].case_id == "BD_01_5d"
    assert cases[0].scan == case_dir / "scan.nii.gz"
    assert cases[0].label == case_dir / "scan_lesionIAM.nii.gz"


def test_epoch_log_metrics_include_precision_recall_and_voxel_recovery():
    text = _format_epoch_metrics(
        [
            {
                "split": "validation",
                "dice_mean": 0.6176,
                "precision_mean": 0.7,
                "recall_mean": 0.6,
                "accuracy_mean": 0.9981,
                "tp": 60,
                "target_voxels": 100,
                "pred_voxels": 90,
            }
        ]
    )

    assert "validation Dice: 0.6176" in text
    assert "Prec: 0.7" in text
    assert "Rec: 0.6" in text
    assert "TP/Target: 60.0%" in text
    assert "Pred/Target: 0.9x" in text


def test_ratlesnet_loss_builder_keeps_default_and_supports_tversky():
    torch = pytest.importorskip("torch")

    default_args = SimpleNamespace(
        loss="ce-dice",
        background_class_weight=1.0,
        lesion_class_weight=1.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        focal_tversky_gamma=0.75,
    )

    def upstream(_pred, _target):
        return torch.tensor(2.0)

    default_loss = _build_loss_fn(
        torch=torch,
        upstream_ce_dice_loss=upstream,
        args=default_args,
    )
    assert default_loss is upstream

    tversky_args = SimpleNamespace(**{**vars(default_args), "loss": "tversky"})
    loss_fn = _build_loss_fn(torch=torch, upstream_ce_dice_loss=upstream, args=tversky_args)
    pred = torch.zeros((1, 2, 2, 2, 2), dtype=torch.float32)
    target = torch.zeros((1, 2, 2, 2, 2), dtype=torch.float32)
    pred[:, 0] = 1.0
    target[:, 0] = 1.0
    pred[:, 1, 0, 0, 0] = 0.8
    pred[:, 0, 0, 0, 0] = 0.2
    target[:, 1, 0, 0, 0] = 1.0
    target[:, 0, 0, 0, 0] = 0.0

    loss = loss_fn(pred, target)

    assert float(loss.detach().cpu()) < 0.5


def test_training_grid_builds_ratlesnet_and_an2023_commands(tmp_path):
    config = {
        "output_root": str(tmp_path / "grid"),
        "splits": {
            "lys_train": "/kaggle/working/lys/train",
            "lys_validation": "/kaggle/working/lys/validation",
        },
        "paths": {
            "ratlesnet_repo": "/kaggle/working/RatLesNetv2",
            "ratlesnet_pretrained": "/kaggle/working/RatLesNetv2/model",
            "an2023_model": "/kaggle/working/stroke-lesion-segmentation/lesion_model.pt",
        },
        "defaults": {
            "gpu": 0,
            "save_every": 1,
            "eval_every": 1,
            "export_predictions": "validation",
        },
        "experiments": [
            {
                "name": "rat_direct",
                "kind": "ratlesnetv2",
                "input": "lys_train",
                "validation": "lys_validation",
                "ratlesnet_repo": "ratlesnet_repo",
                "pretrained_model": "ratlesnet_pretrained",
                "require_pretrained": True,
                "epochs": 2,
                "lr": 5e-5,
                "loss": "focal-tversky",
                "tversky_alpha": 0.3,
                "tversky_beta": 0.7,
                "focal_tversky_gamma": 0.75,
            },
            {
                "name": "an_direct",
                "kind": "an2023",
                "input": "lys_train",
                "validation": "lys_validation",
                "model_path": "an2023_model",
                "epochs": 2,
                "lr": 1e-5,
                "metrics_threshold": 0.8,
            },
        ],
    }

    commands = build_experiment_commands(config)

    assert len(commands) == 2
    rat_cmd = commands[0].command
    an_cmd = commands[1].command
    assert "ratlesnetv2_finetune.scripts.finetune_ratlesnetv2" in rat_cmd
    assert "--require-pretrained" in rat_cmd
    assert "--loss" in rat_cmd
    assert "focal-tversky" in rat_cmd
    assert "--focal-tversky-gamma" in rat_cmd
    assert "/kaggle/working/lys/train" in rat_cmd
    assert "ratlesnetv2_finetune.scripts.finetune_an2023" in an_cmd
    assert "--metrics-threshold" in an_cmd
    assert "/kaggle/working/stroke-lesion-segmentation/lesion_model.pt" in an_cmd


def test_summarize_runs_writes_comparison_report(tmp_path):
    run_dir = tmp_path / "grid" / "rat_direct" / "1"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment_metadata.json").write_text(
        json.dumps({"name": "rat_direct", "kind": "ratlesnetv2"})
    )
    (run_dir / "run_status.json").write_text(json.dumps({"status": "completed"}))
    (run_dir / "best_checkpoints.json").write_text(
        json.dumps(
            {
                "validation_dice": {
                    "epoch": 2,
                    "filename": "best_by_validation_dice.model",
                }
            }
        )
    )
    (run_dir / "best_by_validation_dice.model").write_text("weights")
    overlay = run_dir / "latest_validation_qc_overlay.png"
    overlay.write_bytes(b"not-a-real-png")
    (run_dir / "latest_validation_qc_overlay.json").write_text(
        json.dumps({"latest_overlay": str(overlay)})
    )
    with (run_dir / "metrics_epoch.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "epoch",
                "split",
                "n_cases",
                "lr",
                "loss",
                "dice_mean",
                "precision_mean",
                "recall_mean",
                "target_voxels",
                "pred_voxels",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 1,
                "split": "validation",
                "n_cases": 2,
                "lr": 0.00005,
                "loss": 0.8,
                "dice_mean": 0.4,
                "precision_mean": 0.5,
                "recall_mean": 0.6,
                "target_voxels": 100,
                "pred_voxels": 80,
            }
        )
        writer.writerow(
            {
                "epoch": 2,
                "split": "validation",
                "n_cases": 2,
                "lr": 0.00005,
                "loss": 0.7,
                "dice_mean": 0.55,
                "precision_mean": 0.65,
                "recall_mean": 0.7,
                "target_voxels": 100,
                "pred_voxels": 95,
            }
        )

    summaries = summarize_runs(run_dirs=[run_dir], output_dir=tmp_path / "report")

    assert summaries[0].best_validation_dice == pytest.approx(0.55)
    comparison = (tmp_path / "report" / "comparison.csv").read_text()
    assert "rat_direct" in comparison
    assert "0.55" in comparison
    assert (tmp_path / "report" / "report.html").exists()
    assert (tmp_path / "report" / "selected_recommendation.json").exists()


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


def test_prepare_dataset_rejects_duplicate_case_id_across_splits(tmp_path):
    scan = tmp_path / "scan.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    _write_nifti(scan, np.ones((8, 9, 3), dtype=np.float32))
    _write_nifti(mask, np.zeros((8, 9, 3), dtype=np.uint8))
    plan = _write_plan(tmp_path, scan, mask)
    loaded = yaml.safe_load(plan.read_text())
    loaded["splits"]["validation"] = [dict(loaded["splits"]["train"][0])]
    plan.write_text(yaml.safe_dump(loaded))

    with pytest.raises(ValueError, match="Duplicate case_id"):
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


def test_split_prepared_dataset_creates_disjoint_train_validation_test(tmp_path):
    source_dir = tmp_path / "source"
    scan_dir = source_dir / "T2w"
    mask_dir = source_dir / "masks"
    scan_dir.mkdir(parents=True)
    mask_dir.mkdir()
    split_cases = []
    for index in range(4):
        case_id = f"BD_{index:02d}"
        scan = scan_dir / f"{case_id}.nii.gz"
        mask = mask_dir / f"{case_id}_lesion_mask.nii.gz"
        _write_nifti(scan, np.ones((8, 9, 3), dtype=np.float32))
        label = np.zeros((8, 9, 3), dtype=np.uint8)
        label[index % 4 : index % 4 + 1, 2:4, 1] = 1
        _write_nifti(mask, label)
        split_cases.append(
            {
                "animal_id": case_id,
                "study": "LYS",
                "timepoint": "mixed",
                "case_id": case_id,
                "scan_nifti": str(scan),
                "lesion_mask": str(mask),
            }
        )
    plan = _write_empty_plan(tmp_path)
    loaded = yaml.safe_load(plan.read_text())
    loaded["splits"]["train"] = split_cases
    plan.write_text(yaml.safe_dump(loaded))
    dataset_root, _prepared = prepare_dataset(plan, repo_root=tmp_path)

    result = split_prepared_dataset(
        input_root=dataset_root,
        output_root=tmp_path / "split_dataset",
        validation_count=1,
        test_count=1,
        seed=1,
    )

    assert result.counts == {"train": 2, "validation": 1, "test": 1}
    with result.manifest_path.open() as fh:
        rows = list(csv.DictReader(fh))
    by_split = {
        split: {row["case_id"] for row in rows if row["split"] == split}
        for split in ["train", "validation", "test"]
    }
    assert len(by_split["train"] & by_split["validation"]) == 0
    assert len(by_split["train"] & by_split["test"]) == 0
    assert len(by_split["validation"] & by_split["test"]) == 0
    assert (result.output_root / "validation").is_dir()
    assert (result.output_root / "test").is_dir()


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


def test_segmentation_metrics_report_dice_and_accuracy_for_two_class_logits():
    pred = np.zeros((1, 2, 2, 2, 1), dtype=np.float32)
    pred[:, 0, :, :, :] = 1
    target = np.zeros((1, 2, 2, 2, 1), dtype=np.float32)
    target[:, 0, :, :, :] = 1
    lesion_voxels = [(0, 0, 0), (1, 0, 0)]
    for x, y, z in lesion_voxels:
        pred[0, 0, x, y, z] = 0
        pred[0, 1, x, y, z] = 5
        target[0, 0, x, y, z] = 0
        target[0, 1, x, y, z] = 1

    metrics = _segmentation_metrics(pred, target)

    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["target_voxels"] == 2
    assert metrics["pred_voxels"] == 2


def test_metric_plots_are_written_when_matplotlib_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_loss").write_text("1.0\n0.8\n")
    (run_dir / "metrics_epoch.csv").write_text(
        "epoch,split,n_cases,loss,dice_mean,dice_median,iou_mean,iou_median,"
        "accuracy_mean,accuracy_median,balanced_accuracy_mean,balanced_accuracy_median,"
        "precision_mean,precision_median,recall_mean,recall_median,"
        "specificity_mean,specificity_median,tp,fp,tn,fn,target_voxels,pred_voxels\n"
        "1,validation,2,0.9,0.3,0.3,0.18,0.18,0.98,0.98,0.6,0.6,"
        "0.4,0.4,0.3,0.3,0.99,0.99,3,4,100,7,10,7\n"
        "2,validation,2,0.7,0.5,0.5,0.33,0.33,0.99,0.99,0.7,0.7,"
        "0.6,0.6,0.5,0.5,0.99,0.99,5,3,100,5,10,8\n"
        "2,validation_final,2,0.7,0.5,0.5,0.33,0.33,0.99,0.99,0.7,0.7,"
        "0.6,0.6,0.5,0.5,0.99,0.99,5,3,100,5,10,8\n"
        "2,test,2,0.75,0.45,0.45,0.29,0.29,0.98,0.98,0.68,0.68,"
        "0.5,0.5,0.45,0.45,0.99,0.99,4,4,100,6,10,8\n"
    )

    _write_metric_plots(run_dir)

    assert (run_dir / "loss_curves.png").exists()
    assert (run_dir / "validation_metric_curves.png").exists()
    assert (run_dir / "voxel_count_curves.png").exists()
    assert (run_dir / "final_metric_summary.png").exists()


def test_prediction_export_helpers_parse_epochs_and_splits():
    assert _parse_export_splits("validation,test") == {"validation", "test"}
    assert _parse_export_epochs("1,2,final") == {1, 2, "final"}
    assert _should_export_epoch(2, is_final=False, export_epochs={1, 2, "final"})
    assert _should_export_epoch(9, is_final=True, export_epochs={1, 2, "final"})
    assert _should_export_epoch(9, is_final=False, export_epochs={"all"})
    with pytest.raises(ValueError, match="Unknown"):
        _parse_export_splits("bad")


def test_best_checkpoint_writer_saves_dice_and_loss_models(tmp_path):
    class FakeModel:
        def state_dict(self):
            return {"weight": 1}

    class FakeTorch:
        @staticmethod
        def save(_state, path):
            Path(path).write_text("saved")

    summary = {
        "epoch": 2,
        "split": "validation",
        "loss": 0.4,
        "dice_mean": 0.2,
        "precision_mean": 0.3,
        "recall_mean": 0.4,
        "target_voxels": 10,
        "pred_voxels": 8,
    }
    best_state: dict[str, dict] = {}

    improved = _update_best_checkpoints(
        torch=FakeTorch,
        model=FakeModel(),
        run_dir=tmp_path,
        summary=summary,
        best_state=best_state,
        min_dice_delta=0.0,
    )

    assert improved is True
    assert (tmp_path / "best_by_validation_dice.model").exists()
    assert (tmp_path / "best_by_validation_loss.model").exists()
    assert (tmp_path / "best_checkpoints.json").exists()


def test_scheduler_metric_value_uses_validation_dice_or_loss():
    summary = {"dice_mean": 0.53, "loss": 0.42}

    assert _scheduler_metric_value(summary, metric="validation_dice") == (
        "validation_dice",
        0.53,
    )
    assert _scheduler_metric_value(summary, metric="validation_loss") == (
        "validation_loss",
        0.42,
    )

    with pytest.raises(ValueError, match="Dice is unavailable"):
        _scheduler_metric_value({"dice_mean": None, "loss": 0.42}, metric="validation_dice")


def test_prediction_artifacts_write_nifti_and_overlay(tmp_path):
    pytest.importorskip("matplotlib")
    source_dir = tmp_path / "case"
    source_dir.mkdir()
    scan = np.arange(5 * 6 * 3, dtype=np.float32).reshape((5, 6, 3))
    _write_nifti(source_dir / "scan.nii.gz", scan)
    x = scan[..., np.newaxis]
    target = np.zeros((1, 2, 5, 6, 3), dtype=np.float32)
    pred = np.zeros((1, 2, 5, 6, 3), dtype=np.float32)
    target[:, 0, ...] = 1.0
    pred[:, 0, ...] = 2.0
    target[0, 0, 2, 3, 1] = 0.0
    target[0, 1, 2, 3, 1] = 1.0
    pred[0, 0, 2, 3, 1] = -2.0
    pred[0, 1, 2, 3, 1] = 4.0

    class FakeData:
        list = [source_dir]

    record = _save_prediction_artifacts(
        x=x,
        y=target,
        pred=pred,
        data=FakeData(),
        index=0,
        case_id="case",
        out_dir=tmp_path / "export",
        threshold=0.5,
    )

    assert Path(record["scan"]).exists()
    assert Path(record["target_mask"]).exists()
    assert Path(record["pred_mask"]).exists()
    assert Path(record["lesion_probability"]).exists()
    assert Path(record["overlay"]).exists()
    saved_pred = np.asanyarray(nib.load(record["pred_mask"]).dataobj)
    assert saved_pred.shape == scan.shape
    assert int(saved_pred.sum()) == 1


def test_prediction_artifacts_align_transposed_output_to_reference_grid(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.image as mpimg

    source_dir = tmp_path / "case"
    source_dir.mkdir()
    scan = np.arange(5 * 6 * 3, dtype=np.float32).reshape((5, 6, 3))
    _write_nifti(source_dir / "scan.nii.gz", scan)
    x = np.zeros((3, 5, 6, 1), dtype=np.float32)
    target = np.zeros((1, 2, 3, 5, 6), dtype=np.float32)
    pred = np.zeros((1, 2, 3, 5, 6), dtype=np.float32)
    target[:, 0, ...] = 1.0
    pred[:, 0, ...] = 2.0
    target[0, 0, 1, 2, 3] = 0.0
    target[0, 1, 1, 2, 3] = 1.0
    pred[0, 0, 1, 2, 3] = -2.0
    pred[0, 1, 1, 2, 3] = 4.0

    class FakeData:
        list = [source_dir]

    record = _save_prediction_artifacts(
        x=x,
        y=target,
        pred=pred,
        data=FakeData(),
        index=0,
        case_id="case",
        out_dir=tmp_path / "export",
        threshold=0.5,
    )

    saved_pred = np.asanyarray(nib.load(record["pred_mask"]).dataobj)
    saved_probability = np.asanyarray(nib.load(record["lesion_probability"]).dataobj)
    assert saved_pred.shape == scan.shape
    assert saved_probability.shape == scan.shape
    assert int(saved_pred.sum()) == 1
    assert saved_pred[2, 3, 1] == 1

    overlay = mpimg.imread(record["overlay"])
    assert overlay.shape[:2] == (6, 5)


def test_transpose_to_shape_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="Cannot align"):
        _transpose_to_shape(np.zeros((3, 5, 6)), (5, 5, 3))


def test_latest_qc_overlay_is_published_at_run_root(tmp_path):
    export_dir = tmp_path / "prediction_exports" / "validation" / "epoch_002" / "case_a"
    export_dir.mkdir(parents=True)
    overlay = export_dir / "overlay.png"
    overlay.write_bytes(b"png")

    _publish_latest_qc_overlay(
        tmp_path,
        {"case_id": "case_a", "overlay": str(overlay)},
        split="validation",
        epoch=2,
        label="epoch_002",
    )

    assert (tmp_path / "latest_validation_qc_overlay.png").read_bytes() == b"png"
    assert (tmp_path / "latest_qc_overlay.png").read_bytes() == b"png"
    metadata = json.loads((tmp_path / "latest_qc_overlay.json").read_text())
    assert metadata["split"] == "validation"
    assert metadata["epoch"] == 2
    assert metadata["case_id"] == "case_a"


def test_run_status_records_interrupt_metadata(tmp_path):
    _write_run_status(
        tmp_path / "run_status.json",
        status="interrupted",
        completed_epoch=3,
        interrupted_epoch=4,
        reason="KeyboardInterrupt",
    )

    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status == {
        "completed_epoch": 3,
        "interrupted_epoch": 4,
        "reason": "KeyboardInterrupt",
        "status": "interrupted",
    }


def test_cloud_command_plan_includes_pretrained_model():
    plan = build_cloud_command_plan(
        ratlesnet_repo="/content/RatLesNetv2",
        train_input="/content/dataset/train",
        validation_input="/content/dataset/validation",
        test_input="/content/dataset/test",
        output_dir="/content/runs",
        pretrained_model="/content/pretrained/RatLesNetv2.model",
        require_pretrained=True,
        epochs=3,
        lr=5e-5,
        gpu=0,
        load_memory=0,
        save_every=1,
        eval_only=True,
        eval_every=1,
        metrics_threshold=0.4,
        loss="weighted-ce-dice",
        background_class_weight=1.0,
        lesion_class_weight=5.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        focal_tversky_gamma=0.75,
        early_stop_patience=4,
        lr_scheduler="reduce-on-plateau",
        lr_scheduler_metric="validation_dice",
        lr_plateau_patience=2,
        lr_plateau_factor=0.5,
        lr_plateau_min_delta=0.01,
        min_lr=1e-6,
        export_predictions="validation",
        export_prediction_limit=2,
        export_prediction_epochs="1,final",
        max_train_cases=1,
        max_validation_cases=1,
        max_test_cases=1,
    )

    command = format_command(plan.finetune_command)

    assert "git" in plan.clone_command[0]
    assert "--pretrained-model /content/pretrained/RatLesNetv2.model" in command
    assert "--require-pretrained" in command
    assert "--validation /content/dataset/validation" in command
    assert "--test /content/dataset/test" in command
    assert "--epochs 3" in command
    assert "--lr 5e-05" in command
    assert "--save-every 1" in command
    assert "--eval-only" in command
    assert "--eval-every 1" in command
    assert "--metrics-threshold 0.4" in command
    assert "--loss weighted-ce-dice" in command
    assert "--background-class-weight 1.0" in command
    assert "--lesion-class-weight 5.0" in command
    assert "--tversky-alpha 0.3" in command
    assert "--tversky-beta 0.7" in command
    assert "--focal-tversky-gamma 0.75" in command
    assert "--early-stop-patience 4" in command
    assert "--lr-scheduler reduce-on-plateau" in command
    assert "--lr-scheduler-metric validation_dice" in command
    assert "--lr-plateau-patience 2" in command
    assert "--lr-plateau-factor 0.5" in command
    assert "--lr-plateau-min-delta 0.01" in command
    assert "--min-lr 1e-06" in command
    assert "--export-predictions validation" in command
    assert "--export-prediction-limit 2" in command
    assert "--export-prediction-epochs 1,final" in command
    assert "--max-train-cases 1" in command
    assert "--max-validation-cases 1" in command
    assert "--max-test-cases 1" in command
