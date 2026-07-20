import csv
import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ratlesnetv2_finetune.scripts.compare_oof_candidates import (
    compare_oof_candidates,
)
from ratlesnetv2_finetune.scripts.create_oof_qc_contact_sheet import (
    create_oof_qc_contact_sheet,
    extract_direct_ce_oof_artifact,
)
from ratlesnetv2_finetune.scripts.prepare_architecture_comparator import (
    export_nnunet_validation,
    prepare_external_dataset,
    prepare_paper_folds,
    prepare_target_dataset,
)
from ratlesnetv2_finetune.scripts.train_an2023_unet_adapted import (
    build_an2023_adapted_unet,
    train_an2023_unet_adapted,
)

SPACING = (0.07, 0.07, 0.5)


def _write_nifti(path: Path, data: np.ndarray) -> None:
    affine = np.diag([*SPACING, 1.0])
    image = nib.Nifti1Image(data, affine)
    image.header.set_zooms(SPACING if data.ndim == 3 else (*SPACING, 1.0))
    nib.save(image, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _prepared_dataset(root: Path, count: int) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        case_id = f"case_{index:02d}"
        case_root = root / "train" / "LYS" / "mixed" / case_id
        case_root.mkdir(parents=True)
        scan = np.arange(12 * 12 * 4, dtype=np.float32).reshape(12, 12, 4, 1)
        scan += index
        label = np.zeros((12, 12, 4), dtype=np.uint8)
        label[2 + index % 2 : 5 + index % 2, 3:6, 1:3] = 1
        _write_nifti(case_root / "scan.nii.gz", scan)
        _write_nifti(case_root / "scan_lesionIAM.nii.gz", label)
        _write_nifti(case_root / "scan_lesion.nii.gz", label)
        rows.append(
            {
                "case_id": case_id,
                "animal_id": case_id,
                "split": "train",
                "study": "LYS",
                "timepoint": "mixed",
                "case_dir": str(case_root),
                "scan_path": str(case_root / "scan.nii.gz"),
                "label_path": str(case_root / "scan_lesionIAM.nii.gz"),
            }
        )
    _write_csv(root / "manifest.csv", rows)
    return rows


def _assignments(path: Path, rows: list[dict[str, object]]) -> None:
    assignments = []
    for index, row in enumerate(rows):
        development = index < len(rows) - 2
        assignments.append(
            {
                "case_id": row["case_id"],
                "outer_split": "development" if development else "test",
                "cv_fold": str(index % 5) if development else "",
            }
        )
    _write_csv(path, assignments)


def test_prepare_target_uses_preserved_folds_and_omits_locked_test(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = _prepared_dataset(prepared, 12)
    assignments = tmp_path / "split_assignments.csv"
    _assignments(assignments, rows)
    output = tmp_path / "Dataset701_LYSDevelopmentV1"

    report = prepare_target_dataset(
        input_root=prepared,
        split_assignments=assignments,
        output_root=output,
    )

    assert report["n_training"] == 10
    assert report["n_locked_test_omitted"] == 2
    assert report["locked_test_materialized"] is False
    assert len(list((output / "imagesTr").glob("*.nii.gz"))) == 10
    assert len(list((output / "labelsTr").glob("*.nii.gz"))) == 10
    assert not (output / "imagesTs").exists()
    splits = json.loads((output / "splits_final.json").read_text())
    assert len(splits) == 5
    assert all(len(split["val"]) == 2 for split in splits)
    assert all(set(split["train"]).isdisjoint(split["val"]) for split in splits)
    converted = nib.load(str(next((output / "imagesTr").glob("*.nii.gz"))))
    assert converted.shape == (12, 12, 4)

    assert (
        prepare_target_dataset(
            input_root=prepared,
            split_assignments=assignments,
            output_root=output,
        )["split_assignments_sha256"]
        == report["split_assignments_sha256"]
    )


def test_prepare_target_rejects_changed_split_provenance(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = _prepared_dataset(prepared, 12)
    assignments = tmp_path / "split_assignments.csv"
    _assignments(assignments, rows)
    output = tmp_path / "Dataset701_LYSDevelopmentV1"
    prepare_target_dataset(
        input_root=prepared,
        split_assignments=assignments,
        output_root=output,
    )
    assignment_rows = _read_csv(assignments)
    assignment_rows[0]["cv_fold"] = "1"
    _write_csv(assignments, assignment_rows)

    with pytest.raises(RuntimeError, match="provenance differs"):
        prepare_target_dataset(
            input_root=prepared,
            split_assignments=assignments,
            output_root=output,
        )


def test_prepare_external_uses_all_manifest_cases(tmp_path):
    prepared = tmp_path / "external"
    prepared.mkdir()
    _prepared_dataset(prepared, 3)
    output = tmp_path / "Dataset702_ExternalMouseV0"

    report = prepare_external_dataset(input_root=prepared, output_root=output)

    assert report["n_training"] == 3
    assert report["target_lys_used"] is False
    assert len(_read_csv(output / "case_mapping.csv")) == 3
    assert not (output / "splits_final.json").exists()


def test_prepare_external_preserves_source_train_validation_split(tmp_path):
    prepared = tmp_path / "external"
    prepared.mkdir()
    rows = _prepared_dataset(prepared, 3)
    validation_row = rows[-1]
    source = prepared / "train/LYS/mixed" / str(validation_row["case_id"])
    destination = prepared / "validation/LYS/mixed" / str(validation_row["case_id"])
    destination.parent.mkdir(parents=True)
    source.rename(destination)
    validation_row["split"] = "validation"
    validation_row["case_dir"] = str(destination)
    validation_row["scan_path"] = str(destination / "scan.nii.gz")
    validation_row["label_path"] = str(destination / "scan_lesionIAM.nii.gz")
    _write_csv(prepared / "manifest.csv", rows)
    output = tmp_path / "Dataset702_ExternalMouseV0"

    report = prepare_external_dataset(input_root=prepared, output_root=output)

    assert report["source_split_counts"] == {"train": 2, "validation": 1}
    splits = json.loads((output / "splits_final.json").read_text())
    assert len(splits) == 1
    assert len(splits[0]["train"]) == 2
    assert len(splits[0]["val"]) == 1


def test_prepare_paper_folds_are_symlink_only_and_omit_locked_test(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = _prepared_dataset(prepared, 12)
    assignments = tmp_path / "split_assignments.csv"
    _assignments(assignments, rows)
    output = tmp_path / "paper_folds"

    report = prepare_paper_folds(
        input_root=prepared,
        split_assignments=assignments,
        output_root=output,
    )

    assert report["n_development"] == 10
    assert report["n_locked_test_omitted"] == 2
    for fold in range(5):
        scans = list((output / f"fold_{fold}").rglob("scan.nii.gz"))
        assert len(scans) == 10
        assert all(path.is_symlink() for path in scans)
        observed = {path.parent.name for path in scans}
        assert "case_10" not in observed
        assert "case_11" not in observed


def test_export_nnunet_probabilities_restores_native_orientation(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = _prepared_dataset(prepared, 12)
    assignments = tmp_path / "split_assignments.csv"
    _assignments(assignments, rows)
    dataset = tmp_path / "Dataset701_LYSDevelopmentV1"
    prepare_target_dataset(
        input_root=prepared,
        split_assignments=assignments,
        output_root=dataset,
    )
    fold = 0
    mapping = [row for row in _read_csv(dataset / "case_mapping.csv") if row["cv_fold"] == "0"]
    validation = tmp_path / "validation"
    validation.mkdir()
    for row in mapping:
        target_image = nib.load(row["source_label"])
        target = np.asarray(target_image.dataobj).astype(np.uint8)
        lesion_probability = np.where(target > 0, 0.9, 0.1).astype(np.float32)
        probabilities = np.stack(
            (1 - lesion_probability, lesion_probability),
            axis=0,
        ).transpose((0, 3, 2, 1))
        safe_id = row["nnunet_case_id"]
        np.savez_compressed(validation / f"{safe_id}.npz", probabilities=probabilities)
        _write_nifti(validation / f"{safe_id}.nii.gz", target)

    output = tmp_path / "oof"
    result = export_nnunet_validation(
        validation_root=validation,
        case_mapping=dataset / "case_mapping.csv",
        fold=fold,
        output_root=output,
        candidate="nnunet_test",
    )

    assert result["n_cases"] == len(mapping)
    records = _read_csv(output / "prediction_export_manifest.csv")
    probability = nib.load(records[0]["lesion_probability"])
    target = nib.load(records[0]["target_mask"])
    assert probability.shape == target.shape == (12, 12, 4)
    assert np.allclose(probability.affine, target.affine)
    assert np.array_equal(np.asarray(probability.dataobj) >= 0.5, np.asarray(target.dataobj) > 0)


def test_an2023_adapted_network_preserves_native_shape():
    torch = pytest.importorskip("torch")
    model = build_an2023_adapted_unet(torch)
    value = torch.randn(1, 1, 17, 19, 5)

    output = model(value)

    assert output.shape == (1, 2, 17, 19, 5)
    assert sum(parameter.numel() for parameter in model.parameters()) == 42442


def test_an2023_adapted_one_epoch_writes_complete_probability_contract(tmp_path):
    pytest.importorskip("torch")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    _prepared_dataset(prepared, 3)
    train_root = prepared / "train_subset"
    validation_root = prepared / "validation_subset"
    train_root.mkdir()
    validation_root.mkdir()
    source_cases = sorted((prepared / "train/LYS/mixed").iterdir())
    for destination, sources in (
        (train_root, source_cases[:2]),
        (validation_root, source_cases[2:]),
    ):
        for source in sources:
            staged = destination / source.name
            staged.mkdir()
            for source_file in source.iterdir():
                (staged / source_file.name).symlink_to(source_file)
    output = tmp_path / "run"

    result = train_an2023_unet_adapted(
        train_root=train_root,
        validation_root=validation_root,
        output_root=output,
        epochs=1,
        early_stop_patience=1,
        gradient_accumulation=1,
        noise_std=0,
        device_name="cpu",
        amp=False,
    )

    assert result == 0
    assert json.loads((output / "run_status.json").read_text())["status"] == "completed"
    assert (output / "an2023_adapted_unet.model").is_file()
    manifest = _read_csv(
        output / "prediction_exports/validation/final/prediction_export_manifest.csv"
    )
    assert len(manifest) == 1
    assert Path(manifest[0]["lesion_probability"]).is_file()


def test_compare_oof_candidates_writes_paired_evidence(tmp_path):
    metrics = []
    for index in range(10):
        metrics.append(
            {
                "case_id": f"case_{index:02d}",
                "cv_fold": index % 5,
                "cohort": "cohort_a" if index < 5 else "cohort_b",
                "dice": 0.5 + index / 100,
                "precision": 0.6,
                "recall": 0.7,
                "absolute_volume_error_mm3": 2.0,
                "volume_error_mm3": 1.0,
                "hd95_mm": 0.5,
                "surface_dice": 0.8,
            }
        )
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, metrics)
    candidate_metrics = [dict(row, dice=float(row["dice"]) + 0.05) for row in metrics]
    _write_csv(candidate, candidate_metrics)

    summary = compare_oof_candidates(
        reference_name="reference",
        reference_csv=reference,
        candidate_name="candidate",
        candidate_csv=candidate,
        output_root=tmp_path / "comparison",
        bootstrap_samples=100,
    )

    assert summary["n_cases"] == 10
    assert summary["candidate_minus_reference_mean_dice"] == pytest.approx(0.05)
    assert summary["candidate_fold_wins"] == 5
    assert (tmp_path / "comparison/paired_cases.csv").is_file()

    candidate_metrics[0]["cv_fold"] = 4
    _write_csv(candidate, candidate_metrics)
    with pytest.raises(ValueError, match="metadata differs"):
        compare_oof_candidates(
            reference_name="reference",
            reference_csv=reference,
            candidate_name="candidate",
            candidate_csv=candidate,
            output_root=tmp_path / "comparison_with_split_drift",
            bootstrap_samples=100,
        )


def test_create_oof_qc_contact_sheet_uses_selected_threshold(tmp_path):
    pytest.importorskip("matplotlib")
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    manifest_rows = []
    metric_rows = []
    for index in range(10):
        case_id = f"case_{index:02d}"
        case_root = prediction_root / case_id
        case_root.mkdir()
        scan = np.arange(12 * 12 * 4, dtype=np.float32).reshape(12, 12, 4)
        target = np.zeros((12, 12, 4), dtype=np.uint8)
        target[2:6, 3:7, 1:3] = 1
        prediction = target.astype(bool)
        prediction[: index // 2, :2, 1] = True
        probability = np.where(prediction, 0.8, 0.2).astype(np.float32)
        _write_nifti(case_root / "scan.nii.gz", scan)
        _write_nifti(case_root / "target_mask.nii.gz", target)
        _write_nifti(case_root / "lesion_probability.nii.gz", probability)
        intersection = int(np.logical_and(prediction, target).sum())
        dice = 2 * intersection / (int(prediction.sum()) + int(target.sum()))
        manifest_rows.append(
            {
                "split": "validation",
                "case_id": f"{case_root}/",
                "case_dir": str(case_root),
                "export_dir": str(case_root),
                "scan": str(case_root / "scan.nii.gz"),
                "target_mask": str(case_root / "target_mask.nii.gz"),
                "lesion_probability": str(case_root / "lesion_probability.nii.gz"),
            }
        )
        metric_rows.append({"case_id": case_id, "dice": dice})
    manifest = prediction_root / "prediction_export_manifest.csv"
    metrics = tmp_path / "selected_threshold_case_metrics.csv"
    threshold = tmp_path / "selected_threshold.json"
    _write_csv(manifest, manifest_rows)
    _write_csv(metrics, metric_rows)
    threshold.write_text(json.dumps({"selected_threshold": 0.5}))

    result = create_oof_qc_contact_sheet(
        prediction_manifests=[manifest],
        threshold_json=threshold,
        case_metrics_csv=metrics,
        output_png=tmp_path / "qc.png",
        case_count=6,
    )

    assert result["n_oof_cases"] == 10
    assert result["n_displayed_cases"] == 6
    assert result["selected_threshold"] == 0.5
    assert result["locked_test_used"] is False
    assert (tmp_path / "qc.png").stat().st_size > 0
    selected = _read_csv(tmp_path / "qc.csv")
    assert len(selected) == 6
    assert {row["qc_selection"] for row in selected} == {
        "lowest_dice",
        "distribution_reference",
    }


def test_extract_direct_ce_oof_artifact_extracts_only_qc_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    threshold = source / "selected_threshold.json"
    metrics = source / "selected_threshold_case_metrics.csv"
    threshold.write_text(json.dumps({"selected_threshold": 0.4}))
    _write_csv(metrics, [{"case_id": "case_00", "dice": 0.5}])
    manifests = []
    for fold in range(5):
        manifest = source / f"fold_{fold}_manifest.csv"
        _write_csv(manifest, [{"split": "validation", "case_id": f"case_{fold}"}])
        manifests.append(manifest)
    unrelated = source / "checkpoint.pth"
    unrelated.write_bytes(b"model")
    bundle = tmp_path / "artifact.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(
            threshold,
            arcname="thresholds/direct_ce_dice/selected_threshold.json",
        )
        archive.add(
            metrics,
            arcname=(
                "thresholds/direct_ce_dice/selected_threshold_case_metrics.csv"
            ),
        )
        for fold, manifest in enumerate(manifests):
            archive.add(
                manifest,
                arcname=(
                    f"runs/direct_ce_dice/fold_{fold}/prediction_exports/"
                    "validation/final/prediction_export_manifest.csv"
                ),
            )
        archive.add(unrelated, arcname="runs/direct_ce_dice/fold_0/checkpoint.pth")

    prediction_root, extracted_threshold, extracted_metrics = (
        extract_direct_ce_oof_artifact(
            artifact_bundle=bundle,
            staging_root=tmp_path / "staging",
        )
    )

    assert len(list(prediction_root.rglob("prediction_export_manifest.csv"))) == 5
    assert json.loads(extracted_threshold.read_text())["selected_threshold"] == 0.4
    assert _read_csv(extracted_metrics)[0]["case_id"] == "case_00"
    assert not list((tmp_path / "staging").rglob("checkpoint.pth"))
