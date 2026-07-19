import csv
import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ratlesnetv2_finetune.scripts.audit_prepared_dataset import (
    audit_prepared_dataset,
)
from ratlesnetv2_finetune.scripts.calibrate_probability_threshold import (
    calibrate_probability_threshold,
)
from ratlesnetv2_finetune.scripts.create_grouped_cv import (
    _infer_subject_ids,
    create_grouped_cv,
)
from ratlesnetv2_finetune.scripts.evaluate_probability_ensemble import (
    evaluate_probability_ensemble,
)
from ratlesnetv2_finetune.scripts.finetune_ratlesnetv2 import (
    _lesion_probability_map,
    _prediction_to_binary,
)
from ratlesnetv2_finetune.scripts.normalize_prepared_dataset import (
    normalize_prepared_dataset,
)
from ratlesnetv2_finetune.scripts.package_prepared_dataset import (
    package_prepared_dataset,
)

SPACING = (0.07, 0.07, 0.5)


def _write_nifti(path: Path, data: np.ndarray, spacing=SPACING) -> None:
    affine = np.diag([*spacing, 1.0])
    image = nib.Nifti1Image(data, affine)
    zooms = spacing if data.ndim == 3 else (*spacing, 1.0)
    image.header.set_zooms(zooms)
    nib.save(image, path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def _make_prepared_dataset(root: Path, case_count: int = 13) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(case_count):
        case_id = f"case_{index:02d}"
        timepoint = "D7" if index == case_count - 1 else "D1"
        case_dir = root / "train" / "LYS" / timepoint / case_id
        case_dir.mkdir(parents=True)
        scan = np.ones((12, 10, 4, 1), dtype=np.float32) * (index + 1)
        label = np.zeros((12, 10, 4), dtype=np.uint8)
        lesion_size = index % 4 + 1
        label[2 : 2 + lesion_size, 3:5, 1] = 1
        _write_nifti(case_dir / "scan.nii.gz", scan)
        _write_nifti(case_dir / "scan_lesionIAM.nii.gz", label)
        _write_nifti(case_dir / "scan_lesion.nii.gz", label)
        rows.append(
            {
                "case_id": case_id,
                "animal_id": case_id,
                "split": "train",
                "study": "LYS",
                "timepoint": timepoint,
                "case_dir": str(case_dir),
                "scan_path": str(case_dir / "scan.nii.gz"),
                "label_path": str(case_dir / "scan_lesionIAM.nii.gz"),
                "lesion_voxels": int(label.sum()),
                "lesion_volume_mm3": float(label.sum() * np.prod(SPACING)),
            }
        )
    _write_csv(root / "manifest.csv", rows)
    return rows


def _reviewed_metadata(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for index, row in enumerate(rows):
        # The final D7 case shares its mouse with case_00 and must follow it.
        subject_id = "mouse_00" if index == len(rows) - 1 else f"mouse_{index:02d}"
        result.append(
            {
                "case_id": row["case_id"],
                "subject_id": subject_id,
                "cohort": f"cohort_{index % 3}",
                "timepoint": row["timepoint"],
                "acquisition_protocol": "rare_t2",
                "old_mask_version": "LYS_v0",
                "new_mask_version": "LYS_v1",
                "label_version": "LYS_v1",
                "reviewer": "reviewer_a",
                "correction_reason": "reviewed_no_change",
                "model_prediction_viewed_during_review": "false",
                "qc_status": "approved",
                "exclude": "false",
                "exclude_reason": "",
            }
        )
    return result


def _make_kaggle_altered_dataset(root: Path, *, alias_matches: bool = True) -> None:
    rows = []
    affine = np.diag([*SPACING, 1.0])
    for index, case_id in enumerate(("plain_case", "truncated_gzip_case")):
        case_dir = root / "train" / "LYS" / "mixed" / case_id
        case_dir.mkdir(parents=True)
        scan = np.arange(8 * 9 * 3, dtype=np.float32).reshape(8, 9, 3, 1)
        label = np.zeros((8, 9, 3), dtype=np.uint8)
        label[2:4, 3:6, index : index + 1] = 1
        alias = label.copy()
        if not alias_matches and index == 1:
            alias[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(scan, affine), case_dir / "scan.nii")
        if index == 0:
            nib.save(nib.Nifti1Image(label, affine), case_dir / "scan_lesionIAM.nii")
            nib.save(nib.Nifti1Image(alias, affine), case_dir / "scan_lesion.nii")
        else:
            iam_staged = root / "iam.nii.gz"
            alias_staged = root / "alias.nii.gz"
            nib.save(nib.Nifti1Image(label, affine), iam_staged)
            nib.save(nib.Nifti1Image(alias, affine), alias_staged)
            iam_staged.replace(case_dir / "scan_lesionIAM.n")
            alias_staged.replace(case_dir / "scan_lesion.nii.g")
        rows.append(
            {
                "case_id": case_id,
                "animal_id": case_id,
                "split": "train",
                "study": "LYS",
                "timepoint": "mixed",
                "case_dir": str(case_dir),
                "scan_path": str(case_dir / "scan.nii"),
                "label_path": str(case_dir / "scan_lesionIAM.nii"),
            }
        )
    _write_csv(root / "manifest.csv", rows)


def test_normalize_prepared_dataset_recovers_plain_and_truncated_gzip(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _make_kaggle_altered_dataset(source)

    result = normalize_prepared_dataset(
        source_root=source,
        output_base=tmp_path / "normalized",
        dataset_name="Synthetic",
        expected_cases=2,
    )

    assert len(list(result.rglob("scan.nii.gz"))) == 2
    assert len(list(result.rglob("scan_lesionIAM.nii.gz"))) == 2
    assert len(list(result.rglob("scan_lesion.nii.gz"))) == 2
    report = _read_csv(result / "normalization_report.csv")
    assert {row["source_iam_transport"] for row in report} == {"plain", "gzip"}
    assert {row["iam_alias_identical"] for row in report} == {"True"}


def test_normalize_prepared_dataset_rejects_mask_alias_disagreement(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _make_kaggle_altered_dataset(source, alias_matches=False)

    with pytest.raises(RuntimeError, match="IAM and alias lesion masks are not identical"):
        normalize_prepared_dataset(
            source_root=source,
            output_base=tmp_path / "normalized",
            dataset_name="Synthetic",
            expected_cases=2,
        )
    assert not (tmp_path / "normalized" / "Synthetic").exists()


def test_two_class_predictions_honor_probability_threshold_without_double_softmax():
    probabilities = np.zeros((1, 2, 2, 2, 2), dtype=np.float32)
    probabilities[:, 0] = 0.4
    probabilities[:, 1] = 0.6

    assert _prediction_to_binary(probabilities, threshold=0.5).all()
    assert not _prediction_to_binary(probabilities, threshold=0.8).any()
    assert np.allclose(_lesion_probability_map(probabilities), 0.6)


def test_conservative_subject_inference_groups_only_clear_d1_d7_pairs():
    th08_base = "Thrombin_08_PhIND__JD_TH08_S1_C1S3"
    th08_d7 = f"{th08_base}D7"
    stz_d1 = "ThrombinSTZ_02_PhIND__JD-PM-thrombin02STZ-D1-CH4"
    stz_d7 = "ThrombinSTZ_02_PhIND__JD-PM-thrombin02STZ-D7-CH4"
    unmatched = "Thrombin_08_PhIND__JD_TH08_S2_C23S4D7"

    subject, rule = _infer_subject_ids(
        {th08_base, th08_d7, stz_d1, stz_d7, unmatched}
    )

    assert subject[th08_base] == subject[th08_d7]
    assert subject[stz_d1] == subject[stz_d7]
    assert subject[unmatched].startswith("single::")
    assert rule[unmatched] == "singleton_no_clear_pair"


def test_qc_and_grouped_split_are_subject_disjoint(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = _make_prepared_dataset(prepared)

    qc_root = tmp_path / "qc"
    qc_rows = audit_prepared_dataset(
        input_root=prepared,
        output_root=qc_root,
        expected_spacing=SPACING,
        write_overlays=False,
    )
    assert len(qc_rows) == len(rows)
    template = _read_csv(qc_root / "metadata_template.csv")
    assert {row["subject_id"] for row in template} == {"TODO"}
    assert {row["qc_status"] for row in template} == {"needs_review"}

    metadata_path = tmp_path / "reviewed_metadata.csv"
    _write_csv(metadata_path, _reviewed_metadata(rows))
    split_root = tmp_path / "splits"
    summary = create_grouped_cv(
        input_root=prepared,
        metadata_path=metadata_path,
        output_root=split_root,
        test_fraction=0.2,
        folds=3,
        copy_mode="symlink",
    )
    assert summary["n_cases"] == len(rows)
    assignments = _read_csv(split_root / "split_assignments.csv")
    subject_partitions: dict[str, set[str]] = {}
    subject_folds: dict[str, set[str]] = {}
    for row in assignments:
        subject_partitions.setdefault(row["subject_id"], set()).add(row["outer_split"])
        if row["cv_fold"]:
            subject_folds.setdefault(row["subject_id"], set()).add(row["cv_fold"])
    assert all(len(values) == 1 for values in subject_partitions.values())
    assert all(len(values) == 1 for values in subject_folds.values())
    assert len({row["cv_fold"] for row in assignments if row["cv_fold"]}) == 3
    fold_zero = split_root / "folds" / "fold_0"
    assert list((fold_zero / "train").rglob("scan.nii.gz"))
    assert list((fold_zero / "validation").rglob("scan.nii.gz"))

    inferred_root = tmp_path / "inferred_splits"
    inferred_summary = create_grouped_cv(
        input_root=prepared,
        metadata_path=None,
        output_root=inferred_root,
        test_fraction=0.2,
        folds=3,
        infer_longitudinal_subjects=True,
        copy_mode="symlink",
    )
    assert inferred_summary["subject_grouping_mode"] == "conservative_case_id_inference"
    assert (inferred_root / "inferred_subject_groups.csv").is_file()


def test_threshold_calibration_uses_validation_only_and_finds_best_threshold(tmp_path):
    export_root = tmp_path / "exports"
    export_root.mkdir()
    records = []
    for index in range(2):
        case_dir = export_root / f"case_{index}"
        case_dir.mkdir()
        target = np.zeros((5, 5, 2), dtype=np.uint8)
        target[1:3, 1:3, 0] = 1
        probability = np.full(target.shape, 0.1, dtype=np.float32)
        probability[target > 0] = 0.7
        probability[4, 4, 1] = 0.4
        target_path = case_dir / "target_mask.nii.gz"
        probability_path = case_dir / "lesion_probability.nii.gz"
        _write_nifti(target_path, target)
        _write_nifti(probability_path, probability)
        records.append(
            {
                "case_id": f"case_{index}",
                "split": "validation",
                "target_mask": str(target_path),
                "lesion_probability": str(probability_path),
            }
        )
    manifest = export_root / "prediction_export_manifest.csv"
    _write_csv(manifest, records)

    selection = calibrate_probability_threshold(
        prediction_manifests=[manifest],
        output_root=tmp_path / "calibration",
        thresholds=[0.3, 0.5, 0.8],
        bootstrap_samples=20,
    )
    assert selection["selected_threshold"] == pytest.approx(0.5)
    assert selection["locked_test_used"] is False
    assert selection["mean_dice"] == pytest.approx(1.0)

    records[0]["split"] = "test"
    _write_csv(manifest, records)
    with pytest.raises(ValueError, match="validation.*only"):
        calibrate_probability_threshold(
            prediction_manifests=[manifest],
            output_root=tmp_path / "rejected",
            thresholds=[0.5],
        )


def test_locked_test_ensemble_requires_oof_threshold_and_averages_five_models(tmp_path):
    threshold_json = tmp_path / "selected_threshold.json"
    threshold_json.write_text(
        json.dumps(
            {
                "selected_threshold": 0.5,
                "selection_data": "out_of_fold_validation_only",
                "locked_test_used": False,
            }
        )
    )
    manifests = []
    for model_index in range(5):
        model_root = tmp_path / f"model_{model_index}"
        model_root.mkdir()
        records = []
        for case_index in range(2):
            case_dir = model_root / f"case_{case_index}"
            case_dir.mkdir()
            target = np.zeros((5, 5, 2), dtype=np.uint8)
            target[1:3, 1:3, 0] = 1
            probability = np.full(target.shape, 0.1 + model_index * 0.01, dtype=np.float32)
            probability[target > 0] = 0.7 + model_index * 0.01
            target_path = case_dir / "target_mask.nii.gz"
            probability_path = case_dir / "lesion_probability.nii.gz"
            _write_nifti(target_path, target)
            _write_nifti(probability_path, probability)
            records.append(
                {
                    "case_id": f"case_{case_index}",
                    "split": "test",
                    "target_mask": str(target_path),
                    "lesion_probability": str(probability_path),
                }
            )
        manifest = model_root / "prediction_export_manifest.csv"
        _write_csv(manifest, records)
        manifests.append(manifest)

    output = tmp_path / "locked_test"
    summary = evaluate_probability_ensemble(
        prediction_manifests=manifests,
        threshold_json=threshold_json,
        output_root=output,
        expected_models=5,
        bootstrap_samples=20,
    )

    assert summary["mean_dice"] == pytest.approx(1.0)
    assert summary["evaluation_data"] == "locked_test_once"
    assert summary["postprocessing"] == "none"
    assert len(_read_csv(output / "locked_test_case_metrics.csv")) == 2
    assert len(_read_csv(output / "ensemble_prediction_manifest.csv")) == 2
    with pytest.raises(FileExistsError, match="Do not rerun"):
        evaluate_probability_ensemble(
            prediction_manifests=manifests,
            threshold_json=threshold_json,
            output_root=output,
            expected_models=5,
        )


def test_versioned_package_excludes_cases_and_uses_portable_paths(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = _make_prepared_dataset(prepared, case_count=3)
    excluded = str(rows[1]["case_id"])
    archive_path = tmp_path / "LYS_v1.tar.gz"

    result = package_prepared_dataset(
        input_root=prepared,
        output_path=archive_path,
        dataset_name="LYS_T2w_manual_v1",
        label_version="LYS_v1",
        excluded_cases={excluded},
    )

    assert result["n_cases"] == 2
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        manifest_file = archive.extractfile("LYS_T2w_manual_v1/manifest.csv")
        assert manifest_file is not None
        manifest_rows = list(csv.DictReader(line.decode() for line in manifest_file))
    assert excluded not in {row["case_id"] for row in manifest_rows}
    assert all(row["label_version"] == "LYS_v1" for row in manifest_rows)
    assert all(not Path(row["case_dir"]).is_absolute() for row in manifest_rows)
    assert not any(excluded in name for name in names)
