import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO / "scripts/run_ihc_quantification.py"
    spec = importlib.util.spec_from_file_location("run_ihc_quantification", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_signoff(path: Path, threshold: float = 500.0) -> None:
    path.write_text(
        json.dumps(
            {
                "animal_id": "BD_08_5D",
                "panel": "A",
                "approval_status": "approved_control_calibrated",
                "threshold_status_for_export": "approved",
                "approved": True,
                "igg_fitc_threshold": threshold,
            }
        )
    )


def test_quantification_exploratory_commands_use_selected_target_sections():
    runner = _load_runner()
    commands = runner.build_quantification_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="A",
        threshold=250,
        exploratory=True,
        tissue_mode="auto_if_missing",
        use_default_signoff=False,
    )

    assert len(commands) == 3
    assert {cmd.section_id for cmd in commands} == {"section_01", "section_03", "section_06"}
    assert {cmd.series_index for cmd in commands} == {2, 4, 7}
    assert all(cmd.source_role == "target" for cmd in commands)
    assert all(cmd.threshold == 250 for cmd in commands)
    assert all(cmd.threshold_status == "exploratory_not_final" for cmd in commands)
    assert all(cmd.tissue_mode == "auto_if_missing" for cmd in commands)
    assert all("export_measurements.groovy" in cmd.argv[-1] for cmd in commands)
    assert any(
        ",250.0,8.0,tissue_v1,section_01,exploratory_not_final,"
        "artifact_exclude,auto_if_missing,BD_08_5D,target"
        in part
        for part in commands[0].argv
    )


def test_quantification_rejects_unsigned_final_threshold():
    runner = _load_runner()
    with pytest.raises(ValueError, match="require --exploratory"):
        runner.build_quantification_commands(
            REPO / "config/animals/BD_08_5D.yml",
            panels="A",
            threshold=250,
            use_default_signoff=False,
        )


def test_quantification_uses_approved_signoff(tmp_path: Path):
    runner = _load_runner()
    signoff = tmp_path / "signoff.json"
    _write_signoff(signoff, threshold=500)

    commands = runner.build_quantification_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="A",
        signoff_path=signoff,
        use_default_signoff=False,
    )

    assert len(commands) == 3
    assert all(cmd.threshold == 500 for cmd in commands)
    assert all(cmd.threshold_status == "approved" for cmd in commands)
    assert all(cmd.tissue_mode == "require_reviewed" for cmd in commands)
    assert any(
        ",500.0,8.0,tissue_v1,section_01,approved,artifact_exclude,"
        "require_reviewed,BD_08_5D,target"
        in part
        for part in commands[0].argv
    )


def test_quantification_requires_threshold_or_signoff():
    runner = _load_runner()
    with pytest.raises(ValueError, match="No approved threshold sign-off"):
        runner.build_quantification_commands(
            REPO / "config/animals/BD_08_5D.yml",
            panels="A",
            exploratory=True,
            use_default_signoff=False,
        )


def test_write_tidy_measurements_preserves_quantification_provenance(tmp_path: Path):
    runner = _load_runner()
    raw = tmp_path / "raw.csv"
    tidy = tmp_path / "tidy.csv"
    pd.DataFrame(
        [
            {
                "source_animal": "BD_08_5D",
                "source_role": "target",
                "image": "BD_08_5D.vsi - EFI 40x_02",
                "panel": "A",
                "section_id": "section_01",
                "region": "tissue_v1",
                "tissue_annotation": "tissue_v1",
                "tissue_qc": "rough_tissue_auto_unreviewed",
                "artifact_annotation_names": "artifact_exclude",
                "artifact_annotation_count": 0,
                "igg_fitc_channel_index": 1,
                "igg_fitc_threshold": 250,
                "threshold_status": "exploratory_not_final",
                "downsample": 8,
                "igg_fitc_pos_area_um2": 25.0,
                "total_area_um2": 100.0,
                "tissue_area_um2": 100.0,
                "artifact_excluded_area_um2": 0.0,
                "dapi_count": -1,
                "qc_flag": (
                    "v1_tissue_area_only;threshold_exploratory_not_final;"
                    "rough_tissue_auto_unreviewed"
                ),
            }
        ]
    ).to_csv(raw, index=False)

    runner.write_tidy_measurements(raw, tidy)
    rows = pd.read_csv(tidy)

    assert len(rows) == 2
    fitc = rows[rows["measure"] == "igg_fitc_pct_positive_area"].iloc[0]
    assert fitc["value"] == 25.0
    assert fitc["source_animal"] == "BD_08_5D"
    assert fitc["section_id"] == "section_01"
    assert fitc["tissue_qc"] == "rough_tissue_auto_unreviewed"
    assert fitc["threshold_status"] == "exploratory_not_final"
    assert fitc["igg_fitc_threshold"] == 250


def test_quantification_accepts_explicit_auto_candidate_status():
    runner = _load_runner()
    commands = runner.build_quantification_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="A",
        threshold=250,
        exploratory=True,
        tissue_mode="auto_if_missing",
        use_default_signoff=False,
        threshold_status_override="threshold_auto_candidate_exploratory",
    )

    assert {cmd.threshold_status for cmd in commands} == {
        "threshold_auto_candidate_exploratory"
    }
