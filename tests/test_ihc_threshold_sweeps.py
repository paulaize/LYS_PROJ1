import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_runner():
    path = REPO / "scripts/run_ihc_threshold_sweeps.py"
    spec = importlib.util.spec_from_file_location("run_ihc_threshold_sweeps", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_threshold_sweep_commands_include_target_and_control_panel_a():
    runner = _load_runner()
    commands = runner.build_sweep_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="A",
    )

    assert len(commands) == 6
    assert {cmd.source_animal for cmd in commands} == {"BD_08_5D", "C6S5"}
    assert {cmd.source_role for cmd in commands} == {"target", "no_lys241_control"}
    assert {
        (cmd.source_animal, cmd.section_id, cmd.series_index) for cmd in commands
    } == {
        ("BD_08_5D", "section_01", 2),
        ("BD_08_5D", "section_03", 4),
        ("BD_08_5D", "section_06", 7),
        ("C6S5", "section_05", 6),
        ("C6S5", "section_06", 7),
        ("C6S5", "section_07", 8),
    }
    assert all(cmd.panel == "A" for cmd in commands)
    assert all("export_threshold_sweep.groovy" in cmd.argv[-1] for cmd in commands)
    assert any("auto_if_missing,BD_08_5D,target" in part for part in commands[0].argv)
    assert any("target,artifact_exclude" in part for part in commands[0].argv)


def test_threshold_sweep_commands_include_all_unreviewed_panel_b_controls():
    runner = _load_runner()
    commands = runner.build_sweep_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="B",
        include_target=False,
    )

    assert len(commands) == 8
    assert {cmd.source_animal for cmd in commands} == {"C6S5"}
    assert {cmd.source_role for cmd in commands} == {"no_lys241_control"}
    assert {cmd.section_id for cmd in commands} == {
        "section_01",
        "section_02",
        "section_03",
        "section_04",
        "section_05",
        "section_06",
        "section_07",
        "section_08",
    }


def test_threshold_sweep_commands_use_panel_b_target_and_unreviewed_controls():
    runner = _load_runner()
    commands = runner.build_sweep_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="B",
    )

    assert len(commands) == 11
    target = [cmd for cmd in commands if cmd.source_role == "target"]
    controls = [cmd for cmd in commands if cmd.source_role == "no_lys241_control"]
    assert {cmd.section_id for cmd in target} == {"section_05", "section_06", "section_08"}
    assert {cmd.series_index for cmd in target} == {6, 7, 9}
    assert len(controls) == 8
    assert all(cmd.panel == "B" for cmd in commands)


def test_threshold_sweep_commands_can_select_one_section_and_limit():
    runner = _load_runner()
    commands = runner.build_sweep_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="A",
        sections="section_01",
        limit=1,
    )

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.source_animal == "BD_08_5D"
    assert cmd.section_id == "section_01"
    assert cmd.series_index == 2
    assert any(",32.0,tissue_v1,section_01," in part for part in cmd.argv)
    assert any(",target,artifact_exclude" in part for part in cmd.argv)


def test_threshold_review_thumbnail_commands_write_to_review_tree():
    runner = _load_runner()
    commands = runner.build_review_thumbnail_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="A",
        sections="section_01",
        limit=2,
    )

    assert len(commands) == 2
    assert {cmd.source_animal for cmd in commands} == {"BD_08_5D", "C6S5"}
    assert {cmd.source_role for cmd in commands} == {"target", "no_lys241_control"}
    assert all(cmd.panel == "A" for cmd in commands)
    assert all(cmd.section_id == "section_01" for cmd in commands)
    assert all(cmd.series_index == 2 for cmd in commands)
    assert all("ihc_threshold_review" in str(cmd.output_dir) for cmd in commands)
    assert all("export_threshold_review_thumbnails.groovy" in cmd.argv[-1] for cmd in commands)
    assert any("/panel_A/BD_08_5D_target/section_01" in part for part in commands[0].argv)
    assert any(",64.0,tissue_v1,section_01,auto_if_missing," in part for part in commands[0].argv)
    assert any(",32.0,tissue_v1,section_01,auto_if_missing," in part for part in commands[1].argv)
    assert any(",target,1400,artifact_exclude" in part for part in commands[0].argv)


def test_section_qc_commands_write_to_qc_tree_and_shared_manifest():
    runner = _load_runner()
    commands = runner.build_section_qc_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="A",
        sections="section_01",
        limit=2,
    )

    assert len(commands) == 2
    assert {cmd.source_animal for cmd in commands} == {"BD_08_5D", "C6S5"}
    assert {cmd.source_role for cmd in commands} == {"target", "no_lys241_control"}
    assert all(cmd.panel == "A" for cmd in commands)
    assert all(cmd.section_id == "section_01" for cmd in commands)
    assert all(cmd.series_index == 2 for cmd in commands)
    assert {cmd.expected_channels for cmd in commands} == {4}
    assert all("ihc_section_qc" in str(cmd.output_dir) for cmd in commands)
    assert {cmd.manifest_csv.name for cmd in commands} == {"ihc_section_qc_manifest.csv"}
    assert all("export_section_qc_thumbnail.groovy" in cmd.argv[-1] for cmd in commands)
    assert any("/panel_A/BD_08_5D_target/section_01" in part for part in commands[0].argv)
    assert any(",section_01,2,BD_08_5D,target,1400,4" in part for part in commands[0].argv)


def test_diagnostic_commands_use_metadata_script():
    runner = _load_runner()
    commands = runner.build_diagnostic_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="A",
        sections="section_01",
        limit=1,
        include_controls=False,
    )

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.source_animal == "BD_08_5D"
    assert cmd.panel == "A"
    assert cmd.section_id == "section_01"
    assert cmd.series_index == 2
    assert cmd.output_csv.name == "ihc_qupath_diagnostics.csv"
    assert "diagnose_image.groovy" in cmd.argv[-1]
    assert any("A," in part and ",1,section_01,BD_08_5D,target" in part for part in cmd.argv)
