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

    assert len(commands) == 16
    assert {cmd.source_animal for cmd in commands} == {"BD_08_5D", "C6S5"}
    assert {cmd.source_role for cmd in commands} == {"target", "no_lys241_control"}
    assert {cmd.series_index for cmd in commands} == {2, 3, 4, 5, 6, 7, 8, 9}
    assert all(cmd.panel == "A" for cmd in commands)
    assert all("export_threshold_sweep.groovy" in cmd.argv[-1] for cmd in commands)
    assert any("auto_if_missing,BD_08_5D,target" in part for part in commands[0].argv)


def test_threshold_sweep_commands_can_skip_target():
    runner = _load_runner()
    commands = runner.build_sweep_commands(
        REPO / "config/animals/BD_08_5D.yml",
        panels="B",
        include_target=False,
    )

    assert len(commands) == 8
    assert {cmd.source_animal for cmd in commands} == {"C6S5"}
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
