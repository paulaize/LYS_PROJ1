"""Run exploratory IHC IgG-FITC threshold sweeps through QuPath.

This is automation for the first-pass calibration stage only. It uses rough
auto tissue annotations in memory and writes CSVs flagged as not final. Final
v1 measurements still require reviewed tissue_v1 and an approved threshold.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class SweepCommand:
    source_animal: str
    source_role: str
    panel: str
    section_id: str
    series_index: int
    output_csv: Path
    argv: list[str]

    def shell(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class DiagnosticCommand:
    source_animal: str
    source_role: str
    panel: str
    section_id: str
    series_index: int
    output_csv: Path
    argv: list[str]

    def shell(self) -> str:
        return shlex.join(self.argv)


def _section_items(
    qupath_cfg: dict,
    requested_sections: str | None = None,
) -> list[tuple[int, str]]:
    series_indices = [int(v) for v in qupath_cfg.get("section_series_indices", [])]
    section_by_series = {
        int(k): str(v) for k, v in (qupath_cfg.get("section_ids_by_series_index") or {}).items()
    }
    selected = set(qupath_cfg.get("selected_section_ids") or [])
    excluded = set(qupath_cfg.get("excluded_section_ids") or [])
    requested = (
        {section.strip() for section in requested_sections.split(",") if section.strip()}
        if requested_sections
        else set()
    )

    items: list[tuple[int, str]] = []
    for series in series_indices:
        section_id = section_by_series.get(series, f"series_{series}")
        if selected and section_id not in selected:
            continue
        if section_id in excluded:
            continue
        if requested and section_id not in requested:
            continue
        items.append((series, section_id))
    return items


def _panel_filter(all_panels: Iterable[str], requested: str | None) -> list[str]:
    panels = sorted(str(p) for p in all_panels)
    if requested in (None, "", "all"):
        return panels
    wanted = {p.strip() for p in requested.split(",") if p.strip()}
    missing = wanted - set(panels)
    if missing:
        raise ValueError(f"Requested panel(s) not in config: {sorted(missing)}")
    return [p for p in panels if p in wanted]


def build_sweep_commands(
    config_path: str | Path,
    *,
    panels: str | None = None,
    sections: str | None = None,
    limit: int | None = None,
    include_target: bool = True,
    include_controls: bool = True,
) -> list[SweepCommand]:
    from src.config import load_config

    cfg = load_config(config_path)
    ihc_cfg = cfg.animal.get("ihc", {})
    panel_cfgs = ihc_cfg.get("panels") or {}
    selected_panels = _panel_filter(panel_cfgs.keys(), panels)

    qupath_exe = cfg.pipeline.get("ihc", {}).get("qupath_executable")
    if not qupath_exe:
        raise ValueError("ihc.qupath_executable is unset in config/pipeline.yml")
    qupath = str(
        cfg.resolve_input_path(qupath_exe)
        if not Path(qupath_exe).is_absolute()
        else qupath_exe
    )

    cal_cfg = cfg.pipeline.get("ihc", {}).get("threshold_calibration", {})
    thresholds = cal_cfg.get("candidate_thresholds") or []
    if not thresholds:
        raise ValueError("ihc.threshold_calibration.candidate_thresholds is unset")
    threshold_arg = ";".join(str(v) for v in thresholds)

    downsample = float(
        cal_cfg.get("downsample", cfg.pipeline.get("ihc", {}).get("default_downsample", 8.0))
    )
    tissue_name = ihc_cfg.get("qupath", {}).get("tissue_annotation_name", "tissue_v1")
    script = cfg.repo_root / "src/ihc/qupath/export_threshold_sweep.groovy"
    commands: list[SweepCommand] = []

    def add_source(source_cfg, source_role: str, panel: str) -> None:
        source_ihc = source_cfg.animal.get("ihc", {})
        source_panel = source_cfg.panel_config(panel)
        source_qupath = source_ihc.get("qupath", {})
        channel_index = source_cfg.panel_igg_fitc_channel_index(panel)
        out_csv = cfg.work_dir() / f"ihc_threshold_sweep_panel_{panel}.csv"

        for vsi in source_panel.get("vsi_files") or []:
            vsi_path = source_cfg.resolve_input_path(vsi)
            if vsi_path is None:
                continue
            for series_index, section_id in _section_items(source_qupath, sections):
                args = ",".join(
                    [
                        panel,
                        str(out_csv),
                        str(channel_index),
                        threshold_arg,
                        str(downsample),
                        tissue_name,
                        section_id,
                        "auto_if_missing",
                        source_cfg.animal_id,
                        source_role,
                    ]
                )
                argv = [
                    qupath,
                    "script",
                    "--image",
                    str(vsi_path),
                    "--server",
                    f"[--classname,BioFormatsServerBuilder,--series,{series_index}]",
                    "--args",
                    args,
                    str(script),
                ]
                commands.append(
                    SweepCommand(
                        source_animal=source_cfg.animal_id,
                        source_role=source_role,
                        panel=panel,
                        section_id=section_id,
                        series_index=series_index,
                        output_csv=out_csv,
                        argv=argv,
                    )
                )

    if include_target:
        for panel in selected_panels:
            add_source(cfg, "target", panel)

    if include_controls:
        for control in ihc_cfg.get("threshold_controls") or []:
            control_config = control.get("config")
            if not control_config:
                continue
            control_cfg = load_config(control_config, repo_root=cfg.repo_root)
            role = str(control.get("role", "control"))
            for panel in selected_panels:
                add_source(control_cfg, role, panel)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        commands = commands[:limit]

    return commands


def build_diagnostic_commands(
    config_path: str | Path,
    *,
    panels: str | None = None,
    sections: str | None = None,
    limit: int | None = None,
    include_target: bool = True,
    include_controls: bool = True,
) -> list[DiagnosticCommand]:
    from src.config import load_config

    cfg = load_config(config_path)
    ihc_cfg = cfg.animal.get("ihc", {})
    panel_cfgs = ihc_cfg.get("panels") or {}
    selected_panels = _panel_filter(panel_cfgs.keys(), panels)

    qupath_exe = cfg.pipeline.get("ihc", {}).get("qupath_executable")
    if not qupath_exe:
        raise ValueError("ihc.qupath_executable is unset in config/pipeline.yml")
    qupath = str(
        cfg.resolve_input_path(qupath_exe)
        if not Path(qupath_exe).is_absolute()
        else qupath_exe
    )

    tissue_name = ihc_cfg.get("qupath", {}).get("tissue_annotation_name", "tissue_v1")
    _ = tissue_name  # keep diagnostics tied to the same v1 QuPath config block.
    script = cfg.repo_root / "src/ihc/qupath/diagnose_image.groovy"
    out_csv = cfg.work_dir() / "ihc_qupath_diagnostics.csv"
    commands: list[DiagnosticCommand] = []

    def add_source(source_cfg, source_role: str, panel: str) -> None:
        source_ihc = source_cfg.animal.get("ihc", {})
        source_panel = source_cfg.panel_config(panel)
        source_qupath = source_ihc.get("qupath", {})
        channel_index = source_cfg.panel_igg_fitc_channel_index(panel)

        for vsi in source_panel.get("vsi_files") or []:
            vsi_path = source_cfg.resolve_input_path(vsi)
            if vsi_path is None:
                continue
            for series_index, section_id in _section_items(source_qupath, sections):
                args = ",".join(
                    [
                        panel,
                        str(out_csv),
                        str(channel_index),
                        section_id,
                        source_cfg.animal_id,
                        source_role,
                    ]
                )
                argv = [
                    qupath,
                    "script",
                    "--image",
                    str(vsi_path),
                    "--server",
                    f"[--classname,BioFormatsServerBuilder,--series,{series_index}]",
                    "--args",
                    args,
                    str(script),
                ]
                commands.append(
                    DiagnosticCommand(
                        source_animal=source_cfg.animal_id,
                        source_role=source_role,
                        panel=panel,
                        section_id=section_id,
                        series_index=series_index,
                        output_csv=out_csv,
                        argv=argv,
                    )
                )

    if include_target:
        for panel in selected_panels:
            add_source(cfg, "target", panel)

    if include_controls:
        for control in ihc_cfg.get("threshold_controls") or []:
            control_config = control.get("config")
            if not control_config:
                continue
            control_cfg = load_config(control_config, repo_root=cfg.repo_root)
            role = str(control.get("role", "control"))
            for panel in selected_panels:
                add_source(control_cfg, role, panel)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        commands = commands[:limit]

    return commands


def run_threshold_sweeps(
    config_path: str | Path,
    *,
    panels: str | None = None,
    sections: str | None = None,
    limit: int | None = None,
    run: bool = False,
    append: bool = False,
    timeout_seconds: int | None = 600,
    include_target: bool = True,
    include_controls: bool = True,
) -> list[SweepCommand]:
    commands = build_sweep_commands(
        config_path,
        panels=panels,
        sections=sections,
        limit=limit,
        include_target=include_target,
        include_controls=include_controls,
    )
    if not commands:
        raise ValueError("No IHC threshold sweep commands were generated.")

    outputs = {cmd.output_csv for cmd in commands}
    for out in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        if run and out.exists() and not append:
            out.unlink()

    for i, cmd in enumerate(commands, start=1):
        print(
            f"[{i}/{len(commands)}] {cmd.source_animal} {cmd.source_role} "
            f"panel {cmd.panel} {cmd.section_id} series {cmd.series_index}"
        )
        print(cmd.shell())
        if run:
            try:
                subprocess.run(cmd.argv, check=True, timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"QuPath threshold sweep timed out after {timeout_seconds}s for "
                    f"{cmd.source_animal} panel {cmd.panel} {cmd.section_id} "
                    f"series {cmd.series_index}. Run ihc-diagnose on the same section "
                    "to determine whether QuPath/Bio-Formats opening or the pixel "
                    "sweep is the slow step."
                ) from exc

    if not run:
        print(
            "\nDry run only. Add RUN_ARGS=--run to execute these QuPath commands. "
            "For first testing, prefer RUN_ARGS=\"--panel A --section section_01 "
            "--target-only --limit 1 --run\"."
        )
    return commands


def run_diagnostics(
    config_path: str | Path,
    *,
    panels: str | None = None,
    sections: str | None = None,
    limit: int | None = None,
    run: bool = False,
    append: bool = False,
    timeout_seconds: int | None = 180,
    include_target: bool = True,
    include_controls: bool = True,
) -> list[DiagnosticCommand]:
    commands = build_diagnostic_commands(
        config_path,
        panels=panels,
        sections=sections,
        limit=limit,
        include_target=include_target,
        include_controls=include_controls,
    )
    if not commands:
        raise ValueError("No IHC diagnostic commands were generated.")

    outputs = {cmd.output_csv for cmd in commands}
    for out in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        if run and out.exists() and not append:
            out.unlink()

    for i, cmd in enumerate(commands, start=1):
        print(
            f"[{i}/{len(commands)}] diagnose {cmd.source_animal} {cmd.source_role} "
            f"panel {cmd.panel} {cmd.section_id} series {cmd.series_index}"
        )
        print(cmd.shell())
        if run:
            try:
                subprocess.run(cmd.argv, check=True, timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"QuPath diagnostic timed out after {timeout_seconds}s for "
                    f"{cmd.source_animal} panel {cmd.panel} {cmd.section_id} "
                    f"series {cmd.series_index}. This points to slow direct "
                    ".vsi/Bio-Formats series opening, before the threshold sweep itself."
                ) from exc

    if not run:
        print(
            "\nDry run only. Add RUN_ARGS=--run to execute these QuPath diagnostics. "
            "For first testing, use RUN_ARGS=\"--panel A --section section_01 "
            "--target-only --limit 1 --run\"."
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exploratory IHC threshold sweeps.")
    parser.add_argument("--config", required=True, help="config/animals/<id>.yml")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Only open the configured VSI series and write metadata; do not sweep pixels",
    )
    parser.add_argument("--panel", default=None, help="Panel to run: A, B, or A,B")
    parser.add_argument(
        "--section",
        default=None,
        help="Section id(s) to run, e.g. section_01 or section_01,section_02",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N commands")
    parser.add_argument("--run", action="store_true", help="Execute QuPath commands")
    parser.add_argument("--append", action="store_true", help="Append to existing sweep CSVs")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help=(
            "Per-QuPath-command timeout when --run is used; defaults to 180 "
            "for diagnostics, 600 for sweeps"
        ),
    )
    parser.add_argument("--target-only", action="store_true", help="Skip configured controls")
    parser.add_argument("--controls-only", action="store_true", help="Skip target animal")
    args = parser.parse_args()

    if args.target_only and args.controls_only:
        raise ValueError("--target-only and --controls-only are mutually exclusive")

    timeout_seconds = args.timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = 180 if args.diagnose else 600

    run_kwargs = {
        "panels": args.panel,
        "sections": args.section,
        "limit": args.limit,
        "run": args.run,
        "append": args.append,
        "timeout_seconds": timeout_seconds,
        "include_target": not args.controls_only,
        "include_controls": not args.target_only,
    }
    if args.diagnose:
        run_diagnostics(args.config, **run_kwargs)
    else:
        run_threshold_sweeps(args.config, **run_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
