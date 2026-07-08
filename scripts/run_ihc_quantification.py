"""Run v1 IHC IgG-FITC whole-section quantification through QuPath.

This script is the bridge between manual threshold review and first draft data.
It can run in two modes:

* final/reviewed mode: requires an approved threshold sign-off and an existing
  reviewed tissue annotation in a QuPath project/open image context.
* exploratory mode: requires an explicit threshold or sign-off, may create a
  rough tissue ROI when ``--tissue-mode auto_if_missing`` is used, and writes
  QC flags marking the rows as not final.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_ihc_threshold_sweeps import (  # noqa: E402
    _artifact_annotation_arg,
    _panel_filter,
    _section_items,
)


@dataclass(frozen=True)
class QuantificationCommand:
    source_animal: str
    source_role: str
    panel: str
    section_id: str
    series_index: int
    output_csv: Path
    tidy_csv: Path
    threshold: float
    threshold_status: str
    tissue_mode: str
    argv: list[str]

    def shell(self) -> str:
        return shlex.join(self.argv)


TIDY_FIELDS = [
    "source_animal",
    "source_role",
    "image",
    "panel",
    "section_id",
    "region",
    "marker",
    "cell_type",
    "measure",
    "value",
    "unit",
    "area_mm2",
    "n_cells",
    "tissue_annotation",
    "tissue_qc",
    "artifact_annotation_names",
    "artifact_annotation_count",
    "artifact_excluded_area_um2",
    "igg_fitc_channel_index",
    "igg_fitc_threshold",
    "threshold_status",
    "downsample",
    "qc_flag",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"IHC threshold sign-off JSON not found: {path}")
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"IHC threshold sign-off must be one JSON object: {path}")
    return data


def _default_signoff_path(cfg, panel: str) -> Path:
    return cfg.work_dir() / f"ihc_threshold_signoff_panel_{panel}.json"


def _validate_signoff(signoff: dict[str, Any], *, cfg, panel: str, path: Path) -> float:
    if signoff.get("animal_id") != cfg.animal_id:
        raise ValueError(
            f"Sign-off animal_id={signoff.get('animal_id')!r} does not match "
            f"config animal {cfg.animal_id!r}: {path}"
        )
    if str(signoff.get("panel")) != str(panel):
        raise ValueError(
            f"Sign-off panel={signoff.get('panel')!r} does not match requested "
            f"panel {panel!r}: {path}"
        )
    if signoff.get("approved") is not True:
        raise ValueError(f"IHC threshold sign-off is not approved: {path}")
    status = str(signoff.get("approval_status", ""))
    if not status.startswith("approved"):
        raise ValueError(f"IHC threshold sign-off has non-approved status {status!r}: {path}")
    threshold = signoff.get("igg_fitc_threshold")
    if threshold is None:
        raise ValueError(f"IHC threshold sign-off is missing igg_fitc_threshold: {path}")
    threshold = float(threshold)
    if threshold <= 0:
        raise ValueError(f"IHC threshold sign-off threshold must be > 0: {path}")
    return threshold


def _resolve_threshold(
    cfg,
    *,
    panel: str,
    threshold: float | None,
    signoff_path: str | Path | None,
    exploratory: bool,
    use_default_signoff: bool,
) -> tuple[float, str, Path | None]:
    chosen_signoff: Path | None = Path(signoff_path) if signoff_path else None
    if chosen_signoff is not None and not chosen_signoff.is_absolute():
        chosen_signoff = cfg.repo_root / chosen_signoff
    if chosen_signoff is None and use_default_signoff:
        default = _default_signoff_path(cfg, panel)
        if default.exists():
            chosen_signoff = default

    signoff_threshold: float | None = None
    signoff_status = "approved"
    if chosen_signoff is not None:
        signoff = _load_json(chosen_signoff)
        signoff_threshold = _validate_signoff(signoff, cfg=cfg, panel=panel, path=chosen_signoff)
        signoff_status = str(signoff.get("threshold_status_for_export", "approved"))
        if signoff_status != "approved":
            raise ValueError(
                f"Only threshold_status_for_export='approved' is accepted for final "
                f"exports; got {signoff_status!r} in {chosen_signoff}"
            )

    if threshold is not None:
        threshold = float(threshold)
        if threshold <= 0:
            raise ValueError("--threshold must be > 0")
        if signoff_threshold is not None and not math.isclose(
            threshold, signoff_threshold, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"Explicit --threshold {threshold:g} does not match sign-off "
                f"threshold {signoff_threshold:g}"
            )
        if signoff_threshold is None and not exploratory:
            raise ValueError(
                "Explicit IgG-FITC thresholds without an approved sign-off require "
                "--exploratory so the output is not mistaken for final data."
            )
        return threshold, "exploratory_not_final" if exploratory else signoff_status, chosen_signoff

    if signoff_threshold is None:
        raise ValueError(
            f"No approved threshold sign-off found for panel {panel}. Provide "
            "--signoff, or use --threshold with --exploratory for draft data."
        )
    threshold_status = "exploratory_not_final" if exploratory else signoff_status
    return signoff_threshold, threshold_status, chosen_signoff


def _qupath_executable(cfg) -> str:
    qupath_exe = cfg.pipeline.get("ihc", {}).get("qupath_executable")
    if not qupath_exe:
        raise ValueError("ihc.qupath_executable is unset in config/pipeline.yml")
    qupath = (
        cfg.resolve_input_path(qupath_exe)
        if not Path(qupath_exe).is_absolute()
        else Path(qupath_exe)
    )
    return str(qupath)


def _default_output_csv(cfg, panels: list[str]) -> Path:
    if len(panels) == 1:
        return cfg.work_dir() / f"ihc_quantification_panel_{panels[0]}.csv"
    return cfg.work_dir() / "ihc_quantification.csv"


def _default_tidy_csv(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_tidy.csv")


def build_quantification_commands(
    config_path: str | Path,
    *,
    panels: str | None = None,
    sections: str | None = None,
    threshold: float | None = None,
    signoff_path: str | Path | None = None,
    exploratory: bool = False,
    tissue_mode: str | None = None,
    downsample: float | None = None,
    output_csv: str | Path | None = None,
    tidy_csv: str | Path | None = None,
    limit: int | None = None,
    use_default_signoff: bool = True,
) -> list[QuantificationCommand]:
    from src.config import load_config

    cfg = load_config(config_path)
    ihc_cfg = cfg.animal.get("ihc", {})
    panel_cfgs = ihc_cfg.get("panels") or {}
    selected_panels = _panel_filter(panel_cfgs.keys(), panels)
    if not selected_panels:
        raise ValueError("No IHC panels were selected for quantification.")
    if signoff_path and len(selected_panels) > 1:
        raise ValueError("--signoff can only be used with one panel at a time")

    qupath = _qupath_executable(cfg)
    export_script = cfg.repo_root / "src/ihc/qupath/export_measurements.groovy"
    raw_out = Path(output_csv) if output_csv else _default_output_csv(cfg, selected_panels)
    tidy_out = Path(tidy_csv) if tidy_csv else _default_tidy_csv(raw_out)
    if not raw_out.is_absolute():
        raw_out = cfg.repo_root / raw_out
    if not tidy_out.is_absolute():
        tidy_out = cfg.repo_root / tidy_out

    cal_cfg = cfg.pipeline.get("ihc", {}).get("threshold_calibration", {})
    export_downsample = float(
        downsample
        if downsample is not None
        else cfg.pipeline.get("ihc", {}).get("default_downsample", cal_cfg.get("downsample", 8.0))
    )
    export_tissue_mode = tissue_mode or ("auto_if_missing" if exploratory else "require_reviewed")
    if export_tissue_mode not in {"require_reviewed", "auto_if_missing"}:
        raise ValueError("tissue_mode must be require_reviewed or auto_if_missing")
    if export_tissue_mode == "auto_if_missing" and not exploratory:
        raise ValueError("tissue_mode=auto_if_missing requires --exploratory")

    tissue_name = ihc_cfg.get("qupath", {}).get("tissue_annotation_name", "tissue_v1")
    qupath_cfg = ihc_cfg.get("qupath", {})
    artifact_arg = _artifact_annotation_arg(qupath_cfg)
    commands: list[QuantificationCommand] = []

    for panel in selected_panels:
        panel_cfg = cfg.panel_config(panel)
        panel_threshold, threshold_status, _ = _resolve_threshold(
            cfg,
            panel=panel,
            threshold=threshold,
            signoff_path=signoff_path,
            exploratory=exploratory,
            use_default_signoff=use_default_signoff,
        )
        channel_index = cfg.panel_igg_fitc_channel_index(panel)
        for vsi in panel_cfg.get("vsi_files") or []:
            vsi_path = cfg.resolve_input_path(vsi)
            if vsi_path is None:
                continue
            for series_index, section_id in _section_items(qupath_cfg, sections, panel_cfg):
                args = ",".join(
                    [
                        panel,
                        str(raw_out),
                        str(channel_index),
                        str(panel_threshold),
                        str(export_downsample),
                        tissue_name,
                        section_id,
                        threshold_status,
                        artifact_arg,
                        export_tissue_mode,
                        cfg.animal_id,
                        "target",
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
                    str(export_script),
                ]
                commands.append(
                    QuantificationCommand(
                        source_animal=cfg.animal_id,
                        source_role="target",
                        panel=panel,
                        section_id=section_id,
                        series_index=series_index,
                        output_csv=raw_out,
                        tidy_csv=tidy_out,
                        threshold=panel_threshold,
                        threshold_status=threshold_status,
                        tissue_mode=export_tissue_mode,
                        argv=argv,
                    )
                )

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        commands = commands[:limit]

    return commands


def write_tidy_measurements(raw_csv: str | Path, tidy_csv: str | Path) -> Path:
    from src.ihc.ingest import ingest_qupath

    raw_csv = Path(raw_csv)
    tidy_csv = Path(tidy_csv)
    rows = ingest_qupath(raw_csv)
    tidy_csv.parent.mkdir(parents=True, exist_ok=True)
    with tidy_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIDY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return tidy_csv


def run_quantification(
    config_path: str | Path,
    *,
    panels: str | None = None,
    sections: str | None = None,
    threshold: float | None = None,
    signoff_path: str | Path | None = None,
    exploratory: bool = False,
    tissue_mode: str | None = None,
    downsample: float | None = None,
    output_csv: str | Path | None = None,
    tidy_csv: str | Path | None = None,
    limit: int | None = None,
    run: bool = False,
    append: bool = False,
    timeout_seconds: int | None = 600,
    use_default_signoff: bool = True,
) -> list[QuantificationCommand]:
    commands = build_quantification_commands(
        config_path,
        panels=panels,
        sections=sections,
        threshold=threshold,
        signoff_path=signoff_path,
        exploratory=exploratory,
        tissue_mode=tissue_mode,
        downsample=downsample,
        output_csv=output_csv,
        tidy_csv=tidy_csv,
        limit=limit,
        use_default_signoff=use_default_signoff,
    )
    if not commands:
        raise ValueError("No IHC quantification commands were generated.")

    outputs = {cmd.output_csv for cmd in commands}
    for out in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        if run and out.exists() and not append:
            out.unlink()

    for i, cmd in enumerate(commands, start=1):
        print(
            f"[{i}/{len(commands)}] quantify {cmd.source_animal} "
            f"{cmd.source_role} panel {cmd.panel} {cmd.section_id} "
            f"series {cmd.series_index} threshold={cmd.threshold:g} "
            f"status={cmd.threshold_status} tissue_mode={cmd.tissue_mode}"
        )
        print(cmd.shell())
        if run:
            try:
                subprocess.run(cmd.argv, check=True, timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"QuPath IHC quantification timed out after {timeout_seconds}s for "
                    f"{cmd.source_animal} panel {cmd.panel} {cmd.section_id} "
                    f"series {cmd.series_index}."
                ) from exc

    if run:
        for out in outputs:
            tidy_out = commands[0].tidy_csv if len(outputs) == 1 else _default_tidy_csv(out)
            written = write_tidy_measurements(out, tidy_out)
            print(f"Wrote tidy IHC quantification CSV: {written}")
    else:
        print(
            "\nDry run only. Add RUN_ARGS=--run to execute these QuPath commands. "
            "For first draft data, use --exploratory --threshold <value> "
            "--tissue-mode auto_if_missing."
        )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v1 IHC IgG-FITC quantification.")
    parser.add_argument("--config", required=True, help="config/animals/<id>.yml")
    parser.add_argument("--panel", default=None, help="Panel to run: A, B, or A,B")
    parser.add_argument(
        "--section",
        default=None,
        help="Section id(s) to run, e.g. section_01 or section_01,section_03",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="IgG-FITC threshold. Without sign-off this requires --exploratory.",
    )
    parser.add_argument("--signoff", default=None, help="Approved threshold sign-off JSON")
    parser.add_argument(
        "--no-default-signoff",
        action="store_true",
        help="Do not auto-load work/<animal>/ihc_threshold_signoff_panel_<panel>.json",
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="Mark outputs as exploratory/not final; required for rough auto tissue.",
    )
    parser.add_argument(
        "--tissue-mode",
        choices=["require_reviewed", "auto_if_missing"],
        default=None,
        help="require_reviewed for final exports, auto_if_missing for draft direct-VSI exports",
    )
    parser.add_argument("--downsample", type=float, default=None, help="Measurement downsample")
    parser.add_argument("--output-csv", default=None, help="Raw QuPath measurement CSV")
    parser.add_argument("--tidy-csv", default=None, help="Tidy Python measurement CSV")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N commands")
    parser.add_argument("--run", action="store_true", help="Execute QuPath commands")
    parser.add_argument("--append", action="store_true", help="Append to existing raw CSV")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Per-QuPath-command timeout when --run is used",
    )
    args = parser.parse_args()

    run_quantification(
        args.config,
        panels=args.panel,
        sections=args.section,
        threshold=args.threshold,
        signoff_path=args.signoff,
        exploratory=args.exploratory,
        tissue_mode=args.tissue_mode,
        downsample=args.downsample,
        output_csv=args.output_csv,
        tidy_csv=args.tidy_csv,
        limit=args.limit,
        run=args.run,
        append=args.append,
        timeout_seconds=args.timeout_seconds,
        use_default_signoff=not args.no_default_signoff,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
