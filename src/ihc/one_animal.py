"""Run the isolated one-animal, two-panel exploratory IHC workflow."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.build_ihc_threshold_review_dashboard import (
    ThresholdSummary,
    build_dashboard,
    summarize_thresholds,
)
from scripts.run_ihc_quantification import build_quantification_commands
from scripts.run_ihc_threshold_sweeps import (
    SECTION_QC_MANIFEST_FIELDS,
    _append_section_qc_failure,
    build_diagnostic_commands,
    build_review_thumbnail_commands,
    build_section_qc_commands,
    build_sweep_commands,
)
from src.config import Config, load_config
from src.ihc.ingest import ingest_qupath
from src.ihc.source_bundle import checksum_olympus_bundle, configured_panel_bundles

METHOD_VERSION = "ihc_one_animal_exploratory_v1"
AUTO_THRESHOLD_STATUS = "threshold_auto_candidate_exploratory"
NO_CANDIDATE_STATUS = "no_exploratory_candidate_available"
PANEL_B_CONTROL_STATUS = "technical_open_only_unreviewed"
CANDIDATE_CRITERIA = {
    "max_control_pct": 1.0,
    "min_target_pct": 0.05,
    "min_target_control_fold": 5.0,
}


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_dir: Path
    exit_code: int
    runtime_seconds: float
    panel_summaries: dict[str, dict[str, Any]]
    failure_count: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run_dir"] = str(self.run_dir)
        return data


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _command_key(command: Any) -> tuple[str, str, str, str]:
    return (
        str(command.source_animal),
        str(command.source_role),
        str(command.panel),
        str(command.section_id),
    )


def _record_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        role = str(record["source_role"])
        role_counts = counts.setdefault(role, {"successful": 0, "failed": 0})
        role_counts["successful" if record["status"] == "success" else "failed"] += 1
    return counts


def _execute_commands(
    commands: Iterable[Any],
    *,
    stage: str,
    run_dir: Path,
    repo_root: Path,
    timeout_seconds: int,
    source_checksums: dict[tuple[str, str], str],
    failures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, str]]]:
    commands = list(commands)
    records: list[dict[str, Any]] = []
    successful: set[tuple[str, str, str, str]] = set()
    log_dir = run_dir / "logs" / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    for index, command in enumerate(commands, start=1):
        key = _command_key(command)
        log_path = log_dir / (
            f"{index:03d}_{key[0]}_{key[1]}_panel_{key[2]}_{key[3]}.log"
        )
        print(
            f"[{stage} {index}/{len(commands)}] {key[0]} {key[1]} "
            f"panel {key[2]} {key[3]}"
        )
        started = time.monotonic()
        status = "success"
        exit_code: int | None = 0
        failure_reason = ""
        with log_path.open("w") as log:
            log.write(shlex.join(command.argv) + "\n\n")
            log.flush()
            try:
                result = subprocess.run(
                    command.argv,
                    cwd=repo_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                exit_code = result.returncode
                if exit_code != 0:
                    status = "failed"
                    failure_reason = f"QuPath exited with status {exit_code}"
            except subprocess.TimeoutExpired:
                status = "failed"
                exit_code = None
                failure_reason = f"Timed out after {timeout_seconds}s"
        record = {
            "stage": stage,
            "source_animal": key[0],
            "source_role": key[1],
            "panel": key[2],
            "section_id": key[3],
            "series_index": int(command.series_index),
            "status": status,
            "exit_code": exit_code,
            "runtime_seconds": round(time.monotonic() - started, 3),
            "log_path": str(log_path),
            "source_bundle_checksum": source_checksums.get((key[0], key[2]), ""),
        }
        records.append(record)
        if status == "success":
            successful.add(key)
        else:
            failure = {**record, "failure_reason": failure_reason}
            failures.append(failure)
            print(f"WARNING: {stage} failed for {key}: {failure_reason}")
    return records, successful


def select_exploratory_candidate(
    rows: list[dict[str, str]],
    *,
    allow_auto_candidate: bool,
) -> tuple[dict[str, Any], list[ThresholdSummary]]:
    """Apply the dashboard calculations and deterministic lowest-pass rule."""
    summaries = summarize_thresholds(rows, **CANDIDATE_CRITERIA)
    passing = [summary.threshold for summary in summaries if summary.candidate]
    selected = min(passing) if passing and allow_auto_candidate else None
    if selected is not None:
        status = AUTO_THRESHOLD_STATUS
    elif not passing:
        status = NO_CANDIDATE_STATUS
    else:
        status = "automatic_candidate_selection_disabled"
    result = {
        "status": status,
        "selected_threshold": selected,
        "passing_candidates": passing,
        "candidate_criteria": CANDIDATE_CRITERIA,
        "selection_rule": (
            "select the lowest passing configured threshold independently per panel"
        ),
        "formal_signoff_created": False,
        "approved": False,
    }
    return result, summaries


def _summary_rows(summaries: list[ThresholdSummary]) -> list[dict[str, Any]]:
    return [
        {
            "threshold": summary.threshold,
            "target_mean_pct_positive_area": summary.target_mean,
            "target_min_pct_positive_area": summary.target_min,
            "target_max_pct_positive_area": summary.target_max,
            "control_mean_pct_positive_area": summary.control_mean,
            "control_min_pct_positive_area": summary.control_min,
            "control_max_pct_positive_area": summary.control_max,
            "target_control_fold": summary.separation_fold,
            "passes_exploratory_criteria": summary.candidate,
            "candidate_reason": summary.reason,
        }
        for summary in summaries
    ]


def _validate_configuration(
    cfg: Config,
    panels: tuple[str, ...],
) -> dict[str, Config]:
    if cfg.animal_id != "BD_08_5D" or cfg.timepoint != "5d":
        raise ValueError(
            "The one-animal v1 runner is currently scoped to BD_08_5D at timepoint 5d"
        )
    controls = cfg.animal.get("ihc", {}).get("threshold_controls") or []
    if not controls:
        raise ValueError("At least one configured IHC threshold control is required")
    source_configs = {cfg.animal_id: cfg}
    for control in controls:
        config_path = control.get("config")
        if not config_path:
            raise ValueError("Every IHC threshold control needs a config path")
        control_cfg = load_config(config_path, repo_root=cfg.repo_root)
        source_configs[control_cfg.animal_id] = control_cfg
    if "C6S5" not in source_configs:
        raise ValueError("The current v1 exploratory runner requires configured control C6S5")

    for panel in panels:
        for source_cfg in source_configs.values():
            panel_cfg = source_cfg.panel_config(panel)
            channel_index = source_cfg.panel_igg_fitc_channel_index(panel)
            marker = str((panel_cfg.get("channel_map") or {}).get(channel_index, ""))
            if marker != "IgG-FITC":
                raise ValueError(
                    f"{source_cfg.animal_id} panel {panel}: channel {channel_index} "
                    f"is {marker!r}, not confirmed IgG-FITC"
                )
            configured_panel_bundles(source_cfg, panel)
    return source_configs


def _checksum_sources(
    source_configs: dict[str, Config],
    panels: tuple[str, ...],
) -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    bundle_rows: list[dict[str, Any]] = []
    source_checksums: dict[tuple[str, str], str] = {}
    for animal_id in sorted(source_configs):
        source_cfg = source_configs[animal_id]
        for panel in panels:
            configured = configured_panel_bundles(source_cfg, panel)
            if len(configured) != 1:
                raise ValueError(
                    f"{animal_id} panel {panel}: the v1 runner expects exactly one VSI bundle"
                )
            bundle = configured[0]
            print(f"[source checksum] {animal_id} panel {panel}: {bundle['vsi_path']}")
            manifest = checksum_olympus_bundle(
                bundle["vsi_path"],
                bundle["companion_dir"],
                configured_vsi_path=bundle["configured_vsi_path"],
                configured_companion_dir=bundle["configured_companion_dir"],
            )
            manifest["animal_id"] = animal_id
            manifest["panel"] = panel
            bundle_rows.append(manifest)
            source_checksums[(animal_id, panel)] = manifest[
                "combined_source_bundle_sha256"
            ]
    overall_payload = json.dumps(
        [
            {
                "animal_id": row["animal_id"],
                "panel": row["panel"],
                "combined_source_bundle_sha256": row[
                    "combined_source_bundle_sha256"
                ],
            }
            for row in bundle_rows
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest = {
        "schema_version": 1,
        "bundles": bundle_rows,
        "overall_source_set_sha256": hashlib.sha256(overall_payload).hexdigest(),
    }
    return manifest, source_checksums


def _append_section_qc_failures(
    commands: list[Any],
    records: list[dict[str, Any]],
) -> None:
    by_key = {_command_key(command): command for command in commands}
    for record in records:
        if record["status"] == "success":
            continue
        command = by_key[
            (
                record["source_animal"],
                record["source_role"],
                record["panel"],
                record["section_id"],
            )
        ]
        _append_section_qc_failure(
            command,
            section_qc_status="failed_qupath_open",
            section_qc_notes=(
                f"One-animal runner exit={record['exit_code']}; log={record['log_path']}"
            ),
        )


def _positive_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _technical_eligibility(
    *,
    panel: str,
    diagnostic_commands: list[Any],
    diagnostic_success: set[tuple[str, str, str, str]],
    section_qc_success: set[tuple[str, str, str, str]],
    diagnostic_csv: Path,
    section_qc_csv: Path,
    source_configs: dict[str, Config],
    source_checksums: dict[tuple[str, str], str],
    failures: list[dict[str, Any]],
) -> tuple[set[tuple[str, str, str, str]], list[dict[str, Any]]]:
    diagnostics = {_row_key(row): row for row in _read_csv(diagnostic_csv)}
    qc_rows = _read_csv(section_qc_csv)
    qc_by_key = {_row_key(row): row for row in qc_rows}
    eligible: set[tuple[str, str, str, str]] = set()
    for command in diagnostic_commands:
        key = _command_key(command)
        reasons: list[str] = []
        diag = diagnostics.get(key)
        qc = qc_by_key.get(key)
        if key not in diagnostic_success:
            reasons.append("diagnostic_command_failed")
        if key not in section_qc_success:
            reasons.append("section_qc_command_failed")
        if diag is None:
            reasons.append("missing_diagnostic_row")
        else:
            expected_channels = len(
                source_configs[key[0]].panel_config(panel).get("channel_map") or {}
            )
            if int(float(diag.get("channels", 0) or 0)) != expected_channels:
                reasons.append("unexpected_channel_count")
            if diag.get("channel_status") != "ok":
                reasons.append("configured_igg_fitc_channel_invalid")
            if not _positive_number(diag.get("pixel_width_um")) or not _positive_number(
                diag.get("pixel_height_um")
            ):
                reasons.append("invalid_pixel_calibration")
        if qc is None:
            reasons.append("missing_section_qc_row")
        elif qc.get("qc_flag") not in {"", "ok"}:
            reasons.append(f"section_qc_{qc.get('qc_flag')}")
        if reasons:
            failures.append(
                {
                    "stage": "technical_validation",
                    "source_animal": key[0],
                    "source_role": key[1],
                    "panel": key[2],
                    "section_id": key[3],
                    "series_index": int(command.series_index),
                    "status": "failed",
                    "exit_code": "",
                    "runtime_seconds": 0,
                    "log_path": "",
                    "source_bundle_checksum": source_checksums.get((key[0], key[2]), ""),
                    "failure_reason": ";".join(sorted(set(reasons))),
                }
            )
        else:
            eligible.add(key)

    for row in qc_rows:
        key = _row_key(row)
        selection = source_configs[key[0]].panel_config(panel).get("section_selection") or {}
        row["source_bundle_checksum"] = source_checksums.get((key[0], panel), "")
        row["technical_validation_status"] = (
            PANEL_B_CONTROL_STATUS
            if key in eligible and panel == "B" and key[1] != "target"
            else "technical_open_valid"
            if key in eligible
            else "section_qc_only_unreviewed"
        )
        row["control_selection_status"] = (
            PANEL_B_CONTROL_STATUS
            if panel == "B" and key[1] != "target"
            else str(selection.get("status", "pending_review"))
        )
        selected = {str(value) for value in selection.get("selected_section_ids") or []}
        row["selected_for_threshold_calibration"] = (
            "true" if key[3] in selected else "false"
        )
        row["selected_for_final_export"] = "false"
    return eligible, qc_rows


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("source_animal", "")),
        str(row.get("source_role", "")),
        str(row.get("panel", "")),
        str(row.get("section_id", "")),
    )


def _filter_commands(
    commands: Iterable[Any],
    eligible: set[tuple[str, str, str, str]],
) -> list[Any]:
    return [command for command in commands if _command_key(command) in eligible]


def _deduplicate_rows(
    rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    label: str,
) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key in seen:
            raise ValueError(f"Duplicate {label} row for key {key}")
        seen.add(key)
        result.append(row)
    return result


def _control_selection_status(source_configs: dict[str, Config], panel: str) -> str:
    control_panel = source_configs["C6S5"].panel_config(panel)
    selection = control_panel.get("section_selection") or {}
    return str(
        selection.get("exploratory_control_selection_status")
        or selection.get("status")
        or "pending_review"
    )


def _enrich_sweep_rows(
    rows: list[dict[str, str]],
    *,
    cfg: Config,
    panel: str,
    source_configs: dict[str, Config],
    source_checksums: dict[tuple[str, str], str],
    code_version: str,
) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        role = str(row.get("source_role", ""))
        source_animal = str(row.get("source_animal", ""))
        selection = source_configs[source_animal].panel_config(panel).get(
            "section_selection"
        ) or {}
        row = dict(row)
        row.update(
            {
                "study_id": cfg.pipeline.get("project", {}).get("name", "LYS_PROJ1"),
                "source_bundle_checksum": source_checksums[(source_animal, panel)],
                "technical_validation_status": (
                    PANEL_B_CONTROL_STATUS
                    if panel == "B" and role != "target"
                    else "technical_open_valid"
                ),
                "control_selection_status": (
                    PANEL_B_CONTROL_STATUS
                    if panel == "B" and role != "target"
                    else str(selection.get("status", "pending_review"))
                ),
                "threshold_status": "threshold_sweep_exploratory_not_final",
                "method_version": METHOD_VERSION,
                "config_version": cfg.version,
                "code_version": code_version,
            }
        )
        enriched.append(row)
    return sorted(
        enriched,
        key=lambda row: (
            row["source_role"] != "target",
            row["source_animal"],
            row["section_id"],
            float(row["threshold"]),
        ),
    )


def _acquisition_warning(diagnostic_rows: list[dict[str, str]]) -> dict[str, Any]:
    by_role: dict[str, list[float]] = {"target": [], "control": []}
    for row in diagnostic_rows:
        try:
            value = float(row["pixel_width_um"])
        except (KeyError, TypeError, ValueError):
            continue
        role = "target" if row.get("source_role") == "target" else "control"
        by_role[role].append(value)
    target = sum(by_role["target"]) / len(by_role["target"]) if by_role["target"] else None
    control = (
        sum(by_role["control"]) / len(by_role["control"])
        if by_role["control"]
        else None
    )
    pixel_size_mismatch = bool(
        target is not None
        and control is not None
        and not math.isclose(target, control, rel_tol=0.05, abs_tol=0.0)
    )
    return {
        "status": "acquisition_compatibility_unconfirmed",
        "target_mean_pixel_width_um": target,
        "control_mean_pixel_width_um": control,
        "pixel_size_mismatch_target_control": pixel_size_mismatch,
        "scientific_approval_blocked": True,
    }


def _enrich_measurement_rows(
    rows: list[dict[str, Any]],
    *,
    cfg: Config,
    panel: str,
    source_checksum: str,
    control_selection_status: str,
    code_version: str,
) -> list[dict[str, Any]]:
    flags = cfg.animal.get("interpretation_flags", {})
    enriched: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        qc_parts = [part for part in str(row.get("qc_flag", "")).split(";") if part]
        for required_flag in (
            "exploratory_only",
            "acquisition_compatibility_unconfirmed",
            "panel_section_matching_unconfirmed",
        ):
            if required_flag not in qc_parts:
                qc_parts.append(required_flag)
        row.update(
            {
                "study_id": cfg.pipeline.get("project", {}).get("name", "LYS_PROJ1"),
                "animal_id": cfg.animal_id,
                "timepoint": cfg.timepoint,
                "modality": "IHC",
                "source_bundle_checksum": source_checksum,
                "tissue_review_status": row.get("tissue_qc", ""),
                "control_selection_status": control_selection_status,
                "section_qc_status": "technical_open_valid_human_review_pending",
                "anti_igg_specificity_resolved": flags.get(
                    "anti_igg_specificity_resolved", False
                ),
                "fitc_specific_to_lys241": flags.get("fitc_specific_to_lys241", False),
                "fitc_igg_specificity": flags.get("fitc_igg_specificity", ""),
                "specificity_source": flags.get("specificity_source", ""),
                "fitc_spectral_bleedthrough": cfg.panel_config(panel).get(
                    "fitc_spectral_bleedthrough", ""
                ),
                "method_version": METHOD_VERSION,
                "config_version": cfg.version,
                "code_version": code_version,
                "qc_flag": ";".join(qc_parts),
            }
        )
        enriched.append(row)
    return enriched


def _limitations() -> list[str]:
    return [
        (
            "IgG-FITC is a control-calibrated signal/positive-area readout, "
            "not absolute LYS241 concentration."
        ),
        (
            "Automatic thresholds are exploratory candidates only and are not "
            "formal sign-off or approval."
        ),
        (
            "Rough tissue ROIs are automatic and unreviewed; final mode requires "
            "reviewed tissue_v1."
        ),
        "Artifact exclusion is not human-reviewed in this exploratory run.",
        (
            "Acquisition compatibility is unconfirmed; physical calibration does "
            "not prove raw-intensity comparability."
        ),
        (
            "Panel A and Panel B are not confirmed section-matched and are never "
            "pooled or paired by section ID."
        ),
        (
            "Panel B control sections are technical-open-only and not human-selected "
            "or visually approved."
        ),
        (
            "No DAPI density, cell segmentation, morphology, colocalization, atlas, "
            "registration, 3-D volume, or group inference is produced."
        ),
    ]


def run_one_animal(
    config_path: str | Path,
    panels: tuple[str, ...] = ("A", "B"),
    mode: str = "exploratory",
    allow_auto_candidate: bool = True,
    *,
    timeout_seconds: int = 900,
) -> RunSummary:
    """Execute the real, panel-separated exploratory workflow and return its summary."""
    started_wall = datetime.now(UTC)
    started_monotonic = time.monotonic()
    cfg = load_config(config_path)
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = cfg.repo_root / resolved_config_path
    panels = tuple(str(panel).strip() for panel in panels if str(panel).strip())
    if mode != "exploratory":
        raise ValueError(
            "The one-animal runner currently implements exploratory mode only; final "
            "mode still requires human control/tissue/artifact/threshold review."
        )
    if not panels or len(set(panels)) != len(panels):
        raise ValueError("panels must contain one or more unique configured panel names")
    source_configs = _validate_configuration(cfg, panels)
    source_manifest, source_checksums = _checksum_sources(source_configs, panels)

    code_commit = _git_value(cfg.repo_root, "rev-parse", "HEAD")
    dirty = bool(_git_value(cfg.repo_root, "status", "--short"))
    code_version = f"{code_commit}{'+dirty' if dirty else ''}"
    analysis_hash_payload = json.dumps(
        {
            "source_set": source_manifest["overall_source_set_sha256"],
            "config_sha256": _sha256_path(resolved_config_path),
            "pipeline_sha256": _sha256_path(cfg.repo_root / "config/pipeline.yml"),
            "panels": panels,
            "mode": mode,
            "allow_auto_candidate": allow_auto_candidate,
            "code_version": code_version,
        },
        sort_keys=True,
    ).encode()
    analysis_hash = hashlib.sha256(analysis_hash_payload).hexdigest()
    run_id = f"{started_wall.strftime('%Y%m%dT%H%M%SZ')}_{analysis_hash[:12]}"
    run_dir = cfg.work_dir() / "ihc_one_animal" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    source_manifest_path = _write_json(
        run_dir / "source_bundle_checksum_manifest.json", source_manifest
    )
    provenance = {
        "study_id": cfg.pipeline.get("project", {}).get("name", "LYS_PROJ1"),
        "animal_id": cfg.animal_id,
        "timepoint": cfg.timepoint,
        "mode": mode,
        "panels": panels,
        "method_version": METHOD_VERSION,
        "config_path": str(resolved_config_path),
        "config_sha256": _sha256_path(resolved_config_path),
        "pipeline_config_path": str(cfg.repo_root / "config/pipeline.yml"),
        "pipeline_config_sha256": _sha256_path(cfg.repo_root / "config/pipeline.yml"),
        "config_version": cfg.version,
        "code_version": code_version,
        "code_commit": code_commit,
        "working_tree_dirty": dirty,
        "python_version": sys.version,
        "qupath_executable": cfg.pipeline.get("ihc", {}).get("qupath_executable"),
        "threshold_grid": cfg.pipeline.get("ihc", {})
        .get("threshold_calibration", {})
        .get("candidate_thresholds", []),
        "candidate_criteria": CANDIDATE_CRITERIA,
        "automatic_candidate_selection_rule": (
            "lowest passing configured threshold, calculated independently per panel"
        ),
        "panel_A_historical_exploratory_threshold": 250,
        "source_manifest": str(source_manifest_path),
        "interpretation_flags": cfg.animal.get("interpretation_flags", {}),
    }
    _write_json(run_dir / "configuration_and_method_provenance.json", provenance)

    failures: list[dict[str, Any]] = []
    all_stage_records: list[dict[str, Any]] = []
    panel_summaries: dict[str, dict[str, Any]] = {}
    combined_rows: list[dict[str, Any]] = []

    for panel in panels:
        panel_dir = run_dir / f"panel_{panel}"
        panel_dir.mkdir(parents=True)

        diagnostic_commands = build_diagnostic_commands(
            config_path, panels=panel, output_dir=panel_dir
        )
        diagnostic_records, diagnostic_success = _execute_commands(
            diagnostic_commands,
            stage=f"panel_{panel}_diagnostics",
            run_dir=run_dir,
            repo_root=cfg.repo_root,
            timeout_seconds=timeout_seconds,
            source_checksums=source_checksums,
            failures=failures,
        )
        all_stage_records.extend(diagnostic_records)

        section_qc_commands = build_section_qc_commands(
            config_path, panels=panel, output_dir=panel_dir
        )
        section_qc_records, section_qc_success = _execute_commands(
            section_qc_commands,
            stage=f"panel_{panel}_section_qc",
            run_dir=run_dir,
            repo_root=cfg.repo_root,
            timeout_seconds=timeout_seconds,
            source_checksums=source_checksums,
            failures=failures,
        )
        _append_section_qc_failures(section_qc_commands, section_qc_records)
        all_stage_records.extend(section_qc_records)

        diagnostic_csv = panel_dir / "ihc_qupath_diagnostics.csv"
        section_qc_csv = panel_dir / "ihc_section_qc_manifest.csv"
        eligible, enriched_qc = _technical_eligibility(
            panel=panel,
            diagnostic_commands=diagnostic_commands,
            diagnostic_success=diagnostic_success,
            section_qc_success=section_qc_success,
            diagnostic_csv=diagnostic_csv,
            section_qc_csv=section_qc_csv,
            source_configs=source_configs,
            source_checksums=source_checksums,
            failures=failures,
        )
        qc_fields = [
            *SECTION_QC_MANIFEST_FIELDS,
            "source_bundle_checksum",
            "technical_validation_status",
            "control_selection_status",
        ]
        _write_csv(panel_dir / "section_qc_manifest.csv", enriched_qc, fieldnames=qc_fields)

        sweep_commands = _filter_commands(
            build_sweep_commands(config_path, panels=panel, output_dir=panel_dir), eligible
        )
        sweep_records, sweep_success = _execute_commands(
            sweep_commands,
            stage=f"panel_{panel}_threshold_sweep",
            run_dir=run_dir,
            repo_root=cfg.repo_root,
            timeout_seconds=timeout_seconds,
            source_checksums=source_checksums,
            failures=failures,
        )
        all_stage_records.extend(sweep_records)
        raw_sweep_path = panel_dir / f"ihc_threshold_sweep_panel_{panel}.csv"
        raw_sweep_rows = _deduplicate_rows(
            _read_csv(raw_sweep_path),
            key_fields=("source_animal", "source_role", "panel", "section_id", "threshold"),
            label=f"panel {panel} threshold sweep",
        )
        enriched_sweep = _enrich_sweep_rows(
            raw_sweep_rows,
            cfg=cfg,
            panel=panel,
            source_configs=source_configs,
            source_checksums=source_checksums,
            code_version=code_version,
        )
        _write_csv(panel_dir / "threshold_sweep_enriched.csv", enriched_sweep)
        candidate_result, threshold_summaries = select_exploratory_candidate(
            raw_sweep_rows,
            allow_auto_candidate=allow_auto_candidate,
        )
        candidate_result.update(
            {
                "panel": panel,
                "historical_exploratory_threshold": 250 if panel == "A" else None,
                "threshold_sweep_csv": str(raw_sweep_path),
            }
        )
        _write_json(panel_dir / "candidate_result.json", candidate_result)
        sensitivity_rows = _summary_rows(threshold_summaries)
        _write_csv(panel_dir / "threshold_sensitivity.csv", sensitivity_rows)

        review_commands = _filter_commands(
            build_review_thumbnail_commands(
                config_path, panels=panel, output_dir=panel_dir
            ),
            eligible,
        )
        review_records, _ = _execute_commands(
            review_commands,
            stage=f"panel_{panel}_threshold_review",
            run_dir=run_dir,
            repo_root=cfg.repo_root,
            timeout_seconds=timeout_seconds,
            source_checksums=source_checksums,
            failures=failures,
        )
        all_stage_records.extend(review_records)
        dashboard_path: Path | None = None
        if candidate_result["passing_candidates"]:
            dashboard_path, _, _ = build_dashboard(
                config_path,
                panel=panel,
                work_dir=panel_dir,
                **CANDIDATE_CRITERIA,
            )

        measurement_rows: list[dict[str, Any]] = []
        quant_records: list[dict[str, Any]] = []
        selected_threshold = candidate_result["selected_threshold"]
        raw_measurements = panel_dir / f"panel_{panel}_raw_measurements.csv"
        tidy_measurements = panel_dir / f"panel_{panel}_tidy_measurements.csv"
        if selected_threshold is not None:
            quant_commands = build_quantification_commands(
                config_path,
                panels=panel,
                threshold=float(selected_threshold),
                exploratory=True,
                tissue_mode="auto_if_missing",
                output_csv=raw_measurements,
                tidy_csv=tidy_measurements,
                use_default_signoff=False,
                threshold_status_override=AUTO_THRESHOLD_STATUS,
            )
            quant_commands = [
                command
                for command in quant_commands
                if _command_key(command) in eligible
                and _command_key(command) in sweep_success
            ]
            quant_records, _ = _execute_commands(
                quant_commands,
                stage=f"panel_{panel}_quantification",
                run_dir=run_dir,
                repo_root=cfg.repo_root,
                timeout_seconds=timeout_seconds,
                source_checksums=source_checksums,
                failures=failures,
            )
            all_stage_records.extend(quant_records)
            if raw_measurements.exists():
                measurement_rows = _enrich_measurement_rows(
                    ingest_qupath(raw_measurements),
                    cfg=cfg,
                    panel=panel,
                    source_checksum=source_checksums[(cfg.animal_id, panel)],
                    control_selection_status=_control_selection_status(
                        source_configs, panel
                    ),
                    code_version=code_version,
                )
                measurement_rows = _deduplicate_rows(
                    measurement_rows,
                    key_fields=(
                        "animal_id",
                        "panel",
                        "section_id",
                        "measure",
                        "igg_fitc_threshold",
                    ),
                    label=f"panel {panel} measurement",
                )
                measurement_rows.sort(
                    key=lambda row: (row["section_id"], row["measure"])
                )
                _write_csv(tidy_measurements, measurement_rows)
                combined_rows.extend(measurement_rows)

        diagnostics = _read_csv(diagnostic_csv)
        acquisition = _acquisition_warning(diagnostics)
        numeric_sensitivity = sum(
            1
            for row in sensitivity_rows
            if _finite_number(row["target_mean_pct_positive_area"])
            and _finite_number(row["control_mean_pct_positive_area"])
        )
        technical_keys = {_command_key(command) for command in diagnostic_commands}
        technical_failed = technical_keys - eligible
        panel_summaries[panel] = {
            "panel": panel,
            "source_bundle_checksums": {
                animal_id: source_checksums[(animal_id, panel)]
                for animal_id in sorted(source_configs)
            },
            "diagnostics": _record_counts(diagnostic_records),
            "section_qc": _record_counts(section_qc_records),
            "technical_sections": {
                "successful": len(eligible),
                "failed": len(technical_failed),
                "by_role": {
                    role: {
                        "successful": sum(1 for key in eligible if key[1] == role),
                        "failed": sum(1 for key in technical_failed if key[1] == role),
                    }
                    for role in sorted({key[1] for key in technical_keys})
                },
            },
            "threshold_sweep": _record_counts(sweep_records),
            "threshold_review": _record_counts(review_records),
            "quantification": _record_counts(quant_records),
            "numeric_threshold_sensitivity_rows": numeric_sensitivity,
            "candidate_status": candidate_result["status"],
            "selected_exploratory_candidate": selected_threshold,
            "passing_candidates": candidate_result["passing_candidates"],
            "measurement_rows": len(measurement_rows),
            "dashboard_path": str(dashboard_path) if dashboard_path else None,
            "acquisition_compatibility": acquisition,
        }
        _write_json(panel_dir / "panel_summary.json", panel_summaries[panel])

    combined_rows = _deduplicate_rows(
        combined_rows,
        key_fields=("animal_id", "panel", "section_id", "measure", "igg_fitc_threshold"),
        label="combined two-panel measurement",
    )
    combined_rows.sort(key=lambda row: (row["panel"], row["section_id"], row["measure"]))
    combined_path = _write_csv(run_dir / "combined_two_panel_tidy.csv", combined_rows)
    failures_path = _write_csv(
        run_dir / "per_section_failures.csv",
        failures,
        fieldnames=[
            "stage",
            "source_animal",
            "source_role",
            "panel",
            "section_id",
            "series_index",
            "status",
            "exit_code",
            "runtime_seconds",
            "log_path",
            "source_bundle_checksum",
            "failure_reason",
        ],
    )
    _write_json(run_dir / "stage_records.json", all_stage_records)
    limitations = _limitations()
    _write_json(
        run_dir / "qc_and_scientific_limitations.json",
        {
            "scientific_approval": False,
            "run_mode": mode,
            "limitations": limitations,
        },
    )
    no_candidate_panels = [
        panel
        for panel, summary in panel_summaries.items()
        if summary["candidate_status"] == NO_CANDIDATE_STATUS
    ]
    _write_json(
        run_dir / "next_action.json",
        {
            "status": (
                NO_CANDIDATE_STATUS if no_candidate_panels else "human_review_required"
            ),
            "panels_without_candidate": no_candidate_panels,
            "actions": [
                "Inspect section-QC and threshold-review images.",
                "Human-select Panel B control sections.",
                "Review/correct tissue_v1 and artifact_exclude annotations.",
                "Establish acquisition compatibility and approve thresholds separately per panel.",
            ],
        },
    )

    technically_complete = all(
        summary["numeric_threshold_sensitivity_rows"] > 0
        for summary in panel_summaries.values()
    )
    exit_code = 0 if technically_complete else 1
    finished_wall = datetime.now(UTC)
    runtime_seconds = round(time.monotonic() - started_monotonic, 3)
    run_manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "technical_exploratory_complete" if exit_code == 0 else "technical_incomplete",
        "exit_code": exit_code,
        "started_at": started_wall.isoformat(),
        "finished_at": finished_wall.isoformat(),
        "runtime_seconds": runtime_seconds,
        "animal_id": cfg.animal_id,
        "timepoint": cfg.timepoint,
        "panels": list(panels),
        "panel_pooling": False,
        "panel_section_matching": cfg.animal.get("ihc", {}).get(
            "panel_section_matching"
        ),
        "combined_two_panel_tidy_csv": str(combined_path),
        "source_bundle_checksum_manifest": str(source_manifest_path),
        "per_section_failures_csv": str(failures_path),
        "failure_count": len(failures),
        "panel_summaries": panel_summaries,
        "scientific_approval": False,
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)
    return RunSummary(
        run_id=run_id,
        run_dir=run_dir,
        exit_code=exit_code,
        runtime_seconds=runtime_seconds,
        panel_summaries=panel_summaries,
        failure_count=len(failures),
    )
