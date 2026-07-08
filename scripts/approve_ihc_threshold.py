"""Validate a manual IHC threshold dashboard decision and write sign-off files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Threshold decision JSON not found: {path}")
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Threshold decision JSON must contain one object: {path}")
    return data


def _threshold_in(value: float, candidates: list[float]) -> bool:
    return any(
        math.isclose(value, candidate, rel_tol=0.0, abs_tol=1e-6)
        for candidate in candidates
    )


def _section_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_animal", "")),
        str(row.get("source_role", "")),
        str(row.get("section_id", "")),
    )


def _selected_keys(decision: dict[str, Any]) -> set[tuple[str, str, str]]:
    sections = decision.get("section_decisions")
    if not isinstance(sections, list):
        raise ValueError("Decision JSON must include section_decisions as a list")
    selected = set()
    for row in sections:
        if not isinstance(row, dict):
            raise ValueError("Each section_decisions item must be an object")
        key = _section_key(row)
        if not all(key):
            raise ValueError("Each section decision needs source_animal, source_role, section_id")
        if bool(row.get("usable_for_threshold", False)):
            selected.add(key)
    return selected


def _configured_selected_keys(cfg, panel: str) -> set[tuple[str, str, str]]:
    from build_ihc_threshold_review_dashboard import _configured_sections

    return {
        (ref.source_animal, ref.source_role, ref.section_id)
        for ref in _configured_sections(cfg, panel)
        if ref.selected and not ref.excluded
    }


def _selected_by_role(
    selected: set[tuple[str, str, str]],
) -> tuple[list[str], dict[str, list[str]]]:
    target: list[str] = []
    controls: dict[str, list[str]] = {}
    for animal, role, section_id in sorted(selected):
        if role == "target":
            target.append(section_id)
        else:
            controls.setdefault(f"{animal}:{role}", []).append(section_id)
    return target, controls


def _candidate_thresholds_from_sweep(
    sweep_csv: Path,
    *,
    max_control_pct: float,
    min_target_pct: float,
    min_target_control_fold: float,
) -> list[float]:
    from build_ihc_threshold_review_dashboard import _read_sweep_rows, summarize_thresholds

    rows = _read_sweep_rows(sweep_csv)
    summaries = summarize_thresholds(
        rows,
        max_control_pct=max_control_pct,
        min_target_pct=min_target_pct,
        min_target_control_fold=min_target_control_fold,
    )
    return [summary.threshold for summary in summaries if summary.candidate]


def approve_threshold_decision(
    config_path: str | Path,
    *,
    panel: str,
    decision_path: str | Path,
    sweep_csv: str | Path | None = None,
    out_dir: str | Path | None = None,
    max_control_pct: float = 1.0,
    min_target_pct: float = 0.05,
    min_target_control_fold: float = 5.0,
) -> tuple[Path, Path]:
    from src.config import load_config

    cfg = load_config(config_path)
    panel = str(panel)
    decision_path = Path(decision_path)
    decision = _load_json(decision_path)

    if decision.get("animal_id") != cfg.animal_id:
        raise ValueError(
            f"Decision animal_id={decision.get('animal_id')!r} does not match config animal "
            f"{cfg.animal_id!r}"
        )
    if str(decision.get("panel")) != panel:
        raise ValueError(
            f"Decision panel={decision.get('panel')!r} does not match --panel {panel!r}"
        )
    if decision.get("review_type") != "ihc_igg_fitc_threshold_manual_review":
        raise ValueError("Decision JSON has the wrong review_type")
    if decision.get("status") != "manual_review_completed":
        raise ValueError("Decision status must be manual_review_completed")
    if decision.get("approved") is not True:
        raise ValueError("Decision must have approved=true before sign-off")
    if not decision.get("reviewer"):
        raise ValueError("Decision must include reviewer")
    if not decision.get("review_date"):
        raise ValueError("Decision must include review_date")
    if not str(decision.get("notes", "")).strip():
        raise ValueError("Decision must include review notes explaining the threshold choice")

    selected_threshold = decision.get("selected_threshold")
    if selected_threshold is None:
        raise ValueError("Decision must include selected_threshold")
    selected_threshold = float(selected_threshold)

    dashboard_candidates = [float(v) for v in decision.get("candidate_thresholds") or []]
    if not _threshold_in(selected_threshold, dashboard_candidates):
        raise ValueError(
            f"Selected threshold {selected_threshold:g} is not in the dashboard candidates "
            f"{dashboard_candidates}"
        )

    sweep_csv = (
        Path(sweep_csv)
        if sweep_csv
        else cfg.work_dir() / f"ihc_threshold_sweep_panel_{panel}.csv"
    )
    recomputed_candidates = _candidate_thresholds_from_sweep(
        sweep_csv,
        max_control_pct=max_control_pct,
        min_target_pct=min_target_pct,
        min_target_control_fold=min_target_control_fold,
    )
    if not _threshold_in(selected_threshold, recomputed_candidates):
        raise ValueError(
            f"Selected threshold {selected_threshold:g} is not plausible under current sweep data. "
            f"Current candidates: {recomputed_candidates}"
        )

    selected = _selected_keys(decision)
    target_sections, control_sections = _selected_by_role(selected)
    if not target_sections:
        raise ValueError("At least one target section must be selected for threshold sign-off")
    if not control_sections:
        raise ValueError("At least one control section must be selected for threshold sign-off")

    configured = _configured_selected_keys(cfg, panel)
    section_selection_matches_config = selected == configured
    added = sorted(selected - configured)
    removed = sorted(configured - selected)

    artifact_names = (
        cfg.animal.get("ihc", {})
        .get("qupath", {})
        .get("artifact_exclusion_annotation_names", ["artifact_exclude"])
    )
    signoff = {
        "animal_id": cfg.animal_id,
        "panel": panel,
        "review_type": "ihc_igg_fitc_threshold_signoff",
        "approval_status": "approved_control_calibrated",
        "threshold_status_for_export": "approved",
        "approved": True,
        "igg_fitc_threshold": selected_threshold,
        "reviewer": decision["reviewer"],
        "review_date": decision["review_date"],
        "decision_source": str(decision_path),
        "sweep_csv": str(sweep_csv),
        "candidate_thresholds": dashboard_candidates,
        "candidate_criteria": {
            "max_control_pct": max_control_pct,
            "min_target_pct": min_target_pct,
            "min_target_control_fold": min_target_control_fold,
        },
        "target_sections_used": target_sections,
        "control_sections_used": control_sections,
        "section_selection_matches_config": section_selection_matches_config,
        "section_selection_added_vs_config": [list(v) for v in added],
        "section_selection_removed_vs_config": [list(v) for v in removed],
        "tissue_annotation_name": cfg.animal.get("ihc", {})
        .get("qupath", {})
        .get("tissue_annotation_name", "tissue_v1"),
        "artifact_exclusion_annotation_names": artifact_names,
        "notes": decision["notes"],
        "qc_flag": (
            "threshold_approved_control_calibrated;"
            "manual_dashboard_review;"
            "requires_reviewed_tissue_v1_and_artifact_exclude"
        ),
    }

    out_dir = Path(out_dir) if out_dir else cfg.work_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"ihc_threshold_signoff_panel_{panel}.json"
    csv_path = out_dir / f"ihc_threshold_signoff_panel_{panel}.csv"
    json_path.write_text(json.dumps(signoff, indent=2) + "\n")
    _write_signoff_csv(csv_path, signoff)
    return json_path, csv_path


def _write_signoff_csv(path: Path, signoff: dict[str, Any]) -> None:
    row = {
        "animal_id": signoff["animal_id"],
        "panel": signoff["panel"],
        "approval_status": signoff["approval_status"],
        "threshold_status_for_export": signoff["threshold_status_for_export"],
        "igg_fitc_threshold": signoff["igg_fitc_threshold"],
        "reviewer": signoff["reviewer"],
        "review_date": signoff["review_date"],
        "approved": signoff["approved"],
        "target_sections_used": ";".join(signoff["target_sections_used"]),
        "control_sections_used": json.dumps(signoff["control_sections_used"], sort_keys=True),
        "decision_source": signoff["decision_source"],
        "sweep_csv": signoff["sweep_csv"],
        "section_selection_matches_config": signoff["section_selection_matches_config"],
        "artifact_exclusion_annotation_names": ";".join(
            signoff["artifact_exclusion_annotation_names"]
        ),
        "notes": signoff["notes"],
        "qc_flag": signoff["qc_flag"],
    }
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Approve an IHC threshold dashboard decision.")
    parser.add_argument("--config", required=True, help="config/animals/<id>.yml")
    parser.add_argument("--panel", required=True, help="IHC panel, e.g. A")
    parser.add_argument("--decision", required=True, help="Downloaded dashboard decision JSON")
    parser.add_argument("--sweep-csv", default=None, help="Override threshold sweep CSV")
    parser.add_argument("--out-dir", default=None, help="Override output directory")
    parser.add_argument("--max-control-pct", type=float, default=1.0)
    parser.add_argument("--min-target-pct", type=float, default=0.05)
    parser.add_argument("--min-target-control-fold", type=float, default=5.0)
    args = parser.parse_args()

    json_path, csv_path = approve_threshold_decision(
        args.config,
        panel=args.panel,
        decision_path=args.decision,
        sweep_csv=args.sweep_csv,
        out_dir=args.out_dir,
        max_control_pct=args.max_control_pct,
        min_target_pct=args.min_target_pct,
        min_target_control_fold=args.min_target_control_fold,
    )
    print(f"Wrote threshold sign-off JSON: {json_path}")
    print(f"Wrote threshold sign-off CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
