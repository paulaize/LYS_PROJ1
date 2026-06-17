"""IgG-FITC positivity threshold calibration scaffolding.

Specificity and positivity threshold are orthogonal: anti-human IgG confirms
what the FITC signal represents, but controls are still required to decide what
intensity counts as positive.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def calibrate_thresholds(config_path: str | Path) -> Path:
    from src.config import load_config

    cfg = load_config(config_path)
    cal_cfg = cfg.pipeline.get("ihc", {}).get("threshold_calibration", {})
    candidate_thresholds = cal_cfg.get("candidate_thresholds") or []
    if not candidate_thresholds:
        raise ValueError(
            "ihc.threshold_calibration.candidate_thresholds is unset. Add an "
            "exploratory threshold grid before running calibration."
        )

    controls = cfg.animal.get("ihc", {}).get("threshold_controls") or []
    if not controls:
        raise ValueError(
            f"[{cfg.animal_id}] no IHC threshold_controls are configured. Add a "
            "vehicle/no-LYS241/secondary-only control config before calibration."
        )

    rows: list[dict] = []
    for panel in sorted((cfg.animal.get("ihc", {}).get("panels") or {}).keys()):
        panel_cfg = cfg.panel_config(panel)
        control_paths = []
        control_ids = []
        for control in controls:
            control_config = control.get("config")
            if not control_config:
                continue
            control_cfg = load_config(control_config, repo_root=cfg.repo_root)
            control_panel = control_cfg.panel_config(panel)
            control_ids.append(control_cfg.animal_id)
            control_paths.extend(control_panel.get("vsi_files") or [])
        rows.append(
            {
                "animal_id": cfg.animal_id,
                "panel": panel,
                "status": cal_cfg.get("status", "exploratory_no_prior_qupath_threshold"),
                "reviewer": cal_cfg.get("reviewer", ""),
                "tissue_annotation": cfg.animal.get("ihc", {})
                .get("qupath", {})
                .get("tissue_annotation_name", "tissue_v1"),
                "igg_fitc_channel_index": cfg.panel_igg_fitc_channel_index(panel),
                "candidate_thresholds": ";".join(str(v) for v in candidate_thresholds),
                "target_vsi_files": ";".join(panel_cfg.get("vsi_files") or []),
                "control_animal_ids": ";".join(control_ids),
                "control_vsi_files": ";".join(control_paths),
                "approved_threshold": "",
                "approval_status": "not_approved",
                "notes": (
                    "Run export_threshold_sweep.groovy on target and control sections. "
                    "This manifest is exploratory and does not approve a threshold."
                ),
            }
        )

    out = cfg.work_dir() / "ihc_threshold_calibration_manifest.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{cfg.animal_id}] wrote exploratory IHC threshold calibration manifest: {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate IgG-FITC positivity thresholds.")
    parser.add_argument("--config", required=True, help="config/animals/<id>.yml")
    args = parser.parse_args()
    calibrate_thresholds(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
