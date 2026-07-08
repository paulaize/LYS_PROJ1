import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_signoff():
    path = REPO / "scripts/approve_ihc_threshold.py"
    spec = importlib.util.spec_from_file_location("approve_ihc_threshold", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_sweep(path: Path) -> None:
    rows = []
    for threshold, (target, control) in {
        100: (15.45, 15.19),
        250: (3.99, 0.35),
        500: (0.31, 0.003),
        1000: (0.005, 0.00016),
    }.items():
        rows.append(
            {
                "source_role": "target",
                "threshold": threshold,
                "igg_fitc_pct_positive_area": target,
            }
        )
        rows.append(
            {
                "source_role": "no_lys241_control",
                "threshold": threshold,
                "igg_fitc_pct_positive_area": control,
            }
        )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decision(path: Path, *, threshold=500.0, approved=True) -> None:
    data = {
        "animal_id": "BD_08_5D",
        "panel": "A",
        "review_type": "ihc_igg_fitc_threshold_manual_review",
        "review_date": "2026-07-08",
        "reviewer": "Paul-Andréas",
        "status": "manual_review_completed",
        "candidate_thresholds": [250.0, 500.0],
        "selected_threshold": threshold,
        "approved": approved,
        "section_decisions": [
            {
                "source_animal": "BD_08_5D",
                "source_role": "target",
                "section_id": "section_01",
                "usable_for_threshold": True,
            },
            {
                "source_animal": "C6S5",
                "source_role": "no_lys241_control",
                "section_id": "section_05",
                "usable_for_threshold": True,
            },
        ],
        "notes": "Manual review chose the stricter candidate after fold inspection.",
    }
    path.write_text(json.dumps(data) + "\n")


def test_threshold_signoff_writes_canonical_outputs(tmp_path: Path):
    signoff = _load_signoff()
    sweep = tmp_path / "sweep.csv"
    decision = tmp_path / "decision.json"
    out_dir = tmp_path / "out"
    _write_sweep(sweep)
    _decision(decision)

    json_path, csv_path = signoff.approve_threshold_decision(
        REPO / "config/animals/BD_08_5D.yml",
        panel="A",
        decision_path=decision,
        sweep_csv=sweep,
        out_dir=out_dir,
    )

    data = json.loads(json_path.read_text())
    assert data["igg_fitc_threshold"] == 500.0
    assert data["approval_status"] == "approved_control_calibrated"
    assert data["threshold_status_for_export"] == "approved"
    assert data["target_sections_used"] == ["section_01"]
    assert data["control_sections_used"] == {"C6S5:no_lys241_control": ["section_05"]}
    assert data["artifact_exclusion_annotation_names"] == ["artifact_exclude"]
    assert csv_path.exists()


def test_threshold_signoff_rejects_unapproved_decision(tmp_path: Path):
    signoff = _load_signoff()
    sweep = tmp_path / "sweep.csv"
    decision = tmp_path / "decision.json"
    _write_sweep(sweep)
    _decision(decision, approved=False)

    with pytest.raises(ValueError, match="approved=true"):
        signoff.approve_threshold_decision(
            REPO / "config/animals/BD_08_5D.yml",
            panel="A",
            decision_path=decision,
            sweep_csv=sweep,
            out_dir=tmp_path / "out",
        )


def test_threshold_signoff_rejects_non_candidate_threshold(tmp_path: Path):
    signoff = _load_signoff()
    sweep = tmp_path / "sweep.csv"
    decision = tmp_path / "decision.json"
    _write_sweep(sweep)
    _decision(decision, threshold=100.0)

    with pytest.raises(ValueError, match="dashboard candidates"):
        signoff.approve_threshold_decision(
            REPO / "config/animals/BD_08_5D.yml",
            panel="A",
            decision_path=decision,
            sweep_csv=sweep,
            out_dir=tmp_path / "out",
        )
