from pathlib import Path

import pytest

from src.ihc.one_animal import (
    AUTO_THRESHOLD_STATUS,
    NO_CANDIDATE_STATUS,
    _deduplicate_rows,
    select_exploratory_candidate,
)

REPO = Path(__file__).resolve().parents[1]


def _sweep_rows(values):
    rows = []
    for threshold, (target, control) in values.items():
        rows.extend(
            [
                {
                    "source_role": "target",
                    "threshold": str(threshold),
                    "igg_fitc_pct_positive_area": str(target),
                },
                {
                    "source_role": "no_lys241_control",
                    "threshold": str(threshold),
                    "igg_fitc_pct_positive_area": str(control),
                },
            ]
        )
    return rows


def test_auto_candidate_reuses_dashboard_rules_and_selects_lowest_pass():
    result, summaries = select_exploratory_candidate(
        _sweep_rows(
            {
                100: (15.45, 15.19),
                250: (3.99, 0.35),
                500: (0.31, 0.003),
                1000: (0.005, 0.00016),
            }
        ),
        allow_auto_candidate=True,
    )

    assert result["status"] == AUTO_THRESHOLD_STATUS
    assert result["selected_threshold"] == 250
    assert result["passing_candidates"] == [250, 500]
    assert [summary.threshold for summary in summaries] == [100, 250, 500, 1000]
    assert result["formal_signoff_created"] is False


def test_auto_candidate_records_explicit_absence():
    result, _ = select_exploratory_candidate(
        _sweep_rows({100: (0.001, 5.0), 250: (0.001, 2.0)}),
        allow_auto_candidate=True,
    )

    assert result["status"] == NO_CANDIDATE_STATUS
    assert result["selected_threshold"] is None
    assert result["passing_candidates"] == []


def test_combined_rows_reject_duplicate_section_measurements():
    rows = [
        {
            "animal_id": "BD_08_5D",
            "panel": "A",
            "section_id": "section_01",
            "measure": "igg_fitc_positive_area",
            "igg_fitc_threshold": 250,
        },
        {
            "animal_id": "BD_08_5D",
            "panel": "A",
            "section_id": "section_01",
            "measure": "igg_fitc_positive_area",
            "igg_fitc_threshold": 250,
        },
    ]

    with pytest.raises(ValueError, match="Duplicate combined"):
        _deduplicate_rows(
            rows,
            key_fields=(
                "animal_id",
                "panel",
                "section_id",
                "measure",
                "igg_fitc_threshold",
            ),
            label="combined two-panel measurement",
        )
