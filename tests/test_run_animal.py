"""Tests for the v1 output table assembly."""
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.run_animal import write_minimal_table

REPO = Path(__file__).resolve().parents[1]


def test_ihc_rows_carry_resolved_specificity_and_panel_spectral_flags(tmp_path: Path):
    cfg = load_config("config/animals/BD_08_5D.yml", repo_root=REPO)
    cfg.pipeline["paths"]["outputs_root"] = str(tmp_path)

    rows = [
        {
            "panel": "A",
            "region": "tissue_v1",
            "cell_type": "all",
            "measure": "igg_fitc_pct_positive_area",
            "value": 12.5,
            "unit": "percent",
            "area_mm2": 1.0,
            "n_cells": "",
            "qc_flag": "v1_area_only",
        },
        {
            "panel": "B",
            "region": "tissue_v1",
            "cell_type": "all",
            "measure": "igg_fitc_pct_positive_area",
            "value": 9.0,
            "unit": "percent",
            "area_mm2": 1.2,
            "n_cells": "",
            "qc_flag": "v1_area_only",
        },
    ]

    out = write_minimal_table(cfg, None, rows)
    df = pd.read_csv(out)

    panel_a = df[df["panel"] == "A"].iloc[0]
    panel_b = df[df["panel"] == "B"].iloc[0]
    assert bool(panel_a["anti_igg_specificity_resolved"]) is True
    assert bool(panel_a["fitc_specific_to_lys241"]) is True
    assert panel_a["fitc_specificity_source"] == "confirmed_anti_human_IgG (Paul, 2026-06-17)"
    assert panel_a["fitc_igg_specificity"] == "anti_human_confirmed"
    assert panel_a["fitc_spectral_bleedthrough"] == "none"
    assert "fitc_spectral_bleedthrough" not in panel_a["qc_flag"]
    assert panel_b["fitc_spectral_bleedthrough"] == "none"
    assert "fitc_spectral_bleedthrough" not in panel_b["qc_flag"]


def test_mri_rows_leave_ihc_specificity_columns_blank(tmp_path: Path):
    cfg = load_config("config/animals/BD_08_5D.yml", repo_root=REPO)
    cfg.pipeline["paths"]["outputs_root"] = str(tmp_path)
    mri = {
        "raw_mm3": 1.0,
        "corrected_mm3": 0.9,
        "edited": False,
        "dice": 1.0,
        "reviewer": "tester",
        "qc_flag": "needs_human_review",
    }

    out = write_minimal_table(cfg, mri, [])
    df = pd.read_csv(out)

    assert df["anti_igg_specificity_resolved"].isna().all()
    assert df["fitc_specific_to_lys241"].isna().all()
    assert df["fitc_spectral_bleedthrough"].isna().all()
