from pathlib import Path

import pandas as pd
import pytest

from src.ihc.ingest import ingest_qupath


def test_ingest_qupath_uses_igg_fitc_names(tmp_path: Path):
    csv_path = tmp_path / "ihc.csv"
    pd.DataFrame(
        [
            {
                "image": "section_01.vsi",
                "panel": "A",
                "region": "whole_section",
                "igg_fitc_pos_area_um2": 250.0,
                "total_area_um2": 1000.0,
                "dapi_count": -1,
                "qc_flag": "v1_area_only",
            }
        ]
    ).to_csv(csv_path, index=False)

    rows = ingest_qupath(csv_path)
    assert len(rows) == 2
    fitc = [r for r in rows if r["measure"] == "igg_fitc_pct_positive_area"][0]
    assert fitc["marker"] == "IgG-FITC"
    assert fitc["value"] == 25.0
    assert fitc["unit"] == "percent"
    assert fitc["area_mm2"] == 0.001
    assert "lys241" not in fitc["measure"].lower()


def test_ingest_rejects_legacy_fitc_column(tmp_path: Path):
    csv_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        [
            {
                "image": "section_01.vsi",
                "panel": "A",
                "region": "whole_section",
                "fitc_pos_area_um2": 250.0,
                "total_area_um2": 1000.0,
                "dapi_count": -1,
            }
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="igg_fitc_pos_area_um2"):
        ingest_qupath(csv_path)
