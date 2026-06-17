from pathlib import Path

import pandas as pd

from src.ihc.calibrate import calibrate_thresholds

REPO = Path(__file__).resolve().parents[1]


def test_calibration_writes_exploratory_manifest():
    cfg_path = REPO / "config/animals/BD_08_5D.yml"
    out = calibrate_thresholds(cfg_path)
    assert out.exists()

    df = pd.read_csv(out)
    assert set(df["panel"]) == {"A", "B"}
    assert set(df["approval_status"]) == {"not_approved"}
    assert set(df["control_animal_ids"]) == {"C6S5"}
    assert "100;250;500" in df.iloc[0]["candidate_thresholds"]
