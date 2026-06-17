from pathlib import Path

import pytest

from src.ihc.calibrate import calibrate_thresholds

REPO = Path(__file__).resolve().parents[1]


def test_calibration_fails_until_controls_and_policy_exist():
    with pytest.raises(NotImplementedError, match="Specificity is resolved"):
        calibrate_thresholds(REPO / "config/animals/BD_08_5D.yml")
