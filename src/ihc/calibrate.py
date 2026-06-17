"""IgG-FITC positivity threshold calibration.

Specificity and positivity threshold are orthogonal: anti-human IgG confirms
what the FITC signal represents, but controls are still required to decide what
intensity counts as positive.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def calibrate_thresholds(config_path: str | Path) -> None:
    from src.config import load_config

    cfg = load_config(config_path)
    raise NotImplementedError(
        f"[{cfg.animal_id}] IgG-FITC threshold calibration is not implemented yet. "
        "Specificity is resolved as anti-human IgG for LYS241, but positivity "
        "thresholds still require configured vehicle/secondary-only/control images "
        "and an approved calibration policy."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate IgG-FITC positivity thresholds.")
    parser.add_argument("--config", required=True, help="config/animals/<id>.yml")
    args = parser.parse_args()
    calibrate_thresholds(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
