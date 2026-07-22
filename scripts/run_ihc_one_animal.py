"""Thin CLI for the Qt-free Python IHC one-animal orchestrator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ihc.one_animal import run_one_animal  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated exploratory IgG-FITC analysis for one animal and two panels."
    )
    parser.add_argument("--config", required=True, help="config/animals/<id>.yml")
    parser.add_argument("--panels", default="A,B", help="Comma-separated panel names")
    parser.add_argument("--mode", choices=["exploratory"], default="exploratory")
    parser.add_argument("--allow-auto-candidate", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Required safety flag: execute source hashing and real QuPath processing",
    )
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required; this entry point does not substitute mocks or stale data")

    summary = run_one_animal(
        args.config,
        panels=tuple(panel.strip() for panel in args.panels.split(",") if panel.strip()),
        mode=args.mode,
        allow_auto_candidate=args.allow_auto_candidate,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
