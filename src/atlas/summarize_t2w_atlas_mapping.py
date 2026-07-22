"""Validate subject-space atlas labels and summarize native lesion overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.atlas.t2w_mapping import summarize_atlas_mappings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qc-slice-axis", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--require-approved-registration", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_atlas_mappings(
        manifest=Path(args.manifest),
        output_root=Path(args.output),
        qc_slice_axis=args.qc_slice_axis,
        require_approved_registration=args.require_approved_registration,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
