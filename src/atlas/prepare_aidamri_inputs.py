"""Stage frozen RatLesNetV2 inference inputs for AIDAmri v3 registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.atlas.t2w_mapping import prepare_aidamri_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-header-orientation",
        default="LIP",
        help="AIDAmri v3 input orientation encoded by the NIfTI affine",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_aidamri_inputs(
        inference_manifest=Path(args.inference_manifest),
        output_root=Path(args.output),
        expected_header_orientation=args.expected_header_orientation,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

