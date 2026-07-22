"""Validate a completed masked rigid partial-slab atlas registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.atlas.constrained_registration import (
    validate_constrained_rigid_registration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--structures-csv", required=True)
    parser.add_argument("--aidamri-revision", required=True)
    parser.add_argument("--niftyreg-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--registration-runtime-seconds", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_constrained_rigid_registration(
        prepared_root=Path(args.prepared_root),
        structures_csv=Path(args.structures_csv),
        aidamri_revision=args.aidamri_revision,
        niftyreg_revision=args.niftyreg_revision,
        container_image_id=args.container_image_id,
        registration_runtime_seconds=args.registration_runtime_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
