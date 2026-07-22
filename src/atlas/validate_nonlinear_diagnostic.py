"""CLI for validating a bounded nonlinear partial-slab diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.atlas.nonlinear_diagnostic import DIAGNOSTICS, validate_nonlinear_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--structures-csv", required=True, type=Path)
    parser.add_argument("--diagnostic-id", required=True, choices=sorted(DIAGNOSTICS))
    parser.add_argument("--aidamri-revision", required=True)
    parser.add_argument("--niftyreg-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--registration-runtime-seconds", type=float)
    args = parser.parse_args()
    result = validate_nonlinear_diagnostic(
        prepared_root=args.prepared_root,
        structures_csv=args.structures_csv,
        diagnostic_id=args.diagnostic_id,
        aidamri_revision=args.aidamri_revision,
        niftyreg_revision=args.niftyreg_revision,
        container_image_id=args.container_image_id,
        registration_runtime_seconds=args.registration_runtime_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
