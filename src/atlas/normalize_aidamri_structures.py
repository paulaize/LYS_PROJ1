"""CLI for normalizing AIDAmri's split-parental structure lookup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.atlas.aidamri_lookup import normalize_parental_structure_lookup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize AIDAmri split-parental names and acronyms to CSV"
    )
    parser.add_argument("--names-table", required=True)
    parser.add_argument("--acronyms-table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aidamri-revision", required=True)
    parser.add_argument("--right-hemisphere-offset", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = normalize_parental_structure_lookup(
        names_table=Path(args.names_table),
        acronyms_table=Path(args.acronyms_table),
        output_csv=Path(args.output),
        aidamri_revision=args.aidamri_revision,
        right_hemisphere_offset=args.right_hemisphere_offset,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
