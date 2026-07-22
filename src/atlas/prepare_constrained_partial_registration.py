"""Prepare an approved AIDAmri slab for masked rigid subject registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.atlas.constrained_registration import prepare_constrained_registration_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--subject-t2", required=True)
    parser.add_argument("--subject-brain-mask", required=True)
    parser.add_argument("--lesion-mask", required=True)
    parser.add_argument("--template-t2", required=True)
    parser.add_argument("--template-atlas-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slab-edge-padding-mm", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_constrained_registration_inputs(
        selection_json=Path(args.selection_json),
        candidates_csv=Path(args.candidates_csv),
        subject_t2=Path(args.subject_t2),
        subject_brain_mask=Path(args.subject_brain_mask),
        lesion_mask=Path(args.lesion_mask),
        template_t2=Path(args.template_t2),
        template_atlas_labels=Path(args.template_atlas_labels),
        output_root=Path(args.output),
        slab_edge_padding_mm=args.slab_edge_padding_mm,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
