"""Create a hash-bound AP slab review for partial-volume AIDAmri mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.atlas.partial_volume import prepare_partial_volume_slab_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--subject-t2", required=True)
    parser.add_argument("--subject-brain-mask", required=True)
    parser.add_argument("--lesion-mask", required=True)
    parser.add_argument("--template-t2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-orientation", default="LIP")
    parser.add_argument("--top-candidates", type=int, default=6)
    parser.add_argument("--minimum-candidate-separation-mm", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_partial_volume_slab_review(
        case_id=args.case_id,
        subject_t2=Path(args.subject_t2),
        subject_brain_mask=Path(args.subject_brain_mask),
        lesion_mask=Path(args.lesion_mask),
        template_t2=Path(args.template_t2),
        output_root=Path(args.output),
        expected_orientation=args.expected_orientation,
        top_candidates=args.top_candidates,
        minimum_candidate_separation_mm=args.minimum_candidate_separation_mm,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

