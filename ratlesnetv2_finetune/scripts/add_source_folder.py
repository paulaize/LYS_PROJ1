"""Add one local T2w/mask source folder to a RatLesNetV2 dataset YAML plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from ratlesnetv2_finetune.source_folders import add_source_folder_to_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RatLesNetV2 dataset YAML plan to update")
    parser.add_argument(
        "--source-root",
        required=True,
        help="Folder containing one T2w scan subfolder and one lesion-mask subfolder",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Dataset split where discovered cases should be listed",
    )
    parser.add_argument("--study", default="LYS", help="Study folder name for RatLesNetV2 output")
    parser.add_argument(
        "--timepoint",
        default="unknown_timepoint",
        help="Timepoint folder name for RatLesNetV2 output",
    )
    parser.add_argument(
        "--scan-subdir",
        default=None,
        help="Name/path of the scan subfolder. If omitted, auto-detects a unique candidate.",
    )
    parser.add_argument(
        "--mask-subdir",
        default=None,
        help="Name/path of the mask subfolder. If omitted, auto-detects a unique candidate.",
    )
    parser.add_argument(
        "--id-prefix",
        default=None,
        help="Optional prefix for derived animal_id/case_id values when filenames are not unique.",
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[2],
        help="Repository root used to store relative paths when source files are inside the repo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the update summary without writing the YAML plan",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = add_source_folder_to_plan(
        plan_path=args.config,
        source_root=args.source_root,
        split=args.split,
        study=args.study,
        timepoint=args.timepoint,
        scan_subdir=args.scan_subdir,
        mask_subdir=args.mask_subdir,
        id_prefix=args.id_prefix,
        repo_root=args.repo_root,
        dry_run=args.dry_run,
    )

    verb = "Would update" if result.dry_run else "Updated"
    print(f"{verb} dataset plan: {result.plan_path}")
    print(f"Source folder: {result.source_root}")
    print(
        f"Split {result.split}: "
        f"added={result.added} updated={result.updated} unchanged={result.unchanged}"
    )
    for case in result.cases:
        print(f"  {case.case_id}: {case.scan_nifti.name} + {case.lesion_mask.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
