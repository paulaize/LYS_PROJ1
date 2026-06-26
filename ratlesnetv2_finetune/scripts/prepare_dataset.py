"""CLI for preparing RatLesNetV2 training folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from ratlesnetv2_finetune.dataset import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RatLesNetV2 dataset YAML plan")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[2],
        help="Repository root for resolving relative paths",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild an existing prepared dataset under work/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root, prepared = prepare_dataset(
        args.config,
        repo_root=args.repo_root,
        overwrite=args.overwrite,
    )
    print(f"Wrote RatLesNetV2 dataset: {dataset_root}")
    print(f"Wrote manifest: {dataset_root / 'manifest.csv'}")
    for case in prepared:
        print(
            f"  {case.split:10} {case.animal_id:16} "
            f"{case.image_shape} lesion={case.lesion_volume_mm3:.4g} mm^3"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

