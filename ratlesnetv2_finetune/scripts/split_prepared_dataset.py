"""Split an existing prepared RatLesNetV2 dataset into train/validation/test.

This is intended for prepared folders that already contain RatLesNetV2 case
directories plus a ``manifest.csv``. It is useful when an initial Colab upload
archive was exported as all-train smoke-test data and needs explicit held-out
splits before a real training run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitResult:
    input_root: Path
    output_root: Path
    counts: dict[str, int]
    group_counts: dict[str, int]
    manifest_path: Path
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared dataset root with manifest.csv")
    parser.add_argument("--output", required=True, help="Output prepared dataset root")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--group-by",
        choices=["animal_id", "case_id"],
        default="animal_id",
        help="Keep all rows with the same group key in the same split",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to materialize case folders in the split output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output folder",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = split_prepared_dataset(
        input_root=args.input,
        output_root=args.output,
        seed=args.seed,
        group_by=args.group_by,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        validation_count=args.validation_count,
        test_count=args.test_count,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )
    print(f"Wrote split dataset: {result.output_root}")
    print(f"Wrote manifest: {result.manifest_path}")
    print(f"Wrote summary: {result.summary_path}")
    for split in VALID_SPLITS:
        print(
            f"  {split:10} cases={result.counts.get(split, 0):4d} "
            f"groups={result.group_counts.get(split, 0):4d}"
        )
    return 0


def split_prepared_dataset(
    *,
    input_root: str | Path,
    output_root: str | Path,
    seed: int = 20260706,
    group_by: str = "animal_id",
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    validation_count: int | None = None,
    test_count: int | None = None,
    copy_mode: str = "copy",
    overwrite: bool = False,
) -> SplitResult:
    input_root = Path(input_root)
    output_root = Path(output_root)
    if group_by not in {"animal_id", "case_id"}:
        raise ValueError("group_by must be 'animal_id' or 'case_id'")
    if copy_mode not in {"copy", "symlink"}:
        raise ValueError("copy_mode must be 'copy' or 'symlink'")
    if not input_root.is_dir():
        raise FileNotFoundError(f"Prepared dataset root not found: {input_root}")
    if input_root.resolve() == output_root.resolve():
        raise ValueError("Output root must be different from input root")

    rows = _read_manifest(input_root / "manifest.csv")
    groups = _group_rows(rows, group_by=group_by)
    split_by_group = _assign_splits(
        groups=groups,
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        validation_count=validation_count,
        test_count=test_count,
    )

    _prepare_output_root(output_root, overwrite=overwrite)
    output_rows: list[dict[str, Any]] = []
    for group_key, group_rows in groups.items():
        split = split_by_group[group_key]
        for row in group_rows:
            source_dir = _case_dir_from_manifest(input_root, row)
            dest_dir = (
                output_root
                / split
                / _required(row, "study")
                / _required(row, "timepoint")
                / _required(row, "case_id")
            )
            _materialize_case_dir(source_dir, dest_dir, copy_mode=copy_mode)
            output_rows.append(_updated_manifest_row(row, split=split, case_dir=dest_dir))

    manifest_path = output_root / "manifest.csv"
    _write_manifest(manifest_path, output_rows)
    counts = {split: sum(row["split"] == split for row in output_rows) for split in VALID_SPLITS}
    group_counts = {
        split: sum(assigned == split for assigned in split_by_group.values())
        for split in VALID_SPLITS
    }
    summary_path = output_root / "split_summary.json"
    _write_summary(
        summary_path,
        input_root=input_root,
        output_root=output_root,
        seed=seed,
        group_by=group_by,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        validation_count=validation_count,
        test_count=test_count,
        counts=counts,
        group_counts=group_counts,
    )
    return SplitResult(
        input_root=input_root,
        output_root=output_root,
        counts=counts,
        group_counts=group_counts,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset manifest not found: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"Prepared dataset manifest contains no rows: {path}")
    required = {"split", "animal_id", "study", "timepoint", "case_id"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Prepared dataset manifest missing required columns: {missing}")
    return rows


def _group_rows(rows: list[dict[str, str]], *, group_by: str) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = _required(row, group_by)
        groups.setdefault(key, []).append(row)
    return groups


def _assign_splits(
    *,
    groups: dict[str, list[dict[str, str]]],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    validation_count: int | None,
    test_count: int | None,
) -> dict[str, str]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be >= 0 and < 1")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be >= 0 and < 1")
    n_groups = len(groups)
    n_validation = (
        validation_count if validation_count is not None else round(n_groups * validation_fraction)
    )
    n_test = test_count if test_count is not None else round(n_groups * test_fraction)
    if n_validation < 0 or n_test < 0:
        raise ValueError("validation_count and test_count must be >= 0")
    if n_validation + n_test >= n_groups:
        raise ValueError(
            "Requested validation/test split leaves no training groups: "
            f"groups={n_groups}, validation={n_validation}, test={n_test}"
        )

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    test_keys = set(keys[:n_test])
    validation_keys = set(keys[n_test : n_test + n_validation])
    return {
        key: "test" if key in test_keys else "validation" if key in validation_keys else "train"
        for key in keys
    }


def _case_dir_from_manifest(input_root: Path, row: dict[str, str]) -> Path:
    split = _required(row, "split")
    case_dir = (
        input_root
        / split
        / _required(row, "study")
        / _required(row, "timepoint")
        / _required(row, "case_id")
    )
    if case_dir.is_dir():
        return case_dir

    recorded = row.get("case_dir")
    if recorded:
        recorded_path = Path(recorded)
        if recorded_path.is_dir():
            return recorded_path
    raise FileNotFoundError(
        f"Could not find prepared case directory for case_id={row.get('case_id')!r}. "
        f"Checked {case_dir} and manifest case_dir={recorded!r}."
    )


def _prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Split output already exists: {output_root}. Use --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def _materialize_case_dir(source_dir: Path, dest_dir: Path, *, copy_mode: str) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "symlink":
        dest_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
        return
    shutil.copytree(source_dir, dest_dir)


def _updated_manifest_row(row: dict[str, str], *, split: str, case_dir: Path) -> dict[str, str]:
    out = dict(row)
    out["split"] = split
    out["case_dir"] = str(case_dir)
    out["scan_path"] = str(case_dir / "scan.nii.gz")
    out["label_path"] = str(case_dir / "scan_lesionIAM.nii.gz")
    return out


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    *,
    input_root: Path,
    output_root: Path,
    seed: int,
    group_by: str,
    validation_fraction: float,
    test_fraction: float,
    validation_count: int | None,
    test_count: int | None,
    counts: dict[str, int],
    group_counts: dict[str, int],
) -> None:
    payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": seed,
        "group_by": group_by,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "validation_count": validation_count,
        "test_count": test_count,
        "counts": counts,
        "group_counts": group_counts,
    }
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value in (None, "", "TODO"):
        raise ValueError(f"Manifest row is missing required value: {key}")
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
