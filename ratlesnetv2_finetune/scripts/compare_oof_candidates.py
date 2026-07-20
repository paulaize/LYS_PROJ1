"""Create a paired comparison of two pooled OOF threshold reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_oof_candidates(
        reference_name=args.reference_name,
        reference_csv=Path(args.reference),
        candidate_name=args.candidate_name,
        candidate_csv=Path(args.candidate),
        output_root=Path(args.output),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def compare_oof_candidates(
    *,
    reference_name: str,
    reference_csv: Path,
    candidate_name: str,
    candidate_csv: Path,
    output_root: Path,
    bootstrap_samples: int = 2000,
    seed: int = 20260715,
) -> dict[str, Any]:
    """Compare candidates case-by-case and preserve fold/cohort summaries."""
    for name in (reference_name, candidate_name):
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Unsafe candidate name: {name!r}")
    if reference_name == candidate_name:
        raise ValueError("Reference and candidate names must differ")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    if output_root.exists():
        raise FileExistsError(output_root)

    reference = pd.read_csv(reference_csv).set_index("case_id").sort_index()
    candidate = pd.read_csv(candidate_csv).set_index("case_id").sort_index()
    if not reference.index.is_unique or not candidate.index.is_unique:
        raise ValueError("OOF case IDs must be unique")
    if not reference.index.equals(candidate.index):
        reference_only = sorted(set(reference.index) - set(candidate.index))
        candidate_only = sorted(set(candidate.index) - set(reference.index))
        raise ValueError(
            "OOF case sets differ: "
            f"reference_only={reference_only[:10]}, candidate_only={candidate_only[:10]}"
        )

    metadata_columns = [
        column
        for column in ["subject_id", "cv_fold", "cohort", "timepoint", "lesion_volume_bin"]
        if column in reference.columns
    ]
    missing_metadata = [column for column in metadata_columns if column not in candidate.columns]
    if missing_metadata:
        raise ValueError(f"Candidate is missing OOF metadata columns: {missing_metadata}")
    for column in metadata_columns:
        reference_values = reference[column].fillna("").astype(str)
        candidate_values = candidate[column].fillna("").astype(str)
        if not reference_values.equals(candidate_values):
            raise ValueError(f"OOF metadata differs between candidates for column {column!r}")
    paired = reference[metadata_columns].copy()
    metrics = [
        "dice",
        "precision",
        "recall",
        "absolute_volume_error_mm3",
        "volume_error_mm3",
        "hd95_mm",
        "surface_dice",
    ]
    missing = [
        f"{side}:{metric}"
        for side, frame in (("reference", reference), ("candidate", candidate))
        for metric in metrics
        if metric not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing OOF metrics: {missing}")
    for metric in metrics:
        paired[f"{reference_name}_{metric}"] = pd.to_numeric(
            reference[metric], errors="coerce"
        )
        paired[f"{candidate_name}_{metric}"] = pd.to_numeric(
            candidate[metric], errors="coerce"
        )
        paired[f"candidate_minus_reference_{metric}"] = (
            paired[f"{candidate_name}_{metric}"] - paired[f"{reference_name}_{metric}"]
        )
    paired = paired.reset_index()

    dice_difference = paired["candidate_minus_reference_dice"].to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dice_difference), size=(bootstrap_samples, len(paired)))
    bootstrap = dice_difference[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    fold_summary = _group_summary(
        paired,
        group="cv_fold",
        reference_name=reference_name,
        candidate_name=candidate_name,
        metrics=metrics,
    )
    cohort_summary = _group_summary(
        paired,
        group="cohort",
        reference_name=reference_name,
        candidate_name=candidate_name,
        metrics=metrics,
    )
    summary = {
        "reference": reference_name,
        "candidate": candidate_name,
        "n_cases": int(len(paired)),
        "reference_mean_dice": float(paired[f"{reference_name}_dice"].mean()),
        "candidate_mean_dice": float(paired[f"{candidate_name}_dice"].mean()),
        "candidate_minus_reference_mean_dice": float(dice_difference.mean()),
        "paired_dice_difference_ci95_low": float(ci_low),
        "paired_dice_difference_ci95_high": float(ci_high),
        "reference_zero_dice_cases": int((paired[f"{reference_name}_dice"] == 0).sum()),
        "candidate_zero_dice_cases": int((paired[f"{candidate_name}_dice"] == 0).sum()),
        "reference_mean_absolute_volume_error_mm3": float(
            paired[f"{reference_name}_absolute_volume_error_mm3"].mean()
        ),
        "candidate_mean_absolute_volume_error_mm3": float(
            paired[f"{candidate_name}_absolute_volume_error_mm3"].mean()
        ),
        "reference_mean_precision": float(paired[f"{reference_name}_precision"].mean()),
        "candidate_mean_precision": float(paired[f"{candidate_name}_precision"].mean()),
        "reference_mean_recall": float(paired[f"{reference_name}_recall"].mean()),
        "candidate_mean_recall": float(paired[f"{candidate_name}_recall"].mean()),
        "reference_mean_hd95_mm": float(paired[f"{reference_name}_hd95_mm"].mean()),
        "candidate_mean_hd95_mm": float(paired[f"{candidate_name}_hd95_mm"].mean()),
        "reference_mean_surface_dice": float(
            paired[f"{reference_name}_surface_dice"].mean()
        ),
        "candidate_mean_surface_dice": float(
            paired[f"{candidate_name}_surface_dice"].mean()
        ),
        "candidate_fold_wins": int(
            (fold_summary["candidate_minus_reference_dice"] > 0).sum()
        ),
        "reference_fold_wins": int(
            (fold_summary["candidate_minus_reference_dice"] < 0).sum()
        ),
        "fold_ties": int((fold_summary["candidate_minus_reference_dice"] == 0).sum()),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    output_root.mkdir(parents=True)
    paired.to_csv(output_root / "paired_cases.csv", index=False)
    fold_summary.to_csv(output_root / "paired_folds.csv", index=False)
    cohort_summary.to_csv(output_root / "paired_cohorts.csv", index=False)
    (output_root / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _group_summary(
    paired: pd.DataFrame,
    *,
    group: str,
    reference_name: str,
    candidate_name: str,
    metrics: list[str],
) -> pd.DataFrame:
    if group not in paired.columns:
        raise ValueError(f"Required metadata column absent: {group}")
    value_columns = [
        f"{name}_{metric}"
        for metric in metrics
        for name in (reference_name, candidate_name)
    ]
    result = paired.groupby(group, dropna=False)[value_columns].mean().reset_index()
    for metric in metrics:
        result[f"candidate_minus_reference_{metric}"] = (
            result[f"{candidate_name}_{metric}"] - result[f"{reference_name}_{metric}"]
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
