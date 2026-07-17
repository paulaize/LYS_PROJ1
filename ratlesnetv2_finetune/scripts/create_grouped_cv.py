"""Create a locked subject-level test set and grouped cross-validation folds.

Reviewed metadata is preferred. A conservative inference mode is also
available: it groups only clear base/D7 Thrombin_08 pairs and explicit STZ
D1/D7 pairs, while treating every other case as its own subject.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_METADATA_COLUMNS = {
    "case_id",
    "subject_id",
    "cohort",
    "timepoint",
    "acquisition_protocol",
    "old_mask_version",
    "new_mask_version",
    "label_version",
    "reviewer",
    "correction_reason",
    "model_prediction_viewed_during_review",
    "qc_status",
    "exclude",
}


@dataclass(frozen=True)
class SubjectGroup:
    subject_id: str
    rows: list[dict[str, str]]
    features: Counter[str]

    @property
    def case_count(self) -> int:
        return len(self.rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared dataset root with manifest.csv")
    parser.add_argument("--metadata", default=None, help="Reviewed case/subject metadata CSV")
    parser.add_argument(
        "--infer-longitudinal-subjects",
        action="store_true",
        help="Infer only conservative D1/D7 groups when reviewed metadata is unavailable",
    )
    parser.add_argument("--label-version", default="LYS_v1")
    parser.add_argument("--output", required=True, help="Output root for locked test and folds")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--approved-qc-status",
        default="approved",
        help="Comma-separated QC status values allowed into the split",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="symlink",
        help="File symlinks avoid five data copies while remaining visible to recursive scans",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    approved = {value.strip() for value in args.approved_qc_status.split(",") if value.strip()}
    result = create_grouped_cv(
        input_root=Path(args.input),
        metadata_path=Path(args.metadata) if args.metadata else None,
        output_root=Path(args.output),
        test_fraction=args.test_fraction,
        folds=args.folds,
        seed=args.seed,
        approved_qc_status=approved,
        infer_longitudinal_subjects=args.infer_longitudinal_subjects,
        label_version=args.label_version,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )
    print(f"Cases included: {result['n_cases']}")
    print(f"Subjects included: {result['n_subjects']}")
    print(f"Locked test subjects: {result['test_subjects']}")
    print(f"Development subjects: {result['development_subjects']}")
    print(f"Subject grouping: {result['subject_grouping_mode']}")
    print(f"Assignments: {Path(args.output) / 'split_assignments.csv'}")
    return 0


def create_grouped_cv(
    *,
    input_root: Path,
    metadata_path: Path | None,
    output_root: Path,
    test_fraction: float = 0.2,
    folds: int = 5,
    seed: int = 20260715,
    approved_qc_status: set[str] | None = None,
    infer_longitudinal_subjects: bool = False,
    label_version: str = "LYS_v1",
    copy_mode: str = "symlink",
    overwrite: bool = False,
) -> dict[str, Any]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Prepared dataset root not found: {input_root}")
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be > 0 and < 0.5")
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if copy_mode not in {"copy", "symlink"}:
        raise ValueError("copy_mode must be copy or symlink")
    if (metadata_path is None) == (not infer_longitudinal_subjects):
        raise ValueError(
            "Provide exactly one of --metadata or --infer-longitudinal-subjects"
        )

    manifest = _read_csv(input_root / "manifest.csv")
    inferred_groups: list[dict[str, str]] = []
    if metadata_path is not None:
        approved_qc_status = approved_qc_status or {"approved"}
        if not approved_qc_status:
            raise ValueError("At least one approved QC status is required")
        joined, excluded = _join_reviewed_metadata(
            manifest,
            metadata_path=metadata_path,
            approved_qc_status=approved_qc_status,
        )
        grouping_mode = "reviewed_metadata"
    else:
        joined, inferred_groups = _join_inferred_metadata(
            manifest,
            label_version=label_version,
        )
        excluded = []
        grouping_mode = "conservative_case_id_inference"
    if not joined:
        raise ValueError("No approved, non-excluded cases remain")

    _add_lesion_volume_bins(joined)
    groups = _subject_groups(joined)
    if len(groups) <= folds:
        raise ValueError(
            f"Need more than {folds} approved subjects to create a locked test plus CV folds"
        )

    outer = _balanced_assignment(
        groups,
        labels=("development", "test"),
        fractions=(1.0 - test_fraction, test_fraction),
        seed=seed,
    )
    development_groups = [group for group in groups if outer[group.subject_id] == "development"]
    if len(development_groups) < folds:
        raise ValueError(
            f"Only {len(development_groups)} development subjects remain for {folds}-fold CV"
        )
    fold_labels = tuple(str(index) for index in range(folds))
    cv_assignment = _balanced_assignment(
        development_groups,
        labels=fold_labels,
        fractions=tuple(1.0 / folds for _ in fold_labels),
        seed=seed + 1,
    )

    _prepare_output(output_root, overwrite=overwrite)
    assignments = _assignment_rows(joined, outer=outer, cv=cv_assignment)
    _write_csv(output_root / "split_assignments.csv", assignments)
    if inferred_groups:
        _write_csv(output_root / "inferred_subject_groups.csv", inferred_groups)
    if excluded:
        _write_csv(output_root / "excluded_cases.csv", excluded)

    source_by_case = {
        _required(row, "case_id"): _case_dir(input_root, row)
        for row in manifest
    }
    development_rows = [
        row
        for row in joined
        if outer[_required(row, "subject_id")] == "development"
    ]
    test_rows = [row for row in joined if outer[_required(row, "subject_id")] == "test"]
    _materialize_partition(
        root=output_root / "development",
        rows=development_rows,
        split_for_case={_required(row, "case_id"): "train" for row in development_rows},
        source_by_case=source_by_case,
        copy_mode=copy_mode,
    )
    _materialize_partition(
        root=output_root / "locked_test",
        rows=test_rows,
        split_for_case={_required(row, "case_id"): "test" for row in test_rows},
        source_by_case=source_by_case,
        copy_mode=copy_mode,
    )

    for fold in range(folds):
        split_for_case = {}
        for row in development_rows:
            subject = _required(row, "subject_id")
            split_for_case[_required(row, "case_id")] = (
                "validation" if cv_assignment[subject] == str(fold) else "train"
            )
        _materialize_partition(
            root=output_root / "folds" / f"fold_{fold}",
            rows=development_rows,
            split_for_case=split_for_case,
            source_by_case=source_by_case,
            copy_mode=copy_mode,
        )

    _validate_assignments(assignments, folds=folds)
    summary = _summary_payload(
        assignments,
        excluded=excluded,
        test_fraction=test_fraction,
        folds=folds,
        seed=seed,
        copy_mode=copy_mode,
        subject_grouping_mode=grouping_mode,
        inferred_longitudinal_groups=sum(
            row["group_size"] != "1" for row in inferred_groups
        ),
    )
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _join_reviewed_metadata(
    manifest: list[dict[str, str]],
    *,
    metadata_path: Path,
    approved_qc_status: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metadata = _read_metadata(metadata_path)
    manifest_ids = {_required(row, "case_id") for row in manifest}
    metadata_ids = set(metadata)
    if manifest_ids != metadata_ids:
        missing = sorted(manifest_ids - metadata_ids)
        extra = sorted(metadata_ids - manifest_ids)
        raise ValueError(
            "Metadata case IDs must exactly match the prepared manifest. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )

    joined: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for row in manifest:
        case_id = _required(row, "case_id")
        merged = {**row, **metadata[case_id]}
        if _as_bool(_required(merged, "exclude")):
            excluded.append(merged)
            continue
        _validate_metadata_row(merged, approved_qc_status=approved_qc_status)
        joined.append(merged)
    return joined, excluded


def _join_inferred_metadata(
    manifest: list[dict[str, str]],
    *,
    label_version: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if label_version in {"", "TODO"}:
        raise ValueError("label_version must be explicit")
    case_ids = {_required(row, "case_id") for row in manifest}
    subject_by_case, rule_by_case = _infer_subject_ids(case_ids)
    joined = []
    for row in manifest:
        case_id = _required(row, "case_id")
        version = str(row.get("label_version") or label_version)
        joined.append(
            {
                **row,
                "subject_id": subject_by_case[case_id],
                "cohort": _inferred_case_family(case_id),
                "timepoint": _encoded_timepoint(case_id),
                "acquisition_protocol": "not_provided",
                "old_mask_version": "not_provided",
                "new_mask_version": version,
                "label_version": version,
                "reviewer": "not_provided",
                "correction_reason": "not_provided",
                "model_prediction_viewed_during_review": "unknown",
                "qc_status": "metadata_bypassed_by_user",
                "exclude": "false",
                "exclude_reason": "",
            }
        )

    grouped_cases: dict[str, list[str]] = defaultdict(list)
    for case_id, subject_id in subject_by_case.items():
        grouped_cases[subject_id].append(case_id)
    report = []
    for subject_id, members in sorted(grouped_cases.items()):
        ordered = sorted(members)
        report.append(
            {
                "subject_id": subject_id,
                "group_size": str(len(ordered)),
                "inference_rule": rule_by_case[ordered[0]],
                "case_ids": ";".join(ordered),
            }
        )
    return joined, report


def _infer_subject_ids(case_ids: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    subject_by_case = {case_id: f"single::{case_id}" for case_id in case_ids}
    rule_by_case = {case_id: "singleton_no_clear_pair" for case_id in case_ids}

    for case_id in sorted(case_ids):
        if not case_id.startswith("Thrombin_08"):
            continue
        base_id = re.sub(r"D7(?:P2)?$", "", case_id, flags=re.IGNORECASE)
        if base_id == case_id or base_id not in case_ids:
            continue
        subject_id = f"paired::Thrombin_08::{base_id}"
        for member in (base_id, case_id):
            subject_by_case[member] = subject_id
            rule_by_case[member] = "exact_base_plus_D7_suffix"

    stz_by_terminal_code: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        if not case_id.startswith("ThrombinSTZ"):
            continue
        match = re.search(r"(C[A-Z]\d+|J\d+)(?:-D[17])?$", case_id, re.IGNORECASE)
        if match:
            stz_by_terminal_code[match.group(1).upper()].append(case_id)
    for terminal_code, members in stz_by_terminal_code.items():
        d1 = [case_id for case_id in members if _encoded_timepoint(case_id) == "D1"]
        d7 = [case_id for case_id in members if _encoded_timepoint(case_id) == "D7"]
        if len(d1) != 1 or len(d7) != 1:
            continue
        subject_id = f"paired::ThrombinSTZ_02::{terminal_code}"
        for member in (*d1, *d7):
            subject_by_case[member] = subject_id
            rule_by_case[member] = "stz_terminal_code_with_explicit_D1_D7"
    return subject_by_case, rule_by_case


def _encoded_timepoint(case_id: str) -> str:
    if re.search(r"D7(?:P2)?$|(?:^|[-_])D7(?:$|[-_])", case_id, re.IGNORECASE):
        return "D7"
    if re.search(r"D1$|(?:^|[-_])D1(?:$|[-_])", case_id, re.IGNORECASE):
        return "D1"
    return "not_encoded"


def _inferred_case_family(case_id: str) -> str:
    match = re.match(r"(ThrombinSTZ_\d+|Thrombin_\d+)", case_id, re.IGNORECASE)
    return match.group(1) if match else "not_encoded"


def _read_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    missing = sorted(REQUIRED_METADATA_COLUMNS - set(rows[0]))
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {missing}")
    by_case: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = _required(row, "case_id")
        if case_id in by_case:
            raise ValueError(f"Duplicate metadata row for case_id={case_id!r}")
        by_case[case_id] = row
    return by_case


def _validate_metadata_row(row: dict[str, str], *, approved_qc_status: set[str]) -> None:
    for key in [
        "subject_id",
        "cohort",
        "timepoint",
        "acquisition_protocol",
        "old_mask_version",
        "new_mask_version",
        "label_version",
        "reviewer",
        "correction_reason",
        "model_prediction_viewed_during_review",
        "qc_status",
    ]:
        _required(row, key)
    _as_bool(_required(row, "model_prediction_viewed_during_review"))
    if row["new_mask_version"] != row["label_version"]:
        raise ValueError(
            f"case_id={row['case_id']!r} has inconsistent new_mask_version="
            f"{row['new_mask_version']!r} and label_version={row['label_version']!r}"
        )
    if row["qc_status"] not in approved_qc_status:
        raise ValueError(
            f"case_id={row['case_id']!r} has qc_status={row['qc_status']!r}; "
            f"allowed values are {sorted(approved_qc_status)}"
        )


def _add_lesion_volume_bins(rows: list[dict[str, str]]) -> None:
    volumes = np.asarray([float(_required(row, "lesion_volume_mm3")) for row in rows])
    positive = volumes[volumes > 0]
    quantiles = np.quantile(positive, [0.25, 0.5, 0.75]) if positive.size else np.asarray([])
    for row, volume in zip(rows, volumes, strict=True):
        if volume <= 0:
            lesion_bin = "negative"
        else:
            lesion_bin = f"q{int(np.searchsorted(quantiles, volume, side='right')) + 1}"
        row["lesion_volume_bin"] = lesion_bin


def _subject_groups(rows: list[dict[str, str]]) -> list[SubjectGroup]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_required(row, "subject_id")].append(row)
    result = []
    for subject_id, subject_rows in sorted(grouped.items()):
        features: Counter[str] = Counter()
        for row in subject_rows:
            for key in ["cohort", "timepoint", "acquisition_protocol", "lesion_volume_bin"]:
                features[f"{key}={_required(row, key)}"] += 1
            features[f"label_positive={float(row['lesion_volume_mm3']) > 0}"] += 1
        result.append(SubjectGroup(subject_id=subject_id, rows=subject_rows, features=features))
    return result


def _balanced_assignment(
    groups: list[SubjectGroup],
    *,
    labels: tuple[str, ...],
    fractions: tuple[float, ...],
    seed: int,
) -> dict[str, str]:
    if len(labels) != len(fractions) or not labels:
        raise ValueError("labels and fractions must have the same nonzero length")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("assignment fractions must sum to 1")
    if len(groups) < len(labels):
        raise ValueError("Not enough subject groups for the requested partitions")

    total_features = sum((group.features for group in groups), Counter())
    total_cases = sum(group.case_count for group in groups)
    rarity = {
        group.subject_id: sum(
            count / max(total_features[feature], 1)
            for feature, count in group.features.items()
        )
        for group in groups
    }
    rng = random.Random(seed)
    tie_break = {group.subject_id: rng.random() for group in groups}
    ordered = sorted(
        groups,
        key=lambda group: (
            -rarity[group.subject_id],
            -group.case_count,
            tie_break[group.subject_id],
        ),
    )

    state = {
        label: {"groups": 0, "cases": 0, "features": Counter()}
        for label in labels
    }
    assignment: dict[str, str] = {}
    target_groups = {
        label: len(groups) * fraction
        for label, fraction in zip(labels, fractions, strict=True)
    }
    target_cases = {
        label: total_cases * fraction
        for label, fraction in zip(labels, fractions, strict=True)
    }
    target_features = {
        label: {feature: count * fraction for feature, count in total_features.items()}
        for label, fraction in zip(labels, fractions, strict=True)
    }

    for group in ordered:
        candidates = list(labels)
        rng.shuffle(candidates)
        selected = min(
            candidates,
            key=lambda label: _assignment_cost(
                state,
                add_to=label,
                group=group,
                labels=labels,
                target_groups=target_groups,
                target_cases=target_cases,
                target_features=target_features,
            ),
        )
        assignment[group.subject_id] = selected
        state[selected]["groups"] += 1
        state[selected]["cases"] += group.case_count
        state[selected]["features"].update(group.features)

    missing_labels = [label for label in labels if state[label]["groups"] == 0]
    if missing_labels:
        raise RuntimeError(f"Balanced assignment produced empty partitions: {missing_labels}")
    return assignment


def _assignment_cost(
    state: dict[str, dict[str, Any]],
    *,
    add_to: str,
    group: SubjectGroup,
    labels: tuple[str, ...],
    target_groups: dict[str, float],
    target_cases: dict[str, float],
    target_features: dict[str, dict[str, float]],
) -> float:
    cost = 0.0
    feature_terms = 0
    for label in labels:
        group_count = state[label]["groups"] + (1 if label == add_to else 0)
        case_count = state[label]["cases"] + (group.case_count if label == add_to else 0)
        cost += 3.0 * ((group_count - target_groups[label]) / max(target_groups[label], 1)) ** 2
        cost += 2.0 * ((case_count - target_cases[label]) / max(target_cases[label], 1)) ** 2
        for feature, target in target_features[label].items():
            observed = state[label]["features"][feature]
            if label == add_to:
                observed += group.features[feature]
            cost += ((observed - target) / max(target, 1)) ** 2
            feature_terms += 1
    return cost / max(feature_terms, 1)


def _assignment_rows(
    rows: list[dict[str, str]],
    *,
    outer: dict[str, str],
    cv: dict[str, str],
) -> list[dict[str, str]]:
    assignments = []
    for row in rows:
        subject = _required(row, "subject_id")
        outer_split = outer[subject]
        assignments.append(
            {
                "case_id": _required(row, "case_id"),
                "subject_id": subject,
                "outer_split": outer_split,
                "cv_fold": cv[subject] if outer_split == "development" else "",
                "cohort": _required(row, "cohort"),
                "timepoint": _required(row, "timepoint"),
                "acquisition_protocol": _required(row, "acquisition_protocol"),
                "lesion_volume_bin": _required(row, "lesion_volume_bin"),
                "lesion_volume_mm3": _required(row, "lesion_volume_mm3"),
                "label_version": _required(row, "label_version"),
                "reviewer": _required(row, "reviewer"),
                "old_mask_version": _required(row, "old_mask_version"),
                "new_mask_version": _required(row, "new_mask_version"),
                "correction_reason": _required(row, "correction_reason"),
                "model_prediction_viewed_during_review": _required(
                    row,
                    "model_prediction_viewed_during_review",
                ),
                "qc_status": _required(row, "qc_status"),
            }
        )
    return sorted(assignments, key=lambda row: row["case_id"])


def _materialize_partition(
    *,
    root: Path,
    rows: list[dict[str, str]],
    split_for_case: dict[str, str],
    source_by_case: dict[str, Path],
    copy_mode: str,
) -> None:
    output_rows = []
    for row in rows:
        case_id = _required(row, "case_id")
        split = split_for_case[case_id]
        destination = (
            root
            / split
            / _required(row, "study")
            / _required(row, "timepoint")
            / case_id
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_mode == "symlink":
            destination.mkdir()
            for source in source_by_case[case_id].iterdir():
                target = destination / source.name
                target.symlink_to(
                    source.resolve(),
                    target_is_directory=source.is_dir(),
                )
        else:
            shutil.copytree(source_by_case[case_id], destination)
        output_rows.append(_updated_manifest_row(row, split=split, case_dir=destination))
    _write_csv(root / "manifest.csv", output_rows)


def _updated_manifest_row(row: dict[str, str], *, split: str, case_dir: Path) -> dict[str, str]:
    result = dict(row)
    result["split"] = split
    result["case_dir"] = str(case_dir)
    result["scan_path"] = str(case_dir / "scan.nii.gz")
    result["label_path"] = str(case_dir / "scan_lesionIAM.nii.gz")
    return result


def _validate_assignments(rows: list[dict[str, str]], *, folds: int) -> None:
    subject_outer: dict[str, set[str]] = defaultdict(set)
    subject_fold: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subject_outer[row["subject_id"]].add(row["outer_split"])
        if row["cv_fold"]:
            subject_fold[row["subject_id"]].add(row["cv_fold"])
    leaking_outer = sorted(subject for subject, values in subject_outer.items() if len(values) != 1)
    leaking_fold = sorted(subject for subject, values in subject_fold.items() if len(values) != 1)
    if leaking_outer or leaking_fold:
        raise RuntimeError(
            f"Subject leakage detected: outer={leaking_outer[:10]}, cv={leaking_fold[:10]}"
        )
    observed_folds = {row["cv_fold"] for row in rows if row["cv_fold"]}
    expected_folds = {str(index) for index in range(folds)}
    if observed_folds != expected_folds:
        raise RuntimeError(
            f"Expected CV folds {sorted(expected_folds)}, got {sorted(observed_folds)}"
        )


def _summary_payload(
    rows: list[dict[str, str]],
    *,
    excluded: list[dict[str, str]],
    test_fraction: float,
    folds: int,
    seed: int,
    copy_mode: str,
    subject_grouping_mode: str,
    inferred_longitudinal_groups: int,
) -> dict[str, Any]:
    subjects = {row["subject_id"] for row in rows}
    development_subjects = {
        row["subject_id"] for row in rows if row["outer_split"] == "development"
    }
    test_subjects = {row["subject_id"] for row in rows if row["outer_split"] == "test"}
    strata: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        outer = row["outer_split"]
        for key in ["cohort", "timepoint", "acquisition_protocol", "lesion_volume_bin"]:
            strata[outer][f"{key}={row[key]}"] += 1
    return {
        "n_cases": len(rows),
        "n_subjects": len(subjects),
        "n_excluded_cases": len(excluded),
        "development_cases": sum(row["outer_split"] == "development" for row in rows),
        "development_subjects": len(development_subjects),
        "test_cases": sum(row["outer_split"] == "test" for row in rows),
        "test_subjects": len(test_subjects),
        "test_fraction_requested": test_fraction,
        "folds": folds,
        "seed": seed,
        "copy_mode": copy_mode,
        "subject_grouping_mode": subject_grouping_mode,
        "inferred_longitudinal_groups": inferred_longitudinal_groups,
        "subject_disjoint": not bool(development_subjects & test_subjects),
        "strata_case_counts": {key: dict(sorted(value.items())) for key, value in strata.items()},
        "locked_test_policy": (
            "Do not use locked_test for preprocessing, threshold, loss, checkpoint, "
            "postprocessing, or visual model-selection decisions."
        ),
    }


def _case_dir(root: Path, row: dict[str, str]) -> Path:
    canonical = (
        root
        / _required(row, "split")
        / _required(row, "study")
        / _required(row, "timepoint")
        / _required(row, "case_id")
    )
    if canonical.is_dir():
        return canonical
    recorded = Path(str(row.get("case_dir", "")))
    if recorded.is_dir():
        return recorded
    raise FileNotFoundError(f"Prepared case directory not found for {_required(row, 'case_id')}")


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Split output exists: {path}; pass --overwrite to rebuild it")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value in {None, "", "TODO"}:
        raise ValueError(f"case_id={row.get('case_id')!r} is missing required metadata: {key}")
    return str(value)


def _as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
