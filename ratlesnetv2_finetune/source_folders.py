"""Add local scan/mask source folders to a RatLesNetV2 dataset plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

NIFTI_SUFFIX = ".nii.gz"
MASK_SUFFIX = "_lesion_mask.nii.gz"
VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class SourceCase:
    animal_id: str
    case_id: str
    study: str
    timepoint: str
    scan_nifti: Path
    lesion_mask: Path


@dataclass(frozen=True)
class PlanUpdateResult:
    plan_path: Path
    source_root: Path
    split: str
    added: int
    updated: int
    unchanged: int
    cases: tuple[SourceCase, ...]
    dry_run: bool = False


def add_source_folder_to_plan(
    *,
    plan_path: str | Path,
    source_root: str | Path,
    split: str = "train",
    study: str = "LYS",
    timepoint: str = "unknown_timepoint",
    scan_subdir: str | Path | None = None,
    mask_subdir: str | Path | None = None,
    id_prefix: str | None = None,
    repo_root: str | Path | None = None,
    dry_run: bool = False,
) -> PlanUpdateResult:
    """Append or update all matched scan/mask pairs from one source folder.

    ``source_root`` must contain one scan subfolder and one mask subfolder, or
    the two subfolders must be supplied explicitly. Scan names are matched to
    lesion masks by the project convention ``case.nii.gz`` ->
    ``case_lesion_mask.nii.gz``.
    """
    split = _validate_split(split)
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
    plan_path = Path(plan_path)
    source_root = Path(source_root)

    cases = discover_source_cases(
        source_root=source_root,
        study=study,
        timepoint=timepoint,
        scan_subdir=scan_subdir,
        mask_subdir=mask_subdir,
        id_prefix=id_prefix,
    )
    plan = _load_plan_for_update(plan_path)
    splits = plan.setdefault("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("Dataset plan key 'splits' must be a mapping")
    for valid_split in sorted(VALID_SPLITS):
        splits.setdefault(valid_split, [])
        if splits[valid_split] is None:
            splits[valid_split] = []
        if not isinstance(splits[valid_split], list):
            raise ValueError(f"Dataset plan key splits.{valid_split} must be a list")

    added, updated, unchanged = _merge_cases(
        splits=splits,
        split=split,
        cases=cases,
        repo_root=repo_root,
    )

    if not dry_run:
        _write_plan(plan_path, plan)

    return PlanUpdateResult(
        plan_path=plan_path,
        source_root=source_root,
        split=split,
        added=added,
        updated=updated,
        unchanged=unchanged,
        cases=tuple(cases),
        dry_run=dry_run,
    )


def discover_source_cases(
    *,
    source_root: str | Path,
    study: str = "LYS",
    timepoint: str = "unknown_timepoint",
    scan_subdir: str | Path | None = None,
    mask_subdir: str | Path | None = None,
    id_prefix: str | None = None,
) -> list[SourceCase]:
    """Return sorted scan/mask pairs discovered below one local source folder."""
    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Source folder not found: {root}")

    scan_dir = _select_subdir(root, scan_subdir, kind="scan")
    mask_dir = _select_subdir(root, mask_subdir, kind="mask")
    if scan_dir == mask_dir:
        raise ValueError(
            f"Scan and mask folders resolved to the same folder: {scan_dir}. "
            "Use separate subfolders."
        )

    scans = {_scan_case_name(path): path for path in _scan_files(scan_dir)}
    masks = {_mask_case_name(path): path for path in _mask_files(mask_dir)}
    if not scans:
        raise FileNotFoundError(f"No T2w scan files matching *{NIFTI_SUFFIX} found in {scan_dir}")
    if not masks:
        raise FileNotFoundError(
            f"No lesion mask files matching *{MASK_SUFFIX} found in {mask_dir}"
        )

    missing_masks = sorted(set(scans) - set(masks))
    extra_masks = sorted(set(masks) - set(scans))
    if missing_masks:
        expected = ", ".join(f"{name}{MASK_SUFFIX}" for name in missing_masks[:5])
        raise FileNotFoundError(
            f"Missing lesion masks in {mask_dir} for {len(missing_masks)} scan(s): {expected}"
        )
    if extra_masks:
        examples = ", ".join(f"{name}{MASK_SUFFIX}" for name in extra_masks[:5])
        raise FileNotFoundError(
            f"Found lesion masks without matching scans in {scan_dir}: {examples}"
        )

    cases: list[SourceCase] = []
    for name in sorted(scans):
        case_id = _case_id(name, id_prefix=id_prefix)
        cases.append(
            SourceCase(
                animal_id=case_id,
                case_id=case_id,
                study=study,
                timepoint=timepoint,
                scan_nifti=scans[name],
                lesion_mask=masks[name],
            )
        )
    return cases


def _merge_cases(
    *,
    splits: dict[str, Any],
    split: str,
    cases: list[SourceCase],
    repo_root: Path,
) -> tuple[int, int, int]:
    added = 0
    updated = 0
    unchanged = 0

    known_case_split: dict[str, str] = {}
    known_case_index: dict[tuple[str, str], int] = {}
    for split_name, split_cases in splits.items():
        if split_name not in VALID_SPLITS or split_cases is None:
            continue
        for index, entry in enumerate(split_cases):
            if not isinstance(entry, dict):
                raise ValueError(f"splits.{split_name}[{index}] must be a mapping")
            case_id = entry.get("case_id") or entry.get("animal_id")
            if case_id is None:
                raise ValueError(f"splits.{split_name}[{index}] is missing case_id/animal_id")
            case_id = str(case_id)
            if case_id in known_case_split:
                raise ValueError(
                    f"Duplicate case_id {case_id!r} already exists in splits "
                    f"{known_case_split[case_id]!r} and {split_name!r}"
                )
            known_case_split[case_id] = str(split_name)
            known_case_index[(str(split_name), case_id)] = index

    split_cases = splits[split]
    for case in cases:
        existing_split = known_case_split.get(case.case_id)
        if existing_split is not None and existing_split != split:
            raise ValueError(
                f"Case {case.case_id!r} already exists in split {existing_split!r}. "
                "Move it manually or choose a unique --id-prefix."
            )

        entry = _case_to_plan_entry(case, repo_root=repo_root)
        existing_index = known_case_index.get((split, case.case_id))
        if existing_index is None:
            split_cases.append(entry)
            added += 1
            known_case_split[case.case_id] = split
            known_case_index[(split, case.case_id)] = len(split_cases) - 1
            continue

        if split_cases[existing_index] == entry:
            unchanged += 1
        else:
            split_cases[existing_index] = entry
            updated += 1

    return added, updated, unchanged


def _case_to_plan_entry(case: SourceCase, *, repo_root: Path) -> dict[str, str]:
    return {
        "animal_id": case.animal_id,
        "study": case.study,
        "timepoint": case.timepoint,
        "case_id": case.case_id,
        "scan_nifti": _format_path(case.scan_nifti, repo_root=repo_root),
        "lesion_mask": _format_path(case.lesion_mask, repo_root=repo_root),
    }


def _select_subdir(root: Path, configured: str | Path | None, *, kind: str) -> Path:
    if configured is not None:
        path = Path(configured)
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_dir():
            raise FileNotFoundError(f"Configured {kind} subfolder not found: {resolved}")
        return resolved

    candidates: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        files = _scan_files(child) if kind == "scan" else _mask_files(child)
        if files:
            candidates.append(child)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        pattern = f"*{NIFTI_SUFFIX}" if kind == "scan" else f"*{MASK_SUFFIX}"
        raise FileNotFoundError(
            f"Could not auto-detect a {kind} subfolder under {root}; "
            f"no immediate subfolder contains files matching {pattern}."
        )
    names = ", ".join(path.name for path in candidates)
    raise ValueError(
        f"Could not auto-detect a unique {kind} subfolder under {root}; "
        f"candidates: {names}. Pass --{kind}-subdir explicitly."
    )


def _scan_files(path: Path) -> list[Path]:
    return sorted(
        file
        for file in path.glob(f"*{NIFTI_SUFFIX}")
        if file.is_file() and not file.name.endswith(MASK_SUFFIX)
    )


def _mask_files(path: Path) -> list[Path]:
    return sorted(file for file in path.glob(f"*{MASK_SUFFIX}") if file.is_file())


def _scan_case_name(path: Path) -> str:
    if not path.name.endswith(NIFTI_SUFFIX) or path.name.endswith(MASK_SUFFIX):
        raise ValueError(f"Not a T2w scan filename by convention: {path.name}")
    return path.name[: -len(NIFTI_SUFFIX)]


def _mask_case_name(path: Path) -> str:
    if not path.name.endswith(MASK_SUFFIX):
        raise ValueError(f"Not a lesion mask filename by convention: {path.name}")
    return path.name[: -len(MASK_SUFFIX)]


def _case_id(name: str, *, id_prefix: str | None) -> str:
    case_id = name if id_prefix in (None, "") else f"{id_prefix}_{name}"
    if case_id in {"", ".", ".."} or "/" in case_id or "\\" in case_id:
        raise ValueError(f"Derived case_id {case_id!r} is not a safe path component")
    return case_id


def _format_path(path: Path, *, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _validate_split(split: str) -> str:
    split = str(split)
    if split not in VALID_SPLITS:
        expected = ", ".join(sorted(VALID_SPLITS))
        raise ValueError(f"Unknown split {split!r}; expected one of: {expected}")
    return split


def _load_plan_for_update(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset plan YAML not found: {path}")
    with path.open() as fh:
        plan = yaml.safe_load(fh) or {}
    if not isinstance(plan, dict):
        raise ValueError(f"Dataset plan {path} did not parse to a mapping")
    return plan


def _write_plan(path: Path, plan: dict[str, Any]) -> None:
    with path.open("w") as fh:
        yaml.safe_dump(plan, fh, sort_keys=False)
