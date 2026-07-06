"""Prepare local T2w/manual-lesion NIfTI data for RatLesNetV2 training.

RatLesNetV2 expects one case per directory, with a 4-D scan
``Height x Width x Slices x Channels`` and a 3-D binary lesion label. This
module makes that contract explicit and writes a manifest so cloud training is
traceable back to source NIfTI files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

DEFAULT_SCAN_FILENAME = "scan.nii.gz"
# Upstream README mentions scan_lesion.nii.gz, but current DataWrapper.py uses this name.
DEFAULT_LABEL_FILENAME = "scan_lesionIAM.nii.gz"
README_LABEL_ALIAS = "scan_lesion.nii.gz"


@dataclass(frozen=True)
class CaseSpec:
    split: str
    animal_id: str
    scan_nifti: Path
    lesion_mask: Path
    study: str = "LYS"
    timepoint: str = "unknown_timepoint"
    case_id: str | None = None

    @property
    def ratlesnet_case_id(self) -> str:
        return self.case_id or self.animal_id


@dataclass(frozen=True)
class PreparedCase:
    split: str
    animal_id: str
    study: str
    timepoint: str
    case_id: str
    case_dir: Path
    scan_path: Path
    label_path: Path
    spacing_mm: tuple[float, float, float]
    image_shape: tuple[int, int, int, int]
    label_voxels: int
    lesion_volume_mm3: float


def load_plan(path: str | Path) -> dict[str, Any]:
    """Read a RatLesNetV2 dataset-preparation YAML plan."""
    plan_path = Path(path)
    with plan_path.open() as fh:
        plan = yaml.safe_load(fh) or {}
    if not isinstance(plan, dict):
        raise ValueError(f"Dataset plan {plan_path} did not parse to a mapping")
    return plan


def prepare_dataset(
    plan_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, list[PreparedCase]]:
    """Prepare all configured cases and write ``manifest.csv``.

    Parameters
    ----------
    plan_path:
        YAML file with output settings and train/validation/test case lists.
    repo_root:
        Base path for relative source and output paths. Defaults to the repo
        root inferred from this file.
    overwrite:
        Replace an existing prepared dataset. False by default to prevent
        accidental replacement of training inputs.
    """
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    plan = load_plan(plan_path)
    output_root = _resolve_path(repo_root, _required(plan, "output_root"))
    dataset_name = _clean_path_part(str(_required(plan, "dataset_name")), "dataset_name")
    dataset_root = output_root / dataset_name
    _ensure_not_inside_data(repo_root, dataset_root)

    filenames = plan.get("ratlesnetv2", {})
    if filenames is None:
        filenames = {}
    if not isinstance(filenames, dict):
        raise ValueError("ratlesnetv2 config must be a mapping")
    scan_filename = str(filenames.get("scan_filename", DEFAULT_SCAN_FILENAME))
    label_filename = str(filenames.get("label_filename", DEFAULT_LABEL_FILENAME))
    write_readme_alias = bool(filenames.get("write_readme_label_alias", True))

    preprocessing = plan.get("preprocessing", {}) or {}
    if not isinstance(preprocessing, dict):
        raise ValueError("preprocessing config must be a mapping")
    expected_spacing = preprocessing.get("expected_spacing_mm")
    expected_spacing_mm = (
        tuple(float(v) for v in expected_spacing) if expected_spacing is not None else None
    )
    spacing_tolerance = float(preprocessing.get("spacing_tolerance_mm", 0.02))
    label_threshold = float(preprocessing.get("label_threshold", 0.5))

    if dataset_root.exists() and not overwrite:
        raise FileExistsError(
            f"Prepared dataset already exists: {dataset_root}. "
            "Use --overwrite only when you intend to rebuild it."
        )
    dataset_root.mkdir(parents=True, exist_ok=True)

    prepared: list[PreparedCase] = []
    for case in iter_case_specs(plan, repo_root=repo_root):
        prepared.append(
            _prepare_case(
                case,
                dataset_root=dataset_root,
                scan_filename=scan_filename,
                label_filename=label_filename,
                write_readme_alias=write_readme_alias,
                expected_spacing_mm=expected_spacing_mm,
                spacing_tolerance=spacing_tolerance,
                label_threshold=label_threshold,
            )
        )

    manifest_path = dataset_root / "manifest.csv"
    _write_manifest(manifest_path, prepared)
    return dataset_root, prepared


def iter_case_specs(plan: dict[str, Any], *, repo_root: Path) -> list[CaseSpec]:
    """Return all configured cases across train/validation/test splits."""
    splits = _required(plan, "splits")
    if not isinstance(splits, dict):
        raise ValueError("splits must be a mapping of split name -> case list")

    cases: list[CaseSpec] = []
    seen_case_ids: dict[str, str] = {}
    for split, split_cases in splits.items():
        split_name = str(split)
        if split_name not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split {split_name!r}; expected train/validation/test")
        if split_cases is None:
            continue
        if not isinstance(split_cases, list):
            raise ValueError(f"splits.{split_name} must be a list")
        for index, raw_case in enumerate(split_cases):
            if not isinstance(raw_case, dict):
                raise ValueError(f"splits.{split_name}[{index}] must be a mapping")
            animal_id = str(_required(raw_case, "animal_id"))
            case_id = str(raw_case["case_id"]) if raw_case.get("case_id") is not None else animal_id
            previous_split = seen_case_ids.get(case_id)
            if previous_split is not None:
                raise ValueError(
                    f"Duplicate case_id {case_id!r} appears in both "
                    f"{previous_split!r} and {split_name!r}; split leakage is not allowed"
                )
            seen_case_ids[case_id] = split_name
            cases.append(
                CaseSpec(
                    split=split_name,
                    animal_id=animal_id,
                    scan_nifti=_resolve_path(repo_root, _required(raw_case, "scan_nifti")),
                    lesion_mask=_resolve_path(repo_root, _required(raw_case, "lesion_mask")),
                    study=str(raw_case.get("study", "LYS")),
                    timepoint=str(raw_case.get("timepoint", "unknown_timepoint")),
                    case_id=case_id,
                )
            )

    if not cases:
        raise ValueError("Dataset plan does not contain any cases")
    return cases


def _prepare_case(
    case: CaseSpec,
    *,
    dataset_root: Path,
    scan_filename: str,
    label_filename: str,
    write_readme_alias: bool,
    expected_spacing_mm: tuple[float, float, float] | None,
    spacing_tolerance: float,
    label_threshold: float,
) -> PreparedCase:
    if not case.scan_nifti.exists():
        raise FileNotFoundError(f"Missing source scan for {case.animal_id}: {case.scan_nifti}")
    if not case.lesion_mask.exists():
        raise FileNotFoundError(f"Missing lesion mask for {case.animal_id}: {case.lesion_mask}")

    split_dir = dataset_root / case.split
    case_dir = (
        split_dir
        / _clean_path_part(case.study, "study")
        / _clean_path_part(case.timepoint, "timepoint")
        / _clean_path_part(case.ratlesnet_case_id, "case_id")
    )
    case_dir.mkdir(parents=True, exist_ok=True)

    scan_img, spacing_mm = _load_scan_as_4d(case.scan_nifti)
    _validate_spacing(spacing_mm, expected_spacing_mm, spacing_tolerance, case.scan_nifti)

    label_img, label_voxels = _load_label_as_binary(
        case.lesion_mask,
        reference_shape=scan_img.shape[:3],
        reference_affine=scan_img.affine,
        reference_header=scan_img.header,
        label_threshold=label_threshold,
    )

    scan_path = case_dir / scan_filename
    label_path = case_dir / label_filename
    nib.save(scan_img, scan_path)
    nib.save(label_img, label_path)
    if write_readme_alias and label_filename != README_LABEL_ALIAS:
        nib.save(label_img, case_dir / README_LABEL_ALIAS)

    lesion_volume_mm3 = float(label_voxels * np.prod(spacing_mm))
    return PreparedCase(
        split=case.split,
        animal_id=case.animal_id,
        study=case.study,
        timepoint=case.timepoint,
        case_id=case.ratlesnet_case_id,
        case_dir=case_dir,
        scan_path=scan_path,
        label_path=label_path,
        spacing_mm=spacing_mm,
        image_shape=tuple(int(v) for v in scan_img.shape),
        label_voxels=label_voxels,
        lesion_volume_mm3=lesion_volume_mm3,
    )


def _load_scan_as_4d(path: Path) -> tuple[nib.Nifti1Image, tuple[float, float, float]]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    spacing_mm = tuple(float(v) for v in img.header.get_zooms()[:3])
    if data.ndim == 3:
        data = data[..., np.newaxis]
    elif data.ndim == 4:
        if data.shape[-1] != 1:
            raise ValueError(
                f"{path} has {data.shape[-1]} channels/modalities. "
                "This branch currently supports one T2w modality."
            )
    else:
        raise ValueError(f"{path} must be a 3-D or 4-D NIfTI, got shape {data.shape}")

    header = img.header.copy()
    header.set_data_shape(data.shape)
    header.set_data_dtype(np.float32)
    header.set_zooms(spacing_mm + (1.0,))
    return nib.Nifti1Image(data.astype(np.float32, copy=False), img.affine, header), spacing_mm


def _load_label_as_binary(
    path: Path,
    *,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    reference_header: nib.Nifti1Header,
    label_threshold: float,
) -> tuple[nib.Nifti1Image, int]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"{path} must be a 3-D binary lesion mask, got shape {data.shape}")
    if tuple(data.shape) != reference_shape:
        raise ValueError(
            f"{path} shape {data.shape} does not match scan shape {reference_shape}. "
            "Do not resample silently; fix the mask/scan pair first."
        )

    binary = (data > label_threshold).astype(np.uint8)
    header = reference_header.copy()
    header.set_data_shape(binary.shape)
    header.set_data_dtype(np.uint8)
    header.set_zooms(reference_header.get_zooms()[:3])
    return nib.Nifti1Image(binary, reference_affine, header), int(binary.sum())


def _write_manifest(path: Path, prepared: list[PreparedCase]) -> None:
    columns = [
        "split",
        "animal_id",
        "study",
        "timepoint",
        "case_id",
        "case_dir",
        "scan_path",
        "label_path",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "image_shape",
        "label_voxels",
        "lesion_volume_mm3",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for case in prepared:
            writer.writerow(
                {
                    "split": case.split,
                    "animal_id": case.animal_id,
                    "study": case.study,
                    "timepoint": case.timepoint,
                    "case_id": case.case_id,
                    "case_dir": str(case.case_dir),
                    "scan_path": str(case.scan_path),
                    "label_path": str(case.label_path),
                    "spacing_x_mm": case.spacing_mm[0],
                    "spacing_y_mm": case.spacing_mm[1],
                    "spacing_z_mm": case.spacing_mm[2],
                    "image_shape": "x".join(str(v) for v in case.image_shape),
                    "label_voxels": case.label_voxels,
                    "lesion_volume_mm3": f"{case.lesion_volume_mm3:.10g}",
                }
            )


def _validate_spacing(
    spacing_mm: tuple[float, float, float],
    expected_spacing_mm: tuple[float, float, float] | None,
    spacing_tolerance: float,
    path: Path,
) -> None:
    if expected_spacing_mm is None:
        return
    bad = [
        (axis, got, expected)
        for axis, got, expected in zip("xyz", spacing_mm, expected_spacing_mm, strict=True)
        if abs(got - expected) > spacing_tolerance
    ]
    if bad:
        detail = ", ".join(
            f"{axis}: got {got:g} mm expected {expected:g} mm"
            for axis, got, expected in bad
        )
        raise ValueError(f"NIfTI spacing check failed for {path}: {detail}")


def _ensure_not_inside_data(repo_root: Path, output_path: Path) -> None:
    data_root = (repo_root / "data").resolve()
    resolved = output_path.resolve()
    if resolved == data_root or resolved.is_relative_to(data_root):
        raise ValueError(f"Refusing to write RatLesNetV2 prepared data under data/: {output_path}")


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else repo_root / p


def _required(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value in (None, "", "TODO"):
        raise ValueError(f"Missing required RatLesNetV2 dataset config key: {key}")
    return value


def _clean_path_part(value: str, field_name: str) -> str:
    text = str(value).strip()
    if text in {"", ".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"{field_name}={value!r} is not a safe single path component")
    return text
