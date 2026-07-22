"""Review gates for mapping partial-volume T2w acquisitions to AIDAmri."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from src.atlas.t2w_mapping import (
    SpatialVolume,
    load_binary_mask,
    load_t2,
    require_same_grid,
    sha256,
)


@dataclass(frozen=True)
class SlabCandidate:
    """One possible correspondence between subject and template AP slices."""

    candidate_id: str
    rank: int
    template_start_index: int
    template_end_index: int
    template_start_offset_mm: float
    template_end_offset_mm: float
    area_profile_rmse: float
    template_slice_indices: tuple[int, ...]


def prepare_partial_volume_slab_review(
    *,
    case_id: str,
    subject_t2: Path,
    subject_brain_mask: Path,
    lesion_mask: Path,
    template_t2: Path,
    output_root: Path,
    expected_orientation: str = "LIP",
    top_candidates: int = 6,
    minimum_candidate_separation_mm: float = 0.25,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rank AP slab candidates without making an anatomical selection.

    Ranking uses only the normalized cross-sectional brain-area profile. It is
    deliberately a review aid: the emitted selection record remains pending
    until a reviewer confirms corresponding anatomical landmarks.
    """
    case_id = _validate_case_id(case_id)
    if top_candidates < 1:
        raise ValueError("top_candidates must be at least 1")
    if minimum_candidate_separation_mm <= 0:
        raise ValueError("minimum_candidate_separation_mm must be positive")

    subject = load_t2(subject_t2.resolve())
    brain = load_binary_mask(subject_brain_mask.resolve())
    lesion = load_binary_mask(lesion_mask.resolve())
    template = load_t2(template_t2.resolve())
    require_same_grid(subject, brain, names=("subject T2w", "subject brain mask"))
    require_same_grid(subject, lesion, names=("subject T2w", "lesion mask"))

    expected_orientation = expected_orientation.upper()
    if subject.orientation != expected_orientation:
        raise ValueError(
            f"Subject orientation is {subject.orientation}, expected {expected_orientation}"
        )
    if template.orientation != expected_orientation:
        raise ValueError(
            f"Template orientation is {template.orientation}, expected {expected_orientation}"
        )

    ap_axis = _ap_axis(subject.orientation)
    if _ap_axis(template.orientation) != ap_axis:
        raise ValueError("Subject and template AP axes do not occupy the same array axis")
    if subject.orientation[ap_axis] != template.orientation[ap_axis]:
        raise ValueError("Subject and template AP array directions disagree")

    template_support = np.isfinite(template.data) & (template.data != 0)
    if not np.any(template_support):
        raise ValueError("Template has no finite nonzero support for slab ranking")
    subject_profile = _cross_sectional_area_profile(brain.data != 0, subject, ap_axis)
    template_profile = _cross_sectional_area_profile(template_support, template, ap_axis)
    candidates = rank_slab_candidates(
        subject_profile_mm2=subject_profile,
        template_profile_mm2=template_profile,
        subject_slice_spacing_mm=subject.spacing_mm[ap_axis],
        template_slice_spacing_mm=template.spacing_mm[ap_axis],
    )
    separation_voxels = max(
        1,
        int(np.ceil(minimum_candidate_separation_mm / template.spacing_mm[ap_axis])),
    )
    review_candidates = select_review_candidates(
        candidates,
        count=min(top_candidates, len(candidates)),
        minimum_start_separation_voxels=separation_voxels,
    )

    output_root = output_root.resolve()
    _prepare_output_root(output_root, overwrite=overwrite)
    ranking_path = output_root / "slab_candidates.csv"
    _write_candidate_csv(ranking_path, candidates)
    score_plot = output_root / "slab_candidate_scores.png"
    _write_score_plot(candidates, score_plot)
    comparison = output_root / "slab_candidate_comparison.png"
    _write_candidate_comparison(
        subject=subject,
        lesion=lesion,
        template=template,
        ap_axis=ap_axis,
        candidates=review_candidates,
        output_path=comparison,
    )
    individual_comparisons = []
    for candidate in review_candidates:
        candidate_path = output_root / f"{candidate.candidate_id}.png"
        _write_candidate_comparison(
            subject=subject,
            lesion=lesion,
            template=template,
            ap_axis=ap_axis,
            candidates=[candidate],
            output_path=candidate_path,
        )
        individual_comparisons.append(str(candidate_path))

    ap_edge_occupancy = _edge_occupancy(brain.data != 0, ap_axis)
    selection_path = output_root / "slab_selection.json"
    selection_record = {
        "schema_version": 1,
        "case_id": case_id,
        "selection_status": "pending",
        "selected_candidate_id": None,
        "selected_template_start_index": None,
        "selected_template_end_index": None,
        "anatomical_landmarks_confirmed": None,
        "reviewer": None,
        "review_date": None,
        "notes": "",
        "subject_t2_sha256": sha256(subject.path),
        "subject_brain_mask_sha256": sha256(brain.path),
        "lesion_mask_sha256": sha256(lesion.path),
        "template_t2_sha256": sha256(template.path),
        "slab_candidates_csv_sha256": sha256(ranking_path),
        "subject_orientation": subject.orientation,
        "template_orientation": template.orientation,
        "ap_axis": ap_axis,
        "ap_array_direction": subject.orientation[ap_axis],
        "review_candidate_ids": [candidate.candidate_id for candidate in review_candidates],
    }
    selection_path.write_text(json.dumps(selection_record, indent=2, sort_keys=True) + "\n")

    summary = {
        "schema_version": 1,
        "case_id": case_id,
        "scientific_status": "requires_human_ap_slab_selection",
        "subject_t2": str(subject.path),
        "subject_brain_mask": str(brain.path),
        "lesion_mask": str(lesion.path),
        "template_t2": str(template.path),
        "subject_shape": list(subject.data.shape),
        "template_shape": list(template.data.shape),
        "subject_spacing_mm": list(subject.spacing_mm),
        "template_spacing_mm": list(template.spacing_mm),
        "subject_ap_extent_center_to_center_mm": (
            (subject.data.shape[ap_axis] - 1) * subject.spacing_mm[ap_axis]
        ),
        "template_ap_extent_center_to_center_mm": (
            (template.data.shape[ap_axis] - 1) * template.spacing_mm[ap_axis]
        ),
        "subject_brain_mask_touches_ap_edges": list(ap_edge_occupancy),
        "template_support_definition": "finite_and_nonzero_template_voxels",
        "ranking_method": "normalized_cross_sectional_brain_area_profile_rmse",
        "ranking_is_anatomical_approval": False,
        "best_geometry_candidate": asdict(candidates[0]),
        "review_candidates": [asdict(candidate) for candidate in review_candidates],
        "slab_candidates_csv": str(ranking_path),
        "slab_candidate_scores": str(score_plot),
        "slab_candidate_comparison": str(comparison),
        "individual_candidate_comparisons": individual_comparisons,
        "slab_selection_json": str(selection_path),
    }
    summary_path = output_root / "slab_review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def rank_slab_candidates(
    *,
    subject_profile_mm2: np.ndarray,
    template_profile_mm2: np.ndarray,
    subject_slice_spacing_mm: float,
    template_slice_spacing_mm: float,
) -> list[SlabCandidate]:
    """Rank every integer template start index whose sampled slab fits."""
    subject_profile = _normalize_profile(subject_profile_mm2, "subject")
    template_profile_mm2 = np.asarray(template_profile_mm2, dtype=np.float64)
    if template_profile_mm2.ndim != 1 or not np.all(np.isfinite(template_profile_mm2)):
        raise ValueError("template profile must be a finite one-dimensional array")
    if subject_slice_spacing_mm <= 0 or template_slice_spacing_mm <= 0:
        raise ValueError("slice spacing must be positive")
    relative_indices = (
        np.arange(subject_profile.size, dtype=np.float64)
        * subject_slice_spacing_mm
        / template_slice_spacing_mm
    )
    max_start = int(np.floor((template_profile_mm2.size - 1) - relative_indices[-1]))
    if max_start < 0:
        raise ValueError("Subject AP extent is larger than the template AP extent")

    raw: list[dict[str, Any]] = []
    for start_index in range(max_start + 1):
        indices = np.rint(start_index + relative_indices).astype(int)
        sampled = _normalize_profile(
            template_profile_mm2[indices],
            f"template candidate starting at {start_index}",
        )
        raw.append(
            {
                "start": start_index,
                "end": int(indices[-1]),
                "indices": tuple(int(value) for value in indices),
                "rmse": float(np.sqrt(np.mean(np.square(subject_profile - sampled)))),
            }
        )
    raw.sort(key=lambda item: (item["rmse"], item["start"]))
    return [
        SlabCandidate(
            candidate_id=f"template_start_{item['start']:04d}",
            rank=rank,
            template_start_index=item["start"],
            template_end_index=item["end"],
            template_start_offset_mm=item["start"] * template_slice_spacing_mm,
            template_end_offset_mm=item["end"] * template_slice_spacing_mm,
            area_profile_rmse=item["rmse"],
            template_slice_indices=item["indices"],
        )
        for rank, item in enumerate(raw, start=1)
    ]


def select_review_candidates(
    candidates: list[SlabCandidate],
    *,
    count: int,
    minimum_start_separation_voxels: int,
) -> list[SlabCandidate]:
    """Select well-spaced high-ranking candidates for visual review."""
    if count < 1 or minimum_start_separation_voxels < 1:
        raise ValueError("candidate count and separation must be positive")
    selected: list[SlabCandidate] = []
    for candidate in candidates:
        if all(
            abs(candidate.template_start_index - other.template_start_index)
            >= minimum_start_separation_voxels
            for other in selected
        ):
            selected.append(candidate)
            if len(selected) == count:
                break
    return selected


def load_approved_slab_selection(
    *,
    selection_json: Path,
    candidates_csv: Path,
    subject_t2: Path,
    subject_brain_mask: Path,
    lesion_mask: Path,
    template_t2: Path,
) -> SlabCandidate:
    """Load a reviewer-approved, hash-current slab decision."""
    selection_json = selection_json.resolve()
    candidates_csv = candidates_csv.resolve()
    record = json.loads(selection_json.read_text())
    if record.get("selection_status") != "approved":
        raise ValueError(f"{selection_json}: slab selection is not approved")
    if record.get("anatomical_landmarks_confirmed") is not True:
        raise ValueError(f"{selection_json}: anatomical landmarks are not confirmed")
    if not record.get("reviewer") or not record.get("review_date"):
        raise ValueError(f"{selection_json}: approval requires reviewer and review_date")
    try:
        date.fromisoformat(str(record["review_date"]))
    except ValueError as error:
        raise ValueError(f"{selection_json}: review_date must be YYYY-MM-DD") from error
    required_hashes = {
        "subject_t2_sha256": sha256(subject_t2.resolve()),
        "subject_brain_mask_sha256": sha256(subject_brain_mask.resolve()),
        "lesion_mask_sha256": sha256(lesion_mask.resolve()),
        "template_t2_sha256": sha256(template_t2.resolve()),
        "slab_candidates_csv_sha256": sha256(candidates_csv),
    }
    for field, expected in required_hashes.items():
        if record.get(field) != expected:
            raise ValueError(f"{selection_json}: {field} is stale or incorrect")

    rows = list(csv.DictReader(candidates_csv.open(newline="")))
    by_id = {row["candidate_id"]: row for row in rows}
    selected_id = record.get("selected_candidate_id")
    if selected_id not in by_id:
        raise ValueError(f"{selection_json}: selected candidate is absent from ranking CSV")
    row = by_id[selected_id]
    candidate = _candidate_from_csv_row(row)
    if record.get("selected_template_start_index") != candidate.template_start_index:
        raise ValueError(f"{selection_json}: selected template start index is inconsistent")
    if record.get("selected_template_end_index") != candidate.template_end_index:
        raise ValueError(f"{selection_json}: selected template end index is inconsistent")
    return candidate


def _cross_sectional_area_profile(
    mask: np.ndarray,
    volume: SpatialVolume,
    ap_axis: int,
) -> np.ndarray:
    in_plane_axes = tuple(axis for axis in range(3) if axis != ap_axis)
    pixel_area = float(np.prod([volume.spacing_mm[axis] for axis in in_plane_axes]))
    return np.count_nonzero(mask, axis=in_plane_axes).astype(np.float64) * pixel_area


def _normalize_profile(profile: np.ndarray, name: str) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 1 or profile.size < 2 or not np.all(np.isfinite(profile)):
        raise ValueError(f"{name} profile must be a finite one-dimensional array")
    maximum = float(np.max(profile))
    if maximum <= 0:
        raise ValueError(f"{name} profile has no positive area")
    return profile / maximum


def _ap_axis(orientation: str) -> int:
    axes = [axis for axis, code in enumerate(orientation) if code in {"A", "P"}]
    if len(axes) != 1:
        raise ValueError(f"Expected exactly one AP axis in orientation {orientation!r}")
    return axes[0]


def _edge_occupancy(mask: np.ndarray, axis: int) -> tuple[bool, bool]:
    first = bool(np.any(np.take(mask, 0, axis=axis)))
    last = bool(np.any(np.take(mask, mask.shape[axis] - 1, axis=axis)))
    return first, last


def _validate_case_id(case_id: str) -> str:
    case_id = case_id.strip()
    if not case_id or case_id in {".", ".."} or "/" in case_id or "\\" in case_id:
        raise ValueError(f"Unsafe or empty case ID: {case_id!r}")
    return case_id


def _prepare_output_root(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_candidate_csv(path: Path, candidates: list[SlabCandidate]) -> None:
    fieldnames = [
        "candidate_id",
        "rank",
        "template_start_index",
        "template_end_index",
        "template_start_offset_mm",
        "template_end_offset_mm",
        "area_profile_rmse",
        "template_slice_indices_json",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            values = {
                key: value
                for key, value in asdict(candidate).items()
                if key != "template_slice_indices"
            }
            writer.writerow(
                {
                    **values,
                    "template_slice_indices_json": json.dumps(candidate.template_slice_indices),
                }
            )


def _candidate_from_csv_row(row: dict[str, str]) -> SlabCandidate:
    indices = tuple(int(value) for value in json.loads(row["template_slice_indices_json"]))
    return SlabCandidate(
        candidate_id=row["candidate_id"],
        rank=int(row["rank"]),
        template_start_index=int(row["template_start_index"]),
        template_end_index=int(row["template_end_index"]),
        template_start_offset_mm=float(row["template_start_offset_mm"]),
        template_end_offset_mm=float(row["template_end_offset_mm"]),
        area_profile_rmse=float(row["area_profile_rmse"]),
        template_slice_indices=indices,
    )


def _write_score_plot(candidates: list[SlabCandidate], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(candidates, key=lambda candidate: candidate.template_start_index)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        [candidate.template_start_offset_mm for candidate in ordered],
        [candidate.area_profile_rmse for candidate in ordered],
        color="black",
    )
    best = candidates[0]
    axis.scatter([best.template_start_offset_mm], [best.area_profile_rmse], color="red")
    axis.set_xlabel("Template slab start offset from AP array index 0 (mm)")
    axis.set_ylabel("Normalized cross-sectional area profile RMSE")
    axis.set_title("Geometry ranking only — anatomical review is still required")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def _write_candidate_comparison(
    *,
    subject: SpatialVolume,
    lesion: SpatialVolume,
    template: SpatialVolume,
    ap_axis: int,
    candidates: list[SlabCandidate],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subject_indices = np.unique(
        np.linspace(0, subject.data.shape[ap_axis] - 1, 6, dtype=int)
    ).tolist()
    subject_display = _intensity_display(subject.data)
    template_display = _intensity_display(template.data)
    figure, axes = plt.subplots(
        len(candidates) * 2,
        len(subject_indices),
        figsize=(16, 4 * len(candidates)),
        squeeze=False,
    )
    for candidate_index, candidate in enumerate(candidates):
        subject_row = axes[candidate_index * 2]
        template_row = axes[candidate_index * 2 + 1]
        for column, subject_index in enumerate(subject_indices):
            template_index = candidate.template_slice_indices[subject_index]
            subject_slice = np.take(subject_display, subject_index, axis=ap_axis)
            lesion_slice = np.take(lesion.data != 0, subject_index, axis=ap_axis)
            template_slice = np.take(template_display, template_index, axis=ap_axis)
            subject_row[column].imshow(subject_slice.T, cmap="gray", origin="lower")
            subject_row[column].imshow(
                np.ma.masked_where(~lesion_slice.T, lesion_slice.T),
                cmap="cool",
                origin="lower",
                alpha=0.3,
            )
            subject_row[column].set_title(f"subject slice {subject_index}")
            template_row[column].imshow(template_slice.T, cmap="gray", origin="lower")
            template_row[column].set_title(f"template slice {template_index}")
            subject_row[column].axis("off")
            template_row[column].axis("off")
        subject_row[0].set_ylabel("subject")
        template_row[0].set_ylabel(
            f"{candidate.candidate_id}\nrank {candidate.rank}; "
            f"RMSE {candidate.area_profile_rmse:.4f}"
        )
    figure.suptitle(
        "Partial-volume AP slab candidates — compare landmarks away from cyan lesion"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def _intensity_display(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    lower, upper = np.percentile(finite, (1, 99))
    if upper <= lower:
        upper = lower + 1.0
    return np.clip((data - lower) / (upper - lower), 0.0, 1.0)
