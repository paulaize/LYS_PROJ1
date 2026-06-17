"""Human-in-the-loop lesion-border correction.

FIXED INTERFACE: keep review_mask(volume, draft_mask, out_path) usable. Optional
keyword arguments are allowed for orchestration, but callers must still be able
to depend on the stable v1 contract.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def record_edit(draft: np.ndarray, corrected: np.ndarray, reviewer: str) -> dict:
    d = dice(draft, corrected)
    edited = bool(d < 1.0)
    return {
        "edited": edited,
        "dice": d,
        "reviewer": reviewer,
        "qc_flag": "reviewed_edited" if edited else "reviewed_no_changes",
        "note": "human review completed",
    }


def _to_editor_order(arr: np.ndarray, slice_axis: int = 2) -> np.ndarray:
    """Return napari-friendly stack order: (slice, row/y, column/x)."""
    if slice_axis != 2:
        raise ValueError("v1 napari editor currently expects slice_axis=2")
    return np.transpose(np.asarray(arr), (2, 1, 0))


def _from_editor_order(arr: np.ndarray, slice_axis: int = 2) -> np.ndarray:
    """Undo _to_editor_order before saving back into NIfTI/header space."""
    if slice_axis != 2:
        raise ValueError("v1 napari editor currently expects slice_axis=2")
    return np.transpose(np.asarray(arr), (2, 1, 0))


def _editor_scale(spacing_mm: tuple[float, float, float] | None, slice_axis: int = 2):
    if spacing_mm is None:
        return None
    if slice_axis != 2:
        raise ValueError("v1 napari editor currently expects slice_axis=2")
    return (float(spacing_mm[2]), float(spacing_mm[1]), float(spacing_mm[0]))


def review_mask(
    volume: np.ndarray,
    draft_mask: np.ndarray,
    out_path: str | Path,
    *,
    reference_img=None,
    reviewer: str | None = None,
    allow_gui: bool = True,
    spacing_mm: tuple[float, float, float] | None = None,
    slice_axis: int = 2,
) -> dict:
    """Open napari to edit draft_mask over volume; save corrected mask to out_path.

    Returns keys: edited, dice, reviewer, qc_flag. If napari is unavailable, the
    draft is saved so the run is reproducible, but qc_flag is
    `needs_human_review`; treat that output as not final.
    """
    reviewer = reviewer or os.environ.get("USER", "unknown")
    out_path = Path(out_path)

    if not allow_gui:
        _save(draft_mask, reference_img, out_path)
        return {
            "edited": False,
            "dice": 1.0,
            "reviewer": reviewer,
            "qc_flag": "needs_human_review",
            "note": "mask editor skipped; draft saved unedited - REVIEW NOT DONE",
        }

    try:
        import napari  # imported lazily; GUI only
    except Exception as exc:  # headless / CI / no display
        _save(draft_mask, reference_img, out_path)
        return {
            "edited": False,
            "dice": 1.0,
            "reviewer": reviewer,
            "qc_flag": "needs_human_review",
            "note": f"napari unavailable ({exc}); draft saved unedited — REVIEW NOT DONE",
        }

    display_volume = _to_editor_order(volume, slice_axis=slice_axis)
    display_mask = _to_editor_order(draft_mask, slice_axis=slice_axis)
    scale = _editor_scale(spacing_mm, slice_axis=slice_axis)

    viewer = napari.Viewer()
    viewer.add_image(display_volume, name="T2", scale=scale)
    labels = viewer.add_labels(display_mask.astype(np.uint8), name="lesion (edit me)", scale=scale)
    print("napari open: paint/erase the lesion layer, then close the window to save.")
    napari.run()

    corrected = _from_editor_order(labels.data > 0, slice_axis=slice_axis)
    _save(corrected, reference_img, out_path)
    return record_edit(draft_mask, corrected, reviewer)


def _save(mask, reference_img, out_path: Path):
    if reference_img is not None:
        from . import io as mri_io

        mri_io.save_mask(np.asarray(mask), reference_img, out_path)
    else:
        out_path = out_path.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.asarray(mask).astype(np.uint8))
