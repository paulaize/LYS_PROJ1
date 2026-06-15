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


def review_mask(
    volume: np.ndarray,
    draft_mask: np.ndarray,
    out_path: str | Path,
    *,
    reference_img=None,
    reviewer: str | None = None,
) -> dict:
    """Open napari to edit draft_mask over volume; save corrected mask to out_path.

    Returns keys: edited, dice, reviewer, qc_flag. If napari is unavailable, the
    draft is saved so the run is reproducible, but qc_flag is
    `needs_human_review`; treat that output as not final.
    """
    reviewer = reviewer or os.environ.get("USER", "unknown")
    out_path = Path(out_path)

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

    viewer = napari.Viewer()
    viewer.add_image(np.asarray(volume), name="T2")
    labels = viewer.add_labels(np.asarray(draft_mask).astype(np.uint8), name="lesion (edit me)")
    print("napari open: paint/erase the lesion layer, then close the window to save.")
    napari.run()

    corrected = labels.data > 0
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
