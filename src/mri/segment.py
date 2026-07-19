"""Lesion segmentation. FIXED INTERFACE — do not change segment_lesion's signature.

v1 backend = "threshold": relative T2 hyperintensity vs the (mirrored) healthy
hemisphere. A future "dl" backend must use a frozen, independently validated
model and retain the same review-first interface.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .sides import ipsi_contra_masks


def segment_lesion(volume: np.ndarray,
                   brain_mask: np.ndarray,
                   *,
                   method: str = "threshold",
                   spacing_mm: tuple[float, float, float] | None = None,
                   k: float = 2.5,
                   min_lesion_mm3: float = 0.5,
                   lesion_side: str | None = None,
                   **kw) -> np.ndarray:
    """Return a binary lesion mask (bool array, same shape as volume).

    Parameters
    ----------
    method : "threshold" (v1) or "dl" (v2, NotImplementedError until Milestone 2).
    k : threshold = contra_mean + k * contra_sd.
    min_lesion_mm3 : drop connected components smaller than this (needs spacing_mm).
    lesion_side : "image_right"/"image_left"/None. None => infer from hyperintensity.
    """
    if method == "threshold":
        return _threshold_backend(
            volume, brain_mask, k=k, min_lesion_mm3=min_lesion_mm3,
            spacing_mm=spacing_mm, lesion_side=lesion_side,
        )
    if method == "dl":
        raise NotImplementedError(
            "The production DL backend is not enabled. Freeze and independently "
            "validate a model, record its weights/version in config, and keep this "
            "review-first interface when implementing it."
        )
    raise ValueError(f"Unknown segmentation method: {method!r}")


def _threshold_backend(volume, brain_mask, *, k, min_lesion_mm3, spacing_mm, lesion_side):
    vol = np.asarray(volume, dtype=np.float32)
    brain = np.asarray(brain_mask, dtype=bool)

    ipsi, contra = ipsi_contra_masks(brain, lesion_side, vol)

    if not contra.any():
        raise ValueError("Empty contralateral hemisphere — check the brain mask / midline.")

    thr = float(vol[contra].mean() + k * vol[contra].std())
    mask = (vol > thr) & ipsi

    # clean up: keep components above the size floor
    mask = _drop_small(mask, min_lesion_mm3, spacing_mm)
    return mask.astype(bool)


def _drop_small(mask, min_lesion_mm3, spacing_mm):
    if not mask.any() or not min_lesion_mm3 or spacing_mm is None:
        return mask
    voxel_vol = float(spacing_mm[0]) * float(spacing_mm[1]) * float(spacing_mm[2])
    min_voxels = max(1, int(round(min_lesion_mm3 / voxel_vol)))
    labeled, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(np.ones_like(labeled), labeled, index=range(1, n + 1))
    keep = {i + 1 for i, s in enumerate(sizes) if s >= min_voxels}
    return np.isin(labeled, list(keep))
