"""Lesion segmentation. FIXED INTERFACE — do not change segment_lesion's signature.

v1 backend = "threshold": relative T2 hyperintensity vs the (mirrored) healthy
hemisphere. v2 backend = "dl": the An et al. 2023 pretrained mouse-T2 model,
QC'd against manual masks first (it false-positives on lesion-free brains).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


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
    lesion_side : "L"/"R"/None. None => infer from hyperintensity.
    """
    if method == "threshold":
        return _threshold_backend(
            volume, brain_mask, k=k, min_lesion_mm3=min_lesion_mm3,
            spacing_mm=spacing_mm, lesion_side=lesion_side,
        )
    if method == "dl":
        raise NotImplementedError(
            "DL backend (An et al. 2023) is Milestone 2. Obtain/QC the model, "
            "set config mri.dl.weights_path, then implement _dl_backend() here. "
            "Keep this signature. QC vs manual masks before trusting outputs; "
            "apply min_lesion_mm3 to suppress control false-positives."
        )
    raise ValueError(f"Unknown segmentation method: {method!r}")


def _threshold_backend(volume, brain_mask, *, k, min_lesion_mm3, spacing_mm, lesion_side):
    vol = np.asarray(volume, dtype=np.float32)
    brain = np.asarray(brain_mask, dtype=bool)

    # Split into left/right hemispheres along the x axis (axis 0 of the array as
    # loaded by nibabel is the first spatial axis). NOTE: this assumes the volume
    # is roughly midline-centered along axis 0. TODO: replace the naive midline
    # with a symmetry-plane fit or the atlas midline (Milestone 3).
    nx = vol.shape[0]
    mid = nx // 2
    left = np.zeros_like(brain)
    left[:mid] = brain[:mid]
    right = np.zeros_like(brain)
    right[mid:] = brain[mid:]

    # Decide ipsi (lesion) vs contra (reference) hemisphere.
    if lesion_side in ("L", "left"):
        ipsi, contra = left, right
    elif lesion_side in ("R", "right"):
        ipsi, contra = right, left
    else:
        # infer: lesion side has higher mean T2 (hyperintense edema)
        lmean = vol[left].mean() if left.any() else -np.inf
        rmean = vol[right].mean() if right.any() else -np.inf
        ipsi, contra = (left, right) if lmean >= rmean else (right, left)

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
