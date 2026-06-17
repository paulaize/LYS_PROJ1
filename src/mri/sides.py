"""MRI side conventions for v1 lesion processing.

For the current Bruker T2 NIfTI as displayed in Fiji, image-right occupies the
higher-index half of array axis 0. Keep this isolated so a later
orientation-aware implementation can replace it without touching callers.
"""
from __future__ import annotations

import numpy as np

IMAGE_RIGHT_ALIASES = {"image_right", "viewer_right", "display_right", "right", "r"}
IMAGE_LEFT_ALIASES = {"image_left", "viewer_left", "display_left", "left", "l"}


def split_image_lr(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (image_right, image_left) masks using the current v1 orientation."""
    brain = np.asarray(mask, dtype=bool)
    mid = brain.shape[0] // 2
    image_right = np.zeros_like(brain, dtype=bool)
    image_right[mid:] = brain[mid:]
    image_left = np.zeros_like(brain, dtype=bool)
    image_left[:mid] = brain[:mid]
    return image_right, image_left


def ipsi_contra_masks(
    brain_mask: np.ndarray,
    lesion_side: str | None,
    volume: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ipsilateral and contralateral masks.

    If lesion_side is unset, infer from mean T2 hyperintensity. v1 should prefer
    an explicit side such as ``image_right`` for stroke animals.
    """
    image_right, image_left = split_image_lr(brain_mask)
    side = "" if lesion_side is None else str(lesion_side).strip().lower()

    if side in IMAGE_RIGHT_ALIASES:
        return image_right, image_left
    if side in IMAGE_LEFT_ALIASES:
        return image_left, image_right
    if side:
        raise ValueError(
            f"Unknown lesion_side {lesion_side!r}; use image_right, image_left, or null."
        )
    if volume is None:
        raise ValueError("volume is required when lesion_side is unset")

    vol = np.asarray(volume, dtype=np.float32)
    rmean = vol[image_right].mean() if image_right.any() else -np.inf
    lmean = vol[image_left].mean() if image_left.any() else -np.inf
    return (image_right, image_left) if rmean >= lmean else (image_left, image_right)
