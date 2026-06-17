"""Edema correction. v1 = Swanson / indirect method.

The infarcted hemisphere swells, so the raw lesion volume overstates the
infarct. The indirect estimate uses the healthy contralateral hemisphere as a
reference:

    corrected_lesion = lesion_measured * (V_contra / V_ipsi)

where V_contra / V_ipsi are hemisphere tissue volumes. This is the standard
edema correction and matches prior lab practice. An atlas-Jacobian map
(Koch et al. 2019) is a Milestone-3 add-on.

NOTE: runs on the HUMAN-CORRECTED mask, so manual edits propagate into the
reported number (intended).
"""
from __future__ import annotations

import numpy as np

from .volume import mask_volume_mm3


def _hemispheres(brain_mask: np.ndarray):
    nx = brain_mask.shape[0]
    mid = nx // 2
    left = np.zeros_like(brain_mask, dtype=bool)
    left[:mid] = brain_mask[:mid] > 0
    right = np.zeros_like(brain_mask, dtype=bool)
    right[mid:] = brain_mask[mid:] > 0
    return left, right


def swanson_corrected_volume(lesion_mask: np.ndarray,
                             brain_mask: np.ndarray,
                             spacing_mm: tuple[float, float, float],
                             lesion_side: str | None = None) -> dict:
    """Return raw + edema-corrected lesion volumes (mm^3) and the swelling ratio.

    TODO: midline is the naive array centre (see segment.py). Replace with the
    atlas midline once registration exists.
    """
    left, right = _hemispheres(brain_mask)
    v_left = mask_volume_mm3(left, spacing_mm)
    v_right = mask_volume_mm3(right, spacing_mm)
    raw = mask_volume_mm3(lesion_mask, spacing_mm)

    # ipsi = hemisphere containing most of the lesion (or as told)
    if lesion_side in ("L", "left"):
        ipsi_is_left = True
    elif lesion_side in ("R", "right"):
        ipsi_is_left = False
    else:
        ipsi_is_left = np.count_nonzero(lesion_mask & left) >= np.count_nonzero(lesion_mask & right)

    v_ipsi, v_contra = (v_left, v_right) if ipsi_is_left else (v_right, v_left)
    ratio = (v_contra / v_ipsi) if v_ipsi > 0 else float("nan")
    corrected = raw * ratio if np.isfinite(ratio) else raw

    return {
        "raw_mm3": raw,
        "corrected_mm3": corrected,
        "swelling_ratio": ratio,
        "v_ipsi_mm3": v_ipsi,
        "v_contra_mm3": v_contra,
    }
