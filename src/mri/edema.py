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

from .sides import ipsi_contra_masks
from .volume import mask_volume_mm3


def swanson_corrected_volume(lesion_mask: np.ndarray,
                             brain_mask: np.ndarray,
                             spacing_mm: tuple[float, float, float],
                             lesion_side: str | None = None) -> dict:
    """Return raw + edema-corrected lesion volumes (mm^3) and the swelling ratio.

    TODO: midline is the naive array centre (see segment.py). Replace with the
    atlas midline once registration exists.
    """
    raw = mask_volume_mm3(lesion_mask, spacing_mm)
    ipsi, contra = ipsi_contra_masks(brain_mask, lesion_side, lesion_mask)
    v_ipsi = mask_volume_mm3(ipsi, spacing_mm)
    v_contra = mask_volume_mm3(contra, spacing_mm)
    ratio = (v_contra / v_ipsi) if v_ipsi > 0 else float("nan")
    corrected = raw * ratio if np.isfinite(ratio) else raw

    return {
        "raw_mm3": raw,
        "corrected_mm3": corrected,
        "swelling_ratio": ratio,
        "v_ipsi_mm3": v_ipsi,
        "v_contra_mm3": v_contra,
    }
