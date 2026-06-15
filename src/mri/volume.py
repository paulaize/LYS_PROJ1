"""Lesion volume from a binary mask + voxel spacing.

CRITICAL: voxels are anisotropic. Volume = voxel_count * (sx*sy*sz). Never
assume isotropic spacing. (AGENTS.md §7)
"""
from __future__ import annotations

import numpy as np


def voxel_volume_mm3(spacing_mm: tuple[float, float, float]) -> float:
    sx, sy, sz = spacing_mm
    return float(sx) * float(sy) * float(sz)


def mask_volume_mm3(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    """Total volume of the positive voxels in a binary mask, in mm^3."""
    n = int(np.count_nonzero(mask))
    return n * voxel_volume_mm3(spacing_mm)


def per_slice_area_mm2(mask: np.ndarray,
                       spacing_mm: tuple[float, float, float],
                       slice_axis: int = 2) -> np.ndarray:
    """Lesion area (mm^2) for each slice along slice_axis.

    Useful for the rostro-caudal lesion profile and for picking IHC-matching
    planes later. In-plane pixel area = product of the two non-slice spacings.
    """
    in_plane = [s for i, s in enumerate(spacing_mm) if i != slice_axis]
    pixel_area = float(in_plane[0]) * float(in_plane[1])
    # nonzero pixels per slice along slice_axis
    counts = np.count_nonzero(np.moveaxis(mask, slice_axis, 0), axis=(1, 2))
    return counts.astype(float) * pixel_area
