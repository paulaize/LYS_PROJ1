"""NIfTI IO and header sanity checks for the MRI track.

The pipeline trusts the header for voxel spacing (voxels are ANISOTROPIC,
~0.07x0.07x0.5 mm). check_header() fails loudly if a loaded volume deviates,
because every downstream volume/area number depends on it.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI not found: {path}")
    return nib.load(str(path))


def get_spacing(img: nib.Nifti1Image) -> tuple[float, float, float]:
    """Voxel spacing in mm (x, y, z) from the header zooms."""
    zooms = img.header.get_zooms()[:3]
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def get_array(img: nib.Nifti1Image) -> np.ndarray:
    return np.asanyarray(img.dataobj)


def check_header(img: nib.Nifti1Image,
                 expected_spacing_mm: tuple[float, float, float],
                 tolerance: float = 0.02) -> tuple[float, float, float]:
    """Return the spacing; warn loudly (ValueError) if it deviates beyond tol.

    Raising is deliberate: a silently-wrong spacing produces a silently-wrong
    lesion volume. If your data legitimately differs, update expected_spacing_mm
    in config/pipeline.yml rather than relaxing this check.
    """
    spacing = get_spacing(img)
    diffs = [abs(a - b) for a, b in zip(spacing, expected_spacing_mm)]
    if any(d > tolerance for d in diffs):
        raise ValueError(
            f"Voxel spacing {spacing} mm deviates from expected "
            f"{tuple(expected_spacing_mm)} mm (tol={tolerance}). "
            "Check the brkraw conversion / reorientation, or update "
            "config/pipeline.yml:mri.expected_spacing_mm if this is intended."
        )
    return spacing


def save_mask(mask: np.ndarray,
              reference: nib.Nifti1Image,
              out_path: str | Path) -> Path:
    """Save a binary/label mask using the reference image's affine+header."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header)
    nib.save(out, str(out_path))
    return out_path
