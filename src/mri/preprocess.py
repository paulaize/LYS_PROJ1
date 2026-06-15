"""Preprocessing for the MRI track: N4 bias correction + a simple brain mask.

v1 keeps this dependency-light (SimpleITK only). A learned rodent brain
extraction (antspynet) is a Milestone-2/3 upgrade — swap it in behind
brain_mask() without changing callers.
"""
from __future__ import annotations

import numpy as np

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - import guard for environments w/o sitk
    sitk = None


def _require_sitk():
    if sitk is None:
        raise ImportError(
            "SimpleITK is required for preprocessing. "
            "Install via env/environment.yml (conda-forge simpleitk)."
        )


def n4_bias_correct(arr: np.ndarray) -> np.ndarray:
    """N4 bias-field correction. Surface-coil RARE has strong gradients that
    wreck global thresholds, so this runs before segmentation by default."""
    _require_sitk()
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    out = corrector.Execute(img, mask)
    return sitk.GetArrayFromImage(out)


def brain_mask(arr: np.ndarray) -> np.ndarray:
    """Crude brain mask via Otsu + largest connected component + fill.

    Good enough for v1 (hemisphere stats, normalization). Replace with
    antspynet rodent brain extraction when accuracy matters (Milestone 2/3).
    """
    _require_sitk()
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    binary = sitk.OtsuThreshold(img, 0, 1, 200)
    binary = sitk.BinaryMorphologicalClosing(binary, [2, 2, 1])
    cc = sitk.ConnectedComponent(binary)
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = cc == 1
    largest = sitk.BinaryFillhole(largest)
    return sitk.GetArrayFromImage(largest).astype(bool)
