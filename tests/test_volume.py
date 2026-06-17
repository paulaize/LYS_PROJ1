"""Pure-arithmetic tests — the v1 stages that must be provably correct.

These intentionally avoid heavy imaging deps (nibabel/SimpleITK/napari) so they
run in any environment, including CI and Codex sandboxes.
"""
import numpy as np
import pytest

from src.mri.edit import dice, review_mask
from src.mri.segment import segment_lesion
from src.mri.volume import mask_volume_mm3, per_slice_area_mm2, voxel_volume_mm3

SPACING = (0.07, 0.07, 0.5)  # anisotropic — the whole point


def test_voxel_volume_is_anisotropic():
    assert voxel_volume_mm3(SPACING) == 0.07 * 0.07 * 0.5
    # an isotropic assumption (0.07^3) would be wrong by ~7x
    assert voxel_volume_mm3(SPACING) != 0.07 ** 3


def test_mask_volume_uses_spacing():
    mask = np.zeros((10, 10, 4), dtype=bool)
    mask[2:5, 2:5, 1:3] = True  # 3*3*2 = 18 voxels
    expected = 18 * voxel_volume_mm3(SPACING)
    assert abs(mask_volume_mm3(mask, SPACING) - expected) < 1e-9


def test_empty_mask_is_zero_volume():
    assert mask_volume_mm3(np.zeros((5, 5, 5), dtype=bool), SPACING) == 0.0


def test_per_slice_area():
    mask = np.zeros((10, 10, 4), dtype=bool)
    mask[:, :, 1] = False
    mask[2:4, 2:7, 2] = True  # slice index 2: 2*5 = 10 px
    areas = per_slice_area_mm2(mask, SPACING, slice_axis=2)
    assert areas.shape == (4,)
    assert abs(areas[2] - 10 * 0.07 * 0.07) < 1e-9
    assert areas[0] == 0.0


def test_dice_identity_and_empty():
    a = np.zeros((4, 4, 4), dtype=bool)
    a[1:3, 1:3, 1:3] = True
    assert dice(a, a) == 1.0
    # two empties => perfect agreement (e.g. a true control)
    z = np.zeros((4, 4, 4), dtype=bool)
    assert dice(z, z) == 1.0


def test_dice_partial_overlap():
    a = np.zeros((4, 4, 4), dtype=bool)
    a[0:2, :, :] = True
    b = np.zeros((4, 4, 4), dtype=bool)
    b[1:3, :, :] = True
    # |A|=|B|=32, overlap=16 -> 2*16/64 = 0.5
    assert abs(dice(a, b) - 0.5) < 1e-9


def test_review_mask_can_skip_gui_for_technical_run(tmp_path):
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    draft = np.zeros((4, 4, 4), dtype=bool)
    draft[1:3, 1:3, 1:3] = True

    meta = review_mask(
        vol,
        draft,
        tmp_path / "lesion_corrected.npy",
        reviewer="tester",
        allow_gui=False,
    )

    assert meta["qc_flag"] == "needs_human_review"
    assert meta["edited"] is False
    assert meta["dice"] == 1.0
    assert (tmp_path / "lesion_corrected.npy").exists()


def test_threshold_segmenter_finds_hyperintensity():
    # synthetic brain: uniform background, a bright blob in the right hemisphere
    vol = np.full((20, 20, 6), 100.0, dtype=np.float32)
    brain = np.ones((20, 20, 6), dtype=bool)
    vol[12:16, 8:12, 2:4] = 400.0  # bright lesion on the right (x>=10)
    mask = segment_lesion(vol, brain, method="threshold", spacing_mm=SPACING,
                          k=2.0, min_lesion_mm3=0.0)
    assert mask.any()
    # detected voxels should sit on the bright blob's hemisphere
    assert mask[12:16, 8:12, 2:4].sum() > 0


def test_dl_backend_not_implemented():
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    brain = np.ones((4, 4, 4), dtype=bool)
    with pytest.raises(NotImplementedError):
        segment_lesion(vol, brain, method="dl")
