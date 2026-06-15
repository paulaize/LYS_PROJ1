# tests/fixtures

Small, committable test data so stages can be verified without the full dataset
or the external drive. Keep everything here **tiny** (a few MB max).

Create when you have access to one real animal:

1. **`t2_small.nii.gz`** — a cropped region of one T2 (RARE) volume around the
   lesion, e.g. with FSL:
   ```bash
   fslroi <full_t2>.nii.gz tests/fixtures/t2_small.nii.gz 80 96 80 96 0 18
   ```
   Keep the correct anisotropic spacing (0.07x0.07x0.5 mm) in the header.

2. **`lesion_small.nii.gz`** — a hand-drawn lesion mask for that crop (ITK-SNAP /
   3D Slicer), used as ground truth for Dice checks of the segmenter.

3. **`ihc_region.ome.tif`** (optional) — a small cropped region exported from one
   `.vsi` in QuPath, for exercising the Groovy measurement path.

`.gitignore` blocks `*.nii.gz` / `*.vsi` / `*.tif` by default. If you want to
commit these small fixtures, add explicit allow rules, e.g.:
```
!tests/fixtures/t2_small.nii.gz
!tests/fixtures/lesion_small.nii.gz
```
Only do this for genuinely small files.

The pure-arithmetic tests in `tests/test_volume.py` and `tests/test_config.py`
run today with **no fixtures** — they use synthetic arrays. Fixtures unlock the
integration-level checks (real header sanity, segmenter-vs-manual Dice).
