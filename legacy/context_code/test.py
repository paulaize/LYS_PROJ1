# %% [1] Imports
from pathlib import Path
import json

import numpy as np
import nibabel as nib # most common for nifti
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from matplotlib.widgets import Slider

# %% 
NIFTI_DIR = Path.home() / "Documents/Lys_BBB/test_data/20241003_094309_FP_THR_03_D1_C25S1_1_1_nifty"

SCAN_PATH = NIFTI_DIR / "sub-FP_THR_03_D1_C25S1_study-FP_THR_03_D1_03102024_scan-6_T1_FLASH_3D_Glymphatic_Sag.nii.gz"
print(f"\nUsing: {SCAN_PATH.name}")

# %% load scana nd metadata
img = nib.load(SCAN_PATH)

data = img.get_fdata()

affine = img.affine
zooms = img.header.get_zooms()

print(f"Shape:            {data.shape}")
print(f"Data type:        {data.dtype}")
print(f"Voxel size (mm):  {tuple(round(z, 4) for z in zooms[:3])}")
print(f"Intensity range:  [{data.min():.1f}, {data.max():.1f}]")
print(f"Mean intensity:   {data.mean():.1f}")
print(f"Median intensity: {np.median(data):.1f}")

# # %%
# This should be integrated because it has a bunch of metadata on the MRI 
# other than the image itself. But for now nothing... Although this data exists in
# the Burker format before conversion and can still be obtained by cli calls
# it will probably give me more about the orientation etc...
# sidecar = SCAN_PATH.with_suffix("").with_suffix(".json")
# print(sidecar)
# if sidecar.exists():
#     meta = json.loads(sidecar.read_text())
#     print("\nKey acquisition parameters:")
#     for k in ("Method", "ProtocolName", "EchoTime", "RepetitionTime",
#              "FlipAngle", "NRepetitions", "NAverages", "SliceThickness"):
#         if k in meta:
#             print(f"  {k}: {meta[k]}")


# # %% Intensity — understand the value distribution
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# axes[0].hist(data.ravel(), bins=200, color="steelblue")
# axes[0].set_title("Full intensity histogram")
# axes[0].set_xlabel("Intensity")
# axes[0].set_ylabel("Voxel count")

# axes[1].hist(data.ravel(), bins=200, color="steelblue")
# axes[1].set_yscale("log")
# axes[1].set_title("Histogram (log y-axis)")
# axes[1].set_xlabel("Intensity")
# axes[1].set_ylabel("Voxel count (log)")

# # Suggest reasonable display window: 1st–99th percentile
# p1, p99 = np.percentile(data, [1, 99])
# for ax in axes:
#     ax.axvline(p1, color="red", linestyle="--", alpha=0.5, label=f"1st pct={p1:.0f}")
#     ax.axvline(p99, color="red", linestyle="--", alpha=0.5, label=f"99th pct={p99:.0f}")
#     ax.legend()

# plt.tight_layout()
# plt.show()

# print(f"\nRecommended display range: vmin={p1:.0f}, vmax={p99:.0f}")

# %% all slices 
nslices = data.shape[2]
ncols = 6
nrows = int(np.ceil(nslices / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
axes = axes.flatten()

vmin, vmax = np.percentile(data, [1, 99])

for i in range(nslices):
    ax = axes[i]
    # ax.imshow(data[:, :, i].T, vmin=vmin, vmax=vmax)
    # bnw
    ax.imshow(data[:, :, i].T,  cmap="gray", origin="lower",
              vmin=vmin, vmax=vmax)
    ax.set_title(f"slice {i} (z={i * zooms[2]:.2f} mm)", fontsize=8)
    ax.axis("off")

plt.suptitle(f"all {nslices} axial slices", y=1.00)
plt.tight_layout()
plt.show()

# %% intensity per slice
slice_means = data.mean(axis=(0, 1))
slice_maxes = data.max(axis=(0, 1))
slice_p95 = np.percentile(data, 95, axis=(0, 1))

x = np.arange(1, len(slice_means) + 1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, slice_means, "o-", label="mean")
ax.plot(x, slice_p95, "s-", label="95th percentile")
ax.plot(x, slice_maxes / 10, "^-", label="max / 10")

ax.set_xlim(0, 20)
ax.set_xticks(np.arange(0, 21, 1))

ax.set_xlabel("Slice n°")
ax.set_ylabel("Intensity %")
ax.set_title("Intensity per slice")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% single-slice view 
# nibable notes --> stroke on T2 appear as bright
SLICE_TO_INSPECT = nslices // 2

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

ax = axes[0]
# ax.imshow(data[:, :, SLICE_TO_INSPECT].T, vmin=vmin, vmax=vmax)
# bnw
ax.imshow(data[:, :, SLICE_TO_INSPECT].T, cmap="gray", origin="lower",
          vmin=vmin, vmax=vmax)
ax.set_title(f"slice {SLICE_TO_INSPECT}", fontsize=12)
ax.axis("off")

ax = axes[1]
p10, p98 = np.percentile(data[data > 0], [15, 95])
# ax.imshow(data[:, :, SLICE_TO_INSPECT].T, vmin=p10, vmax=p98)
# bnw
ax.imshow(data[:, :, SLICE_TO_INSPECT].T, cmap="gray", origin="lower",
          vmin=p10, vmax=p98)
ax.set_title(f"enhanced contrast", fontsize=12)
ax.axis("off")

plt.tight_layout()
plt.show()


# %%
