#!/usr/bin/env python3
"""
Map the Allen mouse brain atlas onto a mouse MRI using brainreg.

Input:  NIfTI subject image (.nii / .nii.gz)

Output: *_atlas_labels.nii.gz  (native grid)
        *_region_lookup.csv
        *_qc_overlay.png
        *_atlas_map.json

brainreg (NiftyReg backend) differences vs. the ANTs version:
  - Specifically tuned for mouse brain registration
  - Accepts voxel_sizes explicitly, so anisotropic thick-slice data is handled cleanly
  - Non-linear (freeform) step should be disabled for ≲20 slices (--affine-only)

Usage examples:
  python mri_atlas_map_brainreg.py brain.nii.gz
  python mri_atlas_map_brainreg.py brain.nii.gz --affine-only
  python mri_atlas_map_brainreg.py brain.nii.gz --query 45,60,10 --labels out/brain_atlas_labels.nii.gz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ helpers --

def load_subject_nifti(subject_path: Path):
    """Return (nibabel image, voxel_sizes_µm tuple, orientation string)."""
    import nibabel as nib
    img = nib.load(str(subject_path))
    zooms = img.header.get_zooms()[:3]                    # mm
    voxel_um = tuple(float(z) * 1000.0 for z in zooms)   # µm — brainreg expects µm
    codes = nib.aff2axcodes(img.affine)                   # e.g. ('R', 'A', 'S')
    orientation = "".join(c.lower() for c in codes)       # -> "ras"
    return img, voxel_um, orientation


def build_structure_lookup(atlas) -> dict:
    lut = {0: {"acronym": "bg", "name": "background / outside brain"}}
    for s in atlas.structures.values():
        lut[int(s["id"])] = {
            "acronym": s.get("acronym", str(s["id"])),
            "name": s.get("name", ""),
        }
    return lut


# ---------------------------------------------------------- registration ---

def register(subject_path: Path, outdir: Path, atlas_name: str,
             voxel_um: tuple, orientation: str, affine_only: bool) -> Path:
    import subprocess, shutil
    if shutil.which("brainreg") is None:
        raise SystemExit("brainreg CLI not found — pip install brainreg")

    print(f"[brainreg] {subject_path.name}  ->  atlas '{atlas_name}'")
    print(f"[brainreg] voxel_sizes={voxel_um} µm   orientation='{orientation}'")
    if affine_only:
        print("[brainreg] affine-only mode (non-linear step disabled)")

    # brainreg CLI: -v expects voxel sizes in µm for each image axis in order
    cmd = [
        "brainreg",
        str(subject_path),
        str(outdir),
        "-v", str(voxel_um[0]), str(voxel_um[1]), str(voxel_um[2]),
        "--atlas", atlas_name,
        "--orientation", orientation,
    ]
    if affine_only:
        cmd += ["--freeform-n-steps", "0"]

    subprocess.run(cmd, check=True)

    labels_tiff = outdir / "registered_atlas.tiff"
    if not labels_tiff.exists():
        raise SystemExit(
            f"Expected brainreg output not found: {labels_tiff}\n"
            f"Check the brainreg log inside {outdir}."
        )
    return labels_tiff


def tiff_to_nifti_labels(labels_tiff: Path, subject_affine,
                          subject_shape: tuple, out_path: Path) -> np.ndarray:
    """
    Load brainreg's registered_atlas.tiff and write a NIfTI on the native grid.
    brainreg resamples its output to match the input resolution, so shapes
    should agree; a warning is printed if they don't.
    """
    try:
        import tifffile
        arr = tifffile.imread(str(labels_tiff)).astype(np.int32)
    except ImportError:
        try:
            import imageio.v3 as iio
            arr = np.asarray(iio.imread(str(labels_tiff))).astype(np.int32)
        except ImportError:
            raise SystemExit("pip install tifffile  (or imageio)")
    import nibabel as nib

    if arr.shape != tuple(subject_shape):
        print(
            f"[warn] label shape {arr.shape} != subject shape {tuple(subject_shape)}; "
            "brainreg may have resampled — affine may be slightly off."
        )
    nib.Nifti1Image(arr, subject_affine).to_filename(str(out_path))
    return arr


# ---------------------------------------------------------- outputs ---------

def write_region_lookup(arr: np.ndarray, lut: dict,
                        voxel_vol_mm3: float, out_csv: Path) -> None:
    import csv
    ids, counts = np.unique(arr.astype(np.int64), return_counts=True)
    rows = []
    for lab, n in zip(ids.tolist(), counts.tolist()):
        info = lut.get(lab, {"acronym": str(lab), "name": "UNKNOWN id"})
        rows.append((lab, info["acronym"], info["name"], n,
                     round(n * voxel_vol_mm3, 6)))
    rows.sort(key=lambda r: -r[3])
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label_id", "acronym", "name", "n_voxels", "volume_mm3"])
        w.writerows(rows)
    print(f"[output] {out_csv.name}: {len(rows)} regions present")


def qc_overlay(subject_nib, labels_arr: np.ndarray, out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import ndimage
    except ImportError:
        print("[qc] matplotlib/scipy missing — skipping QC PNG.")
        return

    vol = np.asarray(subject_nib.dataobj).astype(float)
    lab = labels_arr
    slice_axis = int(np.argmin(vol.shape))
    n = vol.shape[slice_axis]
    idxs = np.unique(
        np.linspace(max(1, int(n * 0.12)), min(n - 2, int(n * 0.88)), 6).astype(int)
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, k in zip(axes.ravel(), idxs):
        sl = np.take(vol, k, axis=slice_axis)
        la = np.take(lab, k, axis=slice_axis)
        edges = la != ndimage.grey_dilation(la, size=(3, 3))
        vmax = np.percentile(sl, 99) if sl.max() > 0 else 1
        ax.imshow(sl.T, cmap="gray", origin="lower", vmax=vmax)
        ax.imshow(np.ma.masked_where(~edges, edges).T, cmap="autumn",
                  origin="lower", alpha=0.9)
        ax.set_title(f"slice {k} (axis {slice_axis})")
        ax.axis("off")
    for ax in axes.ravel()[len(idxs):]:
        ax.axis("off")
    fig.suptitle("Atlas region boundaries over subject MRI (QC)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"[output] {out_png.name}: QC overlay")


def query_label(labels_path: Path, atlas_name: str,
                voxel=None, mm=None) -> None:
    import nibabel as nib
    img = nib.load(str(labels_path))
    arr = np.asarray(img.dataobj)
    if mm is not None:
        ijk = np.linalg.inv(img.affine) @ np.array([*mm, 1.0])
        i, j, k = np.round(ijk[:3]).astype(int)
    else:
        i, j, k = voxel
    try:
        lab = int(arr[i, j, k])
    except IndexError:
        raise SystemExit(f"voxel {(i, j, k)} outside volume {arr.shape}")
    from brainglobe_atlasapi import BrainGlobeAtlas
    lut = build_structure_lookup(BrainGlobeAtlas(atlas_name))
    info = lut.get(lab, {"acronym": str(lab), "name": "UNKNOWN id"})
    print(f"voxel {(i, j, k)} -> id {lab}: {info['acronym']} ({info['name']})")


# ------------------------------------------------------------------- CLI ---

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Map a BrainGlobe atlas onto a mouse MRI using brainreg."
    )
    p.add_argument("subject", type=Path,
                   help="Input NIfTI (.nii or .nii.gz)")
    p.add_argument("-o", "--outdir", type=Path, default=None,
                   help="Output directory (default: <subject_dir>/<stem>_brainreg/)")
    p.add_argument("-a", "--atlas", default="perens_stereotaxic_mri_mouse_25um",
                   help="BrainGlobe atlas name")
    p.add_argument("--affine-only", action="store_true",
                   help="Skip the non-linear freeform step. "
                        "Recommended for thick-slice / few-slice MRI (≲20 slices).")
    p.add_argument("--query", default=None,
                   help="Look up voxel 'i,j,k' in an existing labels file (needs --labels)")
    p.add_argument("--query-mm", default=None,
                   help="Look up coordinate 'x,y,z' mm in an existing labels file (needs --labels)")
    p.add_argument("--labels", type=Path, default=None,
                   help="Existing labels NIfTI for --query / --query-mm")
    args = p.parse_args(argv)

    # ---- query mode (no registration) ------------------------------------
    if args.query or args.query_mm:
        if args.labels is None:
            raise SystemExit("--query / --query-mm needs --labels <file>")
        vox = tuple(int(v) for v in args.query.split(",")) if args.query else None
        mm  = tuple(float(v) for v in args.query_mm.split(",")) if args.query_mm else None
        query_label(args.labels, args.atlas, voxel=vox, mm=mm)
        return 0

    # ---- registration mode -----------------------------------------------
    if not args.subject.exists():
        raise SystemExit(f"subject not found: {args.subject}")

    subject_nib, voxel_um, orientation = load_subject_nifti(args.subject)
    print(f"[subject] shape={subject_nib.shape}  "
          f"voxel_µm={tuple(round(v,1) for v in voxel_um)}  "
          f"orient='{orientation}'")

    stem = args.subject.name
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break

    outdir = args.outdir or (args.subject.parent / f"{stem}_brainreg")
    outdir.mkdir(parents=True, exist_ok=True)

    labels_tiff = register(
        args.subject, outdir, args.atlas,
        voxel_um, orientation, args.affine_only,
    )

    labels_path = outdir / f"{stem}_atlas_labels.nii.gz"
    labels_arr = tiff_to_nifti_labels(
        labels_tiff, subject_nib.affine, subject_nib.shape, labels_path
    )
    print(f"[output] {labels_path.name}: per-voxel region ids (native grid)")
    print(f"[check ] subject {tuple(subject_nib.shape)} == "
          f"labels {labels_arr.shape}  -> "
          f"{'OK' if tuple(subject_nib.shape) == labels_arr.shape else 'MISMATCH'}")

    from brainglobe_atlasapi import BrainGlobeAtlas
    lut = build_structure_lookup(BrainGlobeAtlas(args.atlas))

    zooms = subject_nib.header.get_zooms()[:3]
    voxel_vol_mm3 = float(np.prod(zooms))

    csv_path  = outdir / f"{stem}_region_lookup.csv"
    png_path  = outdir / f"{stem}_qc_overlay.png"
    json_path = outdir / f"{stem}_atlas_map.json"

    write_region_lookup(labels_arr, lut, voxel_vol_mm3, csv_path)
    qc_overlay(subject_nib, labels_arr, png_path)

    json_path.write_text(json.dumps({
        "subject":        str(args.subject),
        "atlas":          args.atlas,
        "backend":        "brainreg (NiftyReg)",
        "affine_only":    args.affine_only,
        "orientation":    orientation,
        "subject_voxel_um":  list(voxel_um),
        "subject_voxel_mm":  [v / 1000 for v in voxel_um],
        "subject_shape":  list(subject_nib.shape),
        "labels_shape":   list(labels_arr.shape),
    }, indent=2), encoding="utf-8")
    print(f"[done]  open {png_path.name} and check the overlay first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())