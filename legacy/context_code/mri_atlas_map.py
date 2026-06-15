#!/usr/bin/env python3
"""
Map the Allen mouse brain atlas (with BrainGlobe) on a mouse MR
--> The idea is that every voxel of the MRI is mapped to a region of the atlas

Input: nifti images

Output: labels.nii.gz (native grid), region_lookup.csv, a QC PNG (proper
   coronal planes)

Logic:
1. Pull a BrainGlobe atlas with an MRI reference + Allen annotations
    --> Gubra T2 Allen atlas (``perens_stereotaxic_mri_mouse_25um``)
2. Build NIfTI images (affine from the atlas orientation) for template
3. Register the atlas TEMPLATE (moving) to the SUBJECT (fixed) using ANTs
4. Reorientation to RAS is done internally
5. ANTs registers in physical/world space so this never changes where voxels actually are
    --> This i think needs to be changed because it doesn't allow moving images
6. Add the markings directly to the nii.gz images so they can be viewed in Fiji.

TO DO:
    [ ] Fix the orientation
    [ ] Make the labels directly open with the image in Fiji
    [ ] Add a validation mechanism for the map
        --> model?
        --> need more images?
        --> use very visible points to do so?
            --> Corpus Callosum, limit of the iso cortex etc.
    [ ] Try an alternative with BrainReg??
    [ ]


"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# direction codes BrainGlobe uses -- each letter maps to (axis index, sign) for building the affine
BRAINGLOBE_AXES = {
    "r": (0, +1),  # Right
    "l": (0, -1),  # Left
    "a": (1, +1),  # Anterior
    "p": (1, -1),  # Posterior
    "s": (2, +1),  # Superior
    "i": (2, -1),  # Inferior
}


def orientation_to_affine(code: str, res_mm) -> np.ndarray:
    """
    Turn a 3-letter BrainGlobe orientation string (e.g. 'rsa') into a NIfTI affine.
    --> each letter tells us which RAS axis that volume axis maps to, and the sign
    """
    code = code.lower().strip()

    if len(code) != 3 or any(c not in BRAINGLOBE_AXES for c in code):
        raise ValueError(f"Unexpected BrainGlobe orientation code: {code!r}")

    affine = np.zeros((4, 4), dtype=float)
    affine[3, 3] = 1.0

    for axis_idx, letter in enumerate(code):
        ras_axis, sign = BRAINGLOBE_AXES[letter]
        affine[ras_axis, axis_idx] = -sign * float(res_mm[axis_idx])

    return affine


def load_atlas(atlas_name: str, cache_dir: Path):
    """
    Download (first run) and load a BrainGlobe atlas, then cache template + annotation as NIfTIs.
    --> returns paths to the two NIfTI files and the atlas object itself
    """
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pip install brainglobe-atlasapi") from e
    import nibabel as nib

    print(f"Loading atlas '{atlas_name}' — this will download it on the first run ...")
    atlas = BrainGlobeAtlas(atlas_name)
    res_mm = [r / 1000.0 for r in atlas.resolution]
    code = getattr(atlas, "orientation", None) or atlas.metadata["orientation"]
    affine = orientation_to_affine(code, res_mm)
    print(f"  orientation: {code}   resolution (mm): {tuple(res_mm)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmpl_path = cache_dir / f"{atlas_name}_template.nii.gz"
    annot_path = cache_dir / f"{atlas_name}_annotation.nii.gz"

    if not tmpl_path.exists():
        ref = np.asarray(atlas.reference, dtype=np.float32)
        nib.Nifti1Image(ref, affine).to_filename(str(tmpl_path))
        print(f"  wrote {tmpl_path.name}  shape={ref.shape}")

    if not annot_path.exists():
        ann = np.asarray(atlas.annotation, dtype=np.int32)
        nib.Nifti1Image(ann, affine).to_filename(str(annot_path))
        print(f"  wrote {annot_path.name}  shape={ann.shape}")

    return tmpl_path, annot_path, atlas


def build_structure_lookup(atlas) -> dict:
    """
    Build a flat id → {acronym, name} dict from the atlas structure tree.
    --> id 0 is always background (outside brain)
    """
    lut = {0: {"acronym": "bg", "name": "background / outside brain"}}

    for s in atlas.structures.values():
        lut[int(s["id"])] = {
            "acronym": s.get("acronym", str(s["id"])),
            "name": s.get("name", ""),
        }

    return lut


# --------------------------------------------------------------------------- #
# Subject preprocessing  (native grid is preserved)
# --------------------------------------------------------------------------- #
def preprocess_subject(subject_path: Path, n4: bool, ants):
    """
    Load the subject MRI and optionally run N4 bias field correction.
    --> native grid is kept throughout — no resampling here
    """
    img = ants.image_read(str(subject_path))

    if img.dimension != 3:
        print(f"Warning: image is {img.dimension}D — expected a 3D volume.",
              file=sys.stderr)

    if n4:
        print("Running N4 bias field correction ...")
        img = ants.n4_bias_field_correction(img)

    return img


def _maybe_reorient(img, reorient, ants):
    """Reorient to RAS if asked — used to give ANTs a sane starting frame."""
    return ants.reorient_image2(img, orientation="RAS") if reorient else img


def register_and_propagate(subject_native, tmpl_path, annot_path,
                           transform, mask_path, reorient, ants):
    """
    Register the atlas template to the subject, then warp the annotation labels
    back onto the native subject grid.
    --> RAS copies are used only to orient the optimizer; output stays on the native grid
    """
    template   = ants.image_read(str(tmpl_path))
    annotation = ants.image_read(str(annot_path))

    subject_reg    = _maybe_reorient(subject_native, reorient, ants)
    template_reg   = _maybe_reorient(template,       reorient, ants)
    annotation_reg = _maybe_reorient(annotation,     reorient, ants)

    fixed_mask = None
    if mask_path is not None:
        fixed_mask = _maybe_reorient(ants.image_read(str(mask_path)), reorient, ants)

    print(f"Registering ({transform}): atlas template → subject — this takes a few minutes ...")
    reg = ants.registration(
        fixed=subject_reg, moving=template_reg,
        type_of_transform=transform, mask=fixed_mask,
    )
    fwd = reg["fwdtransforms"]

    print("Propagating annotation labels onto the native subject grid ...")
    labels = ants.apply_transforms(
        fixed=subject_native, moving=annotation_reg,
        transformlist=fwd, interpolator="genericLabel",
    )
    warped_template = ants.apply_transforms(
        fixed=subject_native, moving=template_reg,
        transformlist=fwd, interpolator="linear",
    )

    return labels, warped_template, reg


def write_region_lookup(labels_img, lut, voxel_vol_mm3, out_csv: Path) -> None:
    """
    Count voxels per atlas region and write a CSV summary.
    --> sorted by voxel count, largest region first
    """
    import csv

    arr = labels_img.numpy().astype(np.int64)
    ids, counts = np.unique(arr, return_counts=True)

    rows = []
    for lab, n in zip(ids.tolist(), counts.tolist()):
        info = lut.get(lab, {"acronym": str(lab), "name": "UNKNOWN id"})
        rows.append((lab, info["acronym"], info["name"], n, round(n * voxel_vol_mm3, 6)))
    rows.sort(key=lambda r: -r[3])

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label_id", "acronym", "name", "n_voxels", "volume_mm3"])
        w.writerows(rows)

    print(f"Wrote {out_csv.name} — {len(rows)} regions found")


def qc_overlay(subject_img, labels_img, out_png: Path) -> None:
    """
    Save a QC PNG showing atlas region boundaries overlaid on the subject MRI.
    --> picks 6 evenly spaced slices along the thinnest axis (usually the acquisition plane)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import ndimage
    except ImportError:
        print("matplotlib or scipy not found — skipping QC image.")
        return

    vol = subject_img.numpy()
    lab = labels_img.numpy()

    slice_axis = int(np.argmin(vol.shape))
    n    = vol.shape[slice_axis]
    idxs = np.unique(
        np.linspace(max(1, int(n * 0.12)), min(n - 2, int(n * 0.88)), 6).astype(int)
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, k in zip(axes.ravel(), idxs):
        sl    = np.take(vol, k, axis=slice_axis).astype(float)
        la    = np.take(lab, k, axis=slice_axis)
        edges = la != ndimage.grey_dilation(la, size=(3, 3))
        vmax  = np.percentile(sl, 99) if sl.max() > 0 else 1

        ax.imshow(sl.T, cmap="gray", origin="lower", vmax=vmax)
        ax.imshow(np.ma.masked_where(~edges, edges).T, cmap="autumn", origin="lower", alpha=0.9)
        ax.set_title(f"slice {k} (axis {slice_axis})")
        ax.axis("off")

    for ax in axes.ravel()[len(idxs):]:
        ax.axis("off")

    fig.suptitle("Atlas region boundaries over subject MRI (QC)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    print(f"Saved QC overlay → {out_png.name}")


def query_label(labels_path, atlas_name, cache_dir, voxel=None, mm=None) -> None:
    """Look up which atlas region a given voxel (or mm coordinate) belongs to."""
    import nibabel as nib

    lab_img = nib.load(str(labels_path))
    arr = np.asarray(lab_img.dataobj)

    if mm is not None:
        ijk = np.linalg.inv(lab_img.affine) @ np.array([*mm, 1.0])
        i, j, k = np.round(ijk[:3]).astype(int)
    else:
        i, j, k = voxel

    try:
        lab = int(arr[i, j, k])
    except IndexError:
        raise SystemExit(f"voxel {(i, j, k)} outside volume {arr.shape}")

    from brainglobe_atlasapi import BrainGlobeAtlas
    lut  = build_structure_lookup(BrainGlobeAtlas(atlas_name))
    info = lut.get(lab, {"acronym": str(lab), "name": "UNKNOWN id"})

    print(f"voxel {(i, j, k)} -> id {lab}: {info['acronym']} ({info['name']})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Map a BrainGlobe (Allen) atlas onto a mouse MRI volume.")
    p.add_argument("subject",        type=Path)
    p.add_argument("-o", "--outdir", type=Path, default=None)
    p.add_argument("-a", "--atlas",  default="perens_stereotaxic_mri_mouse_25um")
    p.add_argument("-t", "--transform", default="SyNRA",
                   help="ANTs type_of_transform (use 'Affine' for thick-slice / low-coverage stacks).")
    p.add_argument("--mask",        type=Path, default=None,
                   help="subject brain mask (focus registration; exclude lesion)")
    p.add_argument("--n4",          action="store_true")
    p.add_argument("--no-reorient", action="store_true",
                   help="skip the internal RAS reorientation step.")
    p.add_argument("--query",       default=None, help="voxel 'i,j,k' (with --labels)")
    p.add_argument("--query-mm",    default=None, help="coord 'x,y,z' mm (--labels)")
    p.add_argument("--labels",      type=Path,    default=None)
    args = p.parse_args(argv)

    cache_dir = Path.home() / ".cache" / "mri_atlas_map"

    # --- query mode: just look up a voxel in an existing labels file ---
    if args.query or args.query_mm:
        if args.labels is None:
            raise SystemExit("--query/--query-mm needs --labels <file>")
        vox = tuple(int(v) for v in args.query.split(","))    if args.query    else None
        mm  = tuple(float(v) for v in args.query_mm.split(",")) if args.query_mm else None
        query_label(args.labels, args.atlas, cache_dir, voxel=vox, mm=mm)
        return 0

    # --- main mode: register atlas to subject ---
    if not args.subject.exists():
        raise SystemExit(f"subject not found: {args.subject}")
    try:
        import ants
    except ImportError as e:
        raise SystemExit("pip install antspyx") from e

    outdir = args.outdir or args.subject.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.subject.name
    for suf in (".nii.gz", ".nii"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    reorient = not args.no_reorient

    # load everything
    tmpl_path, annot_path, atlas = load_atlas(args.atlas, cache_dir)
    lut         = build_structure_lookup(atlas)
    subject_img = preprocess_subject(args.subject, args.n4, ants)

    # register + propagate labels
    labels, warped_tmpl, reg = register_and_propagate(
        subject_img, tmpl_path, annot_path, args.transform,
        args.mask, reorient, ants)

    sp         = subject_img.spacing
    voxel_vol  = float(np.prod(sp[:3]))

    # output paths
    labels_path = outdir / f"{stem}_atlas_labels.nii.gz"
    tmpl_out    = outdir / f"{stem}_atlas_template_in_subject.nii.gz"
    csv_path    = outdir / f"{stem}_region_lookup.csv"
    png_path    = outdir / f"{stem}_qc_overlay.png"
    json_path   = outdir / f"{stem}_atlas_map.json"

    # save labels + warped template
    labels.to_filename(str(labels_path))
    warped_tmpl.to_filename(str(tmpl_out))
    print(f"Labels saved → {labels_path.name}  (per-voxel region ids, native grid)")
    match = subject_img.shape == labels.shape
    print(f"  shape check: subject {tuple(subject_img.shape)} vs labels {tuple(labels.shape)}"
          f"  →  {'OK — overlays the original' if match else 'MISMATCH!'}")

    # region table + QC image
    write_region_lookup(labels, lut, voxel_vol, csv_path)
    qc_overlay(subject_img, labels, png_path)

    # metadata
    json_path.write_text(json.dumps({
        "subject":                  str(args.subject),
        "atlas":                    args.atlas,
        "transform":                args.transform,
        "n4":                       args.n4,
        "reoriented_internally":    reorient,
        "mask":                     str(args.mask) if args.mask else None,
        "subject_voxel_mm":         list(sp[:3]),
        "subject_shape":            list(subject_img.shape),
        "labels_shape":             list(labels.shape),
    }, indent=2), encoding="utf-8")

    print(f"\nDone! Open {png_path.name} to check the overlay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())