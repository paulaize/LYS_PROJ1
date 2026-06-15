#!/usr/bin/env python3
"""
Brainreg-based alternative to an ANTs/BrainGlobe mouse MRI atlas-mapping script.

Goal
----
Input:  a 3D NIfTI MRI volume.
Output: Brainreg output folder plus labels on the original NIfTI grid,
        a region_lookup.csv, a QC overlay PNG, and a run JSON.

Important difference from the ANTs version
------------------------------------------
brainreg is normally an image-stack workflow. This wrapper converts the NIfTI
volume to an ordered TIFF stack, runs the `brainreg` command-line tool, then
converts the warped atlas labels back into a NIfTI image using the original
NIfTI affine.

For thick-slice / low-slice-count MRI, use conservative settings and treat the
result as approximate. Always inspect the QC overlay.

python code/test3.py /Users/paul-andreaslaize/Documents/Lys_BBB/raw_data/Braindistrib_08/20251016_085953_BrainDistrib_08_24h_1_1_nifti/02_T2_haute_resolution_Turbo_reco1.nii.gz \
  -o brainreg_out \
  --atlas perens_stereotaxic_mri_mouse_25um \
  --slice-axis 2 \
  --voxel-um 500 100 100 \
  --orientation psl \
  --conservative
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


# nibabel gives the anatomical direction reached when the voxel index increases.
# BrainGlobe wants the anatomical location of voxel [0, 0, 0], i.e. the opposite.
_POSITIVE_END_TO_ORIGIN = {
    "R": "l",
    "L": "r",
    "A": "p",
    "P": "a",
    "S": "i",
    "I": "s",
}


def _import_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: pip install nibabel") from exc
    return nib


def _import_tifffile():
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: pip install tifffile") from exc
    return tifffile


def _load_nifti(path: Path):
    nib = _import_nibabel()
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise SystemExit(f"Expected a 3D NIfTI after squeeze; got shape {arr.shape}")
    return img, arr.astype(np.float32, copy=False)


def _nifti_axis_origin_code(img) -> str:
    """Return BrainGlobe orientation code for the NIfTI array axes."""
    nib = _import_nibabel()
    axcodes = nib.aff2axcodes(img.affine)
    if any(c is None for c in axcodes):
        raise SystemExit("Could not infer orientation from NIfTI affine; pass --orientation explicitly.")
    try:
        return "".join(_POSITIVE_END_TO_ORIGIN[c] for c in axcodes)
    except KeyError as exc:
        raise SystemExit(f"Unexpected NIfTI axis code {axcodes}; pass --orientation explicitly.") from exc


def _parse_slice_axis(value: str, shape: tuple[int, int, int]) -> int:
    if value == "auto":
        return int(np.argmin(shape))
    axis = int(value)
    if axis not in (0, 1, 2):
        raise SystemExit("--slice-axis must be auto, 0, 1 or 2")
    return axis


def _make_axis_permutation(slice_axis: int) -> list[int]:
    """Move the acquisition/slice axis to BrainGlobe stack axis 0."""
    return [slice_axis] + [a for a in (0, 1, 2) if a != slice_axis]


def _robust_uint16(vol: np.ndarray, invert: bool = False) -> np.ndarray:
    finite = np.isfinite(vol)
    if not finite.any():
        raise SystemExit("Input image contains no finite values.")
    vals = vol[finite]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if hi <= lo:
        return np.zeros(vol.shape, dtype=np.uint16)
    scaled = np.clip((vol - lo) / (hi - lo), 0, 1)
    if invert:
        scaled = 1.0 - scaled
    return np.round(scaled * 65535).astype(np.uint16)


def _export_tiff_stack(vol_stack: np.ndarray, stack_dir: Path, invert: bool) -> None:
    tifffile = _import_tifffile()
    stack_dir.mkdir(parents=True, exist_ok=True)
    for old in stack_dir.glob("*.tif*"):
        old.unlink()
    vol_u16 = _robust_uint16(vol_stack, invert=invert)
    for i in range(vol_u16.shape[0]):
        tifffile.imwrite(str(stack_dir / f"slice_{i:04d}.tiff"), vol_u16[i])


def _brainreg_command(
    stack_dir: Path,
    brainreg_out: Path,
    voxel_um: tuple[float, float, float],
    orientation: str,
    atlas: str,
    args,
) -> list[str]:
    cmd = [
        "brainreg",
        str(stack_dir),
        str(brainreg_out),
        "-v",
        *(f"{v:g}" for v in voxel_um),
        "--orientation",
        orientation,
        "--atlas",
        atlas,
        "--brain_geometry",
        args.brain_geometry,
        "--pre-processing",
        args.pre_processing,
        "--save-original-orientation",
    ]
    if args.n_free_cpus is not None:
        cmd.extend(["--n-free-cpus", str(args.n_free_cpus)])
    if args.debug:
        cmd.append("--debug")
    if args.conservative:
        # Reduce non-linear overfitting risk for sparse/thick-slice volumes.
        cmd.extend([
            "--freeform-use-n-steps", "1",
            "--bending-energy-weight", "0.99",
            "--grid-spacing", "1.0",
            "--smoothing-sigma-floating", "0.5",
            "--smoothing-sigma-reference", "0.5",
        ])
    for extra in args.brainreg_arg or []:
        cmd.extend(shlex.split(extra))
    return cmd


def _find_registered_atlas(brainreg_out: Path) -> Path:
    candidates = [
        brainreg_out / "registered_atlas_original_orientation.tiff",
        brainreg_out / "registered_atlas_original_orientation.tif",
        brainreg_out / "registered_atlas.tiff",
        brainreg_out / "registered_atlas.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    names = ", ".join(p.name for p in candidates)
    raise SystemExit(f"Could not find brainreg warped atlas labels. Looked for: {names}")


def _read_tiff_volume(path: Path) -> np.ndarray:
    tifffile = _import_tifffile()
    arr = tifffile.imread(str(path))
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise SystemExit(f"Expected a 3D label TIFF from brainreg; got shape {arr.shape} at {path}")
    return arr.astype(np.int32, copy=False)


def _nearest_resize(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(arr.shape) == tuple(target_shape):
        return arr
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "brainreg output shape differs from the input stack and scipy is needed "
            "for nearest-neighbour resizing. Install with: pip install scipy"
        ) from exc
    factors = [t / s for t, s in zip(target_shape, arr.shape)]
    return zoom(arr, zoom=factors, order=0).astype(np.int32, copy=False)


def _labels_stack_to_native(
    labels_stack: np.ndarray,
    stack_shape: tuple[int, int, int],
    native_shape: tuple[int, int, int],
    perm: list[int],
) -> np.ndarray:
    labels_stack = _nearest_resize(labels_stack, stack_shape)
    inv_perm = np.argsort(perm)
    native = np.transpose(labels_stack, inv_perm)
    native = _nearest_resize(native, native_shape)
    return native.astype(np.int32, copy=False)


def _save_nifti_like(labels_native: np.ndarray, subject_img, out_path: Path) -> None:
    nib = _import_nibabel()
    header = subject_img.header.copy()
    header.set_data_dtype(np.int32)
    nib.Nifti1Image(labels_native.astype(np.int32), subject_img.affine, header=header).to_filename(str(out_path))


def _build_structure_lookup(atlas_name: str) -> dict[int, dict[str, str]]:
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency: pip install brainglobe-atlasapi") from exc
    atlas = BrainGlobeAtlas(atlas_name)
    lut = {0: {"acronym": "bg", "name": "background / outside brain"}}
    for s in atlas.structures.values():
        lut[int(s["id"])] = {
            "acronym": s.get("acronym", str(s["id"])),
            "name": s.get("name", ""),
        }
    return lut


def _write_region_lookup(labels_native: np.ndarray, lut: dict, voxel_vol_mm3: float, out_csv: Path) -> None:
    ids, counts = np.unique(labels_native.astype(np.int64), return_counts=True)
    rows = []
    for lab, n in zip(ids.tolist(), counts.tolist()):
        info = lut.get(int(lab), {"acronym": str(lab), "name": "UNKNOWN id"})
        rows.append((int(lab), info["acronym"], info["name"], int(n), round(float(n) * voxel_vol_mm3, 6)))
    rows.sort(key=lambda row: -row[3])
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label_id", "acronym", "name", "n_voxels", "volume_mm3"])
        writer.writerows(rows)


def _slice_edges(label_slice: np.ndarray) -> np.ndarray:
    edges = np.zeros(label_slice.shape, dtype=bool)
    edges[:-1, :] |= label_slice[:-1, :] != label_slice[1:, :]
    edges[:, :-1] |= label_slice[:, :-1] != label_slice[:, 1:]
    return edges


def _qc_overlay(subject_vol: np.ndarray, labels_native: np.ndarray, out_png: Path, slice_axis: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[qc] matplotlib missing; skipping QC PNG.")
        return

    n = subject_vol.shape[slice_axis]
    if n <= 1:
        print("[qc] too few slices; skipping QC PNG.")
        return
    idxs = np.unique(np.linspace(max(0, int(n * 0.12)), min(n - 1, int(n * 0.88)), min(6, n)).astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, k in zip(axes.ravel(), idxs):
        img_sl = np.take(subject_vol, k, axis=slice_axis).astype(float)
        lab_sl = np.take(labels_native, k, axis=slice_axis)
        edge = _slice_edges(lab_sl)
        vmax = np.nanpercentile(img_sl, 99) if np.nanmax(img_sl) > 0 else 1
        ax.imshow(img_sl.T, cmap="gray", origin="lower", vmax=vmax)
        ax.imshow(np.ma.masked_where(~edge, edge).T, cmap="autumn", origin="lower", alpha=0.9)
        ax.set_title(f"slice {k} / axis {slice_axis}")
        ax.axis("off")
    for ax in axes.ravel()[len(idxs):]:
        ax.axis("off")
    fig.suptitle("brainreg registered atlas boundaries over subject MRI")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _query_label(labels_path: Path, atlas_name: str, voxel: tuple[int, int, int] | None, mm: tuple[float, float, float] | None) -> None:
    nib = _import_nibabel()
    img = nib.load(str(labels_path))
    arr = np.asarray(img.dataobj)
    if mm is not None:
        ijk = np.linalg.inv(img.affine) @ np.array([*mm, 1.0])
        i, j, k = np.round(ijk[:3]).astype(int)
    elif voxel is not None:
        i, j, k = voxel
    else:
        raise SystemExit("Need --query or --query-mm")
    if not (0 <= i < arr.shape[0] and 0 <= j < arr.shape[1] and 0 <= k < arr.shape[2]):
        raise SystemExit(f"voxel {(i, j, k)} outside label volume {arr.shape}")
    lab = int(arr[i, j, k])
    lut = _build_structure_lookup(atlas_name)
    info = lut.get(lab, {"acronym": str(lab), "name": "UNKNOWN id"})
    print(f"voxel {(i, j, k)} -> id {lab}: {info['acronym']} ({info['name']})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Map a BrainGlobe atlas onto a mouse MRI with brainreg."
    )
    parser.add_argument("subject", type=Path, nargs="?", help="3D NIfTI MRI")
    parser.add_argument("-o", "--outdir", type=Path, default=None)
    parser.add_argument("-a", "--atlas", default="perens_stereotaxic_mri_mouse_25um")
    parser.add_argument("--slice-axis", default="auto", help="NIfTI array axis corresponding to slice/depth: auto, 0, 1, or 2")
    parser.add_argument("--orientation", default="auto", help="BrainGlobe orientation code for the exported stack, e.g. psl. Default: infer from NIfTI affine.")
    parser.add_argument("--voxel-um", nargs=3, type=float, default=None, metavar=("AX0", "AX1", "AX2"), help="Voxel sizes in microns in exported stack axis order. Default: infer from NIfTI header.")
    parser.add_argument("--invert", action="store_true", help="Invert intensity before exporting TIFF stack.")
    parser.add_argument("--pre-processing", default="default", choices=["default", "skip"])
    parser.add_argument("--brain_geometry", default="full", choices=["full", "hemisphere_l", "hemisphere_r"])
    parser.add_argument("--conservative", action="store_true", help="Use less flexible freeform settings for sparse/thick-slice data.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n-free-cpus", type=int, default=None)
    parser.add_argument("--brainreg-arg", action="append", help="Extra raw argument(s) passed to brainreg, e.g. --brainreg-arg '--grid-spacing 1.5'. Can be repeated.")
    parser.add_argument("--keep-stack", action="store_true", help="Keep exported TIFF stack after the run.")
    parser.add_argument("--skip-run", action="store_true", help="Do not run brainreg; reuse existing brainreg output in --outdir/brainreg.")
    parser.add_argument("--query", default=None, help="Query voxel 'i,j,k' in --labels")
    parser.add_argument("--query-mm", default=None, help="Query world coordinate 'x,y,z' in mm in --labels")
    parser.add_argument("--labels", type=Path, default=None, help="Label NIfTI for --query/--query-mm")
    args = parser.parse_args(argv)

    if args.query or args.query_mm:
        if args.labels is None:
            raise SystemExit("--query/--query-mm needs --labels <file>")
        voxel = tuple(int(v) for v in args.query.split(",")) if args.query else None
        mm = tuple(float(v) for v in args.query_mm.split(",")) if args.query_mm else None
        _query_label(args.labels, args.atlas, voxel, mm)
        return 0

    if args.subject is None:
        raise SystemExit("subject NIfTI is required unless using --query")
    if not args.subject.exists():
        raise SystemExit(f"subject not found: {args.subject}")
    if not args.skip_run and shutil.which("brainreg") is None:
        raise SystemExit("Could not find the `brainreg` command. Install it first, e.g. `conda install -c conda-forge brainreg`.")

    subject_img, subject_vol = _load_nifti(args.subject)
    native_shape = tuple(int(x) for x in subject_vol.shape)
    zooms_mm = tuple(float(z) for z in subject_img.header.get_zooms()[:3])
    slice_axis = _parse_slice_axis(args.slice_axis, native_shape)
    perm = _make_axis_permutation(slice_axis)

    stack_vol = np.transpose(subject_vol, perm)
    stack_shape = tuple(int(x) for x in stack_vol.shape)

    original_orientation = _nifti_axis_origin_code(subject_img)
    inferred_orientation = "".join(original_orientation[i] for i in perm)
    orientation = inferred_orientation if args.orientation == "auto" else args.orientation.lower().strip()
    if len(orientation) != 3:
        raise SystemExit("--orientation must be a 3-letter BrainGlobe orientation code, e.g. psl")

    inferred_voxel_um = tuple(float(zooms_mm[i]) * 1000.0 for i in perm)
    voxel_um = tuple(args.voxel_um) if args.voxel_um is not None else inferred_voxel_um

    outdir = args.outdir or (args.subject.parent / f"{args.subject.name.split('.')[0]}_brainreg_map")
    outdir.mkdir(parents=True, exist_ok=True)
    stack_dir = outdir / "brainreg_input_stack"
    brainreg_out = outdir / "brainreg"
    brainreg_out.mkdir(parents=True, exist_ok=True)

    thick_ratio = max(voxel_um) / max(min(voxel_um), 1e-9)
    if min(stack_shape) <= 20 or thick_ratio >= 4:
        print(
            "[warning] Sparse/anisotropic volume detected: "
            f"stack_shape={stack_shape}, voxel_um={voxel_um}. "
            "Use --conservative and inspect the QC carefully.",
            file=sys.stderr,
        )

    if not args.skip_run:
        print(f"[input] native shape={native_shape}, stack shape={stack_shape}, slice_axis={slice_axis}, perm={perm}")
        print(f"[input] orientation={orientation} (inferred={inferred_orientation}), voxel_um={voxel_um}")
        print(f"[export] writing TIFF stack: {stack_dir}")
        _export_tiff_stack(stack_vol, stack_dir, invert=args.invert)

        cmd = _brainreg_command(stack_dir, brainreg_out, voxel_um, orientation, args.atlas, args)
        print("[brainreg] " + " ".join(shlex.quote(c) for c in cmd))
        subprocess.run(cmd, check=True)

    registered_atlas = _find_registered_atlas(brainreg_out)
    print(f"[labels] reading {registered_atlas}")
    labels_stack = _read_tiff_volume(registered_atlas)
    labels_native = _labels_stack_to_native(labels_stack, stack_shape, native_shape, perm)

    stem = args.subject.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    labels_path = outdir / f"{stem}_brainreg_atlas_labels_native.nii.gz"
    csv_path = outdir / f"{stem}_brainreg_region_lookup.csv"
    png_path = outdir / f"{stem}_brainreg_qc_overlay.png"
    json_path = outdir / f"{stem}_brainreg_map.json"

    _save_nifti_like(labels_native, subject_img, labels_path)
    lut = _build_structure_lookup(args.atlas)
    voxel_vol_mm3 = float(np.prod(zooms_mm))
    _write_region_lookup(labels_native, lut, voxel_vol_mm3, csv_path)
    _qc_overlay(subject_vol, labels_native, png_path, slice_axis=slice_axis)

    metadata = {
        "subject": str(args.subject),
        "atlas": args.atlas,
        "native_shape": list(native_shape),
        "stack_shape": list(stack_shape),
        "slice_axis": slice_axis,
        "axis_permutation_native_to_stack": perm,
        "original_nifti_orientation_code": original_orientation,
        "brainreg_orientation": orientation,
        "brainreg_voxel_um": list(voxel_um),
        "inferred_brainreg_orientation": inferred_orientation,
        "inferred_voxel_um": list(inferred_voxel_um),
        "conservative": args.conservative,
        "invert": args.invert,
        "brainreg_output": str(brainreg_out),
        "registered_atlas_source": str(registered_atlas),
        "labels_native": str(labels_path),
        "region_lookup": str(csv_path),
        "qc_overlay": str(png_path),
        "warning": "Sparse/thick-slice data limits registration precision, especially along the slice axis.",
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.keep_stack and stack_dir.exists():
        shutil.rmtree(stack_dir)

    print(f"[output] labels: {labels_path}")
    print(f"[output] lookup: {csv_path}")
    print(f"[output] QC:     {png_path}")
    print(f"[done] Check the QC overlay before trusting per-voxel labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
