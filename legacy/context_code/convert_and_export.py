"""
Convert all Bruker ParaVision raw datasets under RAW_ROOT into NIfTI + JSON
sidecars using brkraw's Python API.

TO DO:
    [ ] Make it more compact 
    [ ] Make it require less processing / do some parallelizing...
    [ ] Also very hard coded for the moment add more parameterization (of the args)
        --> So can call all the scripts at the end better with exact args
    [ ]
    [ ]

"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from brkraw.api import BrukerLoader


RAW_ROOT = Path.home() / "Documents/Lys_BBB/raw_data/Braindistrib_08"
OUT_SUFFIX = "_nifti"
SPACE = "subject_ras"

SPACE_FALLBACKS = ["scanner", "raw"]
# include scl_slope/scl_inter into the data
SCALE_INTENSITIES = True


_BAD = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize(name) -> str:
    return _BAD.sub("_", str(name)).strip("_") or "scan"


def _walk(obj):
    """Yield every (key, value) pair in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def guess_protocol(info, scan_id, loader) -> str:
    """Find a human-readable protocol name for this scan, with fallbacks."""
    sid_keys = {scan_id, str(scan_id), int(scan_id)}
    if isinstance(info, dict):
        scans = info.get("scans") or info.get("scan") or {}
        if isinstance(scans, dict):
            for k, v in scans.items():
                if k in sid_keys and isinstance(v, dict):
                    for kk, vv in _walk(v):
                        if isinstance(kk, str) and "protocol" in kk.lower() and vv:
                            return str(vv)
    try:
        params = loader.search_params(
            "ACQ_protocol_name", scan_id=scan_id, file="acqp"
        )
        for _, v in _walk(params):
            if isinstance(v, str) and v.strip():
                return v
    except Exception:
        pass
    return f"scan{scan_id}"


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# Apply scl_slope/scl_inter to the data, return a float32 Nifti1Image with
# slope=1, inter=0. All other header fields (zooms, units, affine, TR, ...)
# are preserved.

# Important: brkraw's loader.convert() returns an *in-memory* Nifti1Image
# whose .dataobj is a plain numpy array of raw integers. nibabel only
# auto-applies slope/inter on access when .dataobj is an ArrayProxy (i.e.
# when the image was *loaded from disk*). So we have to apply the scaling
# ourselves -- using get_unscaled() when available, otherwise treating
# .dataobj as the raw array, then multiplying through explicitly

def bake_scaling(img: nib.Nifti1Image) -> nib.Nifti1Image:
    slope_h = img.header["scl_slope"]
    inter_h = img.header["scl_inter"]
    slope = float(slope_h) if slope_h and np.isfinite(slope_h) else 1.0
    inter = float(inter_h) if inter_h and np.isfinite(inter_h) else 0.0

    if hasattr(img.dataobj, "get_unscaled"):
        raw = np.asanyarray(img.dataobj.get_unscaled())
    else:
        raw = np.asanyarray(img.dataobj)

    data = (raw.astype(np.float32) * slope + inter).astype(np.float32)

    new_hdr = img.header.copy()
    new_hdr.set_slope_inter(1.0, 0.0)
    new_hdr.set_data_dtype(np.float32)

    # this works but doesn't force Fiji of opening it with mm as the scale
    # because of bio-image it forces microns
    new_hdr.set_xyzt_units(xyz="mm", t="sec")

    new_img = nib.Nifti1Image(data, img.affine, new_hdr)
    new_img.header.set_slope_inter(1.0, 0.0)
    new_img.header.set_xyzt_units(xyz="mm", t="sec")
    return new_img


def save_nifti(img: nib.Nifti1Image, out_path: Path) -> tuple[float, float]:
    """Save img, optionally baking slope/inter. Returns (slope, inter) seen."""
    sl = float(img.header["scl_slope"] or 1.0)
    it = float(img.header["scl_inter"] or 0.0)
    if SCALE_INTENSITIES:
        img = bake_scaling(img)
    img.to_filename(str(out_path))
    return sl, it


def convert_study(study_path: Path, out_dir: Path) -> None:
    print(f"\n=== {study_path.name} ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = BrukerLoader(str(study_path))

    try:
        info = loader.info(scope="full", as_dict=True, show_reco=True)
    except TypeError:
        info = loader.info(scope="full", as_dict=True)
    write_json(out_dir / "study_info.json", info)
    print("  wrote study_info.json")

    for scan_id in loader.avail:
        try:
            scan = loader.get_scan(scan_id)
        except Exception as e:
            print(f"  [scan {scan_id}] could not open: {e}")
            continue

        proto = sanitize(guess_protocol(info, scan_id, loader))

        for reco_id in scan.avail:
            base = f"{int(scan_id):02d}_{proto}_reco{reco_id}"

            # Try preferred space first, then fall back. Some 2D / single-slice
            # scans have a degenerate affine that breaks subject_ras decomposition.
            spaces_to_try = list(dict.fromkeys([SPACE, *SPACE_FALLBACKS]))
            nii = None
            used_space = None
            last_err = None
            for sp in spaces_to_try:
                try:
                    nii = loader.convert(
                        scan_id,
                        reco_id=reco_id,
                        space=sp,
                        xyz_units="mm", # this won't force how fiji opens it
                        t_units="sec",
                    )
                    used_space = sp
                    break
                except Exception as e:
                    last_err = e
                    continue

            if nii is None:
                print(f"  [scan {scan_id} reco {reco_id}] convert failed in "
                      f"{spaces_to_try}: {last_err}")
                continue

            sl_seen, it_seen = 1.0, 0.0
            if isinstance(nii, tuple):
                for i, img in enumerate(nii, start=1):
                    sl_seen, it_seen = save_nifti(
                        img, out_dir / f"{base}_part{i}.nii.gz"
                    )
                tag = f"{base}_part1..{len(nii)}.nii.gz"
            else:
                sl_seen, it_seen = save_nifti(nii, out_dir / f"{base}.nii.gz")
                tag = f"{base}.nii.gz"

            try:
                meta = loader.get_metadata(scan_id, reco_id=reco_id)
            except Exception:
                meta = None
            if isinstance(meta, dict):
                meta.setdefault("BrkRawConversion", {})
                # adds additional metadata for fiji
                meta["BrkRawConversion"].update({
                    "ScaleApplied": bool(SCALE_INTENSITIES),
                    "OriginalSclSlope": sl_seen,
                    "OriginalSclInter": it_seen,
                    "Space": used_space,
                    "SpaceRequested": SPACE,
                })
            write_json(out_dir / f"{base}.json", meta or {})

            bits = []
            if SCALE_INTENSITIES and (sl_seen != 1.0 or it_seen != 0.0):
                bits.append(f"baked slope={sl_seen:g}, inter={it_seen:g}")
            if used_space != SPACE:
                bits.append(f"space={used_space} (fallback)")
            # Voxel sizes (mm, mm, mm[, sec]) --> for Fiji's Set Scale.
            ref = nii[0] if isinstance(nii, tuple) else nii
            zooms = ref.header.get_zooms()
            zooms_str = "x".join(f"{z:.4g}" for z in zooms[:3])
            bits.append(f"voxel={zooms_str} mm")
            extra = f"  ({'; '.join(bits)})" if bits else ""
            print(f"  [scan {scan_id} reco {reco_id}] -> {tag}{extra}")


def main(argv) -> int:
    root = Path(argv[1]).expanduser() if len(argv) > 1 else RAW_ROOT
    # interupt if no raw_data (Bruker folder)
    if not root.exists():
        print(f"raw root does not exist: {root}", file=sys.stderr)
        return 1

    studies = sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.endswith(OUT_SUFFIX)
    )
    if not studies:
        print(f"no Bruker study folders found in {root}", file=sys.stderr)
        return 1

    for study in studies:
        out_dir = study.parent / f"{study.name}{OUT_SUFFIX}"
        try:
            convert_study(study, out_dir)
        except Exception as e:
            print(f"!! {study.name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
