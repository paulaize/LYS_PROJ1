"""Convert the configured Bruker T2 RARE scan to NIfTI for v1 MRI.

This writes only under work/ by default. Raw Bruker inputs under data/ stay
read-only.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def convert_t2(config_path: str | Path, *, force: bool = False, dry_run: bool = False) -> Path:
    from src.config import load_config

    cfg = load_config(config_path)
    if not cfg.has_mri:
        raise ValueError(f"[{cfg.animal_id}] mri.has_mri is false; no Bruker conversion needed.")

    study = cfg.bruker_study
    if study is None:
        raise ValueError(
            f"[{cfg.animal_id}] mri.bruker_study is unset. Add the raw Bruker study folder "
            "before converting scan 2."
        )
    if not study.exists():
        raise FileNotFoundError(f"[{cfg.animal_id}] Bruker study not found: {study}")

    out_path = cfg.t2_nifti
    if out_path is None:
        raise ValueError(
            f"[{cfg.animal_id}] mri.t2_nifti is unset. Set the intended work/ output path."
        )

    brkraw = shutil.which("brkraw")
    if brkraw is None:
        raise RuntimeError("brkraw command not found. Run this inside the lys-bbb conda env.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[{cfg.animal_id}] NIfTI already exists: {out_path}")
        _validate_header(out_path, cfg)
        return out_path

    cmd = [
        brkraw,
        "convert",
        str(study),
        "--scan-id",
        str(cfg.t2_scan_id),
        "--reco-id",
        str(cfg.t2_reco_id),
        "--output",
        str(out_path),
    ]
    if dry_run:
        print(" ".join(cmd))
        return out_path

    print(
        f"[{cfg.animal_id}] converting Bruker scan {cfg.t2_scan_id}/reco {cfg.t2_reco_id} "
        f"-> {out_path}"
    )
    subprocess.run(cmd, check=True)
    if not out_path.exists():
        raise FileNotFoundError(f"brkraw finished but expected NIfTI was not created: {out_path}")
    _validate_header(out_path, cfg)
    return out_path


def _validate_header(nifti_path: Path, cfg) -> None:
    from src.mri import io as mri_io

    img = mri_io.load_nifti(nifti_path)
    spacing = mri_io.check_header(img, cfg.expected_spacing_mm, cfg.spacing_tolerance)
    print(f"[{cfg.animal_id}] validated T2 spacing: {spacing} mm")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert configured Bruker T2 scan to NIfTI.")
    parser.add_argument("--config", required=True, help="config/animals/<id>.yml")
    parser.add_argument("--force", action="store_true", help="overwrite an existing NIfTI")
    parser.add_argument("--dry-run", action="store_true", help="print brkraw command only")
    args = parser.parse_args()
    convert_t2(args.config, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
