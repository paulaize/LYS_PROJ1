"""v1 entry point: run one animal through the thin spine and write a minimal table.

    conda run -n lys-bbb python -m src.run_animal --config config/animals/<id>.yml

v1 does NOT do atlas registration, DL, compartments, or cell-level detection.
It produces:
  - a human-corrected lesion mask + raw/edema-corrected volume if MRI exists,
  - tidy IHC rows from a QuPath export CSV if provided,
  - a single minimal CSV in outputs/<id>/.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import load_config


def run_mri(cfg, *, allow_mask_editor: bool = True) -> dict | None:
    """Return MRI results, or None if this animal has no MRI."""
    if not cfg.has_mri:
        print(f"[{cfg.animal_id}] no MRI (early timepoint) — skipping MRI track.")
        return None

    from .mri import edema, preprocess, segment
    from .mri import io as mri_io
    from .mri.edit import review_mask
    from .mri.volume import mask_volume_mm3, per_slice_area_mm2

    t2 = cfg.t2_nifti
    if t2 is None:
        raise ValueError(f"[{cfg.animal_id}] mri.has_mri is true but mri.t2_nifti is unset.")

    img = mri_io.load_nifti(t2)
    spacing = mri_io.check_header(img, cfg.expected_spacing_mm, cfg.spacing_tolerance)
    arr = mri_io.get_array(img).astype("float32")

    seg_cfg = cfg.pipeline.get("mri", {}).get("segment", {})
    if cfg.pipeline.get("mri", {}).get("n4", {}).get("enabled", True):
        arr = preprocess.n4_bias_correct(arr)
    brain = preprocess.brain_mask(arr)

    draft = segment.segment_lesion(
        arr,
        brain,
        method=seg_cfg.get("method", "threshold"),
        spacing_mm=spacing,
        k=float(seg_cfg.get("k", 2.5)),
        min_lesion_mm3=float(seg_cfg.get("min_lesion_mm3", 0.5)),
        lesion_side=cfg.animal.get("mri", {}).get("lesion_side"),
    )

    work = cfg.work_dir()
    mri_io.save_mask(draft, img, work / "lesion_draft.nii.gz")

    # ✎ human border-correction gate. Fallback is reproducible but QC-flagged.
    edit = review_mask(
        arr,
        draft,
        work / "lesion_corrected.nii.gz",
        reference_img=img,
        reviewer=cfg.reviewer,
        allow_gui=allow_mask_editor,
    )
    corrected_path = work / "lesion_corrected.nii.gz"
    corrected = (
        mri_io.get_array(mri_io.load_nifti(corrected_path)).astype(bool)
        if corrected_path.exists()
        else draft
    )

    vol_raw = mask_volume_mm3(corrected, spacing)
    eo = edema.swanson_corrected_volume(
        corrected,
        brain,
        spacing,
        lesion_side=cfg.animal.get("mri", {}).get("lesion_side"),
    )
    areas = per_slice_area_mm2(corrected, spacing)

    print(
        f"[{cfg.animal_id}] lesion raw={vol_raw:.3f} mm^3, "
        f"corrected={eo['corrected_mm3']:.3f} mm^3 "
        f"(swelling x{eo['swelling_ratio']:.3f}); "
        f"qc={edit.get('qc_flag')} dice={edit['dice']:.3f}"
    )

    return {
        "raw_mm3": vol_raw,
        "corrected_mm3": eo["corrected_mm3"],
        "swelling_ratio": eo["swelling_ratio"],
        "per_slice_area_mm2": areas.tolist(),
        **edit,
    }


def run_ihc(cfg, ihc_csv: Path | None) -> list[dict]:
    if ihc_csv is None or not Path(ihc_csv).exists():
        print(
            f"[{cfg.animal_id}] no IHC export CSV provided — skipping IHC ingest. "
            "Run the QuPath Groovy scripts first; see README."
        )
        return []
    from .ihc.ingest import ingest_qupath

    rows = ingest_qupath(ihc_csv)
    print(f"[{cfg.animal_id}] ingested {len(rows)} IHC measurement rows.")
    return rows


V1_COLUMNS = [
    "animal_id",
    "timepoint",
    "panel",
    "modality",
    "region",
    "cell_type",
    "measure",
    "value",
    "unit",
    "area_mm2",
    "n_cells",
    "edited",
    "edit_dice",
    "reviewer",
    "qc_flag",
    "model_version",
]


def write_minimal_table(cfg, mri: dict | None, ihc_rows: list[dict]) -> Path:
    out = cfg.outputs_dir() / f"{cfg.animal_id}_v1.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=V1_COLUMNS, extrasaction="ignore")
        w.writeheader()
        base = {
            "animal_id": cfg.animal_id,
            "timepoint": cfg.timepoint,
            "model_version": cfg.version,
        }
        if mri is not None:
            for measure, key in [
                ("lesion_volume_raw", "raw_mm3"),
                ("lesion_volume_corrected_swanson", "corrected_mm3"),
            ]:
                w.writerow(
                    {
                        **base,
                        "panel": "",
                        "modality": "MRI",
                        "region": "whole_brain_v1",
                        "cell_type": "",
                        "measure": measure,
                        "value": mri[key],
                        "unit": "mm3",
                        "area_mm2": "",
                        "n_cells": "",
                        "edited": mri["edited"],
                        "edit_dice": mri["dice"],
                        "reviewer": mri.get("reviewer", ""),
                        "qc_flag": mri.get("qc_flag", ""),
                    }
                )
        for r in ihc_rows:
            w.writerow(
                {
                    **base,
                    "panel": r.get("panel", ""),
                    "modality": "IHC",
                    "region": r.get("region", "whole_section"),
                    "cell_type": r.get("cell_type", ""),
                    "measure": r["measure"],
                    "value": r["value"],
                    "unit": r["unit"],
                    "area_mm2": r.get("area_mm2", ""),
                    "n_cells": r.get("n_cells", ""),
                    "edited": "",
                    "edit_dice": "",
                    "reviewer": "",
                    "qc_flag": r.get("qc_flag", ""),
                }
            )
    print(f"[{cfg.animal_id}] wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Run one animal through the v1 pipeline.")
    ap.add_argument("--config", required=True, help="config/animals/<id>.yml")
    ap.add_argument(
        "--pipeline",
        default=None,
        help="config/pipeline.yml (default: alongside repo)",
    )
    ap.add_argument("--ihc-csv", default=None, help="QuPath measurement export CSV (optional)")
    ap.add_argument(
        "--no-mask-editor",
        action="store_true",
        help="save the draft mask as needs_human_review instead of opening napari",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, args.pipeline)
    mri = run_mri(cfg, allow_mask_editor=not args.no_mask_editor)
    ihc_rows = run_ihc(cfg, Path(args.ihc_csv) if args.ihc_csv else None)
    write_minimal_table(cfg, mri, ihc_rows)


if __name__ == "__main__":
    main()
