"""Ingest QuPath measurement exports into tidy rows.

FIXED INTERFACE: keep ingest_qupath(csv_path) stable. v1 expects one row per
(section, annotation) with IgG-FITC positive area and total area. Cell-level
columns can be added in later milestones without changing the function contract.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = {
    "image",
    "panel",
    "region",
    "igg_fitc_pos_area_um2",
    "total_area_um2",
    "dapi_count",
}


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def ingest_qupath(csv_path: str | Path) -> list[dict]:
    """Parse a QuPath export CSV into tidy measurement rows (list of dicts)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"QuPath export not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        legacy = {"fitc_pos_area_um2"} <= set(df.columns)
        hint = " Legacy FITC column found; rename/export as igg_fitc_pos_area_um2." if legacy else ""
        raise ValueError(
            f"QuPath export {csv_path.name} missing columns {sorted(missing)}.{hint} "
            "Check export_measurements.groovy is in sync with EXPECTED_COLUMNS."
        )

    rows: list[dict] = []
    for _, r in df.iterrows():
        total_um2 = _safe_float(r["total_area_um2"])
        total_mm2 = total_um2 / 1e6 if total_um2 > 0 else float("nan")
        pos_um2 = _safe_float(r["igg_fitc_pos_area_um2"])
        pct = (pos_um2 / total_um2 * 100.0) if total_um2 > 0 else float("nan")
        dapi_raw = _safe_float(r["dapi_count"])
        dapi_density = (dapi_raw / total_mm2) if (total_mm2 > 0 and dapi_raw >= 0) else float("nan")

        base = {
            "image": r["image"],
            "panel": r["panel"],
            "region": r["region"],
            "area_mm2": total_mm2,
            "qc_flag": r.get("qc_flag", "v1_area_only"),
        }
        rows.append(
            {
                **base,
                "marker": "IgG-FITC",
                "cell_type": "all",
                "measure": "igg_fitc_pct_positive_area",
                "value": pct,
                "unit": "percent",
                "n_cells": "",
            }
        )
        rows.append(
            {
                **base,
                "marker": "DAPI",
                "cell_type": "nuclei",
                "measure": "dapi_density",
                "value": dapi_density,
                "unit": "nuclei_per_mm2",
                "n_cells": int(dapi_raw) if dapi_raw >= 0 else "",
            }
        )
    return rows
