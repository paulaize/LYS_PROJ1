"""Milestone 4 STUB — merge MRI + IHC into the tidy long table.

Join key: (animal, allen_region, compartment, hemisphere). Output is LONG (one
measure per row) with provenance columns (see AGENTS.md §10). Handle animals
with NO MRI (1h/3h/6h) via histology-proxy compartments, flagged in
compartment_method. Not part of v1.
"""
from __future__ import annotations

# Canonical output schema — the long tidy table the project delivers.
OUTPUT_COLUMNS = [
    "animal_id", "timepoint", "panel", "hemisphere",
    "allen_region", "compartment", "compartment_method", "cell_type",
    "measure", "value", "unit", "n_cells", "area_mm2",
    "model_version", "edited", "edit_dice", "reviewer", "qc_flag",
]


def join_modalities(*args, **kwargs):
    raise NotImplementedError(
        "Milestone 4. Merge MRI compartment/region labels with IHC per-region "
        "measurements on (animal, allen_region, compartment, hemisphere). "
        "Emit the long schema in OUTPUT_COLUMNS. Do NOT add to v1."
    )
