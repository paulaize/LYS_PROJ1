"""Normalize the hemisphere-split AIDAmri parental atlas label tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.atlas.t2w_mapping import sha256


def normalize_parental_structure_lookup(
    *,
    names_table: Path,
    acronyms_table: Path,
    output_csv: Path,
    aidamri_revision: str,
    right_hemisphere_offset: int = 2000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the AIDAmri split-parental labels in the adapter's CSV schema."""
    names_table = names_table.resolve()
    acronyms_table = acronyms_table.resolve()
    output_csv = output_csv.resolve()
    revision = aidamri_revision.strip()
    if not revision:
        raise ValueError("AIDAmri revision is required")
    if right_hemisphere_offset <= 0:
        raise ValueError("Right-hemisphere label offset must be positive")
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_csv}")

    names = _read_integer_keyed_table(names_table, value_name="region name")
    acronyms = _read_integer_keyed_table(acronyms_table, value_name="acronym")

    by_hemisphere: dict[str, dict[int, tuple[int, str]]] = {
        "left": {},
        "right": {},
    }
    for label_id, encoded_name in names.items():
        if encoded_name.startswith("L_"):
            hemisphere = "left"
            base_id = label_id
            region_name = encoded_name[2:]
        elif encoded_name.startswith("R_"):
            hemisphere = "right"
            base_id = label_id - right_hemisphere_offset
            region_name = encoded_name[2:]
        else:
            raise ValueError(
                f"{names_table}: label {label_id} must start with L_ or R_"
            )
        if base_id <= 0:
            raise ValueError(
                f"{names_table}: label {label_id} has invalid base ID {base_id}"
            )
        if base_id in by_hemisphere[hemisphere]:
            raise ValueError(
                f"{names_table}: duplicate {hemisphere} base label ID {base_id}"
            )
        by_hemisphere[hemisphere][base_id] = (label_id, region_name)

    left_ids = set(by_hemisphere["left"])
    right_ids = set(by_hemisphere["right"])
    acronym_ids = set(acronyms)
    if left_ids != right_ids:
        raise ValueError(
            f"{names_table}: left/right parental label pairs do not match"
        )
    if left_ids != acronym_ids:
        missing = sorted(left_ids - acronym_ids)
        extra = sorted(acronym_ids - left_ids)
        raise ValueError(
            f"{acronyms_table}: acronym IDs do not match parental labels; "
            f"missing={missing}, extra={extra}"
        )

    rows: list[dict[str, str | int]] = []
    for hemisphere in ("left", "right"):
        for base_id in sorted(left_ids):
            label_id, encoded_name = by_hemisphere[hemisphere][base_id]
            rows.append(
                {
                    "label_id": label_id,
                    "acronym": acronyms[base_id],
                    "name": encoded_name.replace("_", " "),
                    "hemisphere": hemisphere,
                }
            )
    rows.sort(key=lambda row: int(row["label_id"]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label_id", "acronym", "name", "hemisphere"),
        )
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "schema_version": 1,
        "atlas_id": "AIDAmri_ARA",
        "annotation_variant": "split_parental",
        "aidamri_revision": revision,
        "right_hemisphere_offset": right_hemisphere_offset,
        "names_table": str(names_table),
        "names_table_sha256": sha256(names_table),
        "acronyms_table": str(acronyms_table),
        "acronyms_table_sha256": sha256(acronyms_table),
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256(output_csv),
        "n_base_regions": len(left_ids),
        "n_hemisphere_split_regions": len(rows),
    }
    provenance_path = output_csv.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {**provenance, "provenance_json": str(provenance_path)}


def _read_integer_keyed_table(path: Path, *, value_name: str) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing AIDAmri {value_name} table: {path}")
    records: dict[int, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        key_text, separator, value = line.partition("\t")
        value = value.strip()
        if not separator or not value:
            raise ValueError(
                f"{path}:{line_number}: expected tab-separated label ID and {value_name}"
            )
        try:
            label_id = int(key_text.strip())
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: label ID must be an integer"
            ) from error
        if label_id <= 0:
            raise ValueError(f"{path}:{line_number}: label ID must be positive")
        if label_id in records:
            raise ValueError(f"{path}:{line_number}: duplicate label ID {label_id}")
        records[label_id] = value
    if not records:
        raise ValueError(f"{path}: table is empty")
    return records
