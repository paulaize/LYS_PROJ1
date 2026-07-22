"""Validate and identify Olympus VSI source bundles without modifying them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_olympus_bundle(
    vsi_path: str | Path,
    companion_dir: str | Path,
    *,
    configured_vsi_path: str | None = None,
    configured_companion_dir: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic checksum manifest for one VSI + companion tree.

    Every regular file in the companion directory is covered, including every
    ETS pixel file. The combined digest hashes canonical JSON containing sorted
    relative paths, sizes and per-file SHA-256 digests.
    """
    vsi_path = Path(vsi_path)
    companion_dir = Path(companion_dir)
    if vsi_path.suffix.lower() != ".vsi":
        raise ValueError(f"Olympus source must be a .vsi file: {vsi_path}")
    if not vsi_path.is_file():
        raise FileNotFoundError(f"Olympus VSI source is missing: {vsi_path}")
    if not companion_dir.is_dir():
        raise FileNotFoundError(
            f"Olympus VSI companion directory is missing: {companion_dir}"
        )

    source_root = vsi_path.parent.resolve()
    if companion_dir.parent.resolve() != source_root:
        raise ValueError(
            "Olympus VSI and companion directory must share a parent so relative "
            f"bundle structure is unambiguous: {vsi_path}, {companion_dir}"
        )

    companion_files = sorted(
        (path for path in companion_dir.rglob("*") if path.is_file()),
        key=lambda path: path.resolve().relative_to(source_root).as_posix(),
    )
    ets_files = [path for path in companion_files if path.suffix.lower() == ".ets"]
    if not ets_files:
        raise ValueError(
            f"Olympus companion directory contains no .ets pixel files: {companion_dir}"
        )

    files = [vsi_path, *companion_files]
    entries: list[dict[str, Any]] = []
    for path in files:
        stat = path.stat()
        entries.append(
            {
                "relative_path": path.resolve().relative_to(source_root).as_posix(),
                "size_bytes": stat.st_size,
                "sha256": _sha256_file(path),
            }
        )
    entries.sort(key=lambda row: row["relative_path"])
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    combined_sha256 = hashlib.sha256(canonical).hexdigest()

    return {
        "bundle_format": "olympus_vsi_with_companion_ets",
        "checksum_algorithm": "sha256",
        "combined_checksum_algorithm": "sha256(canonical_sorted_file_manifest_json)",
        "configured_vsi_path": configured_vsi_path or str(vsi_path),
        "configured_companion_dir": configured_companion_dir or str(companion_dir),
        "resolved_vsi_path": str(vsi_path.resolve()),
        "resolved_companion_dir": str(companion_dir.resolve()),
        "companion_dir_relative_path": companion_dir.resolve()
        .relative_to(source_root)
        .as_posix(),
        "file_count": len(entries),
        "ets_file_count": len(ets_files),
        "total_size_bytes": sum(row["size_bytes"] for row in entries),
        "files": entries,
        "combined_source_bundle_sha256": combined_sha256,
    }


def configured_panel_bundles(cfg, panel: str) -> list[dict[str, Any]]:
    """Resolve and validate explicit source-bundle config for one panel."""
    panel_cfg = cfg.panel_config(panel)
    bundles = panel_cfg.get("source_bundles") or []
    if not bundles:
        raise ValueError(
            f"{cfg.animal_id} panel {panel}: source_bundles is unset. Configure both "
            "the .vsi file and its Olympus companion directory."
        )
    configured_vsi = {str(path) for path in panel_cfg.get("vsi_files") or []}
    bundle_vsi = {str(bundle.get("vsi_file", "")) for bundle in bundles}
    if configured_vsi != bundle_vsi:
        raise ValueError(
            f"{cfg.animal_id} panel {panel}: source_bundles VSI paths do not match "
            f"vsi_files ({sorted(bundle_vsi)} != {sorted(configured_vsi)})"
        )

    resolved: list[dict[str, Any]] = []
    for bundle in bundles:
        vsi_config = bundle.get("vsi_file")
        companion_config = bundle.get("companion_dir")
        if not vsi_config or not companion_config:
            raise ValueError(
                f"{cfg.animal_id} panel {panel}: each source bundle needs vsi_file "
                "and companion_dir"
            )
        vsi_path = cfg.resolve_input_path(vsi_config)
        companion_dir = cfg.resolve_input_path(companion_config)
        assert vsi_path is not None and companion_dir is not None
        resolved.append(
            {
                "animal_id": cfg.animal_id,
                "panel": panel,
                "vsi_path": vsi_path,
                "companion_dir": companion_dir,
                "configured_vsi_path": str(vsi_config),
                "configured_companion_dir": str(companion_config),
            }
        )
    return resolved
