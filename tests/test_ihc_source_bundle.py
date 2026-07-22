from pathlib import Path

import pytest

from src.ihc.source_bundle import checksum_olympus_bundle


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    vsi = tmp_path / "sample.vsi"
    companion = tmp_path / "_sample_"
    ets = companion / "stack10001" / "frame_t_0.ets"
    ets.parent.mkdir(parents=True)
    vsi.write_bytes(b"vsi metadata")
    ets.write_bytes(b"pixel bytes")
    return vsi, companion


def test_source_bundle_checksum_covers_vsi_and_relative_ets_structure(tmp_path: Path):
    vsi, companion = _bundle(tmp_path)

    manifest = checksum_olympus_bundle(vsi, companion)

    assert manifest["file_count"] == 2
    assert manifest["ets_file_count"] == 1
    assert manifest["companion_dir_relative_path"] == "_sample_"
    assert [row["relative_path"] for row in manifest["files"]] == [
        "_sample_/stack10001/frame_t_0.ets",
        "sample.vsi",
    ]
    assert len(manifest["combined_source_bundle_sha256"]) == 64


def test_source_bundle_checksum_changes_when_ets_changes(tmp_path: Path):
    vsi, companion = _bundle(tmp_path)
    before = checksum_olympus_bundle(vsi, companion)
    (companion / "stack10001" / "frame_t_0.ets").write_bytes(b"changed pixels")
    after = checksum_olympus_bundle(vsi, companion)

    assert before["combined_source_bundle_sha256"] != after[
        "combined_source_bundle_sha256"
    ]


def test_source_bundle_requires_ets_pixel_files(tmp_path: Path):
    vsi = tmp_path / "sample.vsi"
    companion = tmp_path / "_sample_"
    vsi.write_bytes(b"vsi metadata")
    companion.mkdir()

    with pytest.raises(ValueError, match="no .ets pixel files"):
        checksum_olympus_bundle(vsi, companion)


def test_source_bundle_accepts_paths_reached_through_a_data_symlink(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    vsi, companion = _bundle(source_root)
    linked_root = tmp_path / "data"
    linked_root.symlink_to(source_root, target_is_directory=True)

    manifest = checksum_olympus_bundle(
        linked_root / vsi.name,
        linked_root / companion.name,
    )

    assert manifest["companion_dir_relative_path"] == "_sample_"
    assert manifest["ets_file_count"] == 1
