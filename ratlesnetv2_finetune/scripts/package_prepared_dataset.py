"""Package a prepared dataset as a versioned, portable Kaggle tarball.

Source data are read-only. Excluded cases are removed from both archive members
and the packaged manifest, whose paths are rewritten relative to the archive
root instead of retaining workstation-specific absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import io
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

REQUIRED_CASE_FILES = (
    "scan.nii.gz",
    "scan_lesionIAM.nii.gz",
    "scan_lesion.nii.gz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared dataset root")
    parser.add_argument("--output", required=True, help="Output .tar.gz path")
    parser.add_argument("--dataset-name", required=True, help="Archive root directory name")
    parser.add_argument("--label-version", required=True)
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = package_prepared_dataset(
        input_root=Path(args.input),
        output_path=Path(args.output),
        dataset_name=args.dataset_name,
        label_version=args.label_version,
        excluded_cases=set(args.exclude_case),
        overwrite=args.overwrite,
    )
    print(f"Packaged cases: {result['n_cases']}")
    print(f"Excluded cases: {result['n_excluded']}")
    print(f"Archive: {args.output}")
    return 0


def package_prepared_dataset(
    *,
    input_root: Path,
    output_path: Path,
    dataset_name: str,
    label_version: str,
    excluded_cases: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    excluded_cases = excluded_cases or set()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Prepared dataset root not found: {input_root}")
    if not _safe_component(dataset_name):
        raise ValueError("dataset_name must be one safe path component")
    if label_version in {"", "TODO"}:
        raise ValueError("label_version must be explicit")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Archive exists: {output_path}; pass --overwrite to replace it")
    if output_path.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("Output filename must end in .tar.gz")

    manifest = _read_csv(input_root / "manifest.csv")
    manifest_ids = {_required(row, "case_id") for row in manifest}
    unknown = sorted(excluded_cases - manifest_ids)
    if unknown:
        raise ValueError(f"Excluded case IDs are absent from the manifest: {unknown}")
    included = [row for row in manifest if _required(row, "case_id") not in excluded_cases]
    if not included:
        raise ValueError("No cases remain after exclusions")

    portable_rows = []
    members: list[tuple[Path, str]] = []
    for row in included:
        case_id = _required(row, "case_id")
        source_dir = _case_dir(input_root, row)
        relative_case = PurePosixPath(
            dataset_name,
            _required(row, "split"),
            _required(row, "study"),
            _required(row, "timepoint"),
            case_id,
        )
        for filename in REQUIRED_CASE_FILES:
            source = source_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing required file for {case_id}: {source}")
            members.append((source, str(relative_case / filename)))
        portable = dict(row)
        portable["case_dir"] = str(relative_case)
        portable["scan_path"] = str(relative_case / "scan.nii.gz")
        portable["label_path"] = str(relative_case / "scan_lesionIAM.nii.gz")
        portable["label_version"] = label_version
        portable_rows.append(portable)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.partial")
    if temporary.exists():
        temporary.unlink()
    try:
        with tarfile.open(temporary, mode="w:gz") as archive:
            _add_bytes(
                archive,
                f"{dataset_name}/manifest.csv",
                _csv_bytes(portable_rows),
            )
            for source, archive_name in members:
                archive.add(source, arcname=archive_name, recursive=False)
        _validate_archive(
            temporary,
            dataset_name=dataset_name,
            expected_cases=len(included),
            excluded_cases=excluded_cases,
        )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "n_cases": len(included),
        "n_excluded": len(excluded_cases),
        "dataset_name": dataset_name,
        "label_version": label_version,
    }


def _validate_archive(
    path: Path,
    *,
    dataset_name: str,
    expected_cases: int,
    excluded_cases: set[str],
) -> None:
    with tarfile.open(path, "r:gz") as archive:
        manifest_member = archive.extractfile(f"{dataset_name}/manifest.csv")
        if manifest_member is None:
            raise RuntimeError("Packaged manifest is missing")
        rows = list(csv.DictReader(io.TextIOWrapper(manifest_member, encoding="utf-8")))
        names = {member.name for member in archive.getmembers() if member.isfile()}
    ids = {_required(row, "case_id") for row in rows}
    if len(rows) != expected_cases or ids & excluded_cases:
        raise RuntimeError("Packaged manifest case validation failed")
    expected_files = {
        str(PurePosixPath(row["case_dir"]) / filename)
        for row in rows
        for filename in REQUIRED_CASE_FILES
    }
    if not expected_files.issubset(names):
        missing = sorted(expected_files - names)
        raise RuntimeError(f"Packaged case files are missing: {missing[:10]}")


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def _case_dir(root: Path, row: dict[str, str]) -> Path:
    canonical = (
        root
        / _required(row, "split")
        / _required(row, "study")
        / _required(row, "timepoint")
        / _required(row, "case_id")
    )
    if canonical.is_dir():
        return canonical
    recorded = Path(str(row.get("case_dir", "")))
    if recorded.is_dir():
        return recorded
    raise FileNotFoundError(f"Prepared case directory not found for {_required(row, 'case_id')}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {path}")
    return rows


def _required(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value in {None, "", "TODO"}:
        raise ValueError(f"case_id={row.get('case_id')!r} is missing required value: {key}")
    return str(value)


def _safe_component(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value


if __name__ == "__main__":
    raise SystemExit(main())
