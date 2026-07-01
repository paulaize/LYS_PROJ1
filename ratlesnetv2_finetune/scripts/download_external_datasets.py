"""Download and clean public mouse stroke MRI datasets for RatLesNetV2.

The script writes a clean source folder compatible with
``ratlesnetv2-add-source``:

```
<output-root>/
├── ratlesnetv2_clean_source/
│   ├── T2w/
│   └── masks/
├── manifests/
└── _downloads/              # removed when --delete-archives is used
```

Only native-space manual labels are exported by default. Automated masks,
atlas-space masks, raw Bruker folders, and known duplicate public cases are not
exported as training labels.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

NIFTI_SUFFIX = ".nii.gz"


@dataclass(frozen=True)
class ArchiveSpec:
    key: str
    url: str
    filename: str


ARCHIVES = {
    "an2022": ArchiveSpec(
        key="an2022",
        url="https://zenodo.org/api/records/6379879/files/data_an_et_al_2022.zip/content",
        filename="An2022_data_an_et_al_2022.zip",
    ),
    "knab2025": ArchiveSpec(
        key="knab2025",
        url="https://zenodo.org/api/records/14709930/files/repository_v2.0.zip/content",
        filename="Knab2025_repository_v2.0.zip",
    ),
    "koch2017": ArchiveSpec(
        key="koch2017",
        url="https://zenodo.org/api/records/842677/files/data_koch_et_al_jcbfm_2017.zip/content",
        filename="Koch2017_data_koch_et_al_jcbfm_2017.zip",
    ),
    "mulder2017_cologne1": ArchiveSpec(
        key="mulder2017_cologne1",
        url="https://zenodo.org/api/records/4086450/files/Cologne-Set-1.zip/content",
        filename="Mulder2017_Cologne-Set-1.zip",
    ),
    "mulder2017_cologne2": ArchiveSpec(
        key="mulder2017_cologne2",
        url="https://zenodo.org/api/records/4086450/files/Cologne-Set-2.zip/content",
        filename="Mulder2017_Cologne-Set-2.zip",
    ),
    "mulder2017_leiden_segmentations": ArchiveSpec(
        key="mulder2017_leiden_segmentations",
        url=(
            "https://zenodo.org/api/records/4086450/files/"
            "Leiden-Set-T2maps_Segmentations.zip/content"
        ),
        filename="Mulder2017_Leiden-Set-T2maps_Segmentations.zip",
    ),
}

DATASET_CHOICES = ("an2022", "knab2025", "koch2017", "mulder2017", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="/Volumes/Untitled/external_datasets",
        help="External-dataset root folder to create/update",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_CHOICES,
        default=["all"],
        help="Datasets to download/clean",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Delete downloaded zip archives after successful cleaning",
    )
    parser.add_argument(
        "--overwrite-clean",
        action="store_true",
        help="Rebuild existing cleaned NIfTI outputs",
    )
    parser.add_argument(
        "--include-mulder",
        action="store_true",
        help=(
            "Include full Mulder2017 T2-map/manual-IAM pairs. These are useful "
            "for later expansion but are not geometry-matched to An/Knab/Koch/LYS."
        ),
    )
    parser.add_argument(
        "--include-koch-cropped",
        action="store_true",
        help="Also export Koch cropped-to-20-slices MCAO copies as cropped_native cases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned downloads/extraction steps without downloading or writing outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    selected = _selected_datasets(args.datasets, include_mulder=args.include_mulder)

    if args.dry_run:
        print(f"Would write external datasets under: {output_root}")
        for archive in _archives_for(selected):
            print(f"Would download: {archive.filename}")
        return 0

    downloads = output_root / "_downloads"
    clean_root = output_root / "ratlesnetv2_clean_source"
    manifest_root = output_root / "manifests"
    t2w_dir = clean_root / "T2w"
    mask_dir = clean_root / "masks"
    for directory in [downloads, t2w_dir, mask_dir, manifest_root]:
        directory.mkdir(parents=True, exist_ok=True)

    archive_paths = {
        archive.key: _download_archive(archive, downloads)
        for archive in _archives_for(selected)
    }

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()

    if "an2022" in selected:
        records.extend(
            _process_an2022(
                archive_paths["an2022"],
                t2w_dir=t2w_dir,
                mask_dir=mask_dir,
                seen_source_ids=seen_source_ids,
                skipped=skipped,
                overwrite=args.overwrite_clean,
            )
        )
    if "knab2025" in selected:
        records.extend(
            _process_knab2025(
                archive_paths["knab2025"],
                t2w_dir=t2w_dir,
                mask_dir=mask_dir,
                seen_source_ids=seen_source_ids,
                skipped=skipped,
                overwrite=args.overwrite_clean,
            )
        )
    if "koch2017" in selected:
        records.extend(
            _process_koch2017(
                archive_paths["koch2017"],
                t2w_dir=t2w_dir,
                mask_dir=mask_dir,
                seen_source_ids=seen_source_ids,
                skipped=skipped,
                include_cropped=args.include_koch_cropped,
                overwrite=args.overwrite_clean,
            )
        )
    if "mulder2017" in selected:
        for key in [
            "mulder2017_cologne1",
            "mulder2017_cologne2",
            "mulder2017_leiden_segmentations",
        ]:
            records.extend(
                _process_mulder_archive(
                    archive_paths[key],
                    t2w_dir=t2w_dir,
                    mask_dir=mask_dir,
                    seen_source_ids=seen_source_ids,
                    skipped=skipped,
                    overwrite=args.overwrite_clean,
                )
            )

    _write_csv(manifest_root / "external_dataset_manifest.csv", records)
    _write_csv(manifest_root / "skipped_cases.csv", skipped)
    _write_readme(output_root, records, skipped)

    if args.delete_archives:
        shutil.rmtree(downloads)

    print(f"Wrote clean RatLesNetV2 source folder: {clean_root}")
    print(f"Wrote manifest: {manifest_root / 'external_dataset_manifest.csv'}")
    print(f"Exported cases: {len(records)}")
    print(f"Skipped cases: {len(skipped)}")
    return 0


def _selected_datasets(raw: list[str], *, include_mulder: bool) -> set[str]:
    selected = set(raw)
    if "all" in selected:
        selected = {"an2022", "knab2025", "koch2017"}
        if include_mulder:
            selected.add("mulder2017")
    return selected


def _archives_for(selected: set[str]) -> list[ArchiveSpec]:
    archives: list[ArchiveSpec] = []
    if "an2022" in selected:
        archives.append(ARCHIVES["an2022"])
    if "knab2025" in selected:
        archives.append(ARCHIVES["knab2025"])
    if "koch2017" in selected:
        archives.append(ARCHIVES["koch2017"])
    if "mulder2017" in selected:
        archives.extend(
            [
                ARCHIVES["mulder2017_cologne1"],
                ARCHIVES["mulder2017_cologne2"],
                ARCHIVES["mulder2017_leiden_segmentations"],
            ]
        )
    return archives


def _download_archive(archive: ArchiveSpec, downloads: Path) -> Path:
    dest = downloads / archive.filename
    if dest.exists() and dest.stat().st_size > 0:
        if _is_valid_zip(dest):
            print(f"Using existing archive: {dest}")
            if archive.key == "koch2017":
                return _prepare_koch_archive(dest)
            return dest
        print(f"Resuming incomplete archive: {dest}")
    else:
        print(f"Downloading {archive.filename}")
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--continue-at",
            "-",
            "--output",
            str(dest),
            archive.url,
        ],
        check=True,
    )
    if not _is_valid_zip(dest):
        raise ValueError(f"Downloaded archive is not a valid zip file: {dest}")
    if archive.key == "koch2017":
        return _prepare_koch_archive(dest)
    return dest


def _is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            zf.infolist()
            return True
    except zipfile.BadZipFile:
        return False


def _prepare_koch_archive(path: Path) -> Path:
    """The Koch zip is a >4 GB non-Zip64 archive; repair improves extraction."""
    repaired = path.with_name(f"{path.stem}.repaired{path.suffix}")
    sentinel = "all/dat/20170428AR_TTC_M07/t2.nii"
    if _system_unzip_member_complete(path, sentinel):
        return path
    if repaired.exists() and _system_unzip_member_complete(repaired, sentinel):
        print(f"Using repaired Koch archive: {repaired}")
        return repaired
    print(f"Repairing malformed Koch archive with zip -FF: {path}")
    subprocess.run(["zip", "-FF", str(path), "--out", str(repaired)], check=True)
    if not _system_unzip_member_complete(repaired, sentinel):
        raise ValueError(f"Could not repair Koch archive sufficiently: {path}")
    print(f"Using repaired Koch archive: {repaired}")
    return repaired


def _system_unzip_member_complete(path: Path, member: str) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            expected_size = zf.getinfo(member).file_size
    except (KeyError, zipfile.BadZipFile):
        return False
    completed = subprocess.run(
        ["unzip", "-p", str(path), member],
        check=False,
        capture_output=True,
    )
    return len(completed.stdout) == expected_size


def _process_an2022(
    archive_path: Path,
    *,
    t2w_dir: Path,
    mask_dir: Path,
    seen_source_ids: set[str],
    skipped: list[dict[str, Any]],
    overwrite: bool,
) -> list[dict[str, Any]]:
    print("Processing An2022")
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as zf:
        cases: dict[str, dict[str, str]] = {}
        for name in zf.namelist():
            parts = name.strip("/").split("/")
            if len(parts) != 4:
                continue
            root, cohort, case_id, filename = parts
            if root != "data_repository" or cohort != "data_charite":
                continue
            if filename in {"t2.nii", "masklesion_manual.nii"}:
                cases.setdefault(case_id, {})[filename] = name

        for case_id in sorted(cases):
            files = cases[case_id]
            source_key = f"An2022:{case_id}"
            global_id = f"An2022__{case_id}"
            if source_key in seen_source_ids:
                skipped.append(_skip_row("An2022", case_id, "duplicate_source_id", source_key))
                continue
            if {"t2.nii", "masklesion_manual.nii"} - set(files):
                skipped.append(_skip_row("An2022", case_id, "missing_manual_pair", ""))
                continue
            record = _export_nifti_pair_from_zip(
                zf,
                scan_member=files["t2.nii"],
                mask_member=files["masklesion_manual.nii"],
                global_id=global_id,
                source_dataset="An2022",
                source_record_url="https://zenodo.org/records/6379879",
                source_case_id=case_id,
                label_observer="unknown",
                image_space="native",
                modality="T2w",
                t2w_dir=t2w_dir,
                mask_dir=mask_dir,
                overwrite=overwrite,
            )
            records.append(record)
            seen_source_ids.add(source_key)
            seen_source_ids.add(f"Charite:{case_id}")
    return records


def _process_knab2025(
    archive_path: Path,
    *,
    t2w_dir: Path,
    mask_dir: Path,
    seen_source_ids: set[str],
    skipped: list[dict[str, Any]],
    overwrite: bool,
) -> list[dict[str, Any]]:
    print("Processing Knab2025")
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as zf:
        cases: dict[str, dict[str, str]] = {}
        for name in zf.namelist():
            parts = name.strip("/").split("/")
            if len(parts) != 4:
                continue
            root, dat, case_id, filename = parts
            if root != "repository" or dat != "dat":
                continue
            if filename in {"t2.nii", "masklesion.nii"}:
                cases.setdefault(case_id, {})[filename] = name

        for case_id in sorted(cases):
            files = cases[case_id]
            if f"Charite:{case_id}" in seen_source_ids:
                skipped.append(_skip_row("Knab2025", case_id, "duplicate_of_An2022", case_id))
                continue
            source_key = f"Knab2025:{case_id}"
            if source_key in seen_source_ids:
                skipped.append(_skip_row("Knab2025", case_id, "duplicate_source_id", source_key))
                continue
            if {"t2.nii", "masklesion.nii"} - set(files):
                skipped.append(_skip_row("Knab2025", case_id, "missing_manual_pair", ""))
                continue
            record = _export_nifti_pair_from_zip(
                zf,
                scan_member=files["t2.nii"],
                mask_member=files["masklesion.nii"],
                global_id=f"Knab2025__{case_id}",
                source_dataset="Knab2025",
                source_record_url="https://zenodo.org/records/14709930",
                source_case_id=case_id,
                label_observer="unknown",
                image_space="native",
                modality="T2w",
                t2w_dir=t2w_dir,
                mask_dir=mask_dir,
                overwrite=overwrite,
            )
            records.append(record)
            seen_source_ids.add(source_key)
    return records


def _process_koch2017(
    archive_path: Path,
    *,
    t2w_dir: Path,
    mask_dir: Path,
    seen_source_ids: set[str],
    skipped: list[dict[str, Any]],
    include_cropped: bool,
    overwrite: bool,
) -> list[dict[str, Any]]:
    print("Processing Koch2017")
    records: list[dict[str, Any]] = []
    allowed_groups = {"all/dat"}
    if include_cropped:
        allowed_groups.add("mcao_cropped_to_20slices/dat")
    with zipfile.ZipFile(archive_path) as zf:
        cases: dict[tuple[str, str], dict[str, str]] = {}
        for name in zf.namelist():
            parts = name.strip("/").split("/")
            if len(parts) < 4:
                continue
            group = "/".join(parts[:-2])
            case_id = parts[-2]
            filename = parts[-1]
            if group not in allowed_groups:
                continue
            if filename in {"t2.nii", "masklesion.nii"}:
                cases.setdefault((group, case_id), {})[filename] = name

        for (group, case_id), files in sorted(cases.items()):
            source_key = f"Koch2017:{group}:{case_id}"
            if source_key in seen_source_ids:
                skipped.append(_skip_row("Koch2017", case_id, "duplicate_source_id", source_key))
                continue
            if {"t2.nii", "masklesion.nii"} - set(files):
                skipped.append(_skip_row("Koch2017", case_id, "missing_manual_pair", group))
                continue
            image_space = "cropped_native" if group.startswith("mcao_cropped") else "native"
            if image_space == "cropped_native":
                prefix = "Koch2017Cropped"
            else:
                prefix = "Koch2017"
            try:
                record = _export_nifti_pair_from_zip(
                    zf,
                    scan_member=files["t2.nii"],
                    mask_member=files["masklesion.nii"],
                    global_id=f"{prefix}__{case_id}",
                    source_dataset="Koch2017",
                    source_record_url="https://zenodo.org/records/842677",
                    source_case_id=case_id,
                    label_observer="unknown",
                    image_space=image_space,
                    modality="T2w",
                    t2w_dir=t2w_dir,
                    mask_dir=mask_dir,
                    overwrite=overwrite,
                )
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                skipped.append(_skip_row("Koch2017", case_id, "unreadable_pair", str(exc)))
                continue
            records.append(record)
            seen_source_ids.add(source_key)
    return records


def _process_mulder_archive(
    archive_path: Path,
    *,
    t2w_dir: Path,
    mask_dir: Path,
    seen_source_ids: set[str],
    skipped: list[dict[str, Any]],
    overwrite: bool,
) -> list[dict[str, Any]]:
    print(f"Processing Mulder2017 archive: {archive_path.name}")
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as zf:
        t2maps: dict[str, str] = {}
        masks: dict[str, str] = {}
        for name in zf.namelist():
            filename = Path(name).name
            if not filename.endswith(".mhd"):
                continue
            if "_MHD_T2maps/" in name and filename.endswith("_T2map.mhd"):
                case_id = filename[: -len("_T2map.mhd")]
                t2maps[case_id] = name
            elif "_MHD_Manual_Segmentations_IAM/" in name and filename.endswith(
                "_MANUAL_IAM.mhd"
            ):
                case_id = filename[: -len("_MANUAL_IAM.mhd")]
                masks[case_id] = name

        for case_id in sorted(set(t2maps) & set(masks)):
            source_key = f"Mulder2017:{case_id}"
            if source_key in seen_source_ids:
                skipped.append(_skip_row("Mulder2017", case_id, "duplicate_source_id", source_key))
                continue
            record = _export_mhd_pair_from_zip(
                zf,
                scan_mhd_member=t2maps[case_id],
                mask_mhd_member=masks[case_id],
                global_id=f"Mulder2017__{case_id}",
                source_case_id=case_id,
                t2w_dir=t2w_dir,
                mask_dir=mask_dir,
                overwrite=overwrite,
            )
            records.append(record)
            seen_source_ids.add(source_key)

        for case_id in sorted(set(t2maps) ^ set(masks)):
            skipped.append(
                _skip_row("Mulder2017", case_id, "missing_manual_pair", archive_path.name)
            )
    return records


def _export_nifti_pair_from_zip(
    zf: zipfile.ZipFile,
    *,
    scan_member: str,
    mask_member: str,
    global_id: str,
    source_dataset: str,
    source_record_url: str,
    source_case_id: str,
    label_observer: str,
    image_space: str,
    modality: str,
    t2w_dir: Path,
    mask_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    out_scan = t2w_dir / f"{global_id}{NIFTI_SUFFIX}"
    out_mask = mask_dir / f"{global_id}_lesion_mask{NIFTI_SUFFIX}"
    if out_scan.exists() and out_mask.exists() and not overwrite:
        return _record_from_existing(
            global_id=global_id,
            source_dataset=source_dataset,
            source_record_url=source_record_url,
            source_case_id=source_case_id,
            label_observer=label_observer,
            image_space=image_space,
            modality=modality,
            scan_path=out_scan,
            mask_path=out_mask,
        )

    with tempfile.TemporaryDirectory(prefix="lys_external_") as tmp:
        tmp_dir = Path(tmp)
        scan_tmp = tmp_dir / "scan.nii"
        mask_tmp = tmp_dir / "mask.nii"
        scan_tmp.write_bytes(_read_zip_member(zf, scan_member))
        mask_tmp.write_bytes(_read_zip_member(zf, mask_member))
        _save_clean_nifti_pair(
            scan_tmp,
            mask_tmp,
            out_scan=out_scan,
            out_mask=out_mask,
        )

    return _record_from_existing(
        global_id=global_id,
        source_dataset=source_dataset,
        source_record_url=source_record_url,
        source_case_id=source_case_id,
        label_observer=label_observer,
        image_space=image_space,
        modality=modality,
        scan_path=out_scan,
        mask_path=out_mask,
    )


def _export_mhd_pair_from_zip(
    zf: zipfile.ZipFile,
    *,
    scan_mhd_member: str,
    mask_mhd_member: str,
    global_id: str,
    source_case_id: str,
    t2w_dir: Path,
    mask_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    out_scan = t2w_dir / f"{global_id}{NIFTI_SUFFIX}"
    out_mask = mask_dir / f"{global_id}_lesion_mask{NIFTI_SUFFIX}"
    if out_scan.exists() and out_mask.exists() and not overwrite:
        return _record_from_existing(
            global_id=global_id,
            source_dataset="Mulder2017",
            source_record_url="https://datadryad.org/dataset/doi:10.5061/dryad.1m528",
            source_case_id=source_case_id,
            label_observer="IAM",
            image_space="native",
            modality="T2map",
            scan_path=out_scan,
            mask_path=out_mask,
        )

    with tempfile.TemporaryDirectory(prefix="lys_external_mhd_") as tmp:
        tmp_dir = Path(tmp)
        _extract_mhd_with_raw(zf, scan_mhd_member, tmp_dir)
        _extract_mhd_with_raw(zf, mask_mhd_member, tmp_dir)
        scan_tmp = tmp_dir / Path(scan_mhd_member).name
        mask_tmp = tmp_dir / Path(mask_mhd_member).name
        _save_clean_mhd_pair(
            scan_tmp,
            mask_tmp,
            out_scan=out_scan,
            out_mask=out_mask,
        )

    return _record_from_existing(
        global_id=global_id,
        source_dataset="Mulder2017",
        source_record_url="https://datadryad.org/dataset/doi:10.5061/dryad.1m528",
        source_case_id=source_case_id,
        label_observer="IAM",
        image_space="native",
        modality="T2map",
        scan_path=out_scan,
        mask_path=out_mask,
    )


def _extract_mhd_with_raw(zf: zipfile.ZipFile, mhd_member: str, out_dir: Path) -> None:
    mhd_text = _read_zip_member(zf, mhd_member).decode("utf-8", errors="replace")
    out_mhd = out_dir / Path(mhd_member).name
    out_mhd.write_text(mhd_text)
    raw_name = None
    for line in mhd_text.splitlines():
        if line.strip().startswith("ElementDataFile"):
            raw_name = line.split("=", 1)[1].strip()
            break
    if raw_name is None:
        raise ValueError(f"MHD file does not declare ElementDataFile: {mhd_member}")
    raw_member = str(Path(mhd_member).with_name(raw_name))
    (out_dir / raw_name).write_bytes(_read_zip_member(zf, raw_member))


def _read_zip_member(zf: zipfile.ZipFile, member: str) -> bytes:
    try:
        return zf.read(member)
    except zipfile.BadZipFile as exc:
        if zf.filename is None:
            raise
        print(f"Falling back to system unzip for malformed zip member: {member}")
        completed = subprocess.run(
            ["unzip", "-p", str(zf.filename), member],
            check=False,
            capture_output=True,
        )
        expected_size = zf.getinfo(member).file_size
        if completed.returncode != 0 and len(completed.stdout) != expected_size:
            detail = completed.stderr.decode("utf-8", errors="replace")
            raise ValueError(f"Could not extract {member} from {zf.filename}: {detail}") from exc
        return completed.stdout


def _save_clean_nifti_pair(
    scan_path: Path, mask_path: Path, *, out_scan: Path, out_mask: Path
) -> None:
    scan = nib.load(str(scan_path))
    mask = nib.load(str(mask_path))
    _save_scan_and_binary_mask(scan, mask, out_scan=out_scan, out_mask=out_mask)


def _save_clean_mhd_pair(
    scan_path: Path, mask_path: Path, *, out_scan: Path, out_mask: Path
) -> None:
    import SimpleITK as sitk

    scan_img = sitk.ReadImage(str(scan_path))
    mask_img = sitk.ReadImage(str(mask_path))
    with tempfile.TemporaryDirectory(prefix="lys_external_mhd_nifti_") as tmp:
        tmp_dir = Path(tmp)
        scan_nii = tmp_dir / "scan.nii.gz"
        mask_nii = tmp_dir / "mask.nii.gz"
        sitk.WriteImage(scan_img, str(scan_nii))
        sitk.WriteImage(mask_img, str(mask_nii))
        scan = nib.load(str(scan_nii))
        mask = nib.load(str(mask_nii))
        _save_scan_and_binary_mask(scan, mask, out_scan=out_scan, out_mask=out_mask)


def _save_scan_and_binary_mask(
    scan: nib.spatialimages.SpatialImage,
    mask: nib.spatialimages.SpatialImage,
    *,
    out_scan: Path,
    out_mask: Path,
) -> None:
    if tuple(scan.shape[:3]) != tuple(mask.shape[:3]):
        raise ValueError(f"Scan shape {scan.shape} does not match mask shape {mask.shape}")
    out_scan.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(scan, str(out_scan))
    mask_data = (np.asanyarray(mask.dataobj) > 0).astype(np.uint8)
    header = scan.header.copy()
    header.set_data_shape(mask_data.shape)
    header.set_data_dtype(np.uint8)
    header.set_zooms(scan.header.get_zooms()[:3])
    nib.save(nib.Nifti1Image(mask_data, scan.affine, header), str(out_mask))


def _record_from_existing(
    *,
    global_id: str,
    source_dataset: str,
    source_record_url: str,
    source_case_id: str,
    label_observer: str,
    image_space: str,
    modality: str,
    scan_path: Path,
    mask_path: Path,
) -> dict[str, Any]:
    scan = nib.load(str(scan_path))
    mask = nib.load(str(mask_path))
    mask_data = np.asanyarray(mask.dataobj) > 0
    spacing = tuple(float(v) for v in scan.header.get_zooms()[:3])
    voxels = int(mask_data.sum())
    return {
        "global_case_id": global_id,
        "source_dataset": source_dataset,
        "source_record_url": source_record_url,
        "source_case_id": source_case_id,
        "study": source_dataset,
        "timepoint": _guess_timepoint(source_case_id),
        "species": "mouse",
        "stroke_model": "unknown",
        "modality": modality,
        "scan_path": str(scan_path),
        "mask_path": str(mask_path),
        "label_source": "manual",
        "label_observer": label_observer,
        "image_space": image_space,
        "shape": "x".join(str(int(v)) for v in scan.shape[:3]),
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "mask_voxels": voxels,
        "lesion_volume_mm3": f"{voxels * float(np.prod(spacing)):.10g}",
        "duplicate_of": "",
        "qc_flag": "needs_visual_qc",
    }


def _guess_timepoint(case_id: str) -> str:
    for token in ["24h", "48h", "17h", "19h", "2d", "4h", "8d", "91h"]:
        if token in case_id:
            return token
    return "unknown_timepoint"


def _skip_row(source_dataset: str, source_case_id: str, reason: str, detail: str) -> dict[str, str]:
    return {
        "source_dataset": source_dataset,
        "source_case_id": source_case_id,
        "reason": reason,
        "detail": detail,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    output_root: Path, records: list[dict[str, Any]], skipped: list[dict[str, Any]]
) -> None:
    counts: dict[str, int] = {}
    for row in records:
        counts[row["source_dataset"]] = counts.get(row["source_dataset"], 0) + 1
    lines = [
        "# External Mouse Stroke MRI Datasets",
        "",
        "This folder was generated by",
        "`python -m ratlesnetv2_finetune.scripts.download_external_datasets`.",
        "",
        "Clean source folder for RatLesNetV2 import:",
        "",
        "`ratlesnetv2_clean_source/`",
        "",
        "Exported case counts:",
    ]
    for dataset in sorted(counts):
        lines.append(f"- {dataset}: {counts[dataset]}")
    lines.extend(
        [
            f"- skipped cases: {len(skipped)}",
            "",
            "Only native-space manual labels are intended for training.",
            "All outputs still require visual QC before model training.",
            "",
            "See repository document `docs/ratlesnetv2_external_datasets.md`",
            "for overlap, geometry, and inclusion rules.",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
