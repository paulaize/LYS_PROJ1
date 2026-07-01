"""Stage scattered LYS Bruker studies and Fiji RoiSets into clean NIfTI pairs.

The source hard drive is treated as read-only. For each RoiSet, this script:

1. finds the associated Bruker study folder;
2. identifies the native T2w RARE scan by Bruker metadata;
3. copies only the selected scan folder, subject file, and RoiSet locally;
4. converts the copied Bruker scan to NIfTI with brkraw;
5. converts the copied Fiji RoiSet to a binary *_lesion_mask.nii.gz.

The final folder is compatible with ``ratlesnetv2-add-source``:

```
<output-root>/
├── ratlesnetv2_clean_source/
│   ├── T2w/
│   └── masks/
├── roi_archives/
├── manifests/
└── README.md
```
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from ratlesnetv2_finetune.roiset_to_nifti_mask import convert_roiset_to_nifti_mask

NIFTI_SUFFIX = ".nii.gz"
MASK_SUFFIX = "_lesion_mask.nii.gz"
DEFAULT_SOURCE_DRIVE = Path("/Volumes/Lys T Dec 2021 N1")
DEFAULT_OUTPUT_ROOT = Path.home() / "Desktop" / "LYS_RatLesNetV2_clean_source"
DEFAULT_DISCOVER_SUBROOTS = (
    "ThrombinSTZ_02_PhIND/stz",
    "Thrombin_06/IRM",
    "Thrombin_08_PhIND",
    "Thrombin_09_PhIND",
)
EXPECTED_T2_PROTOCOL = "T2_haute_resolution_Turbo"


@dataclass(frozen=True)
class ScanInfo:
    scan_id: int
    method: str
    protocol: str
    frame_count: int | None
    size_xy: tuple[int, int] | None
    fov_xy_mm: tuple[float, float] | None
    frame_thickness_mm: float | None
    score: int


@dataclass(frozen=True)
class RoiJob:
    roiset_source: Path
    study_source: Path
    project: str
    case_id: str
    roi_token: str
    roiset_stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-drive",
        default=str(DEFAULT_SOURCE_DRIVE),
        help="Read-only source drive containing the scattered LYS data",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Local clean output folder to create on this computer",
    )
    parser.add_argument(
        "--roi-list",
        help="Optional text file containing one RoiSet .zip path per line",
    )
    parser.add_argument(
        "--discover-root",
        action="append",
        help=(
            "Root to search recursively for RoiSet*.zip files. May be repeated. "
            "Defaults to the known LYS project roots under --source-drive."
        ),
    )
    parser.add_argument(
        "--delete-staging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete per-case temporary copied Bruker data after each conversion",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing output folder and skip already completed pairs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted NIfTI/mask outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inventory and write manifests without copying or converting data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_drive = Path(args.source_drive)
    output_root = Path(args.output_root)

    if not source_drive.is_dir():
        raise FileNotFoundError(f"Source drive/folder not found: {source_drive}")
    _prepare_output_root(output_root, resume=args.resume, dry_run=args.dry_run)

    roisets = _load_roiset_paths(
        source_drive=source_drive,
        roi_list=args.roi_list,
        discover_roots=args.discover_root,
    )
    print(f"Discovered RoiSet archives: {len(roisets)}")

    jobs: list[RoiJob] = []
    skipped: list[dict[str, Any]] = []
    scan_inventory: list[dict[str, Any]] = []
    for roiset in roisets:
        try:
            job = _resolve_roiset_job(roiset, source_drive=source_drive)
        except ValueError as exc:
            skipped.append(_skip_row(roiset, "study_mapping_failed", str(exc)))
            continue
        scan_inventory.extend(_scan_inventory_rows(job))
        try:
            _select_t2_scan(job.study_source)
        except ValueError as exc:
            skipped.append(_skip_row(roiset, "t2_scan_selection_failed", str(exc)))
            continue
        jobs.append(job)

    duplicate_keys = _duplicate_output_keys(jobs)
    if duplicate_keys:
        keep: list[RoiJob] = []
        for job in jobs:
            if job.case_id in duplicate_keys:
                skipped.append(
                    _skip_row(
                        job.roiset_source,
                        "duplicate_case_id",
                        f"{job.case_id} maps to more than one RoiSet/study; skipped",
                    )
                )
                continue
            keep.append(job)
        jobs = keep

    if args.dry_run:
        _write_manifests(
            output_root=output_root,
            records=[],
            skipped=skipped,
            scan_inventory=scan_inventory,
            roiset_count=len(roisets),
            dry_run=True,
        )
        print(f"Resolved jobs: {len(jobs)}")
        print(f"Skipped/needs review: {len(skipped)}")
        print(f"Dry-run manifests written under: {output_root / 'manifests'}")
        return 0

    records: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.case_id}")
        try:
            record = _convert_job(
                job,
                output_root=output_root,
                overwrite=args.overwrite,
                delete_staging=args.delete_staging,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch moving, record provenance.
            skipped.append(_skip_row(job.roiset_source, "conversion_failed", str(exc)))
            continue
        records.append(record)

    _write_manifests(
        output_root=output_root,
        records=records,
        skipped=skipped,
        scan_inventory=scan_inventory,
        roiset_count=len(roisets),
        dry_run=False,
    )
    print(f"Wrote clean source folder: {output_root / 'ratlesnetv2_clean_source'}")
    print(f"Converted pairs: {len(records)}")
    print(f"Skipped/needs review: {len(skipped)}")
    return 0


def _prepare_output_root(output_root: Path, *, resume: bool, dry_run: bool) -> None:
    if output_root.exists() and not resume:
        raise FileExistsError(
            f"Output folder already exists: {output_root}. "
            "Choose a new --output-root or pass --resume."
        )
    if dry_run:
        (output_root / "manifests").mkdir(parents=True, exist_ok=True)
        return
    for child in [
        output_root / "ratlesnetv2_clean_source" / "T2w",
        output_root / "ratlesnetv2_clean_source" / "masks",
        output_root / "roi_archives",
        output_root / "manifests",
    ]:
        child.mkdir(parents=True, exist_ok=True)


def _load_roiset_paths(
    *,
    source_drive: Path,
    roi_list: str | None,
    discover_roots: list[str] | None,
) -> list[Path]:
    paths: list[Path] = []
    if roi_list is not None:
        list_path = Path(roi_list)
        for raw in list_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line.rstrip(".")))
    else:
        roots = (
            [Path(root) for root in discover_roots]
            if discover_roots
            else [source_drive / subroot for subroot in DEFAULT_DISCOVER_SUBROOTS]
        )
        for root in roots:
            resolved = root if root.is_absolute() else source_drive / root
            if not resolved.is_dir():
                continue
            paths.extend(sorted(resolved.rglob("RoiSet*.zip")))

    unique: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"RoiSet archive not found: {path}")
        unique[str(path)] = path
    return [unique[key] for key in sorted(unique)]


def _resolve_roiset_job(roiset: Path, *, source_drive: Path) -> RoiJob:
    study = _find_containing_study(roiset.parent)
    roi_token = _roi_token(roiset)
    if study is None:
        study = _match_sibling_study(roiset)
    if study is None:
        raise ValueError("Could not find an associated Bruker study folder")

    project = _project_from_path(roiset, source_drive=source_drive)
    subject_id = _subject_id(study) or study.name
    stem = _safe_id(roiset.stem)
    case_base = _safe_id(subject_id)
    if roiset.name != "RoiSet.zip":
        case_base = f"{case_base}__{stem}"
    case_id = f"{_safe_id(project)}__{case_base}"
    return RoiJob(
        roiset_source=roiset,
        study_source=study,
        project=project,
        case_id=case_id,
        roi_token=roi_token,
        roiset_stem=stem,
    )


def _find_containing_study(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if _is_bruker_study(candidate):
            return candidate
        if candidate == candidate.parent:
            break
    return None


def _match_sibling_study(roiset: Path) -> Path | None:
    token = _roi_token(roiset)
    if not token:
        return None
    candidates: list[Path] = []
    for child in sorted(roiset.parent.iterdir()):
        if not child.is_dir() or not _is_bruker_study(child):
            continue
        haystacks = [child.name.upper()]
        subject_id = _subject_id(child)
        if subject_id:
            haystacks.append(subject_id.upper())
        if any(_token_matches(token, haystack) for haystack in haystacks):
            candidates.append(child)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates[:8])
        raise ValueError(f"Ambiguous Bruker study match for token {token}: {names}")
    return None


def _is_bruker_study(path: Path) -> bool:
    if not (path / "subject").is_file():
        return False
    return any(
        child.is_dir() and child.name.isdigit() and (child / "method").is_file()
        for child in path.iterdir()
    )


def _roi_token(roiset: Path) -> str:
    match = re.search(r"ROISET_?([A-Z0-9]+S[0-9]+|C[0-9]+Q[0-9]+)", roiset.stem.upper())
    if match:
        return match.group(1)
    match = re.search(r"(C[0-9]+S[0-9]+|C[0-9]+Q[0-9]+)", roiset.parent.name.upper())
    return match.group(1) if match else ""


def _token_matches(token: str, text: str) -> bool:
    pattern = rf"(?<![A-Z0-9]){re.escape(token.upper())}(?![A-Z0-9])"
    return re.search(pattern, text.upper()) is not None


def _project_from_path(path: Path, *, source_drive: Path) -> str:
    try:
        relative = path.relative_to(source_drive)
    except ValueError:
        return "LYS"
    for part in relative.parts:
        if part.lower().startswith("thrombin"):
            return part
    return relative.parts[0] if relative.parts else "LYS"


def _subject_id(study: Path) -> str | None:
    subject = study / "subject"
    if not subject.is_file():
        return None
    text = subject.read_text(errors="replace")
    return _param_text_value(text, "SUBJECT_id") or _param_text_value(text, "SUBJECT_name_string")


def _scan_inventory_rows(job: RoiJob) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        selected_t2_id = _select_t2_scan(job.study_source).scan_id
    except ValueError:
        selected_t2_id = None
    for scan in _scan_infos(job.study_source):
        rows.append(
            {
                "case_id": job.case_id,
                "project": job.project,
                "study_source": str(job.study_source),
                "roi_source": str(job.roiset_source),
                "scan_id": scan.scan_id,
                "method": scan.method,
                "protocol": scan.protocol,
                "frame_count": scan.frame_count or "",
                "size_xy": _join_tuple(scan.size_xy),
                "fov_xy_mm": _join_tuple(scan.fov_xy_mm),
                "frame_thickness_mm": scan.frame_thickness_mm or "",
                "t2_candidate_score": scan.score,
                "is_selected_t2": scan.scan_id == selected_t2_id,
            }
        )
    return rows


def _scan_infos(study: Path) -> list[ScanInfo]:
    scans: list[ScanInfo] = []
    def scan_sort_key(item: Path) -> int:
        return int(item.name) if item.name.isdigit() else 10_000

    for child in sorted(study.iterdir(), key=scan_sort_key):
        if not child.is_dir() or not child.name.isdigit():
            continue
        method_path = child / "method"
        visu_path = child / "visu_pars"
        method_text = method_path.read_text(errors="replace") if method_path.is_file() else ""
        visu_text = visu_path.read_text(errors="replace") if visu_path.is_file() else ""
        method = (
            _param_text_value(method_text, "Method")
            or _param_text_value(visu_text, "VisuAcqSequenceName")
            or ""
        )
        protocol = _param_text_value(visu_text, "VisuAcquisitionProtocol") or ""
        frame_count = _param_int(visu_text, "VisuCoreFrameCount")
        size_xy = _param_int_tuple(visu_text, "VisuCoreSize", count=2)
        fov_xy = _param_float_tuple(visu_text, "VisuCoreExtent", count=2)
        thickness = _param_float(visu_text, "VisuCoreFrameThickness")
        score = _t2_score(
            method=method,
            protocol=protocol,
            frame_count=frame_count,
            size_xy=size_xy,
        )
        scans.append(
            ScanInfo(
                scan_id=int(child.name),
                method=method,
                protocol=protocol,
                frame_count=frame_count,
                size_xy=size_xy,
                fov_xy_mm=fov_xy,
                frame_thickness_mm=thickness,
                score=score,
            )
        )
    return scans


def _select_t2_scan(study: Path) -> ScanInfo:
    candidates = sorted(_scan_infos(study), key=lambda scan: (-scan.score, scan.scan_id))
    if not candidates or candidates[0].score <= 0:
        raise ValueError(f"No T2w RARE scan candidate found in {study}")
    if len(candidates) > 1 and candidates[1].score == candidates[0].score:
        raise ValueError(
            f"Ambiguous T2w scan in {study}: scan {candidates[0].scan_id} and "
            f"scan {candidates[1].scan_id} have the same score"
        )
    return candidates[0]


def _t2_score(
    *,
    method: str,
    protocol: str,
    frame_count: int | None,
    size_xy: tuple[int, int] | None,
) -> int:
    score = 0
    if "RARE" in method.upper():
        score += 10
    if EXPECTED_T2_PROTOCOL.lower() in protocol.lower():
        score += 50
    if size_xy == (256, 256):
        score += 5
    if frame_count == 18:
        score += 5
    return score


def _convert_job(
    job: RoiJob,
    *,
    output_root: Path,
    overwrite: bool,
    delete_staging: bool,
) -> dict[str, Any]:
    scan = _select_t2_scan(job.study_source)
    t2w_dir = output_root / "ratlesnetv2_clean_source" / "T2w"
    mask_dir = output_root / "ratlesnetv2_clean_source" / "masks"
    roi_dir = output_root / "roi_archives"
    out_scan = t2w_dir / f"{job.case_id}{NIFTI_SUFFIX}"
    out_mask = mask_dir / f"{job.case_id}{MASK_SUFFIX}"
    out_roi = roi_dir / f"{job.case_id}__{job.roiset_source.name}"

    if out_scan.exists() and out_mask.exists() and out_roi.exists() and not overwrite:
        return _record_from_outputs(
            job,
            scan,
            out_scan=out_scan,
            out_mask=out_mask,
            out_roi=out_roi,
        )
    if not overwrite and (out_scan.exists() or out_mask.exists()):
        raise FileExistsError(f"Partial output exists for {job.case_id}; rerun with --overwrite")

    with tempfile.TemporaryDirectory(prefix=f"lys_{job.case_id}_", dir=str(output_root)) as tmp:
        staging = Path(tmp)
        staged_study = staging / "study"
        staged_study.mkdir()
        shutil.copy2(job.study_source / "subject", staged_study / "subject")
        shutil.copytree(job.study_source / str(scan.scan_id), staged_study / str(scan.scan_id))
        staged_roi = staging / job.roiset_source.name
        shutil.copy2(job.roiset_source, staged_roi)

        out_scan.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "brkraw",
                "convert",
                str(staged_study),
                "--scan-id",
                str(scan.scan_id),
                "--reco-id",
                "1",
                "--output",
                str(out_scan),
            ],
            check=True,
        )
        convert_roiset_to_nifti_mask(
            nifti_path=out_scan,
            roiset_zip=staged_roi,
            output_path=out_mask,
            overwrite=True,
        )
        out_roi.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_roi, out_roi)
        if not delete_staging:
            kept = output_root / "staging_kept" / job.case_id
            kept.parent.mkdir(parents=True, exist_ok=True)
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(staging, kept)

    return _record_from_outputs(job, scan, out_scan=out_scan, out_mask=out_mask, out_roi=out_roi)


def _record_from_outputs(
    job: RoiJob,
    scan: ScanInfo,
    *,
    out_scan: Path,
    out_mask: Path,
    out_roi: Path,
) -> dict[str, Any]:
    scan_img = nib.load(str(out_scan))
    mask_img = nib.load(str(out_mask))
    if scan_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(f"Shape mismatch for {job.case_id}: {scan_img.shape} vs {mask_img.shape}")
    mask_data = np.asanyarray(mask_img.dataobj) > 0
    spacing = tuple(float(value) for value in scan_img.header.get_zooms()[:3])
    voxels = int(mask_data.sum())
    return {
        "case_id": job.case_id,
        "project": job.project,
        "timepoint": _guess_timepoint(job),
        "roi_source_path": str(job.roiset_source),
        "study_source_path": str(job.study_source),
        "local_roiset_path": str(out_roi),
        "scan_nifti": str(out_scan),
        "lesion_mask": str(out_mask),
        "scan_id": scan.scan_id,
        "reco_id": 1,
        "scan_method": scan.method,
        "scan_protocol": scan.protocol,
        "shape": "x".join(str(int(v)) for v in scan_img.shape[:3]),
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "mask_voxels": voxels,
        "lesion_volume_mm3": f"{voxels * float(np.prod(spacing)):.10g}",
        "qc_flag": "needs_visual_qc",
    }


def _guess_timepoint(job: RoiJob) -> str:
    text = f"{job.case_id} {job.study_source.name} {job.roiset_source.parent.name}".upper()
    for pattern in [r"24H", r"48H", r"5D", r"D[0-9]+"]:
        match = re.search(pattern, text)
        if match:
            return match.group(0).lower()
    return "unknown_timepoint"


def _duplicate_output_keys(jobs: list[RoiJob]) -> set[str]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.case_id] = counts.get(job.case_id, 0) + 1
    return {key for key, count in counts.items() if count > 1}


def _write_manifests(
    *,
    output_root: Path,
    records: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    scan_inventory: list[dict[str, Any]],
    roiset_count: int,
    dry_run: bool,
) -> None:
    manifest_root = output_root / "manifests"
    _write_csv(manifest_root / "lys_clean_dataset_manifest.csv", records)
    _write_csv(manifest_root / "lys_skipped_or_needs_review.csv", skipped)
    _write_csv(manifest_root / "lys_scan_inventory.csv", scan_inventory)
    lines = [
        "# LYS RoiSet/Bruker Clean Dataset",
        "",
        "Generated by `python -m ratlesnetv2_finetune.scripts.prepare_lys_roiset_dataset`.",
        "",
        f"Dry run: {dry_run}",
        f"RoiSet archives seen: {roiset_count}",
        f"Converted pairs: {len(records)}",
        f"Skipped/needs review: {len(skipped)}",
        "",
        "The source hard drive was treated as read-only. Bruker conversion used a",
        "temporary local copy of the selected scan folder and subject file.",
        "",
        "Clean source folder for RatLesNetV2 import:",
        "`ratlesnetv2_clean_source/`",
        "",
        "All masks are draft training labels and still require visual QC.",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines))


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _skip_row(roiset: Path, reason: str, detail: str) -> dict[str, str]:
    return {"roi_source_path": str(roiset), "reason": reason, "detail": detail}


def _safe_id(raw: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "unknown"


def _join_tuple(values: tuple[Any, ...] | None) -> str:
    if values is None:
        return ""
    return "x".join(str(value) for value in values)


def _param_text_value(text: str, key: str) -> str | None:
    raw = _param_raw_value(text, key)
    if raw is None:
        return None
    match = re.search(r"<([^>]*)>", raw, flags=re.DOTALL)
    if match:
        return match.group(1).replace("\\", "").strip()
    return raw.strip()


def _param_int(text: str, key: str) -> int | None:
    raw = _param_raw_value(text, key)
    if raw is None:
        return None
    match = re.search(r"-?\d+", raw)
    return int(match.group(0)) if match else None


def _param_float(text: str, key: str) -> float | None:
    raw = _param_raw_value(text, key)
    if raw is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", raw)
    return float(match.group(0)) if match else None


def _param_int_tuple(text: str, key: str, *, count: int) -> tuple[int, ...] | None:
    raw = _param_raw_value(text, key)
    if raw is None:
        return None
    values = [int(value) for value in re.findall(r"-?\d+", raw)]
    if len(values) < count:
        return None
    return tuple(values[:count])


def _param_float_tuple(text: str, key: str, *, count: int) -> tuple[float, ...] | None:
    raw = _param_raw_value(text, key)
    if raw is None:
        return None
    values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", raw)]
    if len(values) < count:
        return None
    return tuple(values[:count])


def _param_raw_value(text: str, key: str) -> str | None:
    lines = text.splitlines()
    prefix = f"##${key}="
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if value and not re.fullmatch(r"\(\s*[\d, ]+\s*\)", value):
            return value
        collected: list[str] = []
        for next_line in lines[index + 1 :]:
            if next_line.startswith("##"):
                break
            if next_line.startswith("$$") or next_line.startswith("@@"):
                continue
            stripped = next_line.strip()
            if stripped:
                collected.append(stripped)
        return " ".join(collected) if collected else value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
