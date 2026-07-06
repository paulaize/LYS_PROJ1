"""Queue LYS RatLesNetV2 lesion masks for manual ITK-SNAP review."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

DEFAULT_SOURCE_ROOT = Path("~/Desktop/LYS_RatLesNetV2_clean_source/ratlesnetv2_clean_source")
DEFAULT_SCAN_SUBDIR = "T2w"
DEFAULT_MASK_SUBDIR = "masks"
DEFAULT_MASK_GLOB = "*_lesion_mask.nii.gz"
DEFAULT_MASK_SUFFIX = "_lesion_mask.nii.gz"


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    source_image: Path
    source_mask: Path
    reviewed_image: Path
    reviewed_mask: Path


def discover_itksnap() -> Path | None:
    """Return a plausible ITK-SNAP executable path on macOS/PATH."""
    candidates = [
        shutil.which("itksnap"),
        shutil.which("ITK-SNAP"),
        "/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP",
        "/Applications/ITK-SNAP.app/Contents/MacOS/itksnap",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def case_from_mask(mask_path: Path, suffix: str = DEFAULT_MASK_SUFFIX) -> str:
    """Infer case ID from a RatLesNetV2 source mask filename."""
    name = mask_path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected lesion-mask filename: {mask_path}")
    return name[: -len(suffix)]


def default_output_root(source_root: Path) -> Path:
    """Place reviewed source folder next to the original clean source folder."""
    return source_root.parent / f"{source_root.name}_reviewed"


def load_grid(path: Path) -> tuple[tuple[int, ...], np.ndarray, tuple[float, ...]]:
    image = nib.load(str(path))
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return tuple(int(v) for v in image.shape), image.affine, spacing


def validate_grid(image_path: Path, mask_path: Path) -> None:
    """Require image and mask to live on the same voxel grid."""
    image_shape, image_affine, _image_spacing = load_grid(image_path)
    mask_shape, mask_affine, _mask_spacing = load_grid(mask_path)
    if image_shape != mask_shape:
        raise ValueError(
            f"Shape mismatch for {mask_path.name}: image {image_shape}, mask {mask_shape}"
        )
    if not np.allclose(image_affine, mask_affine, atol=1e-3):
        raise ValueError(f"Affine mismatch for {mask_path.name}")


def mask_voxel_count(path: Path) -> int:
    image = nib.load(str(path))
    return int((np.asanyarray(image.dataobj) > 0).sum())


def find_cases(
    *,
    source_root: Path,
    output_root: Path,
    scan_subdir: str = DEFAULT_SCAN_SUBDIR,
    mask_subdir: str = DEFAULT_MASK_SUBDIR,
    filters: list[str] | None = None,
    mask_glob: str = DEFAULT_MASK_GLOB,
    mask_suffix: str = DEFAULT_MASK_SUFFIX,
) -> list[ReviewCase]:
    """Find source-folder scan/mask pairs and their reviewed output paths."""
    source_scan_dir = source_root / scan_subdir
    source_mask_dir = source_root / mask_subdir
    reviewed_scan_dir = output_root / scan_subdir
    reviewed_mask_dir = output_root / mask_subdir
    filters_lc = [value.lower() for value in (filters or [])]
    cases: list[ReviewCase] = []
    for source_mask in sorted(source_mask_dir.glob(mask_glob)):
        case_id = case_from_mask(source_mask, suffix=mask_suffix)
        if filters_lc and not any(value in case_id.lower() for value in filters_lc):
            continue
        cases.append(
            ReviewCase(
                case_id=case_id,
                source_image=source_scan_dir / f"{case_id}.nii.gz",
                source_mask=source_mask,
                reviewed_image=reviewed_scan_dir / f"{case_id}.nii.gz",
                reviewed_mask=reviewed_mask_dir / source_mask.name,
            )
        )
    return cases


def prepare_case(
    case: ReviewCase,
    *,
    copy_scans: bool = False,
    overwrite_reviewed: bool = False,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Create the editable reviewed scan/mask pair for one case."""
    if not case.source_image.exists():
        raise FileNotFoundError(f"missing image: {case.source_image}")
    if not case.source_mask.exists():
        raise FileNotFoundError(f"missing source mask: {case.source_mask}")
    validate_grid(case.source_image, case.source_mask)

    case.reviewed_image.parent.mkdir(parents=True, exist_ok=True)
    case.reviewed_mask.parent.mkdir(parents=True, exist_ok=True)
    scan_status = _prepare_scan(case.source_image, case.reviewed_image, copy_scans=copy_scans)
    if skip_existing and case.reviewed_mask.exists():
        return _record(
            case,
            status="skipped_existing_reviewed",
            scan_status=scan_status,
            message="",
        )
    if case.reviewed_mask.exists() and overwrite_reviewed:
        shutil.copy2(case.source_mask, case.reviewed_mask)
        status = "overwrote_reviewed_mask"
    elif case.reviewed_mask.exists():
        status = "existing_reviewed_mask"
    else:
        shutil.copy2(case.source_mask, case.reviewed_mask)
        status = "copied_source_mask"

    validate_grid(case.reviewed_image, case.reviewed_mask)
    return _record(case, status=status, scan_status=scan_status, message="")


def _prepare_scan(source_image: Path, reviewed_image: Path, *, copy_scans: bool) -> str:
    if reviewed_image.exists():
        return "existing_scan"
    if copy_scans:
        shutil.copy2(source_image, reviewed_image)
        return "copied_scan"
    try:
        reviewed_image.symlink_to(source_image.resolve())
        return "linked_scan"
    except OSError:
        shutil.copy2(source_image, reviewed_image)
        return "copied_scan_symlink_failed"


def _record(case: ReviewCase, *, status: str, scan_status: str, message: str) -> dict[str, Any]:
    shape = ""
    spacing = ("", "", "")
    source_voxels: int | str = ""
    reviewed_voxels: int | str = ""
    if case.source_image.exists():
        shape_tuple, _affine, spacing_tuple = load_grid(case.source_image)
        shape = "x".join(str(v) for v in shape_tuple)
        spacing = tuple(f"{value:.10g}" for value in spacing_tuple)
    if case.source_mask.exists():
        source_voxels = mask_voxel_count(case.source_mask)
    if case.reviewed_mask.exists():
        reviewed_voxels = mask_voxel_count(case.reviewed_mask)
    return {
        "case_id": case.case_id,
        "source_image": str(case.source_image),
        "source_mask": str(case.source_mask),
        "reviewed_image": str(case.reviewed_image),
        "reviewed_mask": str(case.reviewed_mask),
        "status": status,
        "scan_status": scan_status,
        "shape": shape,
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "source_mask_voxels": source_voxels,
        "reviewed_mask_voxels": reviewed_voxels,
        "message": message,
    }


def write_manifest(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "source_image",
        "source_mask",
        "reviewed_image",
        "reviewed_mask",
        "status",
        "scan_status",
        "shape",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "source_mask_voxels",
        "reviewed_mask_voxels",
        "message",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def itksnap_command(viewer: Path | str, image: Path, mask: Path) -> list[str]:
    return [str(viewer), "-g", str(image), "-s", str(mask)]


def format_command(cmd: list[str]) -> str:
    return " ".join(f"'{part}'" if " " in part else part for part in cmd)


def wait_for_next_case(process: subprocess.Popen[Any], prompt: str) -> None:
    """Wait for Enter, or fall back to waiting for ITK-SNAP to close.

    ``conda run`` may not provide interactive stdin in every terminal setup.
    In that case ``input`` raises EOFError; waiting on the viewer process keeps
    the one-case-at-a-time queue usable.
    """
    try:
        input(prompt)
    except EOFError:
        print(
            "No interactive stdin is available; after saving the mask, close "
            "this ITK-SNAP window to open the next queued case."
        )
        process.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open LYS RatLesNetV2 T2w images with editable lesion-mask copies "
            "for manual correction in ITK-SNAP."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="source folder containing T2w/ and masks/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="reviewed source folder; default is <source-root>_reviewed next to source",
    )
    parser.add_argument("--scan-subdir", default=DEFAULT_SCAN_SUBDIR)
    parser.add_argument("--mask-subdir", default=DEFAULT_MASK_SUBDIR)
    parser.add_argument("--mask-glob", default=DEFAULT_MASK_GLOB)
    parser.add_argument("--mask-suffix", default=DEFAULT_MASK_SUFFIX)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="case substring to queue; pass multiple times for multiple filters",
    )
    parser.add_argument("--limit", type=int, default=None, help="maximum number of queued cases")
    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help="zero-based index into the discovered queue after filtering",
    )
    parser.add_argument("--viewer", type=Path, default=None, help="path to ITK-SNAP executable")
    parser.add_argument(
        "--copy-scans",
        action="store_true",
        help="copy scans into output-root instead of creating symlinks",
    )
    parser.add_argument(
        "--overwrite-reviewed",
        action="store_true",
        help="replace existing reviewed masks with the original source masks",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip cases that already have a reviewed mask",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="create reviewed copies and manifest but do not launch ITK-SNAP",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the queue and print commands without writing files or launching a viewer",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="launch queued cases without waiting for Enter between cases",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="queue manifest path; default is <output-root>/manifests/mask_review_queue.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = args.source_root.expanduser()
    output_root = (
        args.output_root.expanduser() if args.output_root else default_output_root(source_root)
    )
    manifest = (
        args.manifest.expanduser()
        if args.manifest
        else output_root / "manifests" / "mask_review_queue.csv"
    )
    cases = find_cases(
        source_root=source_root,
        output_root=output_root,
        scan_subdir=args.scan_subdir,
        mask_subdir=args.mask_subdir,
        filters=args.case,
        mask_glob=args.mask_glob,
        mask_suffix=args.mask_suffix,
    )
    cases = cases[max(args.start_at, 0) :]
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    if not cases:
        print("No matching lesion masks found.", file=sys.stderr)
        return 1

    viewer: Path | None = args.viewer.expanduser() if args.viewer else discover_itksnap()
    if viewer is None and not args.dry_run and not args.prepare_only:
        print(
            "Could not find ITK-SNAP. Install it, add it to PATH, or pass "
            "--viewer /path/to/executable.",
            file=sys.stderr,
        )
        print("Use --dry-run to print commands or --prepare-only to create reviewed copies.")
        return 1

    records: list[dict[str, Any]] = []
    launchable: list[ReviewCase] = []
    for case in cases:
        try:
            if args.dry_run:
                validate_grid(case.source_image, case.source_mask)
                record = _record(case, status="dry_run", scan_status="", message="")
                launchable.append(case)
            else:
                record = prepare_case(
                    case,
                    copy_scans=args.copy_scans,
                    overwrite_reviewed=args.overwrite_reviewed,
                    skip_existing=args.skip_existing,
                )
                if record["status"] != "skipped_existing_reviewed":
                    launchable.append(case)
        except Exception as exc:
            record = _record(case, status="failed", scan_status="", message=str(exc))
        records.append(record)
        print(f"{record['status']:>26}  {case.case_id}")
        if record["message"]:
            print(f"  {record['message']}")

    if args.dry_run:
        print("\nmanifest: not written in --dry-run mode")
    else:
        write_manifest(records, manifest)
        print(f"\nmanifest: {manifest}")
        print(f"reviewed source root: {output_root}")

    failed = [record for record in records if record["status"] == "failed"]
    if failed:
        print(f"failed cases: {len(failed)}", file=sys.stderr)

    if args.prepare_only:
        return 1 if failed else 0

    viewer_for_print: Path | str = viewer or "ITK-SNAP"
    for index, case in enumerate(launchable, start=1):
        image = case.reviewed_image if not args.dry_run else case.source_image
        mask = case.reviewed_mask if not args.dry_run else case.source_mask
        cmd = itksnap_command(viewer_for_print, image, mask)
        print(f"\n[{index}/{len(launchable)}] {case.case_id}")
        print(format_command(cmd))
        if args.dry_run:
            continue
        process = subprocess.Popen(cmd)
        if not args.no_prompt and index < len(launchable):
            wait_for_next_case(
                process,
                "After saving this mask in ITK-SNAP, press Enter to open the next case...",
            )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
