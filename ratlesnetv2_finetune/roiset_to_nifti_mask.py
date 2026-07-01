"""Convert a Fiji/ImageJ RoiSet.zip of lesion outlines to a NIfTI mask.

The intended input is one original 3-D T2w NIfTI and one Fiji ROI Manager
``RoiSet.zip`` containing area ROIs drawn on stack slices. The output is a
binary ``*_lesion_mask.nii.gz`` file with the same array shape, affine, and
voxel spacing as the original NIfTI.
"""

from __future__ import annotations

import argparse
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from skimage.draw import ellipse, polygon

NIFTI_SUFFIX = ".nii.gz"
MASK_SUFFIX = "_lesion_mask.nii.gz"

ROI_MAGIC = b"Iout"
ROI_HEADER_SIZE = 64

TYPE_POLYGON = 0
TYPE_RECT = 1
TYPE_OVAL = 2
TYPE_LINE = 3
TYPE_FREELINE = 4
TYPE_POLYLINE = 5
TYPE_NO_ROI = 6
TYPE_FREEHAND = 7
TYPE_TRACED = 8
TYPE_ANGLE = 9
TYPE_POINT = 10

AREA_POLYGON_TYPES = {TYPE_POLYGON, TYPE_FREEHAND, TYPE_TRACED}
NON_AREA_TYPES = {
    TYPE_LINE,
    TYPE_FREELINE,
    TYPE_POLYLINE,
    TYPE_NO_ROI,
    TYPE_ANGLE,
    TYPE_POINT,
}

TYPE_NAMES = {
    TYPE_POLYGON: "polygon",
    TYPE_RECT: "rect",
    TYPE_OVAL: "oval",
    TYPE_LINE: "line",
    TYPE_FREELINE: "freeline",
    TYPE_POLYLINE: "polyline",
    TYPE_NO_ROI: "noRoi",
    TYPE_FREEHAND: "freehand",
    TYPE_TRACED: "traced",
    TYPE_ANGLE: "angle",
    TYPE_POINT: "point",
}


@dataclass(frozen=True)
class ImageJRoi:
    name: str
    roi_type: int
    left: int
    top: int
    right: int
    bottom: int
    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()
    position: int = 0
    z_position: int = 0

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.roi_type, f"unknown({self.roi_type})")


@dataclass(frozen=True)
class ConvertedRoi:
    name: str
    slice_index: int
    voxel_count: int


@dataclass(frozen=True)
class RoiSetConversionResult:
    output_path: Path
    reference_nifti: Path
    roiset_zip: Path
    image_shape: tuple[int, int, int]
    converted_rois: tuple[ConvertedRoi, ...]
    total_voxels: int


def convert_roiset_to_nifti_mask(
    *,
    nifti_path: str | Path,
    roiset_zip: str | Path,
    output_path: str | Path,
    xy_axes: tuple[int, int] = (0, 1),
    slice_source: Literal["auto", "roi-position", "filename"] = "auto",
    filename_index_base: int = 1,
    overwrite: bool = False,
) -> RoiSetConversionResult:
    """Convert all area ROIs in a Fiji ``RoiSet.zip`` to one NIfTI mask."""
    nifti_path = Path(nifti_path)
    roiset_zip = Path(roiset_zip)
    output_path = Path(output_path)
    _validate_inputs(nifti_path=nifti_path, roiset_zip=roiset_zip, output_path=output_path)
    x_axis, y_axis, z_axis = _validate_axes(xy_axes)
    _validate_slice_options(slice_source, filename_index_base)

    reference = nib.load(str(nifti_path))
    if len(reference.shape) != 3:
        raise ValueError(
            f"{nifti_path} must be a 3-D NIfTI for Fiji RoiSet conversion, "
            f"got shape {reference.shape}"
        )
    shape = tuple(int(v) for v in reference.shape)
    x_size = shape[x_axis]
    y_size = shape[y_axis]
    z_size = shape[z_axis]
    mask = np.zeros(shape, dtype=np.uint8)

    converted: list[ConvertedRoi] = []
    for roi in read_roiset_zip(roiset_zip):
        slice_index = _resolve_slice_index(
            roi,
            z_size=z_size,
            slice_source=slice_source,
            filename_index_base=filename_index_base,
        )
        xs, ys = _rasterize_area_roi(roi, x_size=x_size, y_size=y_size)
        if xs.size == 0:
            raise ValueError(f"ROI {roi.name!r} did not rasterize to any in-bounds pixels")

        before = int(mask.sum())
        index = [slice(None), slice(None), slice(None)]
        index[x_axis] = xs
        index[y_axis] = ys
        index[z_axis] = np.full(xs.shape, slice_index, dtype=np.intp)
        mask[tuple(index)] = 1
        converted.append(
            ConvertedRoi(
                name=roi.name,
                slice_index=slice_index,
                voxel_count=int(mask.sum()) - before,
            )
        )

    if not converted:
        raise ValueError(f"No .roi files found in {roiset_zip}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output mask already exists: {output_path}. Pass --overwrite.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = reference.header.copy()
    header.set_data_shape(mask.shape)
    header.set_data_dtype(np.uint8)
    if len(reference.header.get_zooms()) >= 3:
        header.set_zooms(reference.header.get_zooms()[:3])
    nib.save(nib.Nifti1Image(mask, reference.affine, header), str(output_path))

    return RoiSetConversionResult(
        output_path=output_path,
        reference_nifti=nifti_path,
        roiset_zip=roiset_zip,
        image_shape=shape,
        converted_rois=tuple(converted),
        total_voxels=int(mask.sum()),
    )


def read_roiset_zip(path: str | Path) -> list[ImageJRoi]:
    """Read all ``.roi`` entries from a Fiji ROI Manager zip file."""
    roiset_path = Path(path)
    rois: list[ImageJRoi] = []
    with zipfile.ZipFile(roiset_path) as zf:
        roi_names = sorted(name for name in zf.namelist() if name.lower().endswith(".roi"))
        for name in roi_names:
            rois.append(read_imagej_roi(zf.read(name), name=Path(name).name))
    return rois


def read_imagej_roi(data: bytes, *, name: str) -> ImageJRoi:
    """Decode one simple ImageJ/Fiji ROI file."""
    if len(data) < ROI_HEADER_SIZE:
        raise ValueError(f"ROI {name!r} is too small to contain an ImageJ ROI header")
    if data[:4] != ROI_MAGIC:
        raise ValueError(f"ROI {name!r} does not start with ImageJ ROI magic {ROI_MAGIC!r}")

    roi_type = data[6]
    top = _i16(data, 8)
    left = _i16(data, 10)
    bottom = _i16(data, 12)
    right = _i16(data, 14)
    n_coordinates = _u16(data, 16)
    shape_roi_size = _i32(data, 36)
    position = _i32(data, 56)
    header2_offset = _i32(data, 60)
    z_position = 0
    if header2_offset > 0 and header2_offset + 12 <= len(data):
        z_position = _i32(data, header2_offset + 8)

    if shape_roi_size > 0:
        raise ValueError(
            f"ROI {name!r} is an ImageJ composite/shape ROI, which is not supported. "
            "Save lesion outlines as separate polygon/freehand ROIs."
        )
    if roi_type in NON_AREA_TYPES:
        raise ValueError(
            f"ROI {name!r} has non-area type {TYPE_NAMES.get(roi_type, roi_type)!r}; "
            "lesion masks require polygon/freehand/traced/rect/oval area ROIs."
        )

    if roi_type in AREA_POLYGON_TYPES:
        if n_coordinates <= 0:
            raise ValueError(f"ROI {name!r} has no polygon coordinates")
        needed = ROI_HEADER_SIZE + 4 * n_coordinates
        if len(data) < needed:
            raise ValueError(
                f"ROI {name!r} is truncated: expected at least {needed} bytes, got {len(data)}"
            )
        x_base = ROI_HEADER_SIZE
        y_base = ROI_HEADER_SIZE + 2 * n_coordinates
        xs = tuple(left + _u16(data, x_base + 2 * i) for i in range(n_coordinates))
        ys = tuple(top + _u16(data, y_base + 2 * i) for i in range(n_coordinates))
    elif roi_type in {TYPE_RECT, TYPE_OVAL}:
        xs = ()
        ys = ()
    else:
        raise ValueError(f"ROI {name!r} has unsupported type {roi_type}")

    return ImageJRoi(
        name=name,
        roi_type=roi_type,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        x=xs,
        y=ys,
        position=position,
        z_position=z_position,
    )


def default_mask_output_path(
    nifti_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Return ``<stem>_lesion_mask.nii.gz`` for a source NIfTI path."""
    nifti_path = Path(nifti_path)
    name = nifti_path.name
    if not name.endswith(NIFTI_SUFFIX):
        raise ValueError(f"Expected a .nii.gz NIfTI path, got {nifti_path}")
    directory = Path(output_dir) if output_dir is not None else nifti_path.parent
    return directory / f"{name[: -len(NIFTI_SUFFIX)]}{MASK_SUFFIX}"


def _rasterize_area_roi(
    roi: ImageJRoi,
    *,
    x_size: int,
    y_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    shape_yx = (y_size, x_size)
    if roi.roi_type in AREA_POLYGON_TYPES:
        rows, cols = polygon(np.asarray(roi.y), np.asarray(roi.x), shape=shape_yx)
    elif roi.roi_type == TYPE_RECT:
        x0 = max(0, roi.left)
        x1 = min(x_size, roi.right)
        y0 = max(0, roi.top)
        y1 = min(y_size, roi.bottom)
        if x1 <= x0 or y1 <= y0:
            return np.asarray([], dtype=np.intp), np.asarray([], dtype=np.intp)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rows = yy.ravel()
        cols = xx.ravel()
    elif roi.roi_type == TYPE_OVAL:
        center_y = (roi.top + roi.bottom - 1) / 2.0
        center_x = (roi.left + roi.right - 1) / 2.0
        radius_y = max((roi.bottom - roi.top) / 2.0, 0.0)
        radius_x = max((roi.right - roi.left) / 2.0, 0.0)
        rows, cols = ellipse(center_y, center_x, radius_y, radius_x, shape=shape_yx)
    else:
        raise ValueError(f"Cannot rasterize unsupported ROI type {roi.type_name!r}")
    return cols.astype(np.intp, copy=False), rows.astype(np.intp, copy=False)


def _resolve_slice_index(
    roi: ImageJRoi,
    *,
    z_size: int,
    slice_source: Literal["auto", "roi-position", "filename"],
    filename_index_base: int,
) -> int:
    candidates: list[tuple[str, int]] = []
    if slice_source in {"auto", "roi-position"}:
        if roi.z_position > 0:
            candidates.append(("ROI z-position", roi.z_position - 1))
        if roi.position > 0:
            candidates.append(("ROI position", roi.position - 1))
    if slice_source in {"auto", "filename"}:
        filename_index = _slice_index_from_filename(roi.name, index_base=filename_index_base)
        if filename_index is not None:
            candidates.append(("ROI filename", filename_index))

    if not candidates:
        raise ValueError(
            f"Could not determine stack slice for ROI {roi.name!r}. "
            "Save ROIs with Fiji stack positions or use filenames containing slice numbers."
        )

    source, index = candidates[0]
    if index < 0 or index >= z_size:
        raise ValueError(
            f"{source} maps ROI {roi.name!r} to slice index {index}, "
            f"but valid indices are 0..{z_size - 1}"
        )
    return index


def _slice_index_from_filename(name: str, *, index_base: int) -> int | None:
    stem = Path(name).stem
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    return int(matches[-1]) - index_base


def _validate_inputs(*, nifti_path: Path, roiset_zip: Path, output_path: Path) -> None:
    if not nifti_path.is_file():
        raise FileNotFoundError(f"Reference NIfTI not found: {nifti_path}")
    if not nifti_path.name.endswith(NIFTI_SUFFIX):
        raise ValueError(f"Reference NIfTI must end with {NIFTI_SUFFIX}: {nifti_path}")
    if not roiset_zip.is_file():
        raise FileNotFoundError(f"RoiSet.zip not found: {roiset_zip}")
    if roiset_zip.suffix.lower() != ".zip":
        raise ValueError(f"ROI set must be a .zip file: {roiset_zip}")
    if output_path.suffix != ".gz" or not output_path.name.endswith(NIFTI_SUFFIX):
        raise ValueError(f"Output mask must end with {NIFTI_SUFFIX}: {output_path}")
    if not output_path.name.endswith(MASK_SUFFIX):
        raise ValueError(
            f"Output mask filename should end with {MASK_SUFFIX} "
            "so the dataset importer can pair it"
        )


def _validate_axes(xy_axes: tuple[int, int]) -> tuple[int, int, int]:
    if len(xy_axes) != 2 or set(xy_axes) not in ({0, 1}, {0, 2}, {1, 2}):
        raise ValueError("--xy-axes must contain two distinct axes from 0,1,2")
    z_axes = sorted({0, 1, 2} - set(xy_axes))
    return xy_axes[0], xy_axes[1], z_axes[0]


def _validate_slice_options(slice_source: str, filename_index_base: int) -> None:
    if slice_source not in {"auto", "roi-position", "filename"}:
        raise ValueError("slice_source must be one of: auto, roi-position, filename")
    if filename_index_base not in {0, 1}:
        raise ValueError("filename_index_base must be 0 or 1")


def _i16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">h", data, offset)[0]


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from(">H", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from(">i", data, offset)[0]


def _parse_xy_axes(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated axes, for example 0,1")
    try:
        axes = (int(parts[0]), int(parts[1]))
        _validate_axes(axes)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nifti", required=True, help="Original 3-D T2w NIfTI used in Fiji")
    parser.add_argument("--roiset", required=True, help="Fiji ROI Manager RoiSet.zip")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output *_lesion_mask.nii.gz path. Defaults to "
            "work/ratlesnetv2_finetune/masks/ unless --output-dir is supplied."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the default <nifti-stem>_lesion_mask.nii.gz output name",
    )
    parser.add_argument(
        "--xy-axes",
        default="0,1",
        type=_parse_xy_axes,
        help="NIfTI axes corresponding to Fiji ROI x,y pixel coordinates. Default: 0,1",
    )
    parser.add_argument(
        "--slice-source",
        choices=["auto", "roi-position", "filename"],
        default="auto",
        help="How to map .roi files to NIfTI slices. Default: ROI position, then filename.",
    )
    parser.add_argument(
        "--filename-index-base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether slice numbers parsed from filenames are zero- or one-based. Default: 1",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output mask")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(__file__).resolve().parents[1] / "work" / "ratlesnetv2_finetune" / "masks"
    )
    output = (
        Path(args.output)
        if args.output is not None
        else default_mask_output_path(args.nifti, output_dir=default_output_dir)
    )
    result = convert_roiset_to_nifti_mask(
        nifti_path=args.nifti,
        roiset_zip=args.roiset,
        output_path=output,
        xy_axes=args.xy_axes,
        slice_source=args.slice_source,
        filename_index_base=args.filename_index_base,
        overwrite=args.overwrite,
    )

    print(f"Wrote lesion mask: {result.output_path}")
    print(f"Reference shape: {result.image_shape}")
    print(f"Converted ROIs: {len(result.converted_rois)}")
    print(f"Mask voxels: {result.total_voxels}")
    for roi in result.converted_rois:
        print(f"  {roi.name}: slice={roi.slice_index} new_voxels={roi.voxel_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
