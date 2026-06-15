"""
Initial IHC image QC pipeline for MetaMorph-style .nd + multi-channel TIFF folders.

What this script does:
1. Recursively scans a root folder.
2. Parses .nd metadata files.
3. Groups TIFF channels belonging to the same acquisition.
4. Reads each channel image.
5. Exports:
   - nd_metadata.csv
   - acquisition_inventory.csv
   - channel_intensity_stats.csv
   - QC composite PNGs
   - optional single-channel panel PNGs

Expected TIFF naming pattern:
    48h5_w11CY5.TIF
    48h5_w21CY3.TIF
    48h5_w31FITC.TIF
    48h5_w41DAPI.TIF

The important part is:
    <base>_w<number><channel>.TIF

Examples:
    base = 48h5
    channel = CY5, CY3, FITC, DAPI

Author: starter script for IHC lesion-zone QC
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi


# =============================================================================
# USER SETTINGS
# =============================================================================

# Fluorophore → biological marker mapping.
CHANNEL_TO_MARKER = {
    "CY5": "Podocalyxine",
    "CY3": "NeuroTrace",
    "FITC": "LYS241",
    "DAPI": "DAPI",
}

# Pseudo-color for each channel in the QC composite.
# Y=Yellow (R+G), R=Red, G=Green, B=Blue.
# This does NOT affect quantification, only PNG visualization.
CHANNEL_COLORS = {
    "CY5": "Y",
    "CY3": "R",
    "FITC": "G",
    "DAPI": "B",
}

# Maps color letter(s) to RGB array indices.
_COLOR_TO_RGB_IDX = {
    "R": [0],
    "G": [1],
    "B": [2],
    "Y": [0, 1],
    "C": [1, 2],
    "M": [0, 2],
    "W": [0, 1, 2],
}

# Percentile display stretch for QC images.
# This is for visualization only.
DISPLAY_PERCENTILES = (1, 99.8)

# File extensions to consider as TIFF images.
TIFF_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_nd_file(nd_path: Path) -> Dict[str, str]:
    """
    Parse a MetaMorph .nd file.

    Your example:
        "NDInfoFile", Version 2.0
        "Description", Multi Dimensions Experiment
        "NWavelengths", 4
        "WaveName1", "1CY5"
        ...

    Returns a dictionary of metadata.
    """
    metadata = {
        "nd_file": str(nd_path),
        "nd_stem": nd_path.stem,
        "nd_folder": str(nd_path.parent),
    }

    with open(nd_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line == '"EndFile"':
                continue

            # Split only at the first comma.
            if "," not in line:
                key = line.strip().strip('"')
                metadata[key] = ""
                continue

            key, value = line.split(",", maxsplit=1)
            key = key.strip().strip('"')
            value = value.strip().strip('"')

            metadata[key] = value

    return metadata


def clean_wave_name(wave_name: str) -> str:
    """
    Convert MetaMorph wave name like:
        1CY5 -> CY5
        1CY3 -> CY3
        1FITC -> FITC
        1DAPI -> DAPI

    Keeps unknown names but removes leading digits.
    """
    if wave_name is None:
        return ""

    wave_name = str(wave_name).strip().strip('"')
    wave_name = re.sub(r"^\d+", "", wave_name)
    return wave_name.upper()


def parse_tif_name(tif_path: Path) -> Optional[Dict[str, str]]:
    """
    Parse TIFF name such as:
        48h5_w11CY5.TIF
        48h5_w21CY3.TIF
        48h5_w31FITC.TIF
        48h5_w41DAPI.TIF

    Returns:
        {
            "base": "48h5",
            "w_index": "11",
            "channel_raw": "CY5",
            "channel": "CY5"
        }

    If the file does not match, returns None.
    """
    name = tif_path.name

    # Flexible regex:
    # base can contain many characters, then _w, then digits, then channel letters/numbers.
    pattern = re.compile(
        r"^(?P<base>.+?)_w(?P<w_index>\d+)(?P<channel>[A-Za-z0-9]+)\.(tif|tiff)$",
        re.IGNORECASE,
    )

    match = pattern.match(name)
    if not match:
        return None

    channel_raw = match.group("channel")
    channel = channel_raw.upper()

    return {
        "base": match.group("base"),
        "w_index": match.group("w_index"),
        "channel_raw": channel_raw,
        "channel": channel,
    }


# =============================================================================
# IMAGE FUNCTIONS
# =============================================================================

def read_image(path: Path) -> np.ndarray:
    """
    Read a TIFF image as a numpy array.

    If image has extra singleton dimensions, squeeze them.
    """
    img = tiff.imread(path)
    img = np.asarray(img)

    # Remove singleton dimensions, e.g. (1, Y, X) -> (Y, X)
    img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(
            f"Expected a 2D image for {path}, but got shape {img.shape}. "
            "This script is written for single-plane channel TIFFs."
        )

    return img


def normalize_for_display(
    img: np.ndarray,
    p_low: float = 1,
    p_high: float = 99.8,
) -> np.ndarray:
    """
    Percentile-normalize image to 0-1 for display.

    Important:
        Use this only for visualization.
        Do not use percentile-normalized images for quantification.
    """
    img = np.asarray(img)

    low, high = np.percentile(img, (p_low, p_high))

    if high <= low:
        return np.zeros_like(img, dtype=np.float32)

    out = exposure.rescale_intensity(
        img,
        in_range=(low, high),
        out_range=(0, 1),
    )

    return out.astype(np.float32)


"""
just gives empty things:

python code/test2.py --root "/Users/paul-andreaslaize/Desktop/LYS/Braindistrib_IHC/24h/X5/ipsi/lesion core and periph" --out "/Users/paul-andreaslaize/Desktop/LYS/Braindistrib_IHC/24h/qc_output/X5_ipsi_lesion_core_and_periph"

"""

def make_rgb_composite(
    images_by_channel: Dict[str, np.ndarray],
    channel_colors: Dict[str, str],
    display_percentiles: Tuple[float, float] = (1, 99.8),
) -> np.ndarray:
    """
    Make an RGB composite using per-channel pseudo-colors.

    channel_colors maps channel name to a color letter:
        Y=Yellow (R+G), R=Red, G=Green, B=Blue, C=Cyan, M=Magenta, W=White.

    Channels sharing an RGB slot are max-blended (no clipping of weaker signal).
    """
    available = list(images_by_channel.values())
    if not available:
        raise ValueError("No images available to make composite.")

    shape = available[0].shape
    rgb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

    for channel, img in images_by_channel.items():
        color = channel_colors.get(channel, "").upper()
        indices = _COLOR_TO_RGB_IDX.get(color)
        if not indices:
            continue

        norm = normalize_for_display(
            img,
            p_low=display_percentiles[0],
            p_high=display_percentiles[1],
        )

        for idx in indices:
            rgb[:, :, idx] = np.maximum(rgb[:, :, idx], norm)

    return np.clip(rgb, 0, 1)


def save_composite_png(
    rgb: np.ndarray,
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    """
    Save RGB composite as PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    if title:
        plt.title(title, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_channel_panel_png(
    images_by_channel: Dict[str, np.ndarray],
    out_path: Path,
    title: Optional[str] = None,
    display_percentiles: Tuple[float, float] = (1, 99.8),
) -> None:
    """
    Save a simple panel showing each channel separately.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    channels = sorted(images_by_channel.keys())

    if len(channels) == 0:
        return

    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, channel in zip(axes, channels):
        img = images_by_channel[channel]
        norm = normalize_for_display(
            img,
            p_low=display_percentiles[0],
            p_high=display_percentiles[1],
        )

        marker = CHANNEL_TO_MARKER.get(channel, "")
        label = channel if not marker else f"{channel}\n{marker}"

        ax.imshow(norm, cmap="gray")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =============================================================================
# BASIC QUANTIFICATION FUNCTIONS
# =============================================================================

def image_intensity_stats(img: np.ndarray) -> Dict[str, float]:
    """
    Basic whole-image intensity summary.

    These are useful for QC, not final biological conclusions.
    """
    img_float = img.astype(np.float64)

    stats = {
        "height_px": img.shape[0],
        "width_px": img.shape[1],
        "dtype": str(img.dtype),
        "min": float(np.min(img_float)),
        "max": float(np.max(img_float)),
        "mean": float(np.mean(img_float)),
        "median": float(np.median(img_float)),
        "std": float(np.std(img_float)),
        "p01": float(np.percentile(img_float, 1)),
        "p05": float(np.percentile(img_float, 5)),
        "p50": float(np.percentile(img_float, 50)),
        "p95": float(np.percentile(img_float, 95)),
        "p99": float(np.percentile(img_float, 99)),
        "p99_8": float(np.percentile(img_float, 99.8)),
    }

    return stats


def make_simple_dapi_tissue_mask(
    dapi_img: np.ndarray,
    min_object_size_px: int = 500,
    dilation_px: int = 8,
) -> np.ndarray:
    """
    Very simple tissue mask from DAPI.

    This is only for rough QC. It is NOT a final lesion mask.

    Logic:
    1. Otsu threshold DAPI.
    2. Fill holes.
    3. Dilate a bit.
    4. Remove tiny objects.

    Works best if DAPI signal covers the tissue region reasonably well.
    """
    img = dapi_img.astype(np.float32)

    # If image is blank or nearly blank, return empty mask.
    if np.max(img) <= np.min(img):
        return np.zeros_like(dapi_img, dtype=bool)

    try:
        thresh = threshold_otsu(img)
    except ValueError:
        return np.zeros_like(dapi_img, dtype=bool)

    mask = img > thresh

    # Fill holes and dilate to connect nuclear regions into tissue-like area.
    mask = ndi.binary_fill_holes(mask)

    if dilation_px > 0:
        structure = ndi.generate_binary_structure(2, 1)
        mask = ndi.binary_dilation(mask, structure=structure, iterations=dilation_px)

    # Label and remove small objects.
    labels, nlab = ndi.label(mask)
    if nlab == 0:
        return mask.astype(bool)

    sizes = ndi.sum(mask, labels, index=np.arange(1, nlab + 1))
    keep_labels = np.where(sizes >= min_object_size_px)[0] + 1
    cleaned = np.isin(labels, keep_labels)

    return cleaned.astype(bool)


def masked_intensity_stats(img: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Intensity stats inside a binary mask.
    """
    values = img[mask]

    if values.size == 0:
        return {
            "mask_area_px": 0,
            "masked_mean": np.nan,
            "masked_median": np.nan,
            "masked_integrated_density": np.nan,
            "masked_p95": np.nan,
            "masked_p99": np.nan,
        }

    values = values.astype(np.float64)

    return {
        "mask_area_px": int(mask.sum()),
        "masked_mean": float(np.mean(values)),
        "masked_median": float(np.median(values)),
        "masked_integrated_density": float(np.sum(values)),
        "masked_p95": float(np.percentile(values, 95)),
        "masked_p99": float(np.percentile(values, 99)),
    }


# =============================================================================
# DISCOVERY / GROUPING
# =============================================================================

def find_nd_files(root: Path) -> List[Path]:
    """
    Find all .nd files recursively.
    """
    return sorted(root.rglob("*.nd")) + sorted(root.rglob("*.ND"))


def find_tif_files(root: Path) -> List[Path]:
    """
    Find all TIFF files recursively.
    """
    files = []
    for ext in TIFF_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(set(files))


def build_acquisition_groups(root: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Group TIFFs by folder and base name.

    Returns dict keyed by:
        (folder_path_string, base)

    Each value:
        {
            "folder": Path,
            "base": str,
            "channels": {
                "CY5": Path(...),
                "CY3": Path(...),
                ...
            },
            "w_indices": {
                "CY5": "11",
                ...
            }
        }
    """
    groups = {}

    for tif_path in find_tif_files(root):
        parsed = parse_tif_name(tif_path)
        if parsed is None:
            continue

        folder = tif_path.parent
        base = parsed["base"]
        channel = parsed["channel"]
        w_index = parsed["w_index"]

        key = (str(folder), base)

        if key not in groups:
            groups[key] = {
                "folder": folder,
                "base": base,
                "channels": {},
                "w_indices": {},
            }

        groups[key]["channels"][channel] = tif_path
        groups[key]["w_indices"][channel] = w_index

    return groups


def associate_nd_metadata(
    acquisition: Dict,
    nd_metadata_by_folder: Dict[str, List[Dict[str, str]]],
) -> Dict[str, str]:
    """
    Associate .nd metadata to an acquisition.

    Preferred:
        .nd stem matches TIFF base.

    Fallback:
        If only one .nd file exists in the same folder, use it.

    If neither works:
        return empty metadata.
    """
    folder_str = str(acquisition["folder"])
    base = acquisition["base"]

    nd_list = nd_metadata_by_folder.get(folder_str, [])

    if not nd_list:
        return {}

    # Exact stem match.
    for nd_meta in nd_list:
        if nd_meta.get("nd_stem") == base:
            return nd_meta

    # Fallback: only one .nd in folder.
    if len(nd_list) == 1:
        return nd_list[0]

    # Fallback: no clear match.
    return {}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_root(root: Path, out: Path) -> None:
    """
    Main processing function.
    """
    root = root.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    composite_dir = out / "qc_composites"
    panel_dir = out / "qc_channel_panels"
    mask_dir = out / "qc_dapi_tissue_masks"

    print(f"Scanning root: {root}")
    print(f"Output folder: {out}")

    # -------------------------------------------------------------------------
    # Parse .nd files
    # -------------------------------------------------------------------------
    nd_files = find_nd_files(root)

    nd_rows = []
    nd_metadata_by_folder: Dict[str, List[Dict[str, str]]] = {}

    for nd_path in nd_files:
        meta = parse_nd_file(nd_path)
        nd_rows.append(meta)
        nd_metadata_by_folder.setdefault(str(nd_path.parent), []).append(meta)

    nd_df = pd.DataFrame(nd_rows)
    nd_csv = out / "nd_metadata.csv"
    nd_df.to_csv(nd_csv, index=False)

    print(f"Found {len(nd_files)} .nd files")
    print(f"Wrote: {nd_csv}")

    # -------------------------------------------------------------------------
    # Group TIFF acquisitions
    # -------------------------------------------------------------------------
    acquisitions = build_acquisition_groups(root)
    print(f"Found {len(acquisitions)} TIFF acquisition groups")

    inventory_rows = []
    stats_rows = []
    masked_stats_rows = []

    for idx, ((folder_str, base), acq) in enumerate(sorted(acquisitions.items()), start=1):
        folder = acq["folder"]
        channels = acq["channels"]

        rel_folder = folder.relative_to(root)
        rel_folder_str = str(rel_folder)

        nd_meta = associate_nd_metadata(acq, nd_metadata_by_folder)

        # Read images.
        images_by_channel = {}
        read_errors = []

        for channel, path in sorted(channels.items()):
            try:
                images_by_channel[channel] = read_image(path)
            except Exception as e:
                read_errors.append(f"{channel}: {e}")

        if not images_by_channel:
            print(f"[{idx}] Skipping {base}: no readable images")
            continue

        # Check that all channels have matching shapes.
        shapes = {channel: img.shape for channel, img in images_by_channel.items()}
        unique_shapes = set(shapes.values())
        shape_match = len(unique_shapes) == 1

        # Inventory row.
        inv = {
            "acquisition_id": base,
            "relative_folder": rel_folder_str,
            "absolute_folder": str(folder),
            "n_channels_found": len(channels),
            "channels_found": ";".join(sorted(channels.keys())),
            "shape_match": shape_match,
            "image_shape": str(next(iter(unique_shapes))) if unique_shapes else "",
            "read_errors": " | ".join(read_errors),
            "associated_nd_file": nd_meta.get("nd_file", ""),
            "nd_NWavelengths": nd_meta.get("NWavelengths", ""),
            "nd_DoZSeries": nd_meta.get("DoZSeries", ""),
            "nd_DoWave": nd_meta.get("DoWave", ""),
            "nd_StartTime1": nd_meta.get("StartTime1", ""),
        }

        # Add file paths by channel.
        for channel, path in sorted(channels.items()):
            inv[f"{channel}_file"] = str(path)
            inv[f"{channel}_marker_guess"] = CHANNEL_TO_MARKER.get(channel, "")

        inventory_rows.append(inv)

        # Channel stats.
        for channel, img in sorted(images_by_channel.items()):
            row = {
                "acquisition_id": base,
                "relative_folder": rel_folder_str,
                "channel": channel,
                "marker_guess": CHANNEL_TO_MARKER.get(channel, ""),
                "file": str(channels[channel]),
            }
            row.update(image_intensity_stats(img))
            stats_rows.append(row)

        # Make QC composite.
        try:
            rgb = make_rgb_composite(
                images_by_channel,
                channel_colors=CHANNEL_COLORS,
                display_percentiles=DISPLAY_PERCENTILES,
            )

            composite_name = safe_output_name(rel_folder_str, base, suffix="composite.png")
            composite_path = composite_dir / composite_name

            title = f"{base} | {rel_folder_str}"
            save_composite_png(rgb, composite_path, title=title)

        except Exception as e:
            print(f"[{idx}] Could not create composite for {base}: {e}")

        # Make single-channel panel.
        try:
            panel_name = safe_output_name(rel_folder_str, base, suffix="channels.png")
            panel_path = panel_dir / panel_name

            title = f"{base} | {rel_folder_str}"
            save_channel_panel_png(
                images_by_channel,
                panel_path,
                title=title,
                display_percentiles=DISPLAY_PERCENTILES,
            )

        except Exception as e:
            print(f"[{idx}] Could not create channel panel for {base}: {e}")

        # Make rough DAPI tissue mask if DAPI exists.
        # This is only rough QC, not final ROI segmentation.
        if "DAPI" in images_by_channel:
            try:
                dapi_mask = make_simple_dapi_tissue_mask(images_by_channel["DAPI"])

                mask_name = safe_output_name(rel_folder_str, base, suffix="dapi_tissue_mask.tif")
                mask_path = mask_dir / mask_name
                mask_path.parent.mkdir(parents=True, exist_ok=True)

                tiff.imwrite(mask_path, (dapi_mask.astype(np.uint8) * 255))

                # Masked stats for all channels inside rough tissue mask.
                for channel, img in sorted(images_by_channel.items()):
                    row = {
                        "acquisition_id": base,
                        "relative_folder": rel_folder_str,
                        "channel": channel,
                        "marker_guess": CHANNEL_TO_MARKER.get(channel, ""),
                        "mask_type": "rough_dapi_tissue_mask",
                        "mask_file": str(mask_path),
                    }
                    row.update(masked_intensity_stats(img, dapi_mask))
                    masked_stats_rows.append(row)

            except Exception as e:
                print(f"[{idx}] Could not create DAPI tissue mask for {base}: {e}")

        print(
            f"[{idx}/{len(acquisitions)}] {base} | "
            f"{rel_folder_str} | channels: {sorted(channels.keys())}"
        )

    # -------------------------------------------------------------------------
    # Save CSVs
    # -------------------------------------------------------------------------
    inventory_df = pd.DataFrame(inventory_rows)
    stats_df = pd.DataFrame(stats_rows)
    masked_stats_df = pd.DataFrame(masked_stats_rows)

    inventory_csv = out / "acquisition_inventory.csv"
    stats_csv = out / "channel_intensity_stats.csv"
    masked_stats_csv = out / "rough_dapi_masked_intensity_stats.csv"

    inventory_df.to_csv(inventory_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    masked_stats_df.to_csv(masked_stats_csv, index=False)

    print("\nDone.")
    print(f"Wrote: {inventory_csv}")
    print(f"Wrote: {stats_csv}")
    print(f"Wrote: {masked_stats_csv}")
    print(f"QC composites: {composite_dir}")
    print(f"QC channel panels: {panel_dir}")
    print(f"Rough DAPI masks: {mask_dir}")


def safe_output_name(relative_folder: str, base: str, suffix: str) -> str:
    """
    Make a filesystem-safe output filename that preserves folder information.

    Example:
        X10/contra/lesion core + 48h5 + composite.png
    becomes:
        X10__contra__lesion_core__48h5__composite.png
    """
    folder_part = relative_folder.replace("/", "__").replace("\\", "__")
    folder_part = re.sub(r"\s+", "_", folder_part)
    folder_part = re.sub(r"[^A-Za-z0-9_.-]+", "_", folder_part)

    base_part = re.sub(r"\s+", "_", base)
    base_part = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_part)

    return f"{folder_part}__{base_part}__{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initial QC and inventory for MetaMorph .nd + TIFF IHC images."
    )

    parser.add_argument(
        "--root",
        required=True,
        type=str,
        help="Root folder containing X5/X10 or other IHC image folders.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output folder for CSVs and QC PNGs.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    process_root(root, out)


if __name__ == "__main__":
    main()