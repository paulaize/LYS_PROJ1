
# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path

import tifffile  as tiff
from skimage import exposure


# Doesn't seem to work for now
# supposed to be an interactive image viewer
# import napari
# viewer = napari.Viewer()
# viewer.add_image(image)

"""
I want to do a whole work flow to analyse MRI images and IHC (immuhistochemistry) images. This whole workflow should be in python. The idea of this analyse is to calculate the volume of lesions caused by ischemic strokes in mouse brains, see with the IHCs what is where (as in where are the Neuron, the antibody of the drug that is being developed, etc.) [the immunostaining being used: Podo, NeuroTrace, Anti-IgG{the antibod}, IBA1, GFAP], and also to be able to detect the center of the lesion and the peripheral area.
What is had in mind for this workflow is that:

- both the MRI and IHC should be able to match and be well aligned with a brain atlas (i.e. Allen Mouse Brain Atlas)
- The size of the volume of the lesion should be calculated for each MRI (every MRI has an associated IHC that is then analyzed in a second time and can be used for putting information together to have a better picture)
    - This should be done automatically as much as possible from the NIfTi images that i have from my transfer from bruker.
- The IHC should be correctly colored and then analysed to find:
    - La quantité de LYS241 dans les neurones (neurotrace) et dans les cellules endotheliales (podocalyxin) . Puis dans les astrocytes (GFAP) et la microglie (Iba1)
    - Quantification du signal dans le core, la région périlesionnelles et en contra latéral à tous les temps et pour tous les types cellualires
    - % surface positive LYS241 (FITC); Densité DAPI
    - ipsilatéral vs contralatéral 
- L'idée c'est de pouvoir superpositionner les IRMs et les IHCs facilement pour pouvoir en tirer le plus possible

For your context:
- My MRI images are in Bruker format (and I already have something to convert them to NIfTI format)
- My IHC images are .vsi files
- The people i work with who previously did the analyses by hand use ImageJ (and more specifically Fiji) and are now also using QuPath for the IHC image analysis
- I don't want to change there pipeline of analysis too much so how can I do all of this but while using as much as possible QuPath and ImageJ

I want a full plan of how to guide this automation process of analysis and qantification. Before giving a detailed plan i want you to ask me some questions so that you can better guide your response.
Please give me different options at every step. For exemple to calculate the volume of lesions I want multiple porposals of how I could do so etc.
"""


MARKERS = {
    "CY5": {
        "name" : "Podocalyxine",
        "colour" : "Y"
    },
    "CY3": {
        "name" : "NeuroTrace",
        "colour" : "R"
    },
    "FITC": {
        "name" : "LYS241", # Anti-IgG
        "colour" : "G"
    },
    "DAPI": {
        "name" : "DAPI",
        "colour" : "B"
    }
}

# Maps colour letter to RGB channel indices.
_COLOUR_TO_RGB = {
    "R": [0],
    "G": [1],
    "B": [2],
    "Y": [0, 1],
    "C": [1, 2],
    "M": [0, 2],
    "W": [0, 1, 2],
}


def fiji_auto_limits(img: np.ndarray, n_bins: int = 256) -> tuple[float, float]:
    """
    Compute display min/max using Fiji's Auto Brightness/Contrast algorithm.

    Fiji ignores any histogram bin whose count exceeds total_pixels/10
    (almost always the background peak in fluorescence images), then walks
    inward from both ends to find the first bin whose count exceeds
    total_pixels/5000 — those become the display limits.

    This adapts independently to each channel's own signal distribution.
    """
    flat = img.ravel().astype(np.float64)
    n_pixels = flat.size

    hist, edges = np.histogram(flat, bins=n_bins)

    limit = n_pixels / 10          # bins dominated by background — skip
    threshold = max(1.0, n_pixels / 5000.0)  # minimum count to be real signal

    hmin = edges[0]
    for i, count in enumerate(hist):
        if count <= limit and count >= threshold:
            hmin = edges[i]
            break

    hmax = edges[-1]
    for i in range(len(hist) - 1, -1, -1):
        if hist[i] <= limit and hist[i] >= threshold:
            hmax = edges[i + 1]
            break

    if hmax <= hmin:   # fallback: uniform or featureless image
        hmin = float(np.min(flat))
        hmax = float(np.max(flat))

    return float(hmin), float(hmax)


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Auto-contrast stretch using Fiji's algorithm, independent per image.
    For visualization only — do not use for quantification.
    """
    low, high = fiji_auto_limits(img)
    if high <= low:
        return np.zeros_like(img, dtype=np.float32)
    return exposure.rescale_intensity(
        img.astype(np.float32),
        in_range=(low, high),
        out_range=(0.0, 1.0),
    ).clip(0.0, 1.0).astype(np.float32)


def apply_pseudocolor(img: np.ndarray, colour: str) -> np.ndarray:
    """
    Convert a normalised 2-D image to an RGB array using the pseudo-colour
    defined in _COLOUR_TO_RGB.  Returns shape (H, W, 3) float32.
    """
    norm = normalize_for_display(img)
    rgb = np.zeros((*norm.shape, 3), dtype=np.float32)
    for idx in _COLOUR_TO_RGB.get(colour.upper(), []):
        rgb[:, :, idx] = norm
    return rgb


def make_composite(markers_data: dict) -> np.ndarray:
    """
    Merge all channels into a single RGB image using max-blending per channel,
    equivalent to Fiji's Image > Color > Merge Channels.
    Each channel contributes its pseudo-colour after auto-contrast stretch.
    """
    available = [v for v in markers_data.values() if "image" in v]
    if not available:
        raise ValueError("No images to merge.")

    shape = available[0]["image"].shape
    rgb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

    for entry in available:
        norm = normalize_for_display(entry["image"])
        for idx in _COLOUR_TO_RGB.get(entry["info"]["colour"].upper(), []):
            rgb[:, :, idx] = np.maximum(rgb[:, :, idx], norm)

    return np.clip(rgb, 0, 1)


def save_channel_panel(
    markers_data: dict,
    out_path: Path,
    title: str = "",
) -> None:
    """
    Single PNG with Fiji-style layout:
      - Top row : individual channels, each auto-contrasted and pseudo-coloured.
      - Bottom  : merged composite (all channels overlaid via max-blend).
    """
    keys = list(markers_data.keys())
    n = len(keys)

    composite = make_composite(markers_data)

    # GridSpec: top row has n equal columns; bottom spans all of them.
    fig = plt.figure(figsize=(4.5 * n, 9))
    gs = GridSpec(2, n, figure=fig, hspace=0.25, wspace=0.05)

    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        entry = markers_data[key]
        rgb = apply_pseudocolor(entry["image"], entry["info"]["colour"])
        ax.imshow(rgb)
        ax.set_title(f"{key}\n{entry['info']['name']}", fontsize=9)
        ax.axis("off")

    ax_merge = fig.add_subplot(gs[1, :])
    ax_merge.imshow(composite)
    ax_merge.set_title("Merge", fontsize=10)
    ax_merge.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved panel PNG: {out_path}")


def read_image(path: Path) -> np.ndarray:
    """
    Read a TIFF image as a numpy array (2D image)

    To do:
        --> Verify that there is no important losses in quality
    """
    img = tiff.imread(path)
    img = np.asarray(img)

    # Remove 1 Z dimension if it is ther (eg. (1, Y, X) -> (Y, X))
    img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(
            f"Expected a 2D image for {path}, but got shape {img.shape} !"
        )

    return img



def get_nd_file_data(nd_path: Path) -> dict[str, str]:
    """
    Extract all the information from the .nd file
    --> verify that this will always be necessary

    --> from this i can get the different fluo names and do my matching with the global variable
    """
    with open(nd_path, "r", encoding="utf-8") as nd_file:
        
        metadata = {
        "nd_file": str(nd_path),
        "nd_stem": nd_path.stem, # for now this is what isn't always compatible --> need to check with the people
        "nd_folder": str(nd_path.parent),
    }
        
        for line in nd_file:
            text = line.strip()

            if not text or text.__contains__("EndFile"):
                continue

            # Split only at the first comma
            #   this is enough to create the key
            #   might have to verify that it is always the case with all the nd file
            #       --> didn't find any real documentation...
            if "," not in line:
                key = line.strip().strip('"')
                metadata[key] = ""
                continue

            key, value = line.split(",", maxsplit=1)
            key = key.strip().strip('"')
            value = value.strip().strip('"')

            metadata[key] = value

    return metadata

def get_markers_from_nd(nd_data: dict[str, str]) -> dict[str, dict]:
    """
    Extract IHC markers from parsed .nd file data, match to MARKERS,
    and resolve their TIF file paths.

    Returns a dict keyed by marker name (e.g. "CY5") with:
        - "info": entry from MARKERS global
        - "path": resolved Path to the TIF file

    Fix:
        --> this doesn't work well if the names don't correct match
        --> so need to figure out with them
    """
    n_waves = int(nd_data.get("NWavelengths", 0))
    folder = Path(nd_data["nd_folder"])
    stem = nd_data["nd_stem"]

    result = {}
    for i in range(1, n_waves + 1):
        wave_raw = nd_data.get(f"WaveName{i}", "")  # e.g. "1CY5"
        # Strip leading digits to get the marker key (e.g. "CY5")
        marker_key = wave_raw.lstrip("0123456789")

        if marker_key not in MARKERS:
            continue

        tif_name = f"{stem}_w{i}{wave_raw}.TIF"
        result[marker_key] = {
            "info": MARKERS[marker_key],
            "path": folder / tif_name,
        }

    return result




def main() -> None:
    example_nd_path = Path.home() / "Desktop/LYS/Braindistrib_IHC/24h/x5/ipsi/lesion core and periph/24h3.nd"

    nd_data = get_nd_file_data(example_nd_path)
    markers = get_markers_from_nd(nd_data)

    # Load each channel image into the markers dict.
    for key, entry in markers.items():
        path = entry["path"]
        if not path.exists():
            print(f"WARNING: file not found for {key}: {path}")
            continue
        entry["image"] = read_image(path)
        print(f"Loaded {key} ({entry['info']['name']})  from  {path.name}")

    # Keep only channels that were successfully loaded.
    loaded = {k: v for k, v in markers.items() if "image" in v}

    if not loaded:
        print("No images could be loaded — check the paths above.")
        return

    out_path = example_nd_path.parent / f"{example_nd_path.stem}_channel_panel.png"
    save_channel_panel(loaded, out_path, title=example_nd_path.stem)


if __name__ == "__main__":
    main()
