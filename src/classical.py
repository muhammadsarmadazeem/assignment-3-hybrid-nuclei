"""Classical segmentation (Otsu + morphology) and region-feature tables."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import binary_opening, disk, remove_small_objects

REGIONPROPS_PROPS = (
    "label",
    "area",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "equivalent_diameter_area",
    "perimeter",
    "axis_major_length",
    "axis_minor_length",
)


def otsu_segment(gray: np.ndarray, min_size: int = 8, opening_radius: int = 1) -> np.ndarray:
    """Binary nuclei mask from a grayscale image.

    Bright-on-dark fluorescence is assumed (DAPI-like). A morphological
    opening plus a small-object filter removes speckle without eating
    genuine nuclei (typical area ≫ 8 px after 256×256 resize).
    """
    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0
    thresh = float(threshold_otsu(gray))
    binary = gray > thresh
    binary = binary_opening(binary, footprint=disk(opening_radius))
    binary = remove_small_objects(binary, min_size=min_size)
    return binary.astype(np.uint8)


def labeled_objects(mask: np.ndarray) -> np.ndarray:
    return label(mask > 0, connectivity=2)


def region_table(gray: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Per-connected-component features used by Tasks 2 and 4."""
    lab = labeled_objects(mask)
    if lab.max() == 0:
        return pd.DataFrame(columns=list(REGIONPROPS_PROPS))
    table = regionprops_table(lab, intensity_image=gray, properties=REGIONPROPS_PROPS)
    return pd.DataFrame(table)


def summarise_table(df: pd.DataFrame, image_id: str = "") -> Dict[str, float]:
    """Scalar statistics that become the numbers-only LLM input."""
    n = int(len(df))
    if n == 0:
        return {
            "image_id": image_id,
            "n_objects": 0,
            "mean_area": 0.0,
            "median_area": 0.0,
            "std_area": 0.0,
            "mean_eccentricity": 0.0,
            "mean_solidity": 0.0,
            "mean_intensity": 0.0,
            "area_fraction": 0.0,
            "min_area": 0.0,
            "max_area": 0.0,
        }
    area = df["area"].to_numpy(dtype=np.float64)
    return {
        "image_id": image_id,
        "n_objects": n,
        "mean_area": float(area.mean()),
        "median_area": float(np.median(area)),
        "std_area": float(area.std()),
        "mean_eccentricity": float(df["eccentricity"].mean()),
        "mean_solidity": float(df["solidity"].mean()),
        "mean_intensity": float(df["mean_intensity"].mean()),
        "area_fraction": float(area.sum() / (256 * 256)),
        "min_area": float(area.min()),
        "max_area": float(area.max()),
    }


def density_class(n_objects: int, area_fraction: float = 0.0) -> str:
    """Rule-based density label used as the JSON source of truth.

    Object count is the primary cue; a high foreground fraction also marks
    *dense* even when touching nuclei collapse the connected-component count.
    """
    if n_objects >= 45 or area_fraction >= 0.15:
        return "dense"
    if n_objects < 15:
        return "sparse"
    return "normal"


def table_to_text(df: pd.DataFrame, image_id: str) -> Tuple[str, Dict[str, float]]:
    """Natural-language numbers-only summary (no pixels, no diagnoses)."""
    stats = summarise_table(df, image_id)
    if stats["n_objects"] == 0:
        text = (
            f"Image {image_id}: 0 connected components after Otsu + morphological "
            f"cleanup. All region properties are undefined."
        )
        return text, stats
    text = (
        f"Image {image_id}: {stats['n_objects']} connected components. "
        f"Area (px): mean={stats['mean_area']:.1f}, median={stats['median_area']:.1f}, "
        f"std={stats['std_area']:.1f}, min={stats['min_area']:.1f}, max={stats['max_area']:.1f}. "
        f"Mean eccentricity={stats['mean_eccentricity']:.3f}, "
        f"mean solidity={stats['mean_solidity']:.3f}, "
        f"mean intensity={stats['mean_intensity']:.3f} (0–1 grayscale). "
        f"Foreground area fraction={stats['area_fraction']:.4f}."
    )
    return text, stats
