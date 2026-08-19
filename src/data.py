"""Load, grayscale-convert, and resize the synthetic nuclei dataset."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

from .config import DATA_DIR, IMAGE_SIZE, PROCESSED_DIR, ensure_dirs


def load_metadata() -> pd.DataFrame:
    """Return the ground-truth metadata table shipped with the dataset."""
    return pd.read_csv(DATA_DIR / "metadata.csv")


def list_image_ids(split: str) -> List[str]:
    image_dir = DATA_DIR / split / "images"
    return sorted(p.stem for p in image_dir.glob("*.png"))


def raw_image_path(split: str, image_id: str) -> Path:
    return DATA_DIR / split / "images" / f"{image_id}.png"


def raw_mask_path(split: str, image_id: str) -> Path:
    return DATA_DIR / split / "masks" / f"{image_id}.png"


def split_of(image_id: str) -> str:
    return image_id.split("_")[0]


def load_rgb(image_id: str) -> np.ndarray:
    """Load an RGB uint8 image (H, W, 3)."""
    path = raw_image_path(split_of(image_id), image_id)
    return np.array(Image.open(path).convert("RGB"))


def load_mask(image_id: str) -> np.ndarray:
    """Load a binary mask as uint8 {0, 1}."""
    path = raw_mask_path(split_of(image_id), image_id)
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 127).astype(np.uint8)


def to_grayscale_256(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to float grayscale in [0, 1] and resize to 256×256.

    DAPI-like images store almost all signal in the blue channel. ITU-R BT.601
    luminance (``rgb2gray``) down-weights blue (~0.07) and would crush nuclear
    contrast, so we take the blue channel when three channels are present.
    A spatial resize is applied so the pipeline matches the assignment spec
    even if a future image is not already 256×256.
    """
    if rgb.ndim == 3:
        gray = rgb[..., 2].astype(np.float32)
    else:
        gray = rgb.astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0
    if gray.shape[0] != IMAGE_SIZE or gray.shape[1] != IMAGE_SIZE:
        gray = resize(gray, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True, preserve_range=True)
        gray = gray.astype(np.float32)
    return np.clip(gray, 0.0, 1.0)


def save_processed_gray(image_id: str, gray: np.ndarray) -> Path:
    ensure_dirs()
    split = split_of(image_id)
    out_dir = PROCESSED_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{image_id}.png"
    Image.fromarray((gray * 255).astype(np.uint8)).save(path)
    return path


def preprocess_split(split: str) -> List[Path]:
    """Grayscale + 256×256 every image in a split; return saved paths."""
    paths = []
    for image_id in list_image_ids(split):
        gray = to_grayscale_256(load_rgb(image_id))
        paths.append(save_processed_gray(image_id, gray))
    return paths


def intensity_histogram(image_ids: List[str], bins: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Pooled grayscale histogram over the listed images."""
    counts = np.zeros(bins, dtype=np.float64)
    for image_id in image_ids:
        gray = to_grayscale_256(load_rgb(image_id))
        h, edges = np.histogram(gray, bins=bins, range=(0.0, 1.0))
        counts += h
    return counts, np.linspace(0.0, 1.0, bins + 1)
