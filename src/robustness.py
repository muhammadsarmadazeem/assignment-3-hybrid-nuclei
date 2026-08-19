"""Extra-credit robustness: trace a corrupted image through mask → features → narrative."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from .classical import region_table, summarise_table
from .config import DATA_DIR, TABLE_DIR, ensure_dirs
from .data import load_rgb, to_grayscale_256
from .figures import fig_robustness
from .pipeline import features_from_mask, llm_record_from_features
from .train_eval import predict_mask


def load_corrupted(name: str) -> np.ndarray:
    path = DATA_DIR / "test_corrupted" / "images" / name
    rgb = np.array(Image.open(path).convert("RGB"))
    return to_grayscale_256(rgb)


def trace_corruption(model, device, clean_id: str = "test_000", corrupt_name: str = "test_000_blur.png", skip_llm: bool = False) -> Dict:
    """Compare clean vs heavily blurred image at image / mask / feature / narrative stages."""
    ensure_dirs()
    clean_gray = to_grayscale_256(load_rgb(clean_id))
    corr_gray = load_corrupted(corrupt_name)

    clean_mask = predict_mask(model, clean_gray, device)
    corr_mask = predict_mask(model, corr_gray, device)

    clean_stats = features_from_mask(clean_id, clean_gray, clean_mask)
    corr_stats = features_from_mask(clean_id + "_blur", corr_gray, corr_mask)

    # earliest detectable stage: intensity statistics of the pixels themselves
    image_delta = {
        "mean_intensity_clean": float(clean_gray.mean()),
        "mean_intensity_corr": float(corr_gray.mean()),
        "std_intensity_clean": float(clean_gray.std()),
        "std_intensity_corr": float(corr_gray.std()),
        "laplacian_var_clean": float(_laplacian_var(clean_gray)),
        "laplacian_var_corr": float(_laplacian_var(corr_gray)),
    }

    mask_delta = {
        "n_objects_clean": int(clean_stats["n_objects"]),
        "n_objects_corr": int(corr_stats["n_objects"]),
        "mean_area_clean": float(clean_stats["mean_area"]),
        "mean_area_corr": float(corr_stats["mean_area"]),
        "area_fraction_clean": float(clean_stats["area_fraction"]),
        "area_fraction_corr": float(corr_stats["area_fraction"]),
        "mask_pixel_agreement": float(np.mean(clean_mask == corr_mask)),
    }

    if skip_llm:
        clean_llm = {"narrative": clean_stats["feature_text"]}
        corr_llm = {"narrative": corr_stats["feature_text"]}
    else:
        clean_llm = llm_record_from_features(clean_stats, cache_key=f"robust_clean_{clean_id}")
        corr_llm = llm_record_from_features(corr_stats, cache_key=f"robust_corr_{corrupt_name}")

    # Detectability: blur shows up first as collapsed high-frequency energy
    earliest = "image (Laplacian variance / contrast)"
    if image_delta["laplacian_var_corr"] >= 0.7 * image_delta["laplacian_var_clean"]:
        earliest = "U-Net mask (object count / area fraction)"
        if mask_delta["n_objects_corr"] == mask_delta["n_objects_clean"]:
            earliest = "feature table or narrative"

    fig_robustness(
        clean_gray,
        corr_gray,
        clean_mask,
        corr_mask,
        title=f"Robustness: {clean_id} vs {corrupt_name}",
    )

    out = {
        "clean_id": clean_id,
        "corrupt_name": corrupt_name,
        "earliest_detectable_stage": earliest,
        "image_delta": image_delta,
        "mask_delta": mask_delta,
        "clean_json": {k: clean_llm.get(k) for k in ("n_objects", "mean_area", "density_class", "quality_flag", "narrative")},
        "corr_json": {k: corr_llm.get(k) for k in ("n_objects", "mean_area", "density_class", "quality_flag", "narrative")},
    }
    (TABLE_DIR / "robustness.json").write_text(json.dumps(out, indent=2))
    return out


def _laplacian_var(gray: np.ndarray) -> float:
    """Simple sharpness proxy: variance of a 3×3 Laplacian."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray.astype(np.float32)
    pad = np.pad(g, 1, mode="edge")
    acc = np.zeros_like(g)
    for i in range(3):
        for j in range(3):
            acc += k[i, j] * pad[i : i + g.shape[0], j : j + g.shape[1]]
    return float(acc.var())
