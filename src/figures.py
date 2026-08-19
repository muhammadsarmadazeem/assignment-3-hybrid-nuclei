"""Publication-style figures for the report."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_DIR, ensure_dirs
from .data import load_metadata, load_rgb, to_grayscale_256, intensity_histogram, list_image_ids
from .classical import otsu_segment

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "figure.facecolor": "white",
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    ensure_dirs()
    path = FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_eda_samples() -> Path:
    """One sample per density regime (RGB + grayscale)."""
    meta = load_metadata()
    picks = []
    for dens in ("sparse", "normal", "dense", "clustered"):
        row = meta[(meta.split == "train") & (meta.density == dens)].iloc[0]
        picks.append(row)
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.05))
    for i, row in enumerate(picks):
        rgb = load_rgb(row.image_id)
        gray = to_grayscale_256(rgb)
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f"{row.image_id}\n{row.density}, n={int(row.n_objects)}", pad=6)
        axes[0, i].axis("off")
        axes[1, i].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_xlabel("grayscale" if i == 0 else "")
        axes[1, i].axis("off")
    fig.suptitle(
        "EDA: one train image per density regime (top RGB, bottom 256×256 grayscale)",
        y=0.995,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(fig, "eda_samples.png")


def fig_intensity_histogram() -> Path:
    ids = list_image_ids("train")
    counts, edges = intensity_histogram(ids, bins=64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(4.6, 2.05))
    ax.bar(centres, counts / counts.sum(), width=np.diff(edges), color="#3b6ea5", edgecolor="none")
    ax.set_xlabel("Normalised grayscale intensity")
    ax.set_ylabel("Fraction of pixels")
    ax.set_title("Train-set pooled intensity histogram (80 images)")
    ax.set_xlim(0, 1)
    return _save(fig, "eda_histogram.png")


def fig_otsu_example(image_id: str) -> Path:
    gray = to_grayscale_256(load_rgb(image_id))
    mask = otsu_segment(gray)
    fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.15))
    ax[0].imshow(gray, cmap="gray")
    ax[0].set_title(f"{image_id} grayscale")
    ax[0].axis("off")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Otsu + morphological cleanup")
    ax[1].axis("off")
    return _save(fig, f"otsu_{image_id}.png")


def fig_unet_panels(triplets) -> Path:
    """Input / GT / prediction for ≥3 validation images."""
    n = len(triplets)
    fig, axes = plt.subplots(n, 3, figsize=(5.6, 1.45 * n))
    if n == 1:
        axes = np.array([axes])
    col_titles = ["input (grayscale)", "ground-truth mask", "U-Net prediction"]
    for r, (image_id, gray, gt, pred) in enumerate(triplets):
        axes[r, 0].imshow(gray, cmap="gray")
        axes[r, 1].imshow(gt, cmap="gray")
        axes[r, 2].imshow(pred, cmap="gray")
        axes[r, 0].set_ylabel(image_id, fontsize=8)
        for c in range(3):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c])
    fig.suptitle("Validation panels: input, ground truth, U-Net (best BCE+Dice checkpoint)")
    fig.tight_layout()
    return _save(fig, "unet_val_panels.png")


def fig_loss_curves(histories: Dict[str, Dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.15))
    for name, hist in histories.items():
        axes[0].plot(hist["epoch"], hist["train_loss"], label=f"{name} train")
        axes[0].plot(hist["epoch"], hist["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(hist["epoch"], hist["val_dice"], label=name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / validation loss")
    axes[0].legend(ncol=2, fontsize=6)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Dice")
    axes[1].set_title("Validation Dice")
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, "unet_curves.png")


def fig_otsu_vs_unet(rows: List[Dict], examples: Sequence[Dict]) -> Path:
    """Scatter of val Dice plus one qualitative example per method.

    ``examples`` is a list of dicts with keys image_id, gray, gt, otsu, unet, title.
    Overlay encoding: red = method mask, green = ground truth, yellow = overlap.
    """
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.7))
    axes[0, 0].scatter(df["otsu_dice"], df["unet_dice"], s=18, c="#3b6ea5")
    lo = min(df["otsu_dice"].min(), df["unet_dice"].min()) - 0.01
    hi = 1.0
    axes[0, 0].plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    axes[0, 0].set_xlim(lo, hi)
    axes[0, 0].set_ylim(lo, hi)
    axes[0, 0].set_xlabel("Otsu Dice")
    axes[0, 0].set_ylabel("U-Net Dice")
    axes[0, 0].set_title("Val Dice: U-Net vs Otsu")
    axes[0, 0].set_aspect("equal")
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.55,
        "Row 2 overlay:\nred = method\ngreen = GT\nyellow = overlap",
        transform=axes[1, 0].transAxes,
        va="center",
        fontsize=8,
    )
    for col, ex in enumerate(examples[:2], start=1):
        overlay_otsu = np.stack([ex["otsu"], ex["gt"], np.zeros_like(ex["gt"])], axis=-1).astype(float)
        overlay_unet = np.stack([ex["unet"], ex["gt"], np.zeros_like(ex["gt"])], axis=-1).astype(float)
        axes[0, col].imshow(ex["gray"], cmap="gray")
        axes[0, col].set_title(f"{ex['image_id']}: {ex['title']}")
        axes[0, col].axis("off")
        axes[1, col].imshow(overlay_otsu if "Otsu" in ex["title"] else overlay_unet)
        axes[1, col].set_title("method vs GT overlay")
        axes[1, col].axis("off")
    fig.tight_layout()
    return _save(fig, "otsu_vs_unet.png")


def fig_robustness(clean_gray, corr_gray, clean_mask, corr_mask, title: str) -> Path:
    fig, ax = plt.subplots(2, 2, figsize=(4.4, 4.2))
    ax[0, 0].imshow(clean_gray, cmap="gray")
    ax[0, 0].set_title("Clean input")
    ax[0, 1].imshow(corr_gray, cmap="gray")
    ax[0, 1].set_title("Corrupted input")
    ax[1, 0].imshow(clean_mask, cmap="gray")
    ax[1, 0].set_title("U-Net mask (clean)")
    ax[1, 1].imshow(corr_mask, cmap="gray")
    ax[1, 1].set_title("U-Net mask (corrupted)")
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, "robustness.png")


def fig_medsam_panels(panels) -> Path:
    """gray / GT / U-Net / MedSAM (Otsu-box prompts) for a few validation images."""
    n = len(panels)
    fig, axes = plt.subplots(n, 4, figsize=(7.4, 1.9 * n))
    if n == 1:
        axes = np.array([axes])
    titles = ["input", "ground truth", "U-Net", "MedSAM (Otsu boxes)"]
    for r, (image_id, gray, gt, unet, medsam) in enumerate(panels):
        axes[r, 0].imshow(gray, cmap="gray")
        axes[r, 1].imshow(gt, cmap="gray")
        axes[r, 2].imshow(unet, cmap="gray")
        axes[r, 3].imshow(medsam, cmap="gray")
        axes[r, 0].set_ylabel(image_id, fontsize=8)
        for c in range(4):
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c])
    fig.suptitle("Foundation-model extra: MedSAM vs U-Net on validation images")
    fig.tight_layout()
    return _save(fig, "medsam_vs_unet.png")
