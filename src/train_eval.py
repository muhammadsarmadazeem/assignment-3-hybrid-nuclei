"""PyTorch dataset, training loop, validation Dice/IoU, and Otsu comparison."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import (
    BASE_CHANNELS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_DIR,
    PRIMARY_LOSS,
    SEED,
    TABLE_DIR,
    THRESHOLD,
    ensure_dirs,
)
from .data import list_image_ids, load_mask, load_rgb, to_grayscale_256
from .classical import otsu_segment
from .unet_model import LOSS_FNS, SmallUNet, hard_dice, hard_iou


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NucleiDataset(Dataset):
    """Grayscale 256×256 image / binary mask pairs, with optional flip aug."""

    def __init__(self, split: str, augment: bool = False) -> None:
        self.ids = list_image_ids(split)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        gray = to_grayscale_256(load_rgb(image_id))
        mask = load_mask(image_id).astype(np.float32)
        if mask.shape != gray.shape:
            # masks are already 256×256 in this dataset
            pass
        if self.augment:
            if random.random() < 0.5:
                gray = np.fliplr(gray).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.5:
                gray = np.flipud(gray).copy()
                mask = np.flipud(mask).copy()
        x = torch.from_numpy(gray).unsqueeze(0).float()
        y = torch.from_numpy(mask).unsqueeze(0).float()
        return x, y, image_id


def train_one_loss(
    loss_name: str,
    epochs: int = EPOCHS,
    device: torch.device | None = None,
) -> Dict:
    """Train the Lab-4 U-Net with one of {bce, dice, bce_dice}; save best-by-val-Dice."""
    ensure_dirs()
    set_seed()
    device = device or get_device()
    loss_fn = LOSS_FNS[loss_name]

    train_loader = DataLoader(
        NucleiDataset("train", augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(NucleiDataset("val", augment=False), batch_size=BATCH_SIZE)

    model = SmallUNet(in_ch=1, out_ch=1, base=BASE_CHANNELS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_dice = -1.0
    best_path = MODEL_DIR / f"unet_{loss_name}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
        train_loss = running / max(n_seen, 1)

        val_stats = evaluate_loader(model, val_loader, device, loss_fn)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_stats["loss"])
        history["val_dice"].append(val_stats["dice"])
        history["val_iou"].append(val_stats["iou"])

        if val_stats["dice"] > best_dice:
            best_dice = val_stats["dice"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "loss_name": loss_name,
                    "epoch": epoch,
                    "val_dice": best_dice,
                    "val_iou": val_stats["iou"],
                },
                best_path,
            )
        print(
            f"[{loss_name}] epoch {epoch:02d}/{epochs}  "
            f"train={train_loss:.4f}  val_loss={val_stats['loss']:.4f}  "
            f"val_dice={val_stats['dice']:.4f}  val_iou={val_stats['iou']:.4f}"
        )

    # reload best
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    per_image = per_image_metrics(model, "val", device)
    result = {
        "loss_name": loss_name,
        "best_epoch": ckpt["epoch"],
        "val_dice": ckpt["val_dice"],
        "val_iou": ckpt["val_iou"],
        "history": history,
        "per_image": per_image,
        "checkpoint": str(best_path),
    }
    (TABLE_DIR / f"unet_{loss_name}_metrics.json").write_text(json.dumps(result, indent=2, default=_json_default))
    return result


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


@torch.no_grad()
def evaluate_loader(model, loader, device, loss_fn) -> Dict[str, float]:
    model.eval()
    dices, ious, losses = [], [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        losses.append(float(loss_fn(logits, y).item()))
        for i in range(x.size(0)):
            dices.append(hard_dice(probs[i], y[i]))
            ious.append(hard_iou(probs[i], y[i]))
    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
    }


@torch.no_grad()
def per_image_metrics(model, split: str, device) -> List[Dict]:
    model.eval()
    rows = []
    ds = NucleiDataset(split, augment=False)
    for x, y, image_id in DataLoader(ds, batch_size=1):
        x, y = x.to(device), y.to(device)
        probs = torch.sigmoid(model(x))
        rows.append(
            {
                "image_id": image_id[0],
                "dice": hard_dice(probs[0], y[0]),
                "iou": hard_iou(probs[0], y[0]),
            }
        )
    return rows


@torch.no_grad()
def predict_mask(model, gray: np.ndarray, device) -> np.ndarray:
    """Return a uint8 {0,1} mask from a [0,1] grayscale array."""
    model.eval()
    x = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(device)
    probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (probs >= THRESHOLD).astype(np.uint8)


def load_trained(loss_name: str = PRIMARY_LOSS, device=None) -> SmallUNet:
    device = device or get_device()
    path = MODEL_DIR / f"unet_{loss_name}.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SmallUNet(in_ch=1, out_ch=1, base=BASE_CHANNELS).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def otsu_vs_unet_on_val(model, device) -> List[Dict]:
    """Per-image hard Dice/IoU for Otsu and U-Net on the validation split."""
    rows = []
    for image_id in list_image_ids("val"):
        gray = to_grayscale_256(load_rgb(image_id))
        gt = torch.from_numpy(load_mask(image_id).astype(np.float32)).unsqueeze(0)
        otsu = torch.from_numpy(otsu_segment(gray).astype(np.float32)).unsqueeze(0)
        unet = torch.from_numpy(predict_mask(model, gray, device).astype(np.float32)).unsqueeze(0)
        rows.append(
            {
                "image_id": image_id,
                "otsu_dice": hard_dice(otsu, gt),
                "otsu_iou": hard_iou(otsu, gt),
                "unet_dice": hard_dice(unet, gt),
                "unet_iou": hard_iou(unet, gt),
            }
        )
    return rows


def collect_val_triplets(model, device, n: int = 3) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """(id, gray, gt, pred) for n validation images spanning density regimes."""
    meta_order = ["val_000", "val_001", "val_004", "val_005"]  # sparse, normal, dense, clustered
    chosen = [i for i in meta_order if i in list_image_ids("val")][:n]
    if len(chosen) < n:
        chosen = list_image_ids("val")[:n]
    out = []
    for image_id in chosen:
        gray = to_grayscale_256(load_rgb(image_id))
        gt = load_mask(image_id)
        pred = predict_mask(model, gray, device)
        out.append((image_id, gray, gt, pred))
    return out
