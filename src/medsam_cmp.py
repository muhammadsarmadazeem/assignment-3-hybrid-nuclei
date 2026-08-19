"""Extra credit: compare the trained U-Net with pretrained MedSAM (ViT-B).

MedSAM is a promptable foundation model (Ma et al., 2024). It expects a
bounding-box prompt, so it is *not* a drop-in automatic segmenter. We therefore
evaluate two automatic protocols that never look at the ground-truth mask:

1. A single box covering the whole 256×256 field.
2. One box per Otsu connected component, masks OR-ed together.

Protocol (2) is the fairer multi-nuclei setting; (1) shows what happens if
you treat MedSAM like the U-Net (one forward pass, no object prompts).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.measure import regionprops

from .classical import labeled_objects, otsu_segment
from .config import TABLE_DIR, ensure_dirs
from .data import list_image_ids, load_mask, load_rgb, to_grayscale_256
from .unet_model import hard_dice, hard_iou


MEDSAM_ID = "flaviagiammarino/medsam-vit-base"


def _boxes_from_otsu(gray: np.ndarray, pad: int = 2) -> List[List[float]]:
    """Axis-aligned boxes around Otsu components (no ground truth)."""
    lab = labeled_objects(otsu_segment(gray))
    h, w = gray.shape
    boxes = []
    for rp in regionprops(lab):
        minr, minc, maxr, maxc = rp.bbox
        boxes.append(
            [
                float(max(minc - pad, 0)),
                float(max(minr - pad, 0)),
                float(min(maxc + pad, w - 1)),
                float(min(maxr + pad, h - 1)),
            ]
        )
    if not boxes:
        boxes = [[0.0, 0.0, float(w - 1), float(h - 1)]]
    return boxes


def load_medsam(device: torch.device):
    """Load MedSAM from Hugging Face. Raises if transformers/weights are missing."""
    from transformers import SamModel, SamProcessor

    model = SamModel.from_pretrained(MEDSAM_ID)
    processor = SamProcessor.from_pretrained(MEDSAM_ID)
    model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def predict_medsam(model, processor, rgb: np.ndarray, boxes: List[List[float]], device) -> np.ndarray:
    """Binary mask from one or more boxes. Boxes are [x0, y0, x1, y1] in pixel coords."""
    image = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    acc = np.zeros(rgb.shape[:2], dtype=np.uint8)
    # SAM processors take one image and a list of boxes.
    inputs = processor(image, input_boxes=[boxes], return_tensors="pt")
    moved = {}
    for k, v in inputs.items():
        if not hasattr(v, "to"):
            moved[k] = v
            continue
        if v.dtype == torch.float64:
            v = v.float()
        # keep integer size tensors as long; only send floats/bools to MPS as float32
        moved[k] = v.to(device)
    inputs = moved
    outputs = model(**inputs, multimask_output=False)
    # post_process_masks with binarize=True expects *logits* (threshold 0).
    # Passing sigmoid() first makes every pixel > 0 and yields an all-foreground mask.
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
        binarize=True,
    )
    m = masks[0].numpy()
    # squeeze to (n_boxes, H, W)
    while m.ndim > 3:
        m = m.squeeze(1)
    if m.ndim == 2:
        acc = (m > 0).astype(np.uint8)
    else:
        acc = (m.max(axis=0) > 0).astype(np.uint8)
    if acc.shape != rgb.shape[:2]:
        acc = np.array(Image.fromarray(acc * 255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 0
        acc = acc.astype(np.uint8)
    return acc


def compare_medsam_on_val(unet_model, device, n_panel: int = 3) -> Dict:
    """Mean Dice/IoU of MedSAM vs the trained U-Net on the validation split."""
    ensure_dirs()
    medsam, processor = load_medsam(device)
    from .train_eval import predict_mask

    rows = []
    panels = []
    panel_ids = ["val_000", "val_001", "val_004"]
    for image_id in list_image_ids("val"):
        rgb = load_rgb(image_id)
        gray = to_grayscale_256(rgb)
        gt = torch.from_numpy(load_mask(image_id).astype(np.float32)).unsqueeze(0)
        h, w = gray.shape
        full_box = [[0.0, 0.0, float(w - 1), float(h - 1)]]
        otsu_boxes = _boxes_from_otsu(gray)

        m_full = predict_medsam(medsam, processor, rgb, full_box, device)
        m_boxes = predict_medsam(medsam, processor, rgb, otsu_boxes, device)
        unet = predict_mask(unet_model, gray, device)

        def _score(pred: np.ndarray) -> Tuple[float, float]:
            p = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0)
            return hard_dice(p, gt), hard_iou(p, gt)

        fd, fi = _score(m_full)
        bd, bi = _score(m_boxes)
        ud, ui = _score(unet)
        rows.append(
            {
                "image_id": image_id,
                "unet_dice": ud,
                "unet_iou": ui,
                "medsam_fullbox_dice": fd,
                "medsam_fullbox_iou": fi,
                "medsam_otsubox_dice": bd,
                "medsam_otsubox_iou": bi,
                "n_otsu_boxes": len(otsu_boxes),
            }
        )
        if image_id in panel_ids and len(panels) < n_panel:
            panels.append((image_id, gray, load_mask(image_id), unet, m_boxes))
        print(
            f"[medsam] {image_id}: U-Net {ud:.3f}  full-box {fd:.3f}  otsu-box {bd:.3f}  "
            f"(n_boxes={len(otsu_boxes)})"
        )

    summary = {
        "protocol": "MedSAM ViT-B (flaviagiammarino/medsam-vit-base); no GT boxes",
        "mean_unet_dice": float(np.mean([r["unet_dice"] for r in rows])),
        "mean_unet_iou": float(np.mean([r["unet_iou"] for r in rows])),
        "mean_medsam_fullbox_dice": float(np.mean([r["medsam_fullbox_dice"] for r in rows])),
        "mean_medsam_fullbox_iou": float(np.mean([r["medsam_fullbox_iou"] for r in rows])),
        "mean_medsam_otsubox_dice": float(np.mean([r["medsam_otsubox_dice"] for r in rows])),
        "mean_medsam_otsubox_iou": float(np.mean([r["medsam_otsubox_iou"] for r in rows])),
        "per_image": rows,
    }
    (TABLE_DIR / "medsam_vs_unet.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "panels": panels}
