"""Lab 4 U-Net skeleton (Ronneberger-style, 3-level) plus assignment losses.

Architecture matches Lab 4 (DoubleConv + UNet, base=16).
The assignment uses this network on 256×256 nuclei instead of the lab's 128×128
synthetic cells. Skip connections concatenate encoder maps into the matching decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU, the basic U-Net block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """3-level U-Net from Lab 4. Returns raw logits (no sigmoid)."""

    def __init__(self, in_ch=1, out_ch=1, base=16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


# Backwards-compatible name used in train_eval.py
SmallUNet = UNet


def dice_loss(logits, target, eps=1e-7):
    """Soft Dice loss in [0, 1]. Lower is better. (Lab 4)"""
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, target):
    """BCE + Dice (Lab 4)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(logits, target)


def bce_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target)


def dice_coefficient(logits, target, threshold=0.5, eps=1e-7):
    """Hard Dice for evaluation after thresholding. Higher is better. (Lab 4)"""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(logits, target, threshold=0.5, eps=1e-7):
    """Hard IoU for evaluation. (Lab 4)"""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def hard_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Binary Dice on probability/mask tensors already at eval time."""
    p = (pred > 0.5).float()
    t = (target > 0.5).float()
    inter = (p * t).sum()
    return float((2 * inter + eps) / (p.sum() + t.sum() + eps))


def hard_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    p = (pred > 0.5).float()
    t = (target > 0.5).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return float((inter + eps) / (union + eps))


LOSS_FNS = {
    "bce": bce_loss,
    "dice": dice_loss,
    "bce_dice": combined_loss,
}
