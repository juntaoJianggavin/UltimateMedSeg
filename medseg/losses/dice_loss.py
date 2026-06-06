"""Dice Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("dice")
class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation."""
    def __init__(self, smooth=1.0, ignore_index=None, **kwargs):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """pred: B,C,H,W  target: B,H,W (long)"""
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)

        # One-hot encode target
        target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()

        total_loss = 0.0
        count = 0
        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            p = pred_soft[:, c]
            t = target_onehot[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            total_loss += 1.0 - dice
            count += 1

        return total_loss / max(count, 1)
