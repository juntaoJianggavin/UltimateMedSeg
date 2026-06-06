"""Tversky Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("tversky")
class TverskyLoss(nn.Module):
    """Tversky loss: generalization of Dice with alpha/beta for FP/FN weighting."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        """pred: B,C,H,W  target: B,H,W"""
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()

        total_loss = 0.0
        for c in range(1, num_classes):  # skip background
            p = pred_soft[:, c].reshape(-1)
            t = target_onehot[:, c].reshape(-1)
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            total_loss += 1.0 - tversky

        return total_loss / max(num_classes - 1, 1)
