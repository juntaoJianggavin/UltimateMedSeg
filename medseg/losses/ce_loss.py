"""Cross Entropy Loss."""

import torch
import torch.nn as nn
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("ce")
class CELoss(nn.Module):
    """Standard Cross Entropy loss for segmentation."""
    def __init__(self, weight=None, ignore_index=-100, label_smoothing=0.0, **kwargs):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                       label_smoothing=label_smoothing)

    def forward(self, pred, target):
        """pred: B,C,H,W  target: B,H,W (long)"""
        return self.ce(pred, target)
