"""Lovász-Softmax Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_softmax_flat(probas, labels, classes='present'):
    """Multi-class Lovász-Softmax loss."""
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.shape[1]
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            fg_class = 1.0 - probas[:, 0]
        else:
            fg_class = probas[:, c]
        errors = (fg - fg_class).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=probas.device)


@LOSS_REGISTRY.register("lovasz")
class LovaszLoss(nn.Module):
    """Lovász-Softmax loss for multi-class segmentation."""
    def __init__(self, classes='present', **kwargs):
        super().__init__()
        self.classes = classes

    def forward(self, pred, target):
        """pred: B,C,H,W  target: B,H,W"""
        probas = F.softmax(pred, dim=1)
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).reshape(-1, C)
        labels = target.reshape(-1)
        return _lovasz_softmax_flat(probas, labels, self.classes)
