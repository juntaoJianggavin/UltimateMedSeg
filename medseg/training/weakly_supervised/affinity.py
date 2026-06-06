"""Pixel Affinity Propagation Loss.

Ahn & Kwak, CVPR 2018 (AffinityNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("affinity_loss")
class AffinityLoss(nn.Module):
    """Pixel Affinity Propagation Loss.

    Ahn & Kwak, "Learning Pixel-Level Semantic Affinity with Image-Level
    Supervision", CVPR 2018 (AffinityNet).
    """

    def __init__(self, affinity_weight: float = 1.0, propagation_steps: int = 3,
                 temperature: float = 0.5, **kwargs):
        super().__init__()
        self.affinity_weight = affinity_weight
        self.propagation_steps = propagation_steps
        self.temperature = temperature

    def _compute_affinity(self, features):
        B, C, H, W = features.shape
        feat_flat = features.view(B, C, -1)
        feat_norm = F.normalize(feat_flat, dim=1)
        N = min(H * W, 1024)
        indices = torch.randperm(H * W, device=features.device)[:N]
        feat_sub = feat_norm[:, :, indices]
        affinity = torch.bmm(feat_sub.transpose(1, 2), feat_sub) / self.temperature
        return affinity, indices

    def _propagation_loss(self, predictions, affinity, indices):
        B, C_cls, H, W = predictions.shape
        prob = F.softmax(predictions, dim=1)
        prob_flat = prob.view(B, C_cls, -1)
        prob_sub = prob_flat[:, :, indices]
        pred_sim = torch.bmm(prob_sub.transpose(1, 2), prob_sub)
        affinity_target = F.softmax(affinity, dim=-1)
        pred_sim_norm = F.log_softmax(pred_sim, dim=-1)
        return F.kl_div(pred_sim_norm, affinity_target, reduction='batchmean')

    def forward(self, predictions: torch.Tensor, features: torch.Tensor,
                labeled_loss: Optional[torch.Tensor] = None) -> torch.Tensor:
        affinity, indices = self._compute_affinity(features)
        prop_loss = self._propagation_loss(predictions, affinity, indices)
        total_loss = self.affinity_weight * prop_loss
        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss
