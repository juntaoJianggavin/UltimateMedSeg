"""Kullback-Leibler Divergence Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("kl_divergence")
class KLDivergenceLoss(nn.Module):
    """Kullback-Leibler Divergence Loss for segmentation.
    
    Measures the difference between predicted and target probability distributions.
    Useful for knowledge distillation and uncertainty estimation in medical images.
    """
    def __init__(self, temperature=1.0, ignore_index=None, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """pred: B,C,H,W  target: B,H,W (long)"""
        num_classes = pred.shape[1]
        
        # Softmax with temperature
        pred_soft = F.softmax(pred / self.temperature, dim=1)
        
        # One-hot encode target
        target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        pred_soft = pred_soft.clamp(min=eps)
        
        # KL Divergence: sum(target * log(target / pred))
        # Since target is one-hot, this simplifies to: -sum(target * log(pred))
        kl_loss = F.kl_div(
            torch.log(pred_soft),
            target_onehot,
            reduction='batchmean'
        )
        
        return kl_loss * (self.temperature ** 2)
