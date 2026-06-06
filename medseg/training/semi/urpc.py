"""URPC: Uncertainty Rectified Pyramid Consistency for semi-supervised
medical image segmentation.

Luo et al., MIA 2022.
Reference: https://github.com/HiLab-git/SSL4MIS (train_la_urpc)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight,
)


class URPC(BaseSemiMethod):
    """Uncertainty Rectified Pyramid Consistency for semi-supervised
    medical image segmentation.

    Faithful to official SSL4MIS implementation (train_la_urpc.py):
        - Multi-scale predictions via input downsampling:
            x_s = F.interpolate(x, scale=1/2^s), then upsample prediction back
        - softmax_mse consistency between each scale's softmax and mean softmax
          (official uses softmax_mse_loss, NOT KL divergence)
        - Uncertainty from prediction entropy:
            uncertainty = -sum(p * log(p))
            uncertainty_weight = exp(-uncertainty)
        - Consistency weighted by certainty (low uncertainty = high weight)

    Reference:
        Luo et al., "Uncertainty Rectified Pyramid Consistency for
        Semi-Supervised Medical Image Segmentation", MIA 2022.
        Official: https://github.com/HiLab-git/SSL4MIS (train_la_urpc)
        Paper: https://arxiv.org/abs/2012.07042

    Args:
        num_scales: Number of pyramid scales (official default: 4).
        consistency_weight: Weight for pyramid consistency loss (official: 0.1).
        ema_decay: EMA decay for teacher.
        rampup_epochs: Epochs to ramp up.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 num_scales: int = 4,
                 consistency_weight: float = 0.1,
                 ema_decay: float = 0.999,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.num_scales = num_scales
        self.ema_decay = ema_decay
        self.teacher = None

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)

    def _pyramid_forward(self, model, x):
        """Generate predictions at multiple scales via input downsampling.

        Official: downsample INPUT then predict, upsample OUTPUT back.
        """
        pred = model(x)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        _, _, H, W = pred.shape

        preds = [pred]
        for s in range(1, self.num_scales):
            scale = 1.0 / (2 ** s)
            sh, sw = max(1, int(H * scale)), max(1, int(W * scale))
            x_small = F.interpolate(x, size=(sh, sw), mode='bilinear',
                                    align_corners=False)
            p_small = model(x_small)
            if isinstance(p_small, (list, tuple)):
                p_small = p_small[0]
            # Upsample back to original resolution
            p_up = F.interpolate(p_small, size=(H, W), mode='bilinear',
                                 align_corners=False)
            preds.append(p_up)
        return preds

    def train_step(self, labeled_batch, unlabeled_batch, criterion, optimizer,
                   epoch, total_epochs):
        self.model.train()
        self.teacher.eval()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # ---- Supervised: multi-scale CE loss ----
        preds_l = self._pyramid_forward(self.model, images_l)
        sup_loss = sum(criterion(p, labels) for p in preds_l) / self.num_scales

        # ---- Unlabeled: pyramid consistency with uncertainty weighting ----
        # Student multi-scale predictions
        preds_u = self._pyramid_forward(self.model, images_u)

        # Official: softmax at each scale, then mean softmax as reference
        preds_soft = [F.softmax(p, dim=1) for p in preds_u]
        mean_soft = torch.stack(preds_soft).mean(dim=0)  # (B, C, H, W)

        # Uncertainty from mean prediction entropy
        # Official: uncertainty = -sum(p * log(p)), weight = exp(-uncertainty)
        entropy = -(mean_soft * (mean_soft + 1e-8).log()).sum(dim=1)  # (B, H, W)
        uncertainty_weight = torch.exp(-entropy)  # (B, H, W)

        # Official: softmax_mse_loss (MSE between each scale's softmax and mean)
        # NOT KL divergence — official uses MSE on softmax outputs
        consistency_loss = 0.0
        for p_soft in preds_soft:
            # sum over classes, mean over spatial
            mse_per_pixel = ((p_soft - mean_soft) ** 2).sum(dim=1)  # (B, H, W)
            consistency_loss = consistency_loss + (
                mse_per_pixel * uncertainty_weight).mean()
        consistency_loss = consistency_loss / self.num_scales

        w = get_current_consistency_weight(epoch, self.consistency_weight,
                                           self.rampup_epochs)
        total_loss = sup_loss + w * consistency_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": consistency_loss.item(),
            "w": w,
            "avg_uncertainty": entropy.mean().item(),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
