"""Gated CRF Loss for weakly-supervised segmentation.

Obukhov et al., BMVC 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("gated_crf")
class GatedCRFLoss(nn.Module):
    """Gated CRF Loss for weakly-supervised segmentation.

    Obukhov et al., "Gated CRF Loss for Weakly Supervised Semantic Image
    Segmentation", BMVC 2019.
    """

    def __init__(self, crf_weight: float = 1.0, kernel_size: int = 3,
                 dilation: int = 1, sigma_rgb: float = 15.0,
                 sigma_xy: float = 100.0, **kwargs):
        super().__init__()
        self.crf_weight = crf_weight
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy

    def _compute_pairwise_potentials(self, images, predictions):
        B, C_img, H, W = images.shape
        prob = F.softmax(predictions, dim=1)
        half_k = max(self.kernel_size // 2, 1)
        d = max(self.dilation, 1)
        total = predictions.new_zeros(())
        count = 0
        for dy in range(-half_k, half_k + 1):
            for dx in range(-half_k, half_k + 1):
                if dx == 0 and dy == 0:
                    continue
                sdy, sdx = dy * d, dx * d
                xy_gate = float(np.exp(-(sdx * sdx + sdy * sdy) / (2.0 * self.sigma_xy ** 2)))
                src_y = slice(max(0, -sdy), H - max(0, sdy))
                dst_y = slice(max(0, sdy), H - max(0, -sdy))
                src_x = slice(max(0, -sdx), W - max(0, sdx))
                dst_x = slice(max(0, sdx), W - max(0, -sdx))
                img_diff = (images[:, :, src_y, src_x] - images[:, :, dst_y, dst_x]).pow(2).sum(dim=1)
                rgb_gate = torch.exp(-img_diff / (2.0 * self.sigma_rgb ** 2))
                pred_diff = (prob[:, :, src_y, src_x] - prob[:, :, dst_y, dst_x]).pow(2).sum(dim=1)
                total = total + xy_gate * (rgb_gate * pred_diff).mean()
                count += 1
        return total / max(count, 1)

    def forward(self, predictions: torch.Tensor, images: torch.Tensor,
                labeled_loss: Optional[torch.Tensor] = None) -> torch.Tensor:
        crf_loss = self._compute_pairwise_potentials(images, predictions)
        total_loss = self.crf_weight * crf_loss
        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss
