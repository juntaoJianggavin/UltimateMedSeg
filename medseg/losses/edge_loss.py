"""Edge-aware Loss.

This file implements an edge-weighted cross-entropy loss commonly used as
an auxiliary objective in medical image segmentation (e.g. ET-Net, ScribFormer).

We do **not** claim correspondence to a single named paper; the formulation
follows the standard pattern:

    L_edge = mean( w(x) * CE(pred(x), target(x)) )
    w(x)   = 1 + (edge_weight - 1) * edge_map(x)

The edge map is the absolute Sobel response on the GT one-hot foreground
mask (per-class), reduced across classes via maximum.  Compared with the
previous Laplacian-on-target-index formulation this is well-defined for
multi-class segmentation and matches what most "edge loss" implementations
in popular frameworks (MONAI, Meffort/ET-Net, BCDU-Net) actually do.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("edge")
class EdgeLoss(nn.Module):
    """Edge-weighted cross-entropy loss.

    Args:
        edge_weight: weight applied to pixels lying on the GT boundary
                     (must be > 1.0 to upweight edges).  Defaults to 5.0.
        ignore_index: target class to ignore in CE.
    """

    def __init__(
        self,
        edge_weight: float = 5.0,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__()
        assert edge_weight >= 1.0
        self.edge_weight = float(edge_weight)
        self.ignore_index = ignore_index
        # Sobel kernels.
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [0.0, 0.0, 0.0],
                                [1.0, 2.0, 1.0]])
        kernel = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)  # (2,1,3,3)
        self.register_buffer("sobel_kernel", kernel)

    def _edge_map(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Per-pixel edge map in [0, 1] derived from a one-hot GT."""
        with torch.no_grad():
            oh = F.one_hot(target.clamp_min(0), num_classes).permute(0, 3, 1, 2).float()
            B, C, H, W = oh.shape
            sobel = self.sobel_kernel.to(target.device)
            # apply Sobel per class, then take L2 magnitude and max across classes.
            grad = F.conv2d(
                oh.reshape(B * C, 1, H, W), sobel, padding=1
            )  # (B*C, 2, H, W)
            mag = grad.pow(2).sum(dim=1, keepdim=True).sqrt()  # (B*C, 1, H, W)
            mag = mag.view(B, C, H, W).max(dim=1).values         # (B, H, W)
            mag = (mag > 0.5).float()
        return mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        edge = self._edge_map(target, pred.shape[1])
        weight = 1.0 + (self.edge_weight - 1.0) * edge

        ce = F.cross_entropy(
            pred, target, reduction="none", ignore_index=self.ignore_index
        )
        if self.ignore_index is not None:
            valid = (target != self.ignore_index).float()
            denom = valid.sum().clamp_min(1.0)
            return (ce * weight * valid).sum() / denom
        return (ce * weight).mean()
