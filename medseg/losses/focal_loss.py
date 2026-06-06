"""Focal Loss.

Faithful multi-class reimplementation of:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

For multi-class semantic segmentation:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the softmax probability of the true class for every pixel.
"""

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("focal")
class FocalLoss(nn.Module):
    """Focal loss for multi-class segmentation.

    Args:
        alpha: scalar or per-class list/tensor of class weights.  When a
               scalar is given it is broadcast to every class (matching
               common segmentation implementations such as MMSegmentation).
        gamma: focusing parameter (>= 0).  ``gamma=0`` reduces to weighted CE.
        ignore_index: target class to ignore (default -100).
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: Union[float, Sequence[float]] = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
        **kwargs,
    ):
        super().__init__()
        if isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor([float(alpha)]))
            self._scalar_alpha = True
        else:
            self.register_buffer(
                "alpha", torch.tensor(list(alpha), dtype=torch.float32)
            )
            self._scalar_alpha = False
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: (B, C, H, W) logits.  target: (B, H, W) long."""
        C = pred.shape[1]
        target = target.long()

        log_p = F.log_softmax(pred, dim=1)           # (B, C, H, W)
        log_pt = log_p.gather(1, target.clamp_min(0).unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()                            # (B, H, W)

        # Per-pixel alpha lookup.
        if self._scalar_alpha:
            alpha_t = self.alpha.to(pred.device).expand(C)
        else:
            alpha_t = self.alpha.to(pred.device)
            if alpha_t.numel() != C:
                # Allow alpha to be shorter than C (broadcast last value).
                pad = C - alpha_t.numel()
                if pad > 0:
                    alpha_t = torch.cat(
                        [alpha_t, alpha_t[-1:].expand(pad)], dim=0
                    )
        a_t = alpha_t[target.clamp_min(0)]            # (B, H, W)

        focal = -a_t * (1.0 - pt).pow(self.gamma) * log_pt

        if self.ignore_index is not None:
            valid = target != self.ignore_index
            focal = focal * valid.float()
            denom = valid.float().sum().clamp_min(1.0)
        else:
            denom = torch.tensor(focal.numel(), device=pred.device, dtype=focal.dtype)

        if self.reduction == "mean":
            return focal.sum() / denom
        if self.reduction == "sum":
            return focal.sum()
        return focal
