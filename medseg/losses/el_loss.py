"""Exponential Logarithmic Loss.

Faithful reimplementation of:
    Wong et al., "3D Segmentation with Exponential Logarithmic Loss
    for Highly Unbalanced Object Sizes", MICCAI 2018.

Formula:
    L_exp = w_dice * L_exp_dice + w_ce * L_exp_ce

    L_exp_dice = E_class[ (-ln(dice_c))^gamma_dice ]
    L_exp_ce   = E_pixel[ w_l(x) * (-ln(p_l(x)))^gamma_ce ]

    w_l(x) = ( freq_max / freq(l) )^0.5  (frequency-balanced label weight)
    Default gamma_dice = gamma_ce = 0.3, w_dice = w_ce = 1.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("el_loss")
class ELLoss(nn.Module):
    """Exponential Logarithmic Loss (Wong et al., MICCAI 2018)."""

    def __init__(
        self,
        gamma_dice: float = 0.3,
        gamma_ce: float = 0.3,
        w_dice: float = 1.0,
        w_ce: float = 1.0,
        smooth: float = 1e-5,
        ignore_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.gamma_dice = float(gamma_dice)
        self.gamma_ce = float(gamma_ce)
        self.w_dice = float(w_dice)
        self.w_ce = float(w_ce)
        self.smooth = float(smooth)
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: (B, C, H, W) logits.  target: (B, H, W) long."""
        B, C, H, W = pred.shape
        target = target.long()
        probs = F.softmax(pred, dim=1)
        log_probs = F.log_softmax(pred, dim=1)

        # ----- L_exp_dice -----
        dice_terms = []
        with torch.no_grad():
            target_oh = F.one_hot(target.clamp_min(0), C).permute(0, 3, 1, 2).float()
        for c in range(C):
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            p_c = probs[:, c]
            t_c = target_oh[:, c]
            intersection = (p_c * t_c).sum()
            denom = p_c.sum() + t_c.sum()
            dice_c = (2.0 * intersection + self.smooth) / (denom + self.smooth)
            # clamp to (eps, 1) to avoid log(0).
            dice_c = dice_c.clamp(min=self.smooth, max=1.0)
            dice_terms.append((-torch.log(dice_c)).pow(self.gamma_dice))
        L_exp_dice = torch.stack(dice_terms).mean() if dice_terms else pred.new_zeros(())

        # ----- L_exp_ce -----
        # Per-class frequency weight: ( max_freq / freq_c )^0.5
        with torch.no_grad():
            freq = target_oh.sum(dim=(0, 2, 3))
            freq_max = freq.max().clamp_min(1.0)
            label_weight = (freq_max / freq.clamp_min(1.0)).sqrt()  # (C,)
            if self.ignore_index is not None:
                label_weight[self.ignore_index] = 0.0

        log_pt = log_probs.gather(1, target.clamp_min(0).unsqueeze(1)).squeeze(1)
        # (-log p_l(x))^gamma_ce
        ce_term = (-log_pt).clamp_min(self.smooth).pow(self.gamma_ce)
        w_pixel = label_weight[target.clamp_min(0)]
        if self.ignore_index is not None:
            valid = (target != self.ignore_index).float()
            denom = valid.sum().clamp_min(1.0)
            L_exp_ce = (w_pixel * ce_term * valid).sum() / denom
        else:
            L_exp_ce = (w_pixel * ce_term).mean()

        return self.w_dice * L_exp_dice + self.w_ce * L_exp_ce
