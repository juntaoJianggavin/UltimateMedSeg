"""Generalised Wasserstein Dice Loss.

Faithful reimplementation of:
    Fidon et al., "Generalised Wasserstein Dice Score for Imbalanced Multi-class
    Segmentation using Holistic Convolutional Networks", BrainLes/MICCAI 2017.
    Reference code: https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss

Per-pixel Wasserstein distance to a one-hot label using a class-distance
matrix M (shape C x C, ``M[i, j]`` = distance between class i and class j):

    WD(p, t)[x] = sum_l p_x[l] * M[l, t[x]]

The generalised Wasserstein Dice score is then

    GWD = (2 * sum_x (V[x] - WD(p, t)[x]))
          / (2 * sum_x V[x] - sum_x WD(p, t)[x] - sum_x WD_b(p, t)[x])

with V[x] = max_l M[l, t[x]] (paper Eq. 8) and WD_b being the Wasserstein
distance between a 100% wrong prediction and the label.  Loss = 1 - GWD.

The default distance matrix is M = 1 - I (the standard "trivial" choice
used in the official repo when no domain-specific matrix is supplied);
users can pass any symmetric matrix via the ``M`` argument.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("wasserstein_dice")
class WassersteinDiceLoss(nn.Module):
    """Generalised Wasserstein Dice loss (Fidon et al., 2017)."""

    def __init__(
        self,
        M: Optional[Sequence[Sequence[float]]] = None,
        smooth: float = 1e-6,
        ignore_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_index = ignore_index
        self._M = M  # validated lazily once num_classes is known

    def _get_distance_matrix(self, num_classes: int, device, dtype) -> torch.Tensor:
        if self._M is None:
            M = 1.0 - torch.eye(num_classes, device=device, dtype=dtype)
        else:
            M = torch.as_tensor(self._M, device=device, dtype=dtype)
            assert M.shape == (num_classes, num_classes), (
                f"Distance matrix M must be (C,C)={num_classes}x{num_classes}, "
                f"got {tuple(M.shape)}"
            )
        return M

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: (B, C, H, W) logits.  target: (B, H, W) long.

        Implementation follows Fidon 2017 / MONAI's GeneralizedWassersteinDiceLoss:
            GWDS = 2 * TP / (2 * TP + FN + FP), with
              TP = sum_{x in FG} (V[x] - WD[x])
              FN = sum_{x in FG} WD[x]
              FP = sum_{x in BG} WD[x]
        """
        B, C, H, W = pred.shape
        target = target.long()
        probs = F.softmax(pred, dim=1)

        M = self._get_distance_matrix(C, pred.device, pred.dtype)

        # M_t[b, l, h, w] = M[l, target[b, h, w]]
        target_clamped = target.clamp_min(0)
        M_t = M[:, target_clamped]                        # (C, B, H, W)
        M_t = M_t.permute(1, 0, 2, 3).contiguous()        # (B, C, H, W)

        WD = (probs * M_t).sum(dim=1)                     # (B, H, W)
        V = M_t.max(dim=1).values                         # (B, H, W)

        # Foreground / background masks (FG = target != 0).
        fg_mask = (target != 0).float()
        bg_mask = 1.0 - fg_mask
        if self.ignore_index is not None:
            valid = (target != self.ignore_index).float()
            fg_mask = fg_mask * valid
            bg_mask = bg_mask * valid

        tp = ((V - WD) * fg_mask).sum()
        fn = (WD * fg_mask).sum()
        fp = (WD * bg_mask).sum()

        gwds = (2.0 * tp + self.smooth) / (2.0 * tp + fn + fp + self.smooth)
        return 1.0 - gwds
