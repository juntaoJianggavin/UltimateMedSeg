"""Hausdorff Distance Loss.

Faithful reimplementation of:
    Karimi & Salcudean, "Reducing the Hausdorff Distance in Medical Image
    Segmentation with Convolutional Neural Networks", IEEE TMI 2019.

We implement the **DT-based form (HD_{DT})** that the paper recommends as
the most stable variant:

    HD_{DT} = (1 / |Omega|) * sum_x [ (p(x) - q(x))^2 * (d_p(x)^alpha + d_q(x)^alpha) ]

where p is the softmax foreground probability, q is the GT one-hot mask,
and d_p / d_q are the (unsigned) Euclidean distance transforms of p and q
respectively (with q binarized).  alpha defaults to 2.0.

Per the paper, only the GT distance map is critical for stability;
practical implementations (e.g. MONAI) often only use d_q.  We follow
the paper exactly and use both d_p and d_q.
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.ndimage import distance_transform_edt as _edt
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _edt = None
    _HAS_SCIPY = False

from medseg.registry import LOSS_REGISTRY


def _dt_per_class(mask: np.ndarray) -> np.ndarray:
    """Unsigned Euclidean DT of a 0/1 mask (or its complement when empty).

    For each foreground class c we want d(x) = distance to nearest pixel
    that *is not* class c on a one-hot map -- i.e. the Karimi formulation
    treats d_q as distance to the boundary on each side.  Following MONAI
    we compute DT on the foreground mask and on its complement and take
    the maximum, which approximates the boundary distance at every pixel.
    """
    posmask = mask.astype(bool)
    if not posmask.any() or posmask.all():
        return np.zeros_like(mask, dtype=np.float32)
    if _HAS_SCIPY:
        d_outside = _edt(~posmask)
        d_inside = _edt(posmask)
    else:
        d_outside = _bfs_dt(~posmask)
        d_inside = _bfs_dt(posmask)
    return (d_outside + d_inside).astype(np.float32)


def _bfs_dt(mask: np.ndarray) -> np.ndarray:
    """Pure-numpy fallback DT (L1-chamfer)."""
    H, W = mask.shape
    big = float(H + W)
    out = np.full((H, W), big, dtype=np.float32)
    out[mask] = 0
    if (out == big).all():
        return out
    for _ in range(int(big) + 1):
        prev = out.copy()
        out[1:, :] = np.minimum(out[1:, :], prev[:-1, :] + 1)
        out[:-1, :] = np.minimum(out[:-1, :], prev[1:, :] + 1)
        out[:, 1:] = np.minimum(out[:, 1:], prev[:, :-1] + 1)
        out[:, :-1] = np.minimum(out[:, :-1], prev[:, 1:] + 1)
        if (out < big).all():
            break
    return out


def _batch_dt(mask: torch.Tensor) -> torch.Tensor:
    """mask: (B, H, W) 0/1 tensor.  Returns (B, H, W) float DT on same device."""
    arr = mask.detach().cpu().numpy()
    out = np.zeros_like(arr, dtype=np.float32)
    for b in range(arr.shape[0]):
        out[b] = _dt_per_class(arr[b])
    return torch.from_numpy(out).to(mask.device)


@LOSS_REGISTRY.register("hausdorff")
class HausdorffLoss(nn.Module):
    """HD_{DT} loss (Karimi & Salcudean, TMI 2019)."""

    def __init__(
        self,
        alpha: float = 2.0,
        idc: Optional[Sequence[int]] = None,
        ignore_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.idc = list(idc) if idc is not None else None
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: (B, C, H, W) logits.  target: (B, H, W) long."""
        C = pred.shape[1]
        probs = F.softmax(pred, dim=1)

        if self.idc is None:
            idc = [c for c in range(1, C) if c != self.ignore_index]
        else:
            idc = [c for c in self.idc if c != self.ignore_index]
        if len(idc) == 0:
            return pred.new_zeros(())

        target_oh = F.one_hot(target.long().clamp_min(0), C).permute(0, 3, 1, 2).float()

        total = pred.new_zeros(())
        for c in idc:
            p = probs[:, c]                      # (B, H, W)
            q = target_oh[:, c]                  # (B, H, W)
            with torch.no_grad():
                d_q = _batch_dt(q.bool())
                # DT of binarised prediction (threshold at 0.5)
                d_p = _batch_dt((p > 0.5))
                weight = d_p.pow(self.alpha) + d_q.pow(self.alpha)
            err = (p - q).pow(2)
            total = total + (err * weight).mean()
        return total / len(idc)
