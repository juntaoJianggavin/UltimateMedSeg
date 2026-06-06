"""Normalized Surface Dice Loss (continuous DT-based surrogate).

NSD is a *metric* (Nikolov et al., 2018): the fraction of GT/pred surface
points whose distance to the other surface is within a user-defined
tolerance.  As a metric it is non-differentiable, so we cannot use it
directly as a loss.

This module implements the standard differentiable surrogate widely
used in the literature (e.g. nnU-Net, MONAI's ``GeneralizedSurfaceLoss``):

    L_NSD = (1 / |Omega|) * sum_x [ s_p(x) * DT_q^tol(x) + s_q(x) * DT_p^tol(x) ]

where
    s_p(x) = soft surface map of prediction = |grad p|_1 (channel-wise)
    s_q(x) = surface map of GT (one-hot)
    DT^tol = max(DT - tol, 0), i.e. distance beyond the acceptance band.

So pixels inside the tolerance contribute zero, exactly mirroring the
NSD metric's tolerance behaviour, while keeping the loss differentiable
through s_p (which depends on softmax probabilities).
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


def _np_dt(mask: np.ndarray) -> np.ndarray:
    if _HAS_SCIPY:
        if mask.any():
            return _edt(~mask).astype(np.float32)
        return np.full(mask.shape, float(mask.shape[-1] + mask.shape[-2]),
                       dtype=np.float32)
    H, W = mask.shape
    big = float(H + W)
    out = np.full((H, W), big, dtype=np.float32)
    out[mask] = 0.0
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


def _surface_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Soft surface map of a (B, H, W) probability tensor via abs-grad."""
    dx = torch.zeros_like(mask)
    dy = torch.zeros_like(mask)
    dx[..., :, 1:] = (mask[..., :, 1:] - mask[..., :, :-1]).abs()
    dy[..., 1:, :] = (mask[..., 1:, :] - mask[..., :-1, :]).abs()
    return (dx + dy).clamp(0.0, 1.0)


def _dt_to_surface(mask_bool: torch.Tensor) -> torch.Tensor:
    """DT to the surface of a binary GT mask, computed on CPU."""
    arr = mask_bool.detach().cpu().numpy()
    out = np.zeros_like(arr, dtype=np.float32)
    for b in range(arr.shape[0]):
        m = arr[b]
        # surface = boundary pixels of mask (xor with 1-pixel erosion)
        if m.any() and not m.all():
            shifted_x = np.zeros_like(m)
            shifted_x[:, 1:] = m[:, :-1]
            shifted_y = np.zeros_like(m)
            shifted_y[1:, :] = m[:-1, :]
            surface = m & ~(m & shifted_x & shifted_y)
            out[b] = _np_dt(surface)
        else:
            out[b] = 0.0
    return torch.from_numpy(out).to(mask_bool.device)


@LOSS_REGISTRY.register("nsd")
class NSDLoss(nn.Module):
    """Differentiable NSD-style surface loss.

    Args:
        tolerance: distance (in pixels) within which surface mismatches
                   are considered acceptable.
        idc: classes included in the loss (default foreground only).
        ignore_index: optional class to skip.
    """

    def __init__(
        self,
        tolerance: float = 1.0,
        idc: Optional[Sequence[int]] = None,
        ignore_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.tolerance = float(tolerance)
        self.idc = list(idc) if idc is not None else None
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
            p = probs[:, c]
            q = target_oh[:, c]

            s_p = _surface_from_mask(p)
            with torch.no_grad():
                s_q = _surface_from_mask(q)
                dt_q = _dt_to_surface(q.bool())
                dt_p = _dt_to_surface((p > 0.5))
                tol_dt_q = (dt_q - self.tolerance).clamp_min(0.0)
                tol_dt_p = (dt_p - self.tolerance).clamp_min(0.0)

            term1 = (s_p * tol_dt_q).mean()
            term2 = (s_q * tol_dt_p).mean()
            total = total + 0.5 * (term1 + term2)
        return total / len(idc)
