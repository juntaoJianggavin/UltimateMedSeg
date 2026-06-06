"""Boundary Loss.

Faithful reimplementation of:
    Kervadec et al., "Boundary loss for highly unbalanced segmentation",
    MIDL 2019 / Medical Image Analysis 2021.
    Official repo: https://github.com/LIVIAETS/boundary-loss

Formula:
    L_B(theta) = (1 / |Omega|) * sum_{p in Omega}  phi_G(p) * s_theta(p)

where
    s_theta(p) = softmax(logits)[:, foreground_class, p]
    phi_G(p)   = signed distance transform of GT, positive outside the
                 foreground (background side), negative inside.

The signed distance map is computed *per-batch on CPU with SciPy's
distance_transform_edt* (matching the official repo's `one_hot2dist`),
then transferred back to the prediction's device.  Because GT is constant,
gradients only flow through the softmax probabilities — exactly as in
the paper.
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


def _one_hot2dist(seg_onehot: np.ndarray) -> np.ndarray:
    """Compute signed distance maps from a one-hot mask.

    Args:
        seg_onehot: (K, H, W) bool/0-1 numpy array.
    Returns:
        (K, H, W) float32 SDM.  Outside the foreground => positive distance,
        inside the foreground => negative distance (matching official repo).
    """
    K = seg_onehot.shape[0]
    res = np.zeros_like(seg_onehot, dtype=np.float32)
    for k in range(K):
        posmask = seg_onehot[k].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[k] = (_edt(negmask) * negmask
                      - (_edt(posmask) - 1) * posmask)
    return res


def _fallback_sdm(seg_onehot: np.ndarray) -> np.ndarray:
    """Pure-numpy SDM fallback when SciPy is not available.

    Uses an L1 chamfer-like estimate via successive erosions.  Not as
    accurate as Euclidean DT but still produces a valid SDM with the
    correct sign convention so the loss remains usable.
    """
    K, H, W = seg_onehot.shape
    res = np.zeros((K, H, W), dtype=np.float32)
    big = float(H + W)
    for k in range(K):
        m = seg_onehot[k].astype(np.float32)
        if m.sum() == 0:
            res[k] = big
            continue
        # outside positive distance via BFS layers
        outside = 1.0 - m
        d_out = np.full_like(m, big)
        d_out[m > 0] = 0
        for step in range(1, int(big) + 1):
            prev = d_out.copy()
            d_out[1:, :] = np.minimum(d_out[1:, :], prev[:-1, :] + 1)
            d_out[:-1, :] = np.minimum(d_out[:-1, :], prev[1:, :] + 1)
            d_out[:, 1:] = np.minimum(d_out[:, 1:], prev[:, :-1] + 1)
            d_out[:, :-1] = np.minimum(d_out[:, :-1], prev[:, 1:] + 1)
            if (d_out < big).all():
                break
        # inside negative distance similarly
        d_in = np.full_like(m, big)
        d_in[m == 0] = 0
        for step in range(1, int(big) + 1):
            prev = d_in.copy()
            d_in[1:, :] = np.minimum(d_in[1:, :], prev[:-1, :] + 1)
            d_in[:-1, :] = np.minimum(d_in[:-1, :], prev[1:, :] + 1)
            d_in[:, 1:] = np.minimum(d_in[:, 1:], prev[:, :-1] + 1)
            d_in[:, :-1] = np.minimum(d_in[:, :-1], prev[:, 1:] + 1)
            if (d_in < big).all():
                break
        res[k] = d_out * outside - (d_in - (m > 0).astype(np.float32)) * m
    return res


def _compute_sdm_batch(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """target: (B, H, W) long.  Returns (B, K, H, W) float SDM tensor on CPU."""
    target_np = target.detach().cpu().numpy().astype(np.int64)
    B, H, W = target_np.shape
    sdm = np.zeros((B, num_classes, H, W), dtype=np.float32)
    sdm_fn = _one_hot2dist if _HAS_SCIPY else _fallback_sdm
    for b in range(B):
        oh = np.zeros((num_classes, H, W), dtype=np.float32)
        for k in range(num_classes):
            oh[k] = (target_np[b] == k).astype(np.float32)
        sdm[b] = sdm_fn(oh)
    return torch.from_numpy(sdm)


@LOSS_REGISTRY.register("boundary")
class BoundaryLoss(nn.Module):
    """Boundary loss (Kervadec et al., MIDL 2019).

    Args:
        idc: class indices included in the loss (default: all foreground
             classes, i.e. `range(1, num_classes)` resolved at runtime).
        ignore_index: optional class to skip (cannot appear inside ``idc``).
    """

    def __init__(
        self,
        idc: Optional[Sequence[int]] = None,
        ignore_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.idc = list(idc) if idc is not None else None
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred: (B, C, H, W) logits.  target: (B, H, W) long."""
        num_classes = pred.shape[1]
        probs = F.softmax(pred, dim=1)

        if self.idc is None:
            idc = [c for c in range(1, num_classes) if c != self.ignore_index]
        else:
            idc = [c for c in self.idc if c != self.ignore_index]
        if len(idc) == 0:
            return pred.new_zeros(())

        # SDM is computed on CPU and detached from the autograd graph.
        with torch.no_grad():
            dist_maps = _compute_sdm_batch(target, num_classes).to(pred.device)

        pc = probs[:, idc, ...]
        dc = dist_maps[:, idc, ...]
        loss = (pc * dc).mean()
        return loss
