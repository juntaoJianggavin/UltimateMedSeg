"""RKD: Relational Knowledge Distillation (CVPR 2019).

Paper: https://arxiv.org/abs/1904.05068
Reference implementation (not copied): https://github.com/lenscloth/RKD

Algorithm summary
-----------------
Park et al. argue that knowledge lives in the *structure* of the
embedding space rather than in individual outputs. They distil two
relational potentials over a mini-batch of embeddings:

  - distance-wise: pairwise L2 distances between every pair (i, j),
    normalised by the batch's mean non-zero distance for scale
    invariance, then matched with Huber/SmoothL1 between teacher and
    student;
  - angle-wise: for every triplet (i, j, k), take the cosine of the
    angle at vertex j formed by (e_i - e_j) and (e_k - e_j); match
    with SmoothL1.

For dense segmentation features (B, C, H, W) we collapse spatial
positions by global average pool to one embedding per image, then apply
the original formulation.

Default scalar weights from the paper: w_d = 25, w_a = 50.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


def _pdist(e: torch.Tensor, squared: bool = False, eps: float = 1e-12) -> torch.Tensor:
    """Pairwise L2 distance matrix for embeddings of shape (N, D)."""
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    if not squared:
        res = res.sqrt()
    # Zero out the diagonal (self-distance) explicitly.
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


@LOSS_REGISTRY.register("rkd")
class RKDLoss(nn.Module):
    """Distance + Angle relational KD on per-image embeddings.

    Args:
        w_dist: weight on the distance-wise term (paper default 25).
        w_angle: weight on the angle-wise term (paper default 50).
        pool: how to reduce (B, C, H, W) features to (B, C):
            'gap' (default) for global average pool, 'flatten' to use
            (B*H*W, C) embeddings (much heavier, only safe at low res).
    """

    def __init__(
        self,
        w_dist: float = 25.0,
        w_angle: float = 50.0,
        pool: str = 'gap',
        **kwargs,
    ):
        super().__init__()
        self.w_dist = float(w_dist)
        self.w_angle = float(w_angle)
        if pool not in ('gap', 'flatten'):
            raise ValueError(f"pool must be 'gap' or 'flatten', got {pool}")
        self.pool = pool

    @staticmethod
    def _to_embeddings(x: torch.Tensor, mode: str) -> torch.Tensor:
        if x.dim() == 4:
            if mode == 'gap':
                return x.mean(dim=(2, 3))  # (B, C)
            # flatten: (B, C, H, W) -> (B*H*W, C)
            B, C, H, W = x.shape
            return x.permute(0, 2, 3, 1).reshape(-1, C)
        return x  # already (N, D)

    def _distance_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_d = _pdist(t, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td.clamp(min=1e-12)
        d = _pdist(s, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d.clamp(min=1e-12)
        return F.smooth_l1_loss(d, t_d, reduction='mean')

    def _angle_loss(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Triplet cosine: cos(angle at j) between (i-j) and (k-j).
        with torch.no_grad():
            td = t.unsqueeze(0) - t.unsqueeze(1)              # (N, N, D)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        sd = s.unsqueeze(0) - s.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        return F.smooth_l1_loss(s_angle, t_angle, reduction='mean')

    def forward(self, feat_S, feat_T) -> torch.Tensor:
        """
        Args:
            feat_S / feat_T: student / teacher features (B, C, H, W) or
                already-pooled (N, D). Also accepts a list (use last).
        """
        if isinstance(feat_S, (list, tuple)):
            feat_S = feat_S[-1] if feat_S else None
        if isinstance(feat_T, (list, tuple)):
            feat_T = feat_T[-1] if feat_T else None
        if feat_S is None or feat_T is None:
            return torch.tensor(0.0)

        s = self._to_embeddings(feat_S, self.pool)
        t = self._to_embeddings(feat_T, self.pool).detach()

        # RKD needs N >= 2 for distance, N >= 3 for non-degenerate angle.
        if s.shape[0] < 2:
            return s.sum() * 0.0  # keep grad graph

        loss = torch.tensor(0.0, device=s.device)
        if self.w_dist > 0:
            loss = loss + self.w_dist * self._distance_loss(s, t)
        if self.w_angle > 0:
            loss = loss + self.w_angle * self._angle_loss(s, t)
        return loss
