# Reference: https://github.com/hunto/DIST_KD
# Paper: https://arxiv.org/abs/2205.10536
"""DIST: Knowledge Distillation from A Stronger Teacher (NeurIPS 2022).

Per Huang et al. Eq. (8)-(9): replace KL with Pearson correlation
applied along two complementary axes of the (N, C) softmax matrix
(N = pixels for dense pred, C = classes):

  - inter-class relation: Pearson over the class dimension per pixel
    (rows), captures the relative ordering of class scores;
  - intra-class relation: Pearson over the pixel dimension per class
    (columns), captures how the score for one class varies across
    samples / pixels.

Total loss is ``beta * inter + gamma * intra``, both scaled by ``tau^2``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("dist")
class DISTLoss(nn.Module):
    """DIST: Knowledge Distillation from A Stronger Teacher (NeurIPS 2022).

    Official source: hunto/image_classification_sota.
    """

    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 1.0,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"DIST tau must be > 0, got {tau}.")
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.tau = float(tau)

    @staticmethod
    def _cosine_similarity(a, b, eps: float = 1e-8):
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)

    @classmethod
    def _pearson_correlation(cls, a, b, eps: float = 1e-8):
        return cls._cosine_similarity(
            a - a.mean(1).unsqueeze(1),
            b - b.mean(1).unsqueeze(1),
            eps,
        )

    @classmethod
    def _inter_class_relation(cls, y_s, y_t):
        return 1 - cls._pearson_correlation(y_s, y_t).mean()

    @classmethod
    def _intra_class_relation(cls, y_s, y_t):
        return cls._inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_s: student logits, (B, C, H, W) or (N, C)
            z_t: teacher logits, same shape
        """
        if z_s.dim() == 4:
            B, C, H, W = z_s.shape
            z_s = z_s.permute(0, 2, 3, 1).reshape(-1, C)
            z_t = z_t.permute(0, 2, 3, 1).reshape(-1, C)

        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = (self.tau ** 2) * self._inter_class_relation(y_s, y_t)
        intra_loss = (self.tau ** 2) * self._intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss
