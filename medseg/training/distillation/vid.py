"""VID: Variational Information Distillation for Knowledge Transfer (CVPR 2019).

Paper: https://arxiv.org/abs/1904.05835
Reference implementation (not copied): https://github.com/HobbitLong/RepDistiller

Algorithm summary
-----------------
Ahn et al. cast distillation as maximising a tractable variational
lower bound on the mutual information I(t; s) between teacher and
student feature activations. The teacher conditional p(t | s) is
approximated by a heteroscedastic Gaussian

    q(t | s) = N( mu(s),  diag(sigma^2) )

where mu(s) is a small CNN regressor mapping the student feature into
the teacher's channel space, and sigma^2 is a learnable per-channel
variance parameterised through softplus so it stays positive:

    sigma^2_c = log(1 + exp(alpha_c)) + eps

The variational lower bound (up to an additive constant) is the
negative Gaussian log-likelihood, which becomes the loss

    L = 0.5 * mean( log(sigma^2) + (t - mu(s))^2 / sigma^2 )

so the network learns both how to predict the teacher mean and how
much per-channel residual variance to allocate. This couples better
than pure MSE/L2 hint matching when teacher feature magnitudes vary by
channel.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("vid")
class VIDLoss(nn.Module):
    """Variational Information Distillation feature matcher.

    Args:
        student_channels (int): C_s of the student feature.
        teacher_channels (int): C_t of the teacher feature.
        mid_channels (int): width of the mu predictor (default = teacher).
        init_pred_var (float): initial value of sigma^2 (paper uses 5.0,
            so alpha is initialised to inverse-softplus(5.0 - eps)).
        eps (float): numerical floor added to sigma^2.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        mid_channels: int = None,
        init_pred_var: float = 5.0,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = teacher_channels
        self.eps = float(eps)

        # mu(s): two 1x1 convs with a ReLU - the smallest CNN regressor
        # that still adapts channel counts and adds nonlinearity.
        self.mu = nn.Sequential(
            nn.Conv2d(student_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, teacher_channels, 1, bias=False),
        )

        # Learnable per-channel log-alpha so softplus(alpha) + eps == sigma^2.
        # Initialise so that softplus(alpha) ~ init_pred_var - eps.
        init = math.log(math.exp(max(init_pred_var - eps, 1e-3)) - 1.0)
        self.log_alpha = nn.Parameter(
            torch.full((1, teacher_channels, 1, 1), float(init))
        )

    def forward(self, feat_S, feat_T) -> torch.Tensor:
        """
        Args:
            feat_S: (B, C_s, H, W) student feature, or list/tuple thereof.
            feat_T: (B, C_t, H, W) teacher feature, or list/tuple thereof.
        """
        if isinstance(feat_S, (list, tuple)):
            feat_S = feat_S[-1] if feat_S else None
        if isinstance(feat_T, (list, tuple)):
            feat_T = feat_T[-1] if feat_T else None
        if feat_S is None or feat_T is None:
            return torch.tensor(0.0)

        # Spatial alignment so the mu predictor can compare like-for-like.
        if feat_S.shape[-2:] != feat_T.shape[-2:]:
            feat_T = F.interpolate(
                feat_T, size=feat_S.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        pred_mean = self.mu(feat_S)
        pred_var = F.softplus(self.log_alpha) + self.eps  # (1, C_t, 1, 1)

        # Negative Gaussian log-likelihood (drop the constant log(2*pi)/2).
        # Broadcasting takes care of the per-channel variance.
        t = feat_T.detach()
        nll = 0.5 * (torch.log(pred_var) + (pred_mean - t).pow(2) / pred_var)
        return nll.mean()
