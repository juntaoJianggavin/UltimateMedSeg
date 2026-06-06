# Reference: https://github.com/HobbitLong/RepDistiller
# Paper: https://arxiv.org/abs/1707.01219
"""NST: Like What You Like - Neuron Selectivity Transfer (Huang & Wang, 2017).

Algorithm summary
-----------------
NST views each channel of a CNN feature map as the activation
distribution of a single "neuron" over the H*W spatial positions, and
treats the set of channels of one sample as samples from a "neuron
selectivity distribution". Aligning student and teacher selectivity
distributions reduces to a Maximum Mean Discrepancy (MMD) between the
two sets of channel-vectors using a kernel k:

    MMD^2(F^s, F^t) = (1/C_s^2) sum_{i,j} k(f^s_i, f^s_j)
                    + (1/C_t^2) sum_{i,j} k(f^t_i, f^t_j)
                    - (2/(C_s C_t)) sum_{i,j} k(f^s_i, f^t_j)

Per the paper (Sec. 3.3) the polynomial kernel
``k(x, y) = (x . y + c)^d`` (default ``c = 0, d = 2``) on L2-normalised
channel vectors works best for selectivity transfer.

Each (B, C, H*W) feature is first L2-normalised along H*W (so each
channel-vector has unit norm, matching the official "ST" formulation),
then the MMD is computed per sample and averaged over the batch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("nst")
class NSTLoss(nn.Module):
    """Neuron Selectivity Transfer with polynomial-kernel MMD.

    Args:
        kernel: 'poly' (default, paper) or 'linear'.
        poly_c: additive constant in the polynomial kernel (paper c=0).
        poly_d: polynomial degree (paper d=2).
        student_channels / teacher_channels: optional channel counts.
            When set and unequal, a lazy 1x1 conv aligns student to the
            teacher's channel count before the MMD computation.
    """

    def __init__(
        self,
        kernel: str = 'poly',
        poly_c: float = 0.0,
        poly_d: int = 2,
        student_channels: int = None,
        teacher_channels: int = None,
        **kwargs,
    ):
        super().__init__()
        if kernel not in ('poly', 'linear'):
            raise ValueError(
                f"NST kernel must be 'poly' or 'linear', got {kernel}."
            )
        if poly_d <= 0:
            raise ValueError(f"NST poly_d must be > 0, got {poly_d}.")
        self.kernel = kernel
        self.poly_c = float(poly_c)
        self.poly_d = int(poly_d)

        if (student_channels is not None and teacher_channels is not None
                and student_channels != teacher_channels):
            self.align = nn.Conv2d(
                student_channels, teacher_channels,
                kernel_size=1, bias=False,
            )
        else:
            self.align = None

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gram matrix k(x_i, y_j). x: (B, Cx, D), y: (B, Cy, D)."""
        # Pairwise dot products: (B, Cx, Cy)
        dot = torch.bmm(x, y.transpose(1, 2))
        if self.kernel == 'linear':
            return dot
        # Polynomial kernel: (x.y + c)^d
        return (dot + self.poly_c).pow(self.poly_d)

    def _mmd2(self, fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        """Squared MMD between two sets of channel-vectors per sample.

        fs: (B, Cs, D) student channel-vectors (D = H*W).
        ft: (B, Ct, D) teacher channel-vectors.
        Returns: scalar averaged over the batch.
        """
        B, Cs, _ = fs.shape
        Ct = ft.shape[1]
        kss = self._kernel(fs, fs).mean(dim=(1, 2))           # (B,)
        ktt = self._kernel(ft, ft).mean(dim=(1, 2))           # (B,)
        kst = self._kernel(fs, ft).mean(dim=(1, 2))           # (B,)
        return (kss + ktt - 2.0 * kst).mean()

    def forward(
        self,
        feat_S: torch.Tensor,
        feat_T: torch.Tensor,
    ) -> torch.Tensor:
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
            raise RuntimeError(
                "NSTLoss received None for student or teacher features. "
                "Check that feature_layers points to real module names "
                "in both models."
            )
        if feat_S.dim() != 4 or feat_T.dim() != 4:
            raise ValueError(
                f"NST expects 4D feature tensors (B,C,H,W); got "
                f"student={tuple(feat_S.shape)} teacher={tuple(feat_T.shape)}."
            )

        # Spatial alignment so H*W matches.
        if feat_S.shape[-2:] != feat_T.shape[-2:]:
            feat_T = F.interpolate(
                feat_T, size=feat_S.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        # Optional channel alignment via 1x1 conv.
        if self.align is not None:
            feat_S = self.align(feat_S)

        feat_T = feat_T.detach()
        B, Cs, H, W = feat_S.shape
        Ct = feat_T.shape[1]
        # (B, C, H*W) channel-vectors, then L2-normalise per channel
        # (along the spatial axis) so kernels see direction not magnitude.
        fs = F.normalize(feat_S.reshape(B, Cs, H * W), p=2, dim=2)
        ft = F.normalize(feat_T.reshape(B, Ct, H * W), p=2, dim=2)

        return self._mmd2(fs, ft)
