# Reference: https://github.com/szagoruyko/attention-transfer
# Paper: https://arxiv.org/abs/1612.03928
"""AT: Attention Transfer (Zagoruyko & Komodakis, ICLR 2017).

Paper formula (Sec. 2.1, "Activation-based attention transfer"):

    F_attn(A) = sum_c |A_c|^p          (default p=2)

Each (B, C, H, W) feature map is collapsed to a single (B, H, W) spatial
attention map by summing squared activations across channels. The map
is then flattened to (B, H*W) and L2-normalised. Student and teacher
attention vectors are matched with an L2 (Frobenius) loss:

    L_AT = || Q_s / ||Q_s||_2  -  Q_t / ||Q_t||_2 ||_p

where ``Q = vec(F_attn(A))``. Summed over a list of layer pairs the
student then mimics the teacher's spatial focus pattern. This is the
"AT" loss as implemented in the official Zagoruyko & Komodakis repo
(``attention-transfer/utils.py``: ``at`` / ``at_loss``).

This file implements the official AT formula. It is registered under
``"at"`` so it does not collide with the existing simplified
``"attention_mimicry"`` loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Union
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("at")
class ATLoss(nn.Module):
    """Activation-based Attention Transfer (Zagoruyko & Komodakis, 2017).

    Args:
        p: exponent used to build the attention map (paper uses 2).
        eps: numerical floor for the per-sample L2 normalisation.
        resize_to: 'student' or 'teacher' — which spatial size the pair
            is interpolated to before normalisation. Default 'student'.

    Accepts either a single (B, C, H, W) tensor pair, or a list/tuple of
    such pairs to sum AT loss across multiple layers (paper Sec. 3.2).
    """

    def __init__(
        self,
        p: int = 2,
        eps: float = 1e-6,
        resize_to: str = 'student',
        **kwargs,
    ):
        super().__init__()
        if p <= 0:
            raise ValueError(f"AT exponent p must be > 0, got {p}.")
        if resize_to not in ('student', 'teacher'):
            raise ValueError(
                f"resize_to must be 'student' or 'teacher', got {resize_to}."
            )
        self.p = int(p)
        self.eps = float(eps)
        self.resize_to = resize_to

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial attention map: sum_c |x_c|^p -> flatten -> L2 normalise.

        Returns a (B, H*W) tensor on the unit sphere (per sample).
        """
        # |x|^p summed over channels -> (B, H, W). Use abs() so odd p
        # behaves like the paper's "absolute value raised to p".
        a = x.abs().pow(self.p).sum(dim=1)
        a = a.view(a.size(0), -1)
        # Per-sample L2 normalise so magnitude isn't part of the signal.
        return F.normalize(a, p=2, dim=1, eps=self.eps)

    def _pair_loss(
        self, fs: torch.Tensor, ft: torch.Tensor
    ) -> torch.Tensor:
        if fs.dim() != 4 or ft.dim() != 4:
            raise ValueError(
                f"AT expects 4D feature tensors (B,C,H,W); got "
                f"student={tuple(fs.shape)} teacher={tuple(ft.shape)}."
            )
        # Spatial alignment so the two attention maps have the same length.
        if fs.shape[-2:] != ft.shape[-2:]:
            if self.resize_to == 'student':
                ft = F.interpolate(
                    ft, size=fs.shape[-2:], mode='bilinear', align_corners=False,
                )
            else:
                fs = F.interpolate(
                    fs, size=ft.shape[-2:], mode='bilinear', align_corners=False,
                )
        qs = self._attention(fs)
        qt = self._attention(ft.detach())
        # L2 (squared) distance averaged over batch — matches at_loss().
        return (qs - qt).pow(2).mean()

    def forward(
        self,
        feat_S: Union[torch.Tensor, Sequence[torch.Tensor]],
        feat_T: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            feat_S: student feature map(s).
            feat_T: teacher feature map(s). Same length as feat_S when list.
        """
        # Normalise to lists so the multi-layer path is the common case.
        if isinstance(feat_S, torch.Tensor):
            feat_S = [feat_S]
        if isinstance(feat_T, torch.Tensor):
            feat_T = [feat_T]
        feat_S = list(feat_S)
        feat_T = list(feat_T)
        if len(feat_S) == 0 or len(feat_T) == 0:
            raise RuntimeError(
                "ATLoss received an empty feature list. Check that "
                "feature_layers in the config matches real module names "
                "in both teacher and student."
            )
        if len(feat_S) != len(feat_T):
            raise ValueError(
                f"ATLoss expects equal-length feature lists, got "
                f"{len(feat_S)} student vs {len(feat_T)} teacher."
            )

        total = feat_S[0].new_zeros(())
        for fs, ft in zip(feat_S, feat_T):
            total = total + self._pair_loss(fs, ft)
        return total / len(feat_S)
