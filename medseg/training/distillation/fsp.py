# Reference: https://github.com/yoshitomo-matsubara/torchdistill
# Paper: https://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
"""FSP: A Gift from Knowledge Distillation - Flow of Solution Procedure
(Yim et al., CVPR 2017).

Algorithm summary
-----------------
For every pair of feature maps (F1, F2) at two layers of the same
network, FSP defines a Gram-style "flow" matrix

    G[c1, c2] = (1 / (H * W)) * sum_{h,w} F1[c1, h, w] * F2[c2, h, w]

i.e. ``G = F1_flat @ F2_flat^T / (H * W)`` with shape (C1, C2). The
matrix summarises how the activation pattern moves from one layer to
the next - the "flow of solution procedure". Distillation is the L2
(MSE) distance between the teacher's flow and the student's flow,
summed over a set of layer pairs, averaged over the batch:

    L_FSP = mean_i  || G^T_i - G^S_i ||_F^2

If the two student layers have different spatial sizes, F2 is bilinearly
resized to F1's resolution before the Gram product (paper assumes equal
HxW within a "block").

This file implements the paper formula; no source is copied from any
reference repo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("fsp")
class FSPLoss(nn.Module):
    """Flow of Solution Procedure (Yim et al., CVPR 2017).

    Expects two parallel ordered lists of feature maps from student and
    teacher: ``[F1, F2, F3, ...]``. The loss takes consecutive pairs
    ``(F_i, F_{i+1})`` and forms one FSP / Gram matrix per pair on each
    side, then averages the squared Frobenius error between teacher and
    student matrices.

    Args:
        student_channels: list of channel counts for student layers.
            Used to lazily build 1x1 projections so student / teacher
            FSP matrices have matching shape.
        teacher_channels: list of channel counts for teacher layers.
            Required when student_channels is given (same length).
    """

    def __init__(
        self,
        student_channels: Sequence[int] = None,
        teacher_channels: Sequence[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.student_channels = (
            list(student_channels) if student_channels is not None else None
        )
        self.teacher_channels = (
            list(teacher_channels) if teacher_channels is not None else None
        )
        if (self.student_channels is not None) ^ (self.teacher_channels is not None):
            raise ValueError(
                "FSP: must provide both student_channels and teacher_channels, "
                "or neither."
            )
        if self.student_channels is not None and (
            len(self.student_channels) != len(self.teacher_channels)
        ):
            raise ValueError(
                "FSP: student_channels and teacher_channels must have the "
                "same length (one entry per hooked layer)."
            )
        # 1x1 projections built lazily inside forward (only when needed).
        self.projs = nn.ModuleDict()

    @staticmethod
    def _fsp_matrix(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """G = F1 @ F2^T / (H * W).

        Args:
            f1: (B, C1, H, W)
            f2: (B, C2, H, W). Resized to match f1 spatially if needed.
        Returns:
            (B, C1, C2) Gram-style flow matrix.
        """
        if f1.dim() != 4 or f2.dim() != 4:
            raise ValueError(
                f"FSP needs 4D feature maps, got {f1.dim()}D and {f2.dim()}D."
            )
        if f1.shape[-2:] != f2.shape[-2:]:
            f2 = F.interpolate(
                f2, size=f1.shape[-2:], mode='bilinear', align_corners=False,
            )
        B, C1, H, W = f1.shape
        C2 = f2.shape[1]
        f1_flat = f1.reshape(B, C1, H * W)
        f2_flat = f2.reshape(B, C2, H * W)
        # (B, C1, HW) @ (B, HW, C2) -> (B, C1, C2)
        return torch.bmm(f1_flat, f2_flat.transpose(1, 2)) / float(H * W)

    def _proj(self, idx: int, c_in: int, c_out: int, device, dtype):
        """Get / build a 1x1 projection so student channels match teacher's."""
        key = f"proj_{idx}_{c_in}_{c_out}"
        if key not in self.projs:
            conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
            conv = conv.to(device=device, dtype=dtype)
            self.projs[key] = conv
        return self.projs[key]

    def forward(
        self,
        feat_S: Sequence[torch.Tensor],
        feat_T: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            feat_S: ordered list of student feature maps (>= 2 layers).
            feat_T: ordered list of teacher feature maps (>= 2 layers,
                same length as feat_S).
        """
        if isinstance(feat_S, torch.Tensor):
            feat_S = [feat_S]
        if isinstance(feat_T, torch.Tensor):
            feat_T = [feat_T]
        feat_S = list(feat_S)
        feat_T = list(feat_T)
        if len(feat_S) < 2 or len(feat_T) < 2:
            raise RuntimeError(
                "FSPLoss needs at least 2 hooked layers per side to form "
                f"a flow matrix; got {len(feat_S)} student / {len(feat_T)} "
                f"teacher. Set feature_layers to >= 2 modules in the config."
            )
        if len(feat_S) != len(feat_T):
            raise ValueError(
                f"FSP expects equal-length feature lists, got "
                f"{len(feat_S)} student vs {len(feat_T)} teacher."
            )

        # Optionally project student features so per-pair Gram shapes match
        # those of the teacher (so the MSE is well-defined).
        if self.student_channels is not None:
            projected = []
            for i, fs in enumerate(feat_S):
                if self.student_channels[i] != self.teacher_channels[i]:
                    conv = self._proj(
                        i,
                        self.student_channels[i],
                        self.teacher_channels[i],
                        fs.device,
                        fs.dtype,
                    )
                    projected.append(conv(fs))
                else:
                    projected.append(fs)
            feat_S = projected

        total = feat_S[0].new_zeros(())
        n_pairs = 0
        for i in range(len(feat_S) - 1):
            fs1, fs2 = feat_S[i], feat_S[i + 1]
            ft1, ft2 = feat_T[i].detach(), feat_T[i + 1].detach()
            # Cross-check shapes so a misconfigured channel list errors loudly.
            if fs1.shape[1] != ft1.shape[1] or fs2.shape[1] != ft2.shape[1]:
                raise ValueError(
                    "FSP per-pair channel mismatch after projection: "
                    f"student=({fs1.shape[1]},{fs2.shape[1]}), "
                    f"teacher=({ft1.shape[1]},{ft2.shape[1]}). "
                    "Set student_channels / teacher_channels in the config."
                )
            g_s = self._fsp_matrix(fs1, fs2)
            g_t = self._fsp_matrix(ft1, ft2)
            # Mean-squared Frobenius error, averaged over batch.
            total = total + (g_s - g_t).pow(2).mean()
            n_pairs += 1
        return total / max(n_pairs, 1)
