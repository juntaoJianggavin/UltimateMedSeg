# ReviewKD: Distilling Knowledge via Knowledge Review (CVPR 2021)
# Reference: https://github.com/dvlab-research/ReviewKD
# Paper: https://arxiv.org/abs/2104.09044
# Implemented from paper formulas; not a copy of the official repo.
"""ReviewKD: Cross-stage feature review with ABF + HCL.

Chen et al. CVPR 2021. The "review" mechanism aligns each student
intermediate feature with the corresponding teacher feature through a
top-down fusion: deeper student features are upsampled and combined
with shallower ones via an Attention-Based Fusion (ABF) block, and
each fused stack is matched to the teacher feature with a Hierarchical
Context Loss (HCL) that aggregates MSE across a spatial pyramid.

Inputs:
    feat_S / feat_T: ordered lists of student / teacher features,
    shallow -> deep. Channel counts MUST be declared up front via
    the ``student_channels`` / ``teacher_channels`` lists -- the
    module refuses to silently rebuild ABFs when the shapes change.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from medseg.registry import LOSS_REGISTRY


class _ABF(nn.Module):
    """Attention-Based Fusion block (paper Sec. 3.3).

    Takes the current student feature ``x`` (after 1x1 channel proj to
    ``mid_channels``) and, when ``fuse=True``, also the previously
    fused residual ``y`` upsampled to ``x``'s spatial size. Two
    per-pixel attention maps gate the residual sum, and a 3x3 conv
    projects the result to ``out_channels`` -- the teacher's channel
    count at this stage.
    """

    def __init__(self, in_channels, mid_channels, out_channels, fuse: bool):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channels * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x, y=None):
        x = self.conv1(x)
        if self.att_conv is not None:
            if y is None:
                raise RuntimeError("ABF with fuse=True requires residual y.")
            if y.shape[-2:] != x.shape[-2:]:
                y = F.interpolate(y, size=x.shape[-2:], mode='nearest')
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0:1] + y * z[:, 1:2]
        out = self.conv2(x)
        return out, x


def _hcl(fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """Hierarchical Context Loss (paper Sec. 3.4).

    MSE at native resolution plus MSE at a geometrically-shrinking
    spatial pyramid (H, H/2, H/4, ..., 1). Coarser scales are
    down-weighted by a halving factor so the native resolution
    dominates.
    """
    loss = F.mse_loss(fs, ft)
    H = fs.shape[-2]
    sizes = []
    h = H
    while h > 1:
        h = max(h // 2, 1)
        sizes.append(h)
    weight = 1.0
    total = loss
    for s in sizes:
        weight = weight / 2.0
        fs_p = F.adaptive_avg_pool2d(fs, (s, s))
        ft_p = F.adaptive_avg_pool2d(ft, (s, s))
        total = total + weight * F.mse_loss(fs_p, ft_p)
    return total


@LOSS_REGISTRY.register("review_kd")
class ReviewKDLoss(nn.Module):
    """Cross-stage feature review distillation (CVPR 2021)."""

    def __init__(
        self,
        student_channels: Sequence[int] = None,
        teacher_channels: Sequence[int] = None,
        mid_channels: int = None,
        **kwargs,
    ):
        super().__init__()
        if student_channels is None or teacher_channels is None:
            raise ValueError(
                "ReviewKD requires explicit student_channels and "
                "teacher_channels (shallow -> deep) lists."
            )
        s_chs = [int(c) for c in student_channels]
        t_chs = [int(c) for c in teacher_channels]
        if len(s_chs) != len(t_chs):
            raise ValueError(
                f"ReviewKD student/teacher channel lists must be the same "
                f"length; got {len(s_chs)} vs {len(t_chs)}."
            )
        if len(s_chs) < 2:
            raise ValueError(
                "ReviewKD requires at least 2 stages for the top-down review."
            )
        self.student_channels = s_chs
        self.teacher_channels = t_chs
        mid = int(mid_channels) if mid_channels is not None else max(t_chs)
        # ABF per stage; the deepest stage has no incoming fused y.
        self.abfs = nn.ModuleList()
        for i, (sc, tc) in enumerate(zip(s_chs, t_chs)):
            fuse = (i != len(s_chs) - 1)
            self.abfs.append(_ABF(sc, mid, tc, fuse))

    def forward(self, feat_S, feat_T) -> torch.Tensor:
        """
        Args:
            feat_S: list/tuple of (B, C_si, H_i, W_i) student features,
                shallow -> deep.
            feat_T: list/tuple of (B, C_ti, H_i, W_i) teacher features,
                shallow -> deep.
        """
        if not isinstance(feat_S, (list, tuple)) or not isinstance(feat_T, (list, tuple)):
            raise ValueError(
                "ReviewKD expects list/tuple of student and teacher features "
                "(shallow -> deep)."
            )
        feat_S = list(feat_S)
        feat_T = list(feat_T)
        if len(feat_S) != len(self.student_channels):
            raise ValueError(
                f"ReviewKD got {len(feat_S)} student features, expected "
                f"{len(self.student_channels)} (from student_channels list)."
            )
        if len(feat_T) != len(self.teacher_channels):
            raise ValueError(
                f"ReviewKD got {len(feat_T)} teacher features, expected "
                f"{len(self.teacher_channels)} (from teacher_channels list)."
            )
        # Strict per-stage channel check.
        for i, (s, t) in enumerate(zip(feat_S, feat_T)):
            if s.shape[1] != self.student_channels[i]:
                raise ValueError(
                    f"ReviewKD stage {i}: feat_S channels {s.shape[1]} != "
                    f"declared student_channels[{i}]={self.student_channels[i]}."
                )
            if t.shape[1] != self.teacher_channels[i]:
                raise ValueError(
                    f"ReviewKD stage {i}: feat_T channels {t.shape[1]} != "
                    f"declared teacher_channels[{i}]={self.teacher_channels[i]}."
                )

        # Top-down review: walk deep -> shallow, building fused outputs.
        out_deep, residual = self.abfs[-1](feat_S[-1])
        outs = [out_deep]
        for i in range(len(feat_S) - 2, -1, -1):
            out, residual = self.abfs[i](feat_S[i], residual)
            outs.insert(0, out)

        # HCL between each fused output and the corresponding teacher feat.
        total = feat_S[0].new_zeros(())
        for o, t in zip(outs, feat_T):
            t = t.detach()
            if o.shape[-2:] != t.shape[-2:]:
                t = F.interpolate(
                    t, size=o.shape[-2:],
                    mode='bilinear', align_corners=False,
                )
            total = total + _hcl(o, t)
        return total / len(outs)
