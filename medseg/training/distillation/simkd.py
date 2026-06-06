# SimKD (CVPR 2022)
# Reference: https://github.com/DefangChen/SimKD
# Paper: https://arxiv.org/abs/2203.16633
# Implemented from paper formulas; not a copy of the official repo.
"""SimKD: Knowledge Distillation with the Reused Teacher Classifier.

Chen et al. CVPR 2022 argue that the student can give up its own
classifier and instead "reuse" the (frozen) teacher classifier on a
small projector that maps the student penultimate feature into the
teacher penultimate feature space. The actual distillation signal then
collapses to a single L2 (sum-MSE) between the projected student
feature and the teacher feature -- everything else (logit match, label
match) is handled implicitly by the shared classifier.

For dense prediction we treat the deepest captured feature as the
"penultimate" and apply the L2 spatially. Channel-count alignment is
mandatory: ``student_channels`` and ``teacher_channels`` must be
declared so the projector can be built without silent guessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("simkd")
class SimKDLoss(nn.Module):
    """SimKD feature-matching loss with a learnable projector.

    The projector is the lightweight bottleneck used in the official
    repo (Sec. 3.2): 1x1 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU
    -> 1x1 conv -> BN. The output dimension equals the teacher's
    penultimate channel count. The loss is sum-MSE divided by batch
    size, mirroring the paper's reported objective.

    Args:
        student_channels: penultimate channel count of the student
            feature passed in. REQUIRED -- no silent auto-detection.
        teacher_channels: penultimate channel count of the teacher
            feature. REQUIRED.
        factor: bottleneck shrink factor (default 2, matches official).
    """

    def __init__(
        self,
        student_channels: int = None,
        teacher_channels: int = None,
        factor: int = 2,
        **kwargs,
    ):
        super().__init__()
        if student_channels is None or teacher_channels is None:
            raise ValueError(
                "SimKD requires explicit student_channels and teacher_channels "
                "(no silent channel auto-detection)."
            )
        if int(factor) < 1:
            raise ValueError(f"SimKD factor must be >= 1, got {factor}.")
        self.student_channels = int(student_channels)
        self.teacher_channels = int(teacher_channels)
        mid = max(self.teacher_channels // int(factor), 1)
        self.proj = nn.Sequential(
            nn.Conv2d(self.student_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, self.teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.teacher_channels),
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
            raise RuntimeError(
                "SimKDLoss received None for student or teacher features. "
                "Check that feature_layers matches a real module name in "
                "both teacher and student."
            )
        if feat_S.dim() != 4 or feat_T.dim() != 4:
            raise ValueError(
                f"SimKD expects 4D feature tensors (B,C,H,W); got "
                f"student={tuple(feat_S.shape)} teacher={tuple(feat_T.shape)}."
            )
        # Strict channel check: refuse silent mismatch instead of inserting
        # an unplanned projection.
        if feat_S.shape[1] != self.student_channels:
            raise ValueError(
                f"SimKD expected student_channels={self.student_channels}, "
                f"got feat_S.shape[1]={feat_S.shape[1]}."
            )
        if feat_T.shape[1] != self.teacher_channels:
            raise ValueError(
                f"SimKD expected teacher_channels={self.teacher_channels}, "
                f"got feat_T.shape[1]={feat_T.shape[1]}."
            )

        # Spatial alignment so the MSE is well-defined.
        if feat_S.shape[-2:] != feat_T.shape[-2:]:
            feat_T = F.interpolate(
                feat_T, size=feat_S.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        projected = self.proj(feat_S)
        # sum-MSE / N, matching the official simkd_loss reduction.
        N = projected.shape[0]
        return F.mse_loss(projected, feat_T.detach(), reduction='sum') / max(N, 1)
