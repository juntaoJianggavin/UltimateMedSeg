# Reference: https://github.com/irfanICMLL/TorchDistiller
# Paper: https://arxiv.org/abs/2011.13256
"""CWD: Channel-wise Distillation for Semantic Segmentation (ICCV 2021).

Algorithm summary
-----------------
For dense prediction the activation patterns of every channel form a
spatial probability map (after softmax over the H*W positions). CWD
distils per-channel: independently softmax each channel of student and
teacher with temperature T, then minimise KL(student || teacher) for
every channel, scaled by T^2. The softmax is taken across spatial
positions (not across channels), so each channel becomes a per-pixel
distribution.

Paper default temperature: T = 4.0 (Shu et al., Sec. 4.1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("cwd")
class CWDLoss(nn.Module):
    """Channel-wise Knowledge Distillation for Dense Prediction (ICCV 2021).

    Implements equations (3)-(4) of Shu et al. ICCV 2021: for every channel
    c independently, normalise s_c and t_c into spatial probability maps
    via softmax(./T) over H*W, then compute KL(p_s || p_t) and average.

    If student/teacher channel counts differ a 1x1 projection lifts the
    student feature into the teacher's channel space, mirroring the
    practice in CWD-derived dense KD implementations.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        student_channels: int = None,
        teacher_channels: int = None,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"CWD temperature must be > 0, got {temperature}. "
                f"Paper uses T=4.0."
            )
        self.temperature = float(temperature)
        if (student_channels is not None and teacher_channels is not None
                and student_channels != teacher_channels):
            self.proj = nn.Conv2d(
                student_channels, teacher_channels,
                kernel_size=1, stride=1, padding=0, bias=False,
            )
        else:
            self.proj = None

    def forward(
        self, feat_S: torch.Tensor, feat_T: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_S: Student features/logits (B, C_s, H, W).
            feat_T: Teacher features/logits (B, C_t, H, W).
        """
        # Optional 1x1 projection so channel counts match.
        if self.proj is not None:
            feat_S = self.proj(feat_S)
        elif feat_S.shape[1] != feat_T.shape[1]:
            # Fall back: lazily build a projection on first call.
            self.proj = nn.Conv2d(
                feat_S.shape[1], feat_T.shape[1], 1, bias=False,
            ).to(feat_S.device)
            feat_S = self.proj(feat_S)

        # Spatial alignment.
        if feat_S.shape[2:] != feat_T.shape[2:]:
            feat_T = F.interpolate(
                feat_T, size=feat_S.shape[2:],
                mode='bilinear', align_corners=False,
            )

        B, C, H, W = feat_S.shape
        # Reshape so the last dim is the spatial axis: (B, C, H*W).
        s = feat_S.reshape(B, C, -1)
        t = feat_T.detach().reshape(B, C, -1)

        # Per-channel softmax across SPATIAL positions (dim=2 = H*W).
        # This is the defining choice of CWD (vs vanilla spatial KD,
        # which normalises across channels).
        log_s = F.log_softmax(s / self.temperature, dim=2)
        p_t = F.softmax(t / self.temperature, dim=2)

        # Sum KL over spatial positions, average over channels and batch.
        kl = (p_t * (p_t.clamp(min=1e-8).log() - log_s)).sum(dim=2)
        loss = kl.mean() * (self.temperature ** 2)
        return loss
