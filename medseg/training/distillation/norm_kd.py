# NORM: Normalized Knowledge Distillation (ICLR 2023)
# Reference: https://github.com/xyliu7/NORM
# Paper: https://arxiv.org/abs/2305.00367
# Implemented from paper formulas; not a copy of the official repo.
"""NORM: L2-normalise logits before the temperature-softmax KL.

Liu et al. ICLR 2023 observe that the magnitude of teacher / student
logit vectors differs systematically (a stronger teacher tends to
produce sharper, larger-magnitude logits), and that vanilla KD is
sensitive to this scale. NORM removes the degree of freedom by
L2-normalising every per-pixel class-axis logit vector before applying
the temperature softmax, optionally re-scaling by a fixed multiplier
(``scale``); when ``scale`` is left unset the implementation uses
sqrt(num_classes), which keeps the post-normalisation variance close
to 1 across class counts.

The rest of the loss is the standard Hinton soft-target KL, applied
per-pixel for dense segmentation outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("norm_kd")
class NormKDLoss(nn.Module):
    """Normalised logit KD (ICLR 2023).

    Args:
        temperature: temperature for the soft-target softmax.
        scale: optional fixed scale multiplier applied after L2-normalisation.
            If None (default), uses sqrt(num_classes) inferred from logits.
        eps: numerical floor for the L2 normalisation.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        scale: float = None,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"NORM temperature must be > 0, got {temperature}."
            )
        if scale is not None and float(scale) <= 0:
            raise ValueError(
                f"NORM scale must be > 0 when set, got {scale}."
            )
        self.temperature = float(temperature)
        self.scale = None if scale is None else float(scale)
        self.eps = float(eps)

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        """L2 normalise the class-axis (dim=1) of a (N, C) tensor.

        Then rescale by ``scale`` (if set) or sqrt(C) so the post-norm
        magnitudes are comparable to raw logits.
        """
        norm = z.norm(p=2, dim=1, keepdim=True).clamp(min=self.eps)
        z_hat = z / norm
        if self.scale is not None:
            return z_hat * self.scale
        C = z.shape[1]
        return z_hat * float(C) ** 0.5

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            student_output: (B, C, H, W) student logits.
            teacher_output: (B, C, H, W) teacher logits (no grad).
        """
        if student_output.shape[-2:] != teacher_output.shape[-2:]:
            teacher_output = F.interpolate(
                teacher_output, size=student_output.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        B, C, H, W = student_output.shape
        s = student_output.permute(0, 2, 3, 1).reshape(-1, C)
        t = teacher_output.detach().permute(0, 2, 3, 1).reshape(-1, C)

        s_n = self._normalize(s)
        t_n = self._normalize(t)

        T = self.temperature
        log_p_s = F.log_softmax(s_n / T, dim=1)
        p_t = F.softmax(t_n / T, dim=1)
        return F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)
