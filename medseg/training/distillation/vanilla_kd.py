# Reference: https://github.com/peterliht/knowledge-distillation-pytorch
# Paper: https://arxiv.org/abs/1503.02531
"""Vanilla / Hinton Knowledge Distillation (NeurIPS workshop 2014/2015).

Algorithm summary
-----------------
Hinton et al. distil a small student from a large teacher by matching the
soft probability outputs of the teacher at a high temperature T while
keeping the hard cross-entropy on labels. The loss is

    L = (1 - alpha) * CE(student, y)
      + alpha * T^2 * KL(softmax(s/T) || softmax(t/T))

The factor T^2 keeps the soft-target gradient magnitude comparable to the
hard-label gradient as T scales. For semantic segmentation we apply the
loss per-pixel: logits of shape (B, C, H, W) are reshaped to (B*H*W, C).

Paper default temperature: T = 4.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("vanilla_kd")
class VanillaKDLoss(nn.Module):
    """Hinton's logit-matching KD for dense segmentation outputs.

    Args:
        temperature: T in the soft-target softmax. Common: 2-8.
        alpha: weight on the soft KD term, 1-alpha on the hard CE term.
        ce_weight: kept for parity with other KD modules. If train loop
            already supplies a task loss outside, set alpha=1 (pure KD)
            and the (1-alpha) CE inside this loss becomes zero anyway.
        ignore_index: pixels in target equal to this are excluded from CE.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.9,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"Vanilla KD temperature must be > 0, got {temperature}. "
                f"Hinton et al. recommend T in [2, 8] (paper default 4.0)."
            )
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"Vanilla KD alpha must be in [0, 1], got {alpha}."
            )
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.ignore_index = int(ignore_index)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            student_output: (B, C, H, W) student logits.
            teacher_output: (B, C, H, W) teacher logits (no grad).
            target: optional (B, H, W) long mask. If supplied, the loss
                blends the CE term in; otherwise it returns pure soft-KD.
        """
        # Spatial alignment - teacher upsampled/downsampled to student.
        if student_output.shape[-2:] != teacher_output.shape[-2:]:
            teacher_output = F.interpolate(
                teacher_output, size=student_output.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        T = self.temperature
        # Pixel-wise soft target KL: shape (B*H*W, C).
        B, C, H, W = student_output.shape
        s = student_output.permute(0, 2, 3, 1).reshape(-1, C)
        t = teacher_output.detach().permute(0, 2, 3, 1).reshape(-1, C)

        log_p_s = F.log_softmax(s / T, dim=1)
        p_t = F.softmax(t / T, dim=1)
        kd = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)

        if target is None or self.alpha >= 1.0:
            return self.alpha * kd

        # Hard label CE on student (pixel-wise).
        ce = F.cross_entropy(
            student_output, target.long(),
            ignore_index=self.ignore_index,
        )
        return (1.0 - self.alpha) * ce + self.alpha * kd
