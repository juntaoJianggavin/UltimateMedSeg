# TTM / WTTM: Transformed (Weighted) Teacher Matching (ICLR 2024)
# Reference: https://github.com/zkxufo/TTM
# Paper: https://arxiv.org/abs/2402.11148
# Implemented from paper formulas; not a copy of the official repo.
"""TTM/WTTM: drop the temperature transform on the student side.

Zheng & Yang ICLR 2024 reinterpret KD as ``power-transform matching``:
the conventional Hinton KD with temperature T applied on both sides
is equivalent to minimising

    L_KD(T) = T^2 * KL( softmax(z_t / T) || softmax(z_s / T) ),

which implicitly squashes the student distribution. TTM keeps the
teacher transform (softmax(z_t / T)) but removes it from the student,
matching against the raw softmax(z_s) instead. The paper shows this
introduces an inherent Renyi-entropy regulariser on the student and
gives uniformly better empirical generalisation::

    L_TTM = KL( softmax(z_t / T) || softmax(z_s) ).

WTTM additionally weights each sample by an exponential of the
teacher entropy (the ``weighted'' variant in Sec. 4 of the paper).
When ``weighted=True`` the per-pixel loss is multiplied by

    w_i = exp( H(softmax(z_t^i / T)) / log C )

normalised to mean 1 inside the batch. High-entropy teacher
predictions get larger weight, mirroring the idea that ``hard''
samples carry more transferable knowledge.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("ttm_kd")
class TTMKDLoss(nn.Module):
    """Transformed Teacher Matching (ICLR 2024, Zheng & Yang).

    Args:
        temperature: T for the teacher-side power transform.
        weighted: if True, apply per-sample entropy weighting (WTTM).
        weight_temperature: optional softening of the entropy weights;
            the raw entropy is divided by ``log(C)`` then by this value
            before exponentiation, so larger values flatten the weights
            toward 1.0. Defaults to 1.0 (paper).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        weighted: bool = False,
        weight_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"TTM temperature must be > 0, got {temperature}."
            )
        if weight_temperature <= 0:
            raise ValueError(
                f"TTM weight_temperature must be > 0, got {weight_temperature}."
            )
        self.temperature = float(temperature)
        self.weighted = bool(weighted)
        self.weight_temperature = float(weight_temperature)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            student_output: (B, C, H, W) student logits.
            teacher_output: (B, C, H, W) teacher logits (detached inside).
        """
        if student_output.dim() != 4 or teacher_output.dim() != 4:
            raise ValueError(
                "TTM expects 4D logit tensors (B, C, H, W); got "
                f"student={tuple(student_output.shape)} "
                f"teacher={tuple(teacher_output.shape)}."
            )
        if student_output.shape[1] != teacher_output.shape[1]:
            raise ValueError(
                "TTM requires matching class dimensions; got "
                f"student C={student_output.shape[1]}, "
                f"teacher C={teacher_output.shape[1]}."
            )
        if student_output.shape[-2:] != teacher_output.shape[-2:]:
            teacher_output = F.interpolate(
                teacher_output, size=student_output.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        teacher_output = teacher_output.detach()

        B, C, H, W = student_output.shape
        T = self.temperature
        s = student_output.permute(0, 2, 3, 1).reshape(-1, C)
        t = teacher_output.permute(0, 2, 3, 1).reshape(-1, C)

        # Teacher-side temperature transform; student stays at T=1.
        p_t = F.softmax(t / T, dim=1)
        log_p_s = F.log_softmax(s, dim=1)

        # Per-pixel KL(p_t || p_s) = sum_k p_t * (log p_t - log p_s).
        # Avoid the log(0) corner via clamp.
        log_p_t = (p_t.clamp(min=1e-12)).log()
        kl_per_pixel = (p_t * (log_p_t - log_p_s)).sum(dim=1)

        if not self.weighted:
            return kl_per_pixel.mean()

        # WTTM: weight each pixel by exp( H(p_t) / (log C * tau_w) ).
        entropy = -(p_t * log_p_t).sum(dim=1)
        denom = math.log(max(C, 2)) * self.weight_temperature
        w = torch.exp(entropy / denom)
        # Normalise so mean(w) == 1 — keeps the loss scale comparable
        # to the unweighted form.
        w = w / w.mean().clamp(min=1e-8)
        return (w * kl_per_pixel).mean()
