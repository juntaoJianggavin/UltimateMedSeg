# SDD: Scale Decoupled Distillation (CVPR 2024)
# Reference: https://github.com/shicaiwei123/SDD-CVPR2024
# Paper: https://arxiv.org/abs/2403.13512
# Implemented from paper formulas; not a copy of the official repo.
"""SDD: decouple logits along the spatial scale axis before KL.

Wei et al. CVPR 2024 split the dense logits into a *consistent* term
(global average) and *complementary* terms at a series of finer
spatial scales (region average minus global). The standard KD KL is
then applied at every scale: matching the global term transfers
image-level evidence, while matching the per-scale complementary
terms transfers progressively more local structure that the teacher
captures but a smaller student might lose.

For each scale ``s`` in ``scales`` we adaptive-average-pool the
(B, C, H, W) logits to (B, C, H/s, W/s), subtract the global mean,
then compute temperature-scaled KL between student and teacher. The
total loss is ``alpha_consistent`` times the global KL plus
``beta_complement`` times the sum over scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("sdd")
class SDDLoss(nn.Module):
    """Scale-decoupled KD for dense prediction.

    Args:
        temperature: T in the KL soft-target softmax (paper uses 4.0).
        scales: list of spatial divisors for the complementary terms
            (default ``[2, 4]``). Each entry s produces a (H/s, W/s)
            region map. The global (1x1) term is always included.
        alpha_consistent: weight on the global / consistent KL.
        beta_complement: weight on the sum of complementary-scale KLs.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        scales: List[int] = None,
        alpha_consistent: float = 1.0,
        beta_complement: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"SDD temperature must be > 0, got {temperature}.")
        self.temperature = float(temperature)
        self.scales = list(scales) if scales is not None else [2, 4]
        if any(int(s) <= 0 for s in self.scales):
            raise ValueError(
                f"SDD scales must all be positive integers, got {self.scales}."
            )
        self.scales = [int(s) for s in self.scales]
        self.alpha = float(alpha_consistent)
        self.beta = float(beta_complement)

    def _kl(self, s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
        """Pixel-wise temperature-softmax KL averaged with `batchmean`."""
        B, C = s_logits.shape[:2]
        s = s_logits.permute(0, 2, 3, 1).reshape(-1, C)
        t = t_logits.permute(0, 2, 3, 1).reshape(-1, C)
        T = self.temperature
        log_p_s = F.log_softmax(s / T, dim=1)
        p_t = F.softmax(t / T, dim=1)
        return F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)

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
        if student_output.dim() != 4 or teacher_output.dim() != 4:
            raise ValueError(
                "SDD expects 4D logit tensors (B, C, H, W); got "
                f"student={tuple(student_output.shape)} "
                f"teacher={tuple(teacher_output.shape)}."
            )
        if student_output.shape[-2:] != teacher_output.shape[-2:]:
            teacher_output = F.interpolate(
                teacher_output, size=student_output.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        teacher_output = teacher_output.detach()
        H, W = student_output.shape[-2:]

        # Consistent (global) term: 1x1 region per channel per sample.
        global_s = F.adaptive_avg_pool2d(student_output, 1)
        global_t = F.adaptive_avg_pool2d(teacher_output, 1)
        total = self.alpha * self._kl(global_s, global_t)

        # Complementary (per-scale) terms.
        comp_sum = student_output.new_zeros(())
        for sc in self.scales:
            out_h = max(H // sc, 1)
            out_w = max(W // sc, 1)
            pooled_s = F.adaptive_avg_pool2d(student_output, (out_h, out_w))
            pooled_t = F.adaptive_avg_pool2d(teacher_output, (out_h, out_w))
            # Subtract broadcast global term so we only distil the
            # complementary (region - global) signal at this scale.
            cp_s = pooled_s - global_s
            cp_t = pooled_t - global_t
            comp_sum = comp_sum + self._kl(cp_s, cp_t)
        total = total + self.beta * comp_sum / max(len(self.scales), 1)
        return total
