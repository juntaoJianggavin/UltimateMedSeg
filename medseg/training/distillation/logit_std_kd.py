# LSKD: Logit Standardization in Knowledge Distillation (CVPR 2024)
# Reference: https://github.com/sunshangquan/logit-standardization-KD
# Paper: https://arxiv.org/abs/2403.01427
# Implemented from paper formulas; not a copy of the official repo.
"""LSKD: Z-score standardize the per-sample logit vector before KD.

Sun et al. CVPR 2024 observe that the absolute magnitude/mean of a
logit vector is a degree of freedom that the softmax itself already
absorbs, yet the conventional KD KL still penalises the student for
matching the teacher's (arbitrary) magnitude. They propose a
weighted Z-score transform applied along the class axis of every
sample before the standard temperature-softmax KL::

    z'_k = (z_k - mean_k z) / (std_k z + eps)

For dense prediction we treat every pixel as a sample, so the
standardisation is independent across (B, H, W). The student/teacher
soft targets are then computed with the same base temperature T and
the loss is the usual T^2 * KL(p_t || p_s).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("logit_std_kd")
class LogitStdKDLoss(nn.Module):
    """Logit Standardisation KD (CVPR 2024, Sun et al.).

    Args:
        temperature: base softmax temperature T (paper uses 2.0-4.0).
        eps: numerical floor for the per-sample standard deviation.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        eps: float = 1e-7,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"LSKD temperature must be > 0, got {temperature}."
            )
        if eps <= 0:
            raise ValueError(f"LSKD eps must be > 0, got {eps}.")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def _standardize(self, z: torch.Tensor) -> torch.Tensor:
        """Z-score standardise (N, C) logits along the class axis.

        Equation (rebracketed from Sun et al. Sec. 3.3):
            z'_k = (z_k - mu) / (sigma + eps)
        where mu, sigma are the per-row (per-sample) mean and std
        with biased (population) variance, matching the paper.
        """
        if z.dim() != 2:
            raise ValueError(
                f"LSKD._standardize expects (N, C); got shape {tuple(z.shape)}."
            )
        mu = z.mean(dim=1, keepdim=True)
        # Population std (unbiased=False) matches the paper definition.
        sigma = z.std(dim=1, keepdim=True, unbiased=False)
        return (z - mu) / (sigma + self.eps)

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
                "LSKD expects 4D logit tensors (B, C, H, W); got "
                f"student={tuple(student_output.shape)} "
                f"teacher={tuple(teacher_output.shape)}."
            )
        if student_output.shape[1] != teacher_output.shape[1]:
            raise ValueError(
                "LSKD requires matching class dimensions; got "
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
        s = student_output.permute(0, 2, 3, 1).reshape(-1, C)
        t = teacher_output.permute(0, 2, 3, 1).reshape(-1, C)

        s_std = self._standardize(s)
        t_std = self._standardize(t)

        T = self.temperature
        log_p_s = F.log_softmax(s_std / T, dim=1)
        p_t = F.softmax(t_std / T, dim=1)
        return F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)
