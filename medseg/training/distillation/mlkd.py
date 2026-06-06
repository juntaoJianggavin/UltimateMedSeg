# MLKD: Multi-Level Logit Distillation (CVPR 2023)
# Reference: https://github.com/Jin-Ying/Multi-Level-Logit-Distillation
# Paper: https://arxiv.org/abs/2302.13652
# Implemented from paper formulas; not a copy of the official repo.
"""MLKD: align logit predictions at instance, batch and class levels.

Jin et al. CVPR 2023 argue that vanilla KD only matches the
per-sample class distribution and discards two other ``views'' of
the same logits that the teacher also predicts well: how samples
relate to each other (batch-level correlation matrix) and how
classes relate to each other (class-level correlation matrix). MLKD
distils all three jointly and averages the loss over a set of
temperatures to reduce sensitivity to T::

    L = (1 / |T|) sum_{T in temperatures} [
            alpha_ins * L_instance(T)
          + alpha_bat * L_batch(T)
          + alpha_cls * L_class(T)
        ]

For dense prediction we collapse spatial pixels into a per-image
class distribution via spatial mean-pool (so batch correlation is
B x B). The instance term remains per-pixel KL — it still operates
on the fine-grained spatial logits.

Sub-losses (per temperature T):

* ``L_instance(T) = T^2 * KL( p_t(T) || p_s(T) )`` summed across
  pixels then averaged over the batch.
* ``L_batch(T)    = T^2 * || M_s - M_t ||_F^2 / B^2`` where M is
  the B x B Gram of L2-normalised per-image class distributions.
* ``L_class(T)    = T^2 * || N_s - N_t ||_F^2 / C^2`` where N is
  the C x C Gram of L2-normalised class-wise activation vectors
  (transposed view).
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("mlkd")
class MLKDLoss(nn.Module):
    """Multi-Level Logit Distillation (CVPR 2023, Jin et al.).

    Args:
        temperatures: list of temperatures to average over (paper:
            ``[2, 3, 4, 5, 6]``).
        alpha_instance: weight on the per-pixel KL term.
        alpha_batch: weight on the B x B batch correlation term.
        alpha_class: weight on the C x C class correlation term.
    """

    def __init__(
        self,
        temperatures: List[float] = None,
        alpha_instance: float = 1.0,
        alpha_batch: float = 1.0,
        alpha_class: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if temperatures is None:
            temperatures = [2.0, 3.0, 4.0, 5.0, 6.0]
        temperatures = [float(t) for t in temperatures]
        if any(t <= 0 for t in temperatures):
            raise ValueError(
                f"MLKD temperatures must all be > 0, got {temperatures}."
            )
        if not temperatures:
            raise ValueError("MLKD requires at least one temperature.")
        self.temperatures = temperatures
        self.alpha_instance = float(alpha_instance)
        self.alpha_batch = float(alpha_batch)
        self.alpha_class = float(alpha_class)

    @staticmethod
    def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """L2-normalise along dim=1 — used before building Gram matrices."""
        return x / x.norm(dim=1, keepdim=True).clamp(min=eps)

    def _instance_loss(
        self, s: torch.Tensor, t: torch.Tensor, T: float
    ) -> torch.Tensor:
        """Per-pixel KL between teacher and student soft predictions."""
        log_p_s = F.log_softmax(s / T, dim=1)
        p_t = F.softmax(t / T, dim=1)
        return F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)

    def _batch_loss(
        self, s_avg: torch.Tensor, t_avg: torch.Tensor, T: float
    ) -> torch.Tensor:
        """Frobenius distance between BxB sample-correlation matrices."""
        s_n = self._normalize_rows(s_avg)
        t_n = self._normalize_rows(t_avg)
        M_s = s_n @ s_n.t()
        M_t = t_n @ t_n.t()
        B = s_avg.shape[0]
        return ((M_s - M_t) ** 2).sum() * (T * T) / (B * B)

    def _class_loss(
        self, s_avg: torch.Tensor, t_avg: torch.Tensor, T: float
    ) -> torch.Tensor:
        """Frobenius distance between CxC class-correlation matrices.

        Built from the transpose of the per-image probability matrix
        so each row is a class-by-sample activation vector before
        normalisation.
        """
        s_c = self._normalize_rows(s_avg.t())
        t_c = self._normalize_rows(t_avg.t())
        N_s = s_c @ s_c.t()
        N_t = t_c @ t_c.t()
        C = s_avg.shape[1]
        return ((N_s - N_t) ** 2).sum() * (T * T) / (C * C)

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
                "MLKD expects 4D logit tensors (B, C, H, W); got "
                f"student={tuple(student_output.shape)} "
                f"teacher={tuple(teacher_output.shape)}."
            )
        if student_output.shape[1] != teacher_output.shape[1]:
            raise ValueError(
                "MLKD requires matching class dimensions; got "
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

        # Flat per-pixel matrices for the instance term.
        s_flat = student_output.permute(0, 2, 3, 1).reshape(-1, C)
        t_flat = teacher_output.permute(0, 2, 3, 1).reshape(-1, C)

        total = student_output.new_zeros(())
        for T in self.temperatures:
            # Per-image average class distribution for batch/class terms.
            # Compute from the softmax of pooled logits at temperature T,
            # matching the paper's "predictive distribution" view.
            p_s_full = F.softmax(student_output / T, dim=1)
            p_t_full = F.softmax(teacher_output / T, dim=1)
            s_avg = p_s_full.mean(dim=(2, 3))  # (B, C)
            t_avg = p_t_full.mean(dim=(2, 3))

            l_ins = self._instance_loss(s_flat, t_flat, T)
            l_bat = self._batch_loss(s_avg, t_avg, T) if B > 1 else s_avg.new_zeros(())
            l_cls = self._class_loss(s_avg, t_avg, T) if C > 1 else s_avg.new_zeros(())

            total = total + (
                self.alpha_instance * l_ins
                + self.alpha_batch * l_bat
                + self.alpha_class * l_cls
            )
        return total / len(self.temperatures)
