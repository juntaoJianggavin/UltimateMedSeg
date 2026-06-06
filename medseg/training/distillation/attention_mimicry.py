"""Attention Mimicry Distillation (Zagoruyko & Komodakis, 2016).

Student learns to mimic teacher's spatial attention patterns.
Official concept: https://arxiv.org/abs/1612.03928

Reference implementation: https://github.com/szagoruyko/attention-transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("attention_mimicry")
class AttentionMimicryLoss(nn.Module):
    """Attention mimicry distillation.

    Student learns to mimic teacher's spatial attention patterns.
    Useful for transferring localization knowledge.
    """

    def __init__(self, alpha: float = 0.4, base_loss_fn=None, **kwargs):
        super().__init__()
        self.alpha = alpha
        if base_loss_fn is None:
            from medseg.losses.compound_loss import CompoundLoss
            self.base_loss_fn = CompoundLoss()
        else:
            self.base_loss_fn = base_loss_fn

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_output: Student logits (B, C, H, W)
            teacher_output: Teacher logits (B, C, H, W)
            target: Ground truth (B, H, W)
        """
        task_loss = self.base_loss_fn(student_output, target)

        student_attn = F.softmax(student_output, dim=1)
        teacher_attn = F.softmax(teacher_output, dim=1)

        student_spatial = student_attn.sum(dim=1, keepdim=True)
        teacher_spatial = teacher_attn.sum(dim=1, keepdim=True)

        student_spatial = student_spatial / (student_spatial.max() + 1e-8)
        teacher_spatial = teacher_spatial / (teacher_spatial.max() + 1e-8)

        attn_loss = F.mse_loss(student_spatial, teacher_spatial.detach())
        total_loss = (1 - self.alpha) * task_loss + self.alpha * attn_loss
        return total_loss
