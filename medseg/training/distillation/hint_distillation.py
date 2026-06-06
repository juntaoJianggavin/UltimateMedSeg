"""Hint-based Distillation (FitNets, Romero et al., ICLR 2015).

Official concept: https://arxiv.org/abs/1412.6550
Uses intermediate hints from teacher to guide student learning.

Reference implementation: https://github.com/adri-romsor/FitNets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("hint_distillation")
class HintDistillationLoss(nn.Module):
    """Hint-based distillation for UNet (FitNets style).

    Uses intermediate hints from teacher to guide student learning.
    Particularly effective when student has different architecture.
    """

    def __init__(
        self,
        hint_channels: Optional[List[int]] = None,
        hint_locations: Optional[List[str]] = None,
        temperature: float = 2.0,
        alpha: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.hint_channels = hint_channels or []
        self.hint_locations = hint_locations or []
        self.temperature = temperature
        self.alpha = alpha

        # Regressors to match teacher channels
        self.regressors = nn.ModuleList()
        for ch in self.hint_channels:
            self.regressors.append(nn.Conv2d(ch, ch, 1, bias=False))

    def forward(
        self,
        student_hints: List[torch.Tensor],
        teacher_hints: List[torch.Tensor],
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_hints: List of student intermediate features
            teacher_hints: List of teacher intermediate features
            student_output: Student final output
            teacher_output: Teacher final output
            target: Ground truth
        """
        task_loss = F.cross_entropy(student_output, target.long())

        hint_loss = 0.0
        for idx, (s_hint, t_hint) in enumerate(zip(student_hints, teacher_hints)):
            if idx < len(self.regressors):
                s_hint = self.regressors[idx](s_hint)
            if s_hint.shape[2:] != t_hint.shape[2:]:
                t_hint = F.interpolate(t_hint, size=s_hint.shape[2:],
                                       mode='bilinear', align_corners=False)
            hint_loss += F.mse_loss(s_hint, t_hint.detach())

        hint_loss /= max(len(student_hints), 1)
        total_loss = (1 - self.alpha) * task_loss + self.alpha * hint_loss
        return total_loss
