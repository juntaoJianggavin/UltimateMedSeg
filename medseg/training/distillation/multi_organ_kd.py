"""Multi-Organ Knowledge Distillation for Medical Image Segmentation.

[中文] 多器官知识蒸馏。每个器官类别独立蒸馏，同时蒸馏类别间关系矩阵，
       解决多类别医学分割中的类别不平衡问题。小器官（如肾上腺）获得更高的蒸馏权重。
[EN]   Multi-organ KD with per-class distillation and inter-class relation
       distillation. Addresses class imbalance in multi-organ medical
       segmentation by weighting small organs more heavily.

Algorithm:
    1. Compute per-class KD loss (each organ distilled independently)
    2. Weight inversely proportional to class frequency (small organs -> higher weight)
    3. Compute inter-class relation distillation (correlation matrix matching)
    4. L = sum(alpha_c * KD_c) + lambda_rel * relation_loss

Reference:
    Multi-organ Knowledge Distillation for Medical Image Segmentation.

(Self-contained: no single canonical GitHub source.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("multi_organ_kd")
class MultiOrganKDLoss(nn.Module):
    """Multi-Organ Knowledge Distillation.

    每个器官独立蒸馏 + 类别间关系蒸馏，解决多类别不平衡。
    Per-organ distillation + inter-class relation matching.

    Args:
        temperature: Softmax temperature (default 4.0).
        class_weights: Optional list of per-class weights. If None, uniform.
        relation_weight: Weight for inter-class relation loss (default 0.5).
        balance_mode: 'uniform', 'inverse_freq', or 'custom' (default 'uniform').
    """

    def __init__(
        self,
        temperature: float = 4.0,
        class_weights: list = None,
        relation_weight: float = 0.5,
        balance_mode: str = 'uniform',
        **kwargs,
    ):
        super().__init__()
        self.temperature = temperature
        self.class_weights = class_weights
        self.relation_weight = relation_weight
        self.balance_mode = balance_mode

    def _per_class_kd(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Per-class KD loss with optional class weighting.

        Each class is distilled independently via binary KL divergence.
        """
        T = self.temperature
        B, C, H, W = student_logits.shape

        s_soft = F.softmax(student_logits / T, dim=1)  # (B, C, H, W)
        t_soft = F.softmax(teacher_logits.detach() / T, dim=1)

        # Per-class KL divergence (binary: class c vs rest)
        class_losses = []
        for c in range(C):
            s_c = s_soft[:, c:c+1]  # (B, 1, H, W)
            t_c = t_soft[:, c:c+1]
            # Binary KL for this class
            s_binary = torch.cat([s_c, 1 - s_c], dim=1)
            t_binary = torch.cat([t_c, 1 - t_c], dim=1)
            s_log = torch.log(s_binary + 1e-8)
            kl = F.kl_div(s_log, t_binary, reduction='batchmean')
            class_losses.append(kl)

        class_losses = torch.stack(class_losses)  # (C,)

        # Apply class weights
        if self.class_weights is not None and len(self.class_weights) == C:
            weights = torch.tensor(
                self.class_weights, dtype=student_logits.dtype,
                device=student_logits.device
            )
        elif self.balance_mode == 'inverse_freq':
            # Estimate class frequency from teacher predictions
            t_hard = teacher_logits.detach().argmax(dim=1)  # (B, H, W)
            total_pixels = B * H * W
            freqs = torch.stack([
                (t_hard == c).float().sum() / total_pixels for c in range(C)
            ])
            weights = 1.0 / (freqs + 1e-6)
            weights = weights / weights.sum() * C  # normalize
        else:
            weights = torch.ones(C, device=student_logits.device)

        weighted_loss = (class_losses * weights).sum() / weights.sum()
        return weighted_loss * (T * T)

    def _relation_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Inter-class correlation matrix distillation.

        Compute spatial correlation between each pair of classes and
        match student's correlation matrix to teacher's.
        """
        s_prob = F.softmax(student_logits, dim=1)  # (B, C, H, W)
        t_prob = F.softmax(teacher_logits.detach(), dim=1)

        B, C, H, W = s_prob.shape
        s_flat = s_prob.reshape(B, C, -1)  # (B, C, HW)
        t_flat = t_prob.reshape(B, C, -1)

        # Correlation matrix: (B, C, C)
        s_corr = torch.bmm(s_flat, s_flat.transpose(1, 2)) / (H * W)
        t_corr = torch.bmm(t_flat, t_flat.transpose(1, 2)) / (H * W)

        return F.mse_loss(s_corr, t_corr)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_output: Student logits (B, C, H, W)
            teacher_output: Teacher logits (B, C, H, W)
        """
        per_class_loss = self._per_class_kd(student_output, teacher_output)
        rel_loss = self._relation_loss(student_output, teacher_output)

        return per_class_loss + self.relation_weight * rel_loss
