"""UNet-specific Knowledge Distillation (Hinton et al.).

Logit / feature / attention / multi-scale distillation.
Generic KD — no single official repository.

Reference implementation: https://arxiv.org/abs/1503.02531
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("unet_distillation")
class UNetDistillationLoss(nn.Module):
    """UNet-specific knowledge distillation loss.

    Combines task loss with distillation loss to transfer knowledge
    from teacher UNet to student UNet.
    """

    def __init__(
        self,
        base_loss_fn=None,
        temperature: float = 4.0,
        alpha: float = 0.5,
        distillation_type: str = 'logit',
        feature_weights: Optional[List[float]] = None,
        ignore_index: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_type = distillation_type
        self.feature_weights = feature_weights
        self.ignore_index = ignore_index

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
        student_features: Optional[Dict] = None,
        teacher_features: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_output: Student model output (B, C, H, W)
            teacher_output: Teacher model output (B, C, H, W)
            target: Ground truth labels (B, H, W)
            student_features: Student intermediate features (optional)
            teacher_features: Teacher intermediate features (optional)
        """
        task_loss = self.base_loss_fn(student_output, target)

        if self.distillation_type == 'logit':
            distill_loss = self._logit_distillation(student_output, teacher_output)
        elif self.distillation_type == 'feature':
            distill_loss = self._feature_distillation(student_features, teacher_features)
        elif self.distillation_type == 'attention':
            distill_loss = self._attention_distillation(student_output, teacher_output)
        elif self.distillation_type == 'multi_scale':
            distill_loss = self._multi_scale_distillation(student_features, teacher_features)
        else:
            raise ValueError(f"Unknown distillation_type: {self.distillation_type}")

        total_loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
        return total_loss

    def _device_hint(self):
        """Best-effort device picker for zero-loss returns.

        Tries this module's own parameters first; falls back to CPU if the
        module has no parameters yet. Callers that previously hard-coded
        ``device='cpu'`` here would silently break on GPU runs.
        """
        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return torch.device('cpu')

    def _logit_distillation(self, student_logits, teacher_logits):
        """Logit-based KD (Hinton et al.)."""
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        return kd_loss * (T * T)

    def _feature_distillation(self, student_features, teacher_features):
        """Feature-based distillation."""
        if student_features is None or teacher_features is None:
            return torch.zeros((), device=self._device_hint())
        total_loss = 0.0
        count = 0
        for key in student_features.keys():
            if key in teacher_features:
                s_feat = student_features[key]
                t_feat = teacher_features[key]
                if s_feat.shape != t_feat.shape:
                    t_feat = F.interpolate(t_feat, size=s_feat.shape[2:],
                                           mode='bilinear', align_corners=False)
                total_loss += F.mse_loss(s_feat, t_feat.detach())
                count += 1
        return total_loss / max(count, 1)

    def _attention_distillation(self, student_output, teacher_output):
        """Attention-based distillation (spatial attention maps)."""
        student_attn = self._get_attention_map(student_output)
        teacher_attn = self._get_attention_map(teacher_output)
        return F.mse_loss(student_attn, teacher_attn.detach())

    def _get_attention_map(self, logits):
        attn = logits.exp().sum(dim=1, keepdim=True)
        attn = attn / (attn.max() + 1e-8)
        return attn

    def _multi_scale_distillation(self, student_features, teacher_features):
        """Multi-scale feature distillation with weighted levels."""
        if student_features is None or teacher_features is None:
            return torch.zeros((), device=self._device_hint())
        total_loss = 0.0
        weights = self.feature_weights or [1.0] * len(student_features)
        for idx, (key, s_feat) in enumerate(student_features.items()):
            if key in teacher_features:
                t_feat = teacher_features[key]
                if s_feat.shape != t_feat.shape:
                    t_feat = F.interpolate(t_feat, size=s_feat.shape[2:],
                                           mode='bilinear', align_corners=False)
                total_loss += weights[idx] * F.mse_loss(s_feat, t_feat.detach())
        return total_loss
