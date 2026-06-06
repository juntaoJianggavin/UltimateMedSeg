"""Cross-Modality Knowledge Distillation (CT -> MRI) for Medical Segmentation.

[中文] 跨模态知识蒸馏。将 CT teacher 的分割知识蒸馏到 MRI student，使用 MMD
       (Maximum Mean Discrepancy) 进行模态不变特征对齐，使 student 在目标
       模态上学习 source 模态的解剖结构知识。
[EN]   Cross-modality KD transferring CT teacher knowledge to MRI student.
       Uses Maximum Mean Discrepancy (MMD) for modality-invariant feature
       alignment, enabling anatomical knowledge transfer across modalities.

Algorithm:
    1. Standard KD loss on logits (modality-agnostic)
    2. MMD feature alignment between student and teacher feature spaces
    3. Cross-attention consistency for structural alignment
    4. L = L_kd + lambda_feat * MMD(feat_s, feat_t) + lambda_cross * cross_attn_loss

Reference:
    Cross-Modality Knowledge Distillation for Medical Image Segmentation.

(Self-contained: no single canonical GitHub source.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


def _gaussian_kernel(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute Gaussian kernel matrix for MMD.

    Args:
        x: (B, D) feature vectors.
        sigma: Kernel bandwidth.

    Returns:
        kernel: (B, B) kernel matrix.
    """
    dist = torch.cdist(x, x) ** 2
    return torch.exp(-dist / (2 * sigma ** 2))


def _mmd_loss(
    source_feat: torch.Tensor, target_feat: torch.Tensor, sigma: float = 1.0
) -> torch.Tensor:
    """Compute Maximum Mean Discrepancy between two feature distributions.

    MMD^2 = E[k(s,s)] + E[k(t,t)] - 2*E[k(s,t)]

    Args:
        source_feat: (B, D) source features.
        target_feat: (B, D) target features.
        sigma: Gaussian kernel bandwidth.

    Returns:
        mmd: Scalar MMD loss.
    """
    k_ss = _gaussian_kernel(source_feat, sigma)
    k_tt = _gaussian_kernel(target_feat, sigma)
    k_st = _gaussian_kernel(
        torch.cat([source_feat, target_feat], dim=0), sigma
    )

    B = source_feat.shape[0]
    # E[k(s,s)] - diagonal excluded
    e_ss = (k_ss.sum() - k_ss.diag().sum()) / (B * (B - 1) + 1e-8)
    e_tt = (k_tt.sum() - k_tt.diag().sum()) / (B * (B - 1) + 1e-8)

    # E[k(s,t)] — cross terms from combined kernel
    # Upper-left and lower-right are k(s,s) and k(t,t), off-diagonals are k(s,t)
    k_st_full = _gaussian_kernel(
        torch.cat([source_feat, target_feat], dim=0), sigma
    )
    # Extract cross terms: k_st_full[:B, B:] and k_st_full[B:, :B]
    e_st = k_st_full[:B, B:].mean()

    mmd = e_ss + e_tt - 2 * e_st
    return mmd.clamp(min=0.0)


@LOSS_REGISTRY.register("cross_modality_kd")
class CrossModalityKDLoss(nn.Module):
    """Cross-Modality Knowledge Distillation.

    CT teacher 蒸馏到 MRI student，MMD 特征对齐。
    CT-to-MRI distillation with MMD feature alignment.

    Args:
        temperature: Softmax temperature (default 4.0).
        kd_weight: Logit KD loss weight (default 1.0).
        mmd_weight: MMD feature alignment weight (default 0.5).
        mmd_sigma: Gaussian kernel bandwidth for MMD (default 1.0).
        cross_attn_weight: Cross-attention consistency weight (default 0.3).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        kd_weight: float = 1.0,
        mmd_weight: float = 0.5,
        mmd_sigma: float = 1.0,
        cross_attn_weight: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.temperature = temperature
        self.kd_weight = kd_weight
        self.mmd_weight = mmd_weight
        self.mmd_sigma = mmd_sigma
        self.cross_attn_weight = cross_attn_weight

    def _logit_kd(self, student_logits, teacher_logits):
        """Standard logit distillation."""
        T = self.temperature
        s_soft = F.log_softmax(student_logits / T, dim=1)
        t_soft = F.softmax(teacher_logits.detach() / T, dim=1)
        return F.kl_div(s_soft, t_soft, reduction='batchmean') * (T * T)

    def _feature_mmd(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """MMD alignment between student and teacher feature spaces.

        Uses global average pooled logits as compact feature representation.
        """
        B, C, H, W = student_logits.shape

        # Global average pool to get compact features
        s_feat = F.adaptive_avg_pool2d(student_logits, 1).view(B, C)
        t_feat = F.adaptive_avg_pool2d(teacher_logits.detach(), 1).view(B, C)

        return _mmd_loss(s_feat, t_feat, self.mmd_sigma)

    def _cross_attention_consistency(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attention consistency between modalities.

        Compute spatial attention maps and enforce consistency.
        """
        # Spatial attention: sum across channels
        s_attn = student_logits.softmax(dim=1).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        t_attn = teacher_logits.detach().softmax(dim=1).mean(dim=1, keepdim=True)

        # Normalize
        s_attn = s_attn / (s_attn.amax(dim=(2, 3), keepdim=True) + 1e-8)
        t_attn = t_attn / (t_attn.amax(dim=(2, 3), keepdim=True) + 1e-8)

        return F.mse_loss(s_attn, t_attn)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_output: Student (MRI) logits (B, C, H, W)
            teacher_output: Teacher (CT) logits (B, C, H, W)
        """
        kd_loss = self._logit_kd(student_output, teacher_output)
        mmd_loss = self._feature_mmd(student_output, teacher_output)
        cross_loss = self._cross_attention_consistency(student_output, teacher_output)

        return (
            self.kd_weight * kd_loss
            + self.mmd_weight * mmd_loss
            + self.cross_attn_weight * cross_loss
        )
