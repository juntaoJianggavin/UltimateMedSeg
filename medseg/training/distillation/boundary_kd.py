"""Boundary-Aware Knowledge Distillation for Medical Image Segmentation.

[中文] 边界感知知识蒸馏。在器官边界区域加强蒸馏权重，使用 Laplacian 算子提取
       teacher 预测的边界掩码，在边界处放大 KD loss，确保 student 学习到精确的
       器官边界。
[EN]   Boundary-aware KD that amplifies distillation loss at organ boundaries.
       Uses Laplacian operator to extract boundary masks from teacher predictions,
       ensuring student learns precise organ boundaries.

Algorithm:
    1. Extract boundary mask from teacher softmax via Laplacian edge detection
    2. Compute weighted KD: boundary regions get amplified weight
    3. L = L_kd + lambda_b * boundary_weighted_KD

Reference:
    Boundary-Aware Geometric Encoding for Segmenting Point Clouds +
    Medical KD boundary emphasis.

(Self-contained: no single canonical GitHub source.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


def _laplacian_edge_mask(
    prob: torch.Tensor, kernel_size: int = 3, threshold: float = 0.1
) -> torch.Tensor:
    """Extract boundary mask from softmax using Laplacian operator.

    Args:
        prob: (B, C, H, W) softmax probabilities.
        kernel_size: Laplacian kernel size (3 or 5).
        threshold: Edge threshold for binarization.

    Returns:
        edge_mask: (B, 1, H, W) binary edge mask.
    """
    # Get the dominant class probability
    prob_max, _ = prob.max(dim=1, keepdim=True)  # (B, 1, H, W)

    # Build Laplacian kernel
    if kernel_size == 5:
        lap = torch.tensor([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0],
        ], dtype=prob.dtype, device=prob.device).view(1, 1, 5, 5)
    else:
        lap = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ], dtype=prob.dtype, device=prob.device).view(1, 1, 3, 3)

    pad = kernel_size // 2
    padded = F.pad(prob_max, [pad]*4, mode='reflect')
    edge_response = F.conv2d(padded, lap)
    edge_response = torch.abs(edge_response)

    # Normalize to [0, 1]
    edge_max = edge_response.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
    edge_response = edge_response / edge_max

    # Soft edge mask (differentiable)
    edge_mask = torch.sigmoid((edge_response - threshold) * 10.0)
    return edge_mask


@LOSS_REGISTRY.register("boundary_kd")
class BoundaryKDLoss(nn.Module):
    """Boundary-Aware Knowledge Distillation.

    在器官边界区域加强蒸馏权重，确保 student 学习精确边界。
    Amplifies KD loss at organ boundaries for precise boundary learning.

    Args:
        boundary_weight: Extra weight for boundary regions (default 3.0).
        kd_weight: Base KD loss weight (default 1.0).
        temperature: Softmax temperature (default 4.0).
        edge_kernel_size: Laplacian kernel size 3 or 5 (default 3).
        edge_threshold: Edge binarization threshold (default 0.1).
    """

    def __init__(
        self,
        boundary_weight: float = 3.0,
        kd_weight: float = 1.0,
        temperature: float = 4.0,
        edge_kernel_size: int = 3,
        edge_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.kd_weight = kd_weight
        self.temperature = temperature
        self.edge_kernel_size = edge_kernel_size
        self.edge_threshold = edge_threshold

    def _kd_loss(self, student_logits, teacher_logits):
        """Standard logit KD."""
        T = self.temperature
        s_soft = F.log_softmax(student_logits / T, dim=1)
        t_soft = F.softmax(teacher_logits.detach() / T, dim=1)
        return F.kl_div(s_soft, t_soft, reduction='batchmean') * (T * T)

    def _boundary_kd(self, student_logits, teacher_logits, edge_mask):
        """Boundary-weighted KD loss."""
        T = self.temperature
        s_soft = F.softmax(student_logits / T, dim=1)
        t_soft = F.softmax(teacher_logits.detach() / T, dim=1)

        # Per-pixel KL divergence
        kl = t_soft * (torch.log(t_soft + 1e-8) - torch.log(s_soft + 1e-8))
        kl = kl.sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Weight by boundary mask
        weight_map = 1.0 + self.boundary_weight * edge_mask
        weighted_kl = kl * weight_map

        return weighted_kl.mean() * (T * T)

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
        # Extract boundary mask from teacher
        t_prob = F.softmax(teacher_output.detach(), dim=1)
        edge_mask = _laplacian_edge_mask(
            t_prob, self.edge_kernel_size, self.edge_threshold
        )

        base_kd = self._kd_loss(student_output, teacher_output)
        boundary_kd = self._boundary_kd(student_output, teacher_output, edge_mask)

        return self.kd_weight * base_kd + boundary_kd
