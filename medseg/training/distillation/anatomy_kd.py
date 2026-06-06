"""Anatomy-Aware Knowledge Distillation for Medical Image Segmentation.

[中文] 解剖感知知识蒸馏。利用器官拓扑关系（邻接图 + 包含关系）约束 student 输出，
       保证解剖结构一致性。通过计算类别间空间关系矩阵并蒸馏关系差异。
[EN]   Anatomy-aware KD using organ topology (adjacency graph + containment)
       to constrain student outputs, ensuring anatomical consistency.
       Distills inter-class spatial relationship matrices.

Algorithm:
    1. Compute per-class centroid and spatial extent from teacher/student softmax
    2. Build adjacency matrix (pairwise centroid distances)
    3. Build containment matrix (overlap ratios)
    4. L_anatomy = MSE(adj_teacher, adj_student) + MSE(contain_teacher, contain_student)
    5. L_total = L_kd + lambda * L_anatomy

Reference:
    Anatomy-Aware Knowledge Distillation for Medical Image Segmentation.

(Self-contained: no single canonical GitHub source.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


def _compute_spatial_relations(prob: torch.Tensor) -> torch.Tensor:
    """Compute pairwise spatial relationship matrix from probability maps.

    For each pair of classes (i, j), compute the normalized centroid distance.

    Args:
        prob: (B, C, H, W) softmax probabilities.

    Returns:
        rel_matrix: (B, C, C) pairwise relationship matrix.
    """
    B, C, H, W = prob.shape
    device = prob.device

    # Create coordinate grids
    ys = torch.arange(H, dtype=prob.dtype, device=device).view(1, 1, H, 1) / max(H - 1, 1)
    xs = torch.arange(W, dtype=prob.dtype, device=device).view(1, 1, 1, W) / max(W - 1, 1)

    # Compute centroids per class: (B, C, 2)
    weights = prob + 1e-8
    cy = (weights * ys).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))  # (B, C)
    cx = (weights * xs).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))  # (B, C)
    centroids = torch.stack([cx, cy], dim=-1)  # (B, C, 2)

    # Pairwise centroid distances: (B, C, C)
    diff = centroids.unsqueeze(2) - centroids.unsqueeze(1)  # (B, C, C, 2)
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)  # (B, C, C)

    # Normalize to [0, 1]
    dist_max = dist.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    rel_matrix = 1.0 - dist / (dist_max + 1e-8)

    return rel_matrix


def _compute_containment(prob: torch.Tensor) -> torch.Tensor:
    """Compute class overlap / containment matrix.

    For each pair (i, j), compute the overlap ratio between class i and j.

    Args:
        prob: (B, C, H, W) softmax probabilities.

    Returns:
        contain_matrix: (B, C, C) containment matrix.
    """
    B, C, H, W = prob.shape
    prob_flat = prob.reshape(B, C, -1)  # (B, C, HW)

    # Intersection: min(p_i, p_j)
    # Use softmax values as soft membership
    p_i = prob_flat.unsqueeze(2)  # (B, C, 1, HW)
    p_j = prob_flat.unsqueeze(1)  # (B, 1, C, HW)

    intersection = torch.minimum(p_i, p_j).sum(dim=-1)  # (B, C, C)
    area_i = prob_flat.sum(dim=-1).unsqueeze(2) + 1e-8  # (B, C, 1)

    # Containment: intersection / area_i
    contain_matrix = intersection / area_i  # (B, C, C)

    return contain_matrix


@LOSS_REGISTRY.register("anatomy_kd")
class AnatomyKDLoss(nn.Module):
    """Anatomy-Aware Knowledge Distillation.

    利用器官拓扑关系（邻接距离 + 包含关系）蒸馏解剖学先验。
    Distills anatomical topology (adjacency + containment) from teacher.

    Args:
        anatomy_weight: Weight for anatomy loss (default 1.0).
        kd_weight: Weight for standard KD loss (default 1.0).
        temperature: Temperature for logit KD (default 4.0).
        relation_type: 'adjacency', 'containment', or 'both' (default 'both').
    """

    def __init__(
        self,
        anatomy_weight: float = 1.0,
        kd_weight: float = 1.0,
        temperature: float = 4.0,
        relation_type: str = 'both',
        **kwargs,
    ):
        super().__init__()
        self.anatomy_weight = anatomy_weight
        self.kd_weight = kd_weight
        self.temperature = temperature
        self.relation_type = relation_type

    def _logit_kd(self, student_logits, teacher_logits):
        T = self.temperature
        s_soft = F.log_softmax(student_logits / T, dim=1)
        t_soft = F.softmax(teacher_logits.detach() / T, dim=1)
        return F.kl_div(s_soft, t_soft, reduction='batchmean') * (T * T)

    def _anatomy_loss(self, student_logits, teacher_logits):
        s_prob = F.softmax(student_logits, dim=1)
        t_prob = F.softmax(teacher_logits.detach(), dim=1)

        loss = torch.tensor(0.0, device=student_logits.device)

        if self.relation_type in ('adjacency', 'both'):
            s_adj = _compute_spatial_relations(s_prob)
            t_adj = _compute_spatial_relations(t_prob)
            loss = loss + F.mse_loss(s_adj, t_adj)

        if self.relation_type in ('containment', 'both'):
            s_contain = _compute_containment(s_prob)
            t_contain = _compute_containment(t_prob)
            loss = loss + F.mse_loss(s_contain, t_contain)

        return loss

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
        kd_loss = self._logit_kd(student_output, teacher_output)
        anat_loss = self._anatomy_loss(student_output, teacher_output)
        return self.kd_weight * kd_loss + self.anatomy_weight * anat_loss
