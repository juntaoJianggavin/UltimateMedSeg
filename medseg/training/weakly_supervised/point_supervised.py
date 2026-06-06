"""Point-Supervised Segmentation Loss.

Bearman et al., ECCV 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("point_supervised")
class PointSupervisedLoss(nn.Module):
    """Point-Supervised Segmentation Loss.

    Bearman et al., "What's the Point: Semantic Segmentation with Point
    Annotations", ECCV 2016.
    """

    def __init__(self, objectness_weight: float = 0.5, point_radius: int = 5,
                 ignore_index: int = -1, use_smoothness: bool = False,
                 smoothness_weight: float = 0.1, smoothness_sigma: float = 15.0, **kwargs):
        super().__init__()
        self.objectness_weight = objectness_weight
        self.point_radius = point_radius
        self.ignore_index = ignore_index
        self.use_smoothness = use_smoothness
        self.smoothness_weight = smoothness_weight
        self.smoothness_sigma = smoothness_sigma

    def _expand_points(self, points):
        if self.point_radius <= 0:
            return points
        kernel = 2 * self.point_radius + 1
        dilated_label = torch.full_like(points, self.ignore_index)
        classes = torch.unique(points)
        classes = classes[classes != self.ignore_index]
        for c in classes.tolist():
            m = (points == c).float().unsqueeze(1)
            d = F.max_pool2d(m, kernel_size=kernel, stride=1, padding=self.point_radius).squeeze(1)
            dilated_label = torch.where(d > 0, torch.full_like(points, int(c)), dilated_label)
        original = (points != self.ignore_index)
        return torch.where(original, points, dilated_label)

    def _objectness_prior(self, predictions):
        prob = F.softmax(predictions, dim=1)
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1)
        return entropy.mean()

    def _superpixel_smoothness(self, predictions, images):
        log_prob = F.log_softmax(predictions, dim=1)
        prob = log_prob.exp()
        col_diff = (images[:, :, :, 1:] - images[:, :, :, :-1]).pow(2).sum(dim=1)
        col_w = torch.exp(-col_diff / (2.0 * self.smoothness_sigma ** 2))
        ce_h = -(prob[:, :, :, :-1] * log_prob[:, :, :, 1:]
                 + prob[:, :, :, 1:] * log_prob[:, :, :, :-1]).sum(dim=1)
        loss_h = (col_w * ce_h).mean()
        row_diff = (images[:, :, 1:, :] - images[:, :, :-1, :]).pow(2).sum(dim=1)
        row_w = torch.exp(-row_diff / (2.0 * self.smoothness_sigma ** 2))
        ce_v = -(prob[:, :, :-1, :] * log_prob[:, :, 1:, :]
                 + prob[:, :, 1:, :] * log_prob[:, :, :-1, :]).sum(dim=1)
        loss_v = (row_w * ce_v).mean()
        return loss_h + loss_v

    def forward(self, predictions: torch.Tensor, points: torch.Tensor,
                images: Optional[torch.Tensor] = None,
                labeled_loss: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        expanded = self._expand_points(points)
        point_loss = F.cross_entropy(predictions, expanded, ignore_index=self.ignore_index)
        objectness_loss = self._objectness_prior(predictions)
        total_loss = point_loss + self.objectness_weight * objectness_loss

        if self.use_smoothness:
            if images is None:
                raise ValueError("PointSupervisedLoss(use_smoothness=True) requires ``images``.")
            sm_loss = self._superpixel_smoothness(predictions, images)
            total_loss = total_loss + self.smoothness_weight * sm_loss

        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss
