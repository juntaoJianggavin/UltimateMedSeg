"""Tree Energy Loss for sparsely-annotated semantic segmentation.

Liang et al., CVPR 2022.
Official: https://github.com/megviiresearch/TEL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("tree_energy")
class TreeEnergyLoss(nn.Module):
    """Tree Energy Loss: grid-graph approximation to MST-based tree energy."""

    def __init__(self, energy_weight: float = 1.0, sigma: float = 5.0,
                 tree_loss_weight: Optional[float] = None,
                 tree_topology: str = "knn4", **kwargs):
        super().__init__()
        if tree_loss_weight is not None:
            energy_weight = tree_loss_weight
        self.energy_weight = energy_weight
        self.sigma = sigma
        if tree_topology not in ("knn4", "knn8"):
            raise ValueError(f"tree_topology must be 'knn4' or 'knn8', got {tree_topology!r}")
        self.tree_topology = tree_topology

    def _compute_unary_energy(self, predictions):
        prob = F.softmax(predictions, dim=1)
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1)
        return entropy.mean()

    def _pair_energy(self, prob_a, prob_b, img_a, img_b):
        img_diff = (img_a - img_b).pow(2).sum(dim=1)
        edge_w = torch.exp(-img_diff / (2.0 * self.sigma ** 2))
        lbl_diff = (prob_a - prob_b).pow(2).sum(dim=1)
        return (edge_w * lbl_diff).mean()

    def _compute_pairwise_energy(self, predictions, images):
        prob = F.softmax(predictions, dim=1)
        loss = self._pair_energy(prob[:, :, :, 1:], prob[:, :, :, :-1],
                                 images[:, :, :, 1:], images[:, :, :, :-1])
        loss = loss + self._pair_energy(prob[:, :, 1:, :], prob[:, :, :-1, :],
                                        images[:, :, 1:, :], images[:, :, :-1, :])
        if self.tree_topology == "knn8":
            loss = loss + self._pair_energy(prob[:, :, 1:, :-1], prob[:, :, :-1, 1:],
                                            images[:, :, 1:, :-1], images[:, :, :-1, 1:])
            loss = loss + self._pair_energy(prob[:, :, 1:, 1:], prob[:, :, :-1, :-1],
                                            images[:, :, 1:, 1:], images[:, :, :-1, :-1])
        return loss

    def forward(self, predictions: torch.Tensor, images: torch.Tensor,
                labeled_loss: Optional[torch.Tensor] = None) -> torch.Tensor:
        unary = self._compute_unary_energy(predictions)
        pairwise = self._compute_pairwise_energy(predictions, images)
        total_loss = self.energy_weight * (unary + pairwise)
        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss
