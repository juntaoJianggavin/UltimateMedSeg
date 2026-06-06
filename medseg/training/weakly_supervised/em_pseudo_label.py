"""Expectation-Maximization for pseudo-label refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("em_pseudo_label")
class EMPseudoLabelLoss(nn.Module):
    """Expectation-Maximization for pseudo-label refinement.

    Iteratively refines pseudo-labels and updates model.
    """

    def __init__(self, base_loss_fn=None, num_iterations: int = 5,
                 confidence_threshold: float = 0.7, **kwargs):
        super().__init__()
        self.num_iterations = num_iterations
        self.confidence_threshold = confidence_threshold

        if base_loss_fn is None:
            from medseg.losses.compound_loss import CompoundLoss
            self.base_loss_fn = CompoundLoss()
        else:
            self.base_loss_fn = base_loss_fn

    def forward(self, predictions: torch.Tensor, weak_labels: torch.Tensor,
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        if target is not None:
            return self.base_loss_fn(predictions, target)

        refined_labels = self._e_step(predictions, weak_labels)
        return F.cross_entropy(predictions, refined_labels, ignore_index=-1)

    def _e_step(self, predictions, weak_labels):
        probs = predictions.softmax(dim=1)
        max_probs, pseudo_labels = probs.max(dim=1)
        confident = max_probs > self.confidence_threshold
        pseudo_labels[~confident] = -1
        return pseudo_labels
