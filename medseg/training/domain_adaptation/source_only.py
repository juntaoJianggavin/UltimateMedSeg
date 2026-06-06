"""Source-Only baseline for domain adaptation comparisons.

# Reference: https://github.com/valeoai/ADVENT
# Paper: https://arxiv.org/abs/1811.12833

Algorithm summary:
    Train a segmentation network only on the labelled source domain with
    a standard supervised loss (here cross-entropy + Dice). No use of
    target data, no adaptation. This is the obligatory lower-bound
    baseline against which every DA method should be measured.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("source_only")
class SourceOnlyLoss(nn.Module):
    """Source-only training (no domain adaptation).

    Returns CE + Dice on the source predictions. If the training loop has
    already computed a supervised loss it can pass it through as
    ``labeled_loss`` and this module will simply return it unchanged.

    Args:
        num_classes: number of segmentation classes.
        ce_weight:   weight on the cross-entropy term.
        dice_weight: weight on the soft-Dice term.
        ignore_index: pixel value to ignore in CE.
    """

    def __init__(
        self,
        num_classes: int = 5,
        ce_weight: float = 0.4,
        dice_weight: float = 0.6,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    @staticmethod
    def _soft_dice(pred: torch.Tensor, target_onehot: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
        """Mean soft-Dice loss over classes."""
        prob = F.softmax(pred, dim=1)
        dims = (0, 2, 3)
        intersection = (prob * target_onehot).sum(dim=dims)
        cardinality = prob.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2.0 * intersection + eps) / (cardinality + eps)
        return 1.0 - dice.mean()

    def forward(
        self,
        source_pred: Optional[torch.Tensor] = None,
        source_labels: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            source_pred: source logits, (B, C, H, W).
            source_labels: source ground-truth, (B, H, W) long.
            labeled_loss: optional supervised loss already computed by the
                trainer; if given, returned unchanged.
        """
        if labeled_loss is not None:
            return labeled_loss

        assert source_pred is not None and source_labels is not None, (
            "SourceOnlyLoss needs either ``labeled_loss`` or "
            "(``source_pred`` + ``source_labels``)."
        )

        # CE.
        ce = F.cross_entropy(
            source_pred, source_labels.long(),
            ignore_index=self.ignore_index,
        )

        # Soft-Dice (with ignore-index handling).
        valid = source_labels != self.ignore_index
        labels_clamped = source_labels.clone()
        labels_clamped[~valid] = 0
        oh = F.one_hot(labels_clamped.long(), num_classes=self.num_classes)
        oh = oh.permute(0, 3, 1, 2).float()
        oh = oh * valid.unsqueeze(1).float()
        dice = self._soft_dice(source_pred, oh)

        return self.ce_weight * ce + self.dice_weight * dice
