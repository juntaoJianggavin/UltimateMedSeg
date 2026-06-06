"""SAM-Guided Weakly Supervised Segmentation.

He et al., "Weakly-Supervised Semantic Segmentation with Image-Level
Labels: from Traditional Models to Foundation Models", 2023.

Uses Segment Anything Model (SAM) generated masks as shape priors
to refine weakly-supervised pseudo-labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("sam_guided_weak")
class SAMGuidedWeakLoss(nn.Module):
    """SAM-Guided Weakly Supervised Segmentation Loss."""

    def __init__(
        self,
        sam_weight: float = 1.0,
        pseudo_weight: float = 1.0,
        boundary_refine: bool = True,
        **kwargs
    ):
        super().__init__()
        self.sam_weight = sam_weight
        self.pseudo_weight = pseudo_weight
        self.boundary_refine = boundary_refine

    def _mask_alignment_loss(
        self,
        predictions: torch.Tensor,
        sam_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Align predictions with SAM-generated mask boundaries."""
        prob = F.softmax(predictions, dim=1)  # B, C, H, W
        B, N_masks, H, W = sam_masks.shape

        consistency_loss = torch.tensor(0.0, device=predictions.device)
        count = 0

        for b in range(B):
            for n in range(N_masks):
                mask = sam_masks[b, n]  # H, W
                if mask.sum() < 10:
                    continue
                masked_prob = prob[b, :, mask > 0.5]  # C, P
                if masked_prob.shape[1] > 1:
                    mean_prob = masked_prob.mean(dim=1, keepdim=True)  # C, 1
                    consistency_loss = consistency_loss + (
                        (masked_prob - mean_prob).pow(2).mean()
                    )
                    count += 1

        if count > 0:
            consistency_loss = consistency_loss / count

        return consistency_loss

    def _boundary_refinement(
        self,
        predictions: torch.Tensor,
        sam_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Refine prediction boundaries using SAM mask edges."""
        prob = F.softmax(predictions, dim=1)

        sam_combined = sam_masks.max(dim=1)[0]  # B, H, W
        sam_h = (sam_combined[:, :, 1:] - sam_combined[:, :, :-1]).abs()
        sam_v = (sam_combined[:, 1:, :] - sam_combined[:, :-1, :]).abs()

        pred_h = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs().sum(dim=1)
        pred_v = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs().sum(dim=1)

        boundary_loss = F.mse_loss(pred_h, sam_h) + F.mse_loss(pred_v, sam_v)
        return boundary_loss

    def forward(
        self,
        predictions: torch.Tensor,
        sam_masks: torch.Tensor,
        pseudo_labels: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions (B, C, H, W)
            sam_masks: SAM-generated masks (B, N_masks, H, W)
            pseudo_labels: Pseudo-labels from CAM/other (B, H, W)
            labeled_loss: Optional base loss
        """
        align_loss = self._mask_alignment_loss(predictions, sam_masks)
        total_loss = self.sam_weight * align_loss

        if pseudo_labels is not None:
            pseudo_loss = F.cross_entropy(
                predictions, pseudo_labels, ignore_index=-1
            )
            total_loss = total_loss + self.pseudo_weight * pseudo_loss

        if self.boundary_refine:
            bd_loss = self._boundary_refinement(predictions, sam_masks)
            total_loss = total_loss + 0.5 * bd_loss

        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss

        return total_loss
