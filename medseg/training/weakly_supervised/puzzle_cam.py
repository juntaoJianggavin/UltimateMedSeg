"""Puzzle-CAM: Improved Localization via Matching Partial and Full Features.

Jo & Yu, ICIP 2021.
Official: https://github.com/OFRIN/PuzzleCAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("puzzle_cam")
class PuzzleCAMLoss(nn.Module):
    """Puzzle-CAM loss: class_loss + puzzle_class_loss + alpha * re_loss."""

    def __init__(self, alpha: float = 4.0, alpha_schedule: float = 0.50,
                 num_pieces: int = 4, re_loss_option: str = 'masking',
                 re_loss_type: str = 'L1', **kwargs):
        super().__init__()
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule
        self.num_pieces = num_pieces
        self.re_loss_option = re_loss_option
        self.re_loss_type = re_loss_type  # 'L1' or 'L2'
        self.class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none')
        self.current_iteration = 0
        self.max_iteration = 1

    def _re_loss_fn(self, a, b):
        if self.re_loss_type == 'L2':
            return (a - b).pow(2)
        return torch.abs(a - b)  # L1 (default)

    def update_progress(self, iteration: int, max_iteration: int):
        self.current_iteration = iteration
        self.max_iteration = max(1, max_iteration)

    def _get_alpha(self):
        """Linear warm-up matching the official implementation."""
        if self.alpha_schedule == 0.0:
            return self.alpha
        return min(
            self.alpha * self.current_iteration /
            (self.max_iteration * self.alpha_schedule),
            self.alpha,
        )

    def forward(self, features_full, features_tiled_merged, predictions_full,
                predictions_tiled, image_labels, labeled_loss=None,
                iteration: Optional[int] = None,
                max_iteration: Optional[int] = None):
        if iteration is not None and max_iteration is not None:
            self.update_progress(iteration, max_iteration)

        class_loss = self.class_loss_fn(predictions_full, image_labels.float()).mean()
        p_class_loss = self.class_loss_fn(predictions_tiled, image_labels.float()).mean()

        alpha = self._get_alpha()
        if alpha > 0:
            if self.re_loss_option == 'masking':
                # Per-channel masking: re_loss_fn * class_mask, then mean
                class_mask = image_labels.float().unsqueeze(-1).unsqueeze(-1)
                re_loss = (self._re_loss_fn(features_full, features_tiled_merged)
                           * class_mask).mean()
            elif self.re_loss_option == 'selection':
                # Per-sample, per-class: select only active class features
                re_loss = torch.tensor(0.0, device=features_full.device)
                for b in range(image_labels.shape[0]):
                    class_indices = image_labels[b].nonzero(as_tuple=True)[0]
                    if class_indices.numel() == 0:
                        continue
                    sel_full = features_full[b][class_indices]
                    sel_tiled = features_tiled_merged[b][class_indices]
                    re_loss = re_loss + self._re_loss_fn(sel_full, sel_tiled).mean()
                re_loss = re_loss / max(image_labels.shape[0], 1)
            else:
                re_loss = self._re_loss_fn(
                    features_full, features_tiled_merged).mean()
        else:
            re_loss = torch.tensor(0.0, device=features_full.device)

        total_loss = class_loss + p_class_loss + alpha * re_loss
        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss
