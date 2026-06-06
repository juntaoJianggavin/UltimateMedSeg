"""CBMT: Class-Balanced Mean Teacher for source-free domain adaptation.

# Paper: https://arxiv.org/abs/2405.16860
# Reference: https://github.com/whq-xxh/ADA4MIA

Algorithm summary (from the benchmark description):
    A mean-teacher framework specialised for medical image segmentation
    under source-free / active DA: the student predicts on weakly-augmented
    target images and is regularised against an EMA teacher's predictions
    on the same images, with a consistency loss whose per-pixel weight
    follows the *inverse frequency* of the pseudo-labelled class. To avoid
    noisy per-batch frequency estimates we maintain a running EMA over
    class frequencies instead of recomputing them from a single batch, and
    use a sigmoid ramp-up on the consistency weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("class_balanced_mt")
class CBMTLoss(nn.Module):
    """Class-Balanced Mean Teacher for source-free domain adaptation.

    Reference implementation (not copied):
        https://github.com/whq-xxh/ADA4MIA

    Components:
      * Sigmoid ramp-up for the consistency weight (standard MT recipe).
      * Inverse-frequency class weighting with an EMA buffer over class
        frequencies (avoids the high-variance per-batch estimate of the
        original ad-hoc implementation).
      * MSE consistency between student and teacher softmax outputs.
    """

    def __init__(
        self,
        consistency_weight: float = 1.0,
        num_classes: int = 5,
        rampup_epochs: int = 40,
        freq_momentum: float = 0.99,
        **kwargs,
    ):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.num_classes = num_classes
        self.rampup_epochs = rampup_epochs
        self.current_epoch = 0
        self.freq_momentum = freq_momentum

        # EMA over class frequencies (normalised; starts uniform).
        self.register_buffer(
            "_class_freq_ema",
            torch.full((num_classes,), 1.0 / num_classes),
        )
        self.register_buffer(
            "_freq_initialised",
            torch.tensor(False),
        )

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def _get_rampup_weight(self, epoch):
        """Sigmoid rampup for consistency weight."""
        if epoch < self.rampup_epochs:
            alpha = epoch / float(self.rampup_epochs)
            return float(np.exp(-5.0 * (1.0 - alpha) ** 2))
        return 1.0

    @torch.no_grad()
    def _update_freq_ema(self, pseudo_labels: torch.Tensor):
        """EMA update of class-frequency vector."""
        counts = torch.zeros(self.num_classes, device=pseudo_labels.device)
        for c in range(self.num_classes):
            counts[c] = (pseudo_labels == c).sum()
        freq = counts / (counts.sum() + 1e-6)
        if not bool(self._freq_initialised.item()):
            self._class_freq_ema = freq
            self._freq_initialised = torch.tensor(True, device=pseudo_labels.device)
        else:
            m = self.freq_momentum
            self._class_freq_ema = m * self._class_freq_ema + (1.0 - m) * freq

    def _class_weights_from_ema(self) -> torch.Tensor:
        """Inverse-frequency weights, normalised so they sum to num_classes."""
        inv = 1.0 / (self._class_freq_ema + 1e-6)
        return inv / inv.sum() * self.num_classes

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        labeled_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_pred: Student predictions (B, C, H, W)
            teacher_pred: Teacher (EMA) predictions (B, C, H, W)
            labeled_loss: Supervised loss
        """
        teacher_prob = F.softmax(teacher_pred, dim=1)
        _, pseudo_label = teacher_prob.max(dim=1)

        # Refresh the running class-frequency EMA.
        # NOTE: ``forward`` is called once per training *step* (per minibatch),
        # so this EMA update happens at per-step granularity, which is much
        # smoother than the per-epoch refresh used by some re-implementations
        # and avoids 'class disappears for a whole epoch' artefacts.
        self._update_freq_ema(pseudo_label)
        class_weights = self._class_weights_from_ema()
        weight_map = class_weights[pseudo_label].unsqueeze(1)  # (B, 1, H, W)

        # Consistency loss (MSE between student and teacher softmax).
        student_prob = F.softmax(student_pred, dim=1)
        consistency_loss = (weight_map * (student_prob - teacher_prob) ** 2).mean()

        # Sigmoid rampup.
        weight = self._get_rampup_weight(self.current_epoch)
        consistency_loss = consistency_loss * weight * self.consistency_weight

        total_loss = consistency_loss
        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss

        return total_loss
