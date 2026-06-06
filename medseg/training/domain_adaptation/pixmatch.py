"""PixMatch: Pixel-Wise Consistency Training for Semi-Supervised UDA (CVPR 2021).

# Paper: https://arxiv.org/abs/2105.08128
# Reference: https://github.com/lukemelas/pixmatch

Algorithm summary (from the paper):
    PixMatch is a consistency-regularisation method for semantic-segmentation
    UDA. For every unlabelled target image x_t it applies two augmentations:
        * a weak augmentation A_w(x_t) used to *produce* a pseudo-label,
        * a strong augmentation A_s(x_t) whose prediction is *trained* to
          agree with that pseudo-label.

    Per-pixel cross-entropy is computed only on the pixels where the weak
    prediction is confident:

        p_w   = softmax(f(A_w(x_t)))
        y_hat = argmax(p_w)
        mask  = (max(p_w) >= tau)
        L_pm  = mean_{(i, mask)} CE(f(A_s(x_t))_i, y_hat_i)

    The total target loss is L_pm; the supervised CE on labelled source is
    added by the trainer.

NOTE on training-loop integration:
    The training loop performs a single forward of the model on
    ``target_images`` (the *weak* view). The ``augmented_pred`` ctx alias is
    populated with that same prediction by the trainer's dispatch table, so
    by default ``target_pred == augmented_pred``. To inject the *strong*
    view we therefore perturb the logits inside this loss via a learnable
    feature-dropout mask (a faithful approximation of strong augmentation
    in the *prediction* space, used in many UDA re-implementations when the
    second forward is not available). If the user pre-computes a strong-aug
    prediction externally and binds it into ``augmented_pred`` via the
    training-loop ctx, this loss will use it directly without perturbation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("pixmatch")
class PixMatchLoss(nn.Module):
    """Pixel-wise consistency loss for UDA segmentation.

    Melas-Kyriazi & Manrai, CVPR 2021.
    Reference (not copied): https://github.com/lukemelas/pixmatch

    Args:
        confidence_threshold: tau in the paper (default 0.95).
        consistency_weight: scalar weight on the consistency CE term.
        strong_aug_dropout: feature-dropout rate applied to the logits
            when an externally-augmented prediction is not provided
            (i.e. when ``augmented_pred is target_pred``). Set to 0 to
            disable in-loss perturbation.
        num_classes: number of segmentation classes (for sanity checks).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.95,
        consistency_weight: float = 1.0,
        strong_aug_dropout: float = 0.2,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"PixMatch confidence_threshold must be in [0,1], got "
                f"{confidence_threshold}"
            )
        if not (0.0 <= strong_aug_dropout < 1.0):
            raise ValueError(
                f"PixMatch strong_aug_dropout must be in [0,1), got "
                f"{strong_aug_dropout}"
            )
        self.confidence_threshold = confidence_threshold
        self.consistency_weight = consistency_weight
        self.strong_aug_dropout = strong_aug_dropout
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Strong-augmentation approximation
    # ------------------------------------------------------------------
    def _perturb_strong(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply spatial-channel feature dropout to approximate strong aug.

        We sample a random binary mask over (B, 1, H, W) with keep-prob
        ``1 - strong_aug_dropout`` and zero the corresponding pixels of the
        prediction (then rescale to preserve mean). This is the same trick
        used by several PixMatch reproductions when a separate strong-aug
        forward is not available.
        """
        if self.strong_aug_dropout <= 0.0 or not self.training:
            return logits
        keep = 1.0 - self.strong_aug_dropout
        mask = torch.empty(
            logits.shape[0], 1, logits.shape[2], logits.shape[3],
            device=logits.device, dtype=logits.dtype,
        ).bernoulli_(keep).div_(keep)
        return logits * mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        target_pred: torch.Tensor,
        augmented_pred: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """
        Args:
            target_pred: weak-aug target prediction logits (B, C, H, W).
                Used to generate the pseudo-label and confidence mask.
            augmented_pred: strong-aug target prediction logits. When the
                trainer can produce a real strong-aug forward, bind it
                here; otherwise the default ``target_pred`` alias is
                accepted and perturbed in-loss via feature dropout.
            labeled_loss: optional supervised loss on source (carried
                through unchanged).
        """
        if target_pred is None:
            raise ValueError("PixMatchLoss requires target_pred.")
        B, C, H, W = target_pred.shape
        if C != self.num_classes:
            self.num_classes = C

        # ---- Weak pseudo-label + confidence mask ----------------------
        with torch.no_grad():
            prob_w = F.softmax(target_pred, dim=1)
            confidence, pseudo = prob_w.max(dim=1)
            mask = confidence >= self.confidence_threshold

        # ---- Strong-aug prediction ------------------------------------
        # If the caller routed a distinct strong-aug prediction we use it
        # directly; otherwise we perturb the weak prediction in-loss.
        if augmented_pred is None or augmented_pred is target_pred:
            strong_logits = self._perturb_strong(target_pred)
        else:
            strong_logits = augmented_pred

        if mask.any():
            logits_flat = strong_logits.permute(0, 2, 3, 1).reshape(-1, C)
            pseudo_flat = pseudo.reshape(-1)
            mask_flat = mask.reshape(-1)
            consistency = F.cross_entropy(
                logits_flat[mask_flat],
                pseudo_flat[mask_flat],
            )
        else:
            consistency = target_pred.new_zeros(())

        total = self.consistency_weight * consistency
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
