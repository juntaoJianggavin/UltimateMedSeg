"""Pseudo-Label / Self-Training (Lee, ICML 2013 Workshop).

Paper: https://www.researchgate.net/publication/280581078
Reference implementation (not copied): https://github.com/iBelieveCJM/pseudo_label-pytorch

Algorithm (Lee, "Pseudo-Label: The simple and efficient semi-supervised
learning method for deep neural networks"):

    On every unlabelled sample take the network's current argmax as a hard
    pseudo-label and add a cross-entropy term against it.  The unsupervised
    weight follows the paper's piecewise-linear schedule::

        alpha(t) = 0                       , t < T1
                 = alpha_f * (t - T1) /
                              (T2 - T1)    , T1 <= t < T2
                 = alpha_f                 , t >= T2

    where ``T1, T2`` are epoch thresholds and ``alpha_f`` is the final
    weight (Lee used ``alpha_f = 3``).  Total loss::

        L = CE(f(x_l), y_l)  +  alpha(t) * CE(f(x_u), argmax(f(x_u)))

    A confidence threshold can be added on top to ignore low-confidence
    pseudo-labels -- this is a standard practical extension that has become
    the default in modern self-training pipelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import get_strong_augmentation


def _pseudo_label_schedule(epoch: int, T1: int, T2: int, alpha_f: float) -> float:
    """Lee 2013 Eq. 16 piecewise-linear ramp."""
    if epoch < T1:
        return 0.0
    if epoch >= T2:
        return alpha_f
    return alpha_f * (epoch - T1) / max(T2 - T1, 1)


class PseudoLabel(BaseSemiMethod):
    """Self-training with argmax pseudo-labels and a piecewise-linear ramp.

    Args:
        model: Student network.
        device: Torch device.
        alpha_final: Final unsupervised weight ``alpha_f`` (paper: 3.0).
        T1: Epoch below which the unsup loss is zero (paper: ~100/600).
        T2: Epoch at which ``alpha`` reaches ``alpha_final`` (paper: ~600/600).
            For small ``total_epochs`` we recommend scaling ``T1, T2``
            proportionally (e.g. 10% / 80% of total).
        confidence_threshold: If ``> 0``, only pseudo-labels whose softmax
            confidence is above this value contribute to the loss.  Set to
            ``0.0`` to disable (pure Lee 2013).  Default: ``0.0``.
        use_strong_aug: Whether the student sees a strong-augmented view
            (modern self-training default).  The argmax pseudo-label is
            always taken from the *clean* prediction.
        consistency_weight: kept for BaseSemiMethod backward-compat but
            unused (``alpha_final`` is the actual weight here).
        rampup_epochs: kept for backward-compat -- if both ``T1`` and ``T2``
            are left at their defaults (``-1``) we derive
            ``T1 = rampup_epochs // 4`` and ``T2 = rampup_epochs * 4`` so the
            method still works when only ``rampup_epochs`` is supplied.
        img_size: Image spatial size for strong augmentation.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 alpha_final: float = 3.0,
                 T1: int = -1,
                 T2: int = -1,
                 confidence_threshold: float = 0.0,
                 use_strong_aug: bool = True,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.alpha_final = float(alpha_final)
        # Allow callers to set the schedule directly, otherwise derive from
        # the standard rampup_epochs argument so this method composes with
        # the rest of the framework's config style.
        if T1 < 0 or T2 < 0:
            T1 = max(0, rampup_epochs // 4)
            T2 = max(T1 + 1, rampup_epochs * 4)
        self.T1 = int(T1)
        self.T2 = int(T2)
        self.confidence_threshold = float(confidence_threshold)
        self.use_strong_aug = bool(use_strong_aug)
        self.strong_aug = None

    def build(self) -> None:
        self.strong_aug = get_strong_augmentation(self.img_size)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    # ------------------------------------------------------------------ #
    def train_step(
        self,
        labeled_batch: Dict[str, Any],
        unlabeled_batch: Dict[str, Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # --- supervised CE/Dice on labeled data ---
        pred_l = self._forward(images_l)
        sup_loss = criterion(pred_l, labels)

        # --- argmax pseudo-label from the clean view ---
        with torch.no_grad():
            pred_u_clean = self._forward(images_u)
            probs_clean = F.softmax(pred_u_clean, dim=1)
            max_probs, pseudo = probs_clean.max(dim=1)  # (B, H, W)
            if self.confidence_threshold > 0.0:
                mask = (max_probs >= self.confidence_threshold)
            else:
                mask = torch.ones_like(max_probs, dtype=torch.bool)

        # --- student forward on (optionally) strong-augmented view ---
        if self.use_strong_aug:
            x_student = self.strong_aug(images_u)
        else:
            x_student = images_u
        pred_u_student = self._forward(x_student)

        # masked CE against the argmax pseudo-label
        if mask.any():
            ce = F.cross_entropy(pred_u_student, pseudo, reduction='none')
            unsup_loss = (ce * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        else:
            unsup_loss = torch.zeros((), device=self.device, requires_grad=True)

        alpha = _pseudo_label_schedule(epoch, self.T1, self.T2, self.alpha_final)
        total_loss = sup_loss + alpha * unsup_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": unsup_loss.item(),
            "w": alpha,
            "mask_ratio": mask.float().mean().item(),
        }

    def get_eval_model(self) -> nn.Module:
        return self.model
