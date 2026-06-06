# Reference: https://github.com/vikasverma1077/ICT
# Paper:     https://arxiv.org/abs/1903.03825
"""Interpolation Consistency Training (Verma et al., IJCAI 2019).

Algorithm (Sec. 3 of the paper):
    Sample two unlabeled images u_a, u_b and a mixup coefficient
    lambda ~ Beta(alpha, alpha).  The student is asked to match the
    teacher's *interpolated* prediction at the interpolated point:

        u_mix       = lambda * u_a + (1 - lambda) * u_b
        target      = lambda * softmax(teacher(u_a))
                       + (1 - lambda) * softmax(teacher(u_b))
        L_unsup     = MSE(softmax(student(u_mix)), target)

    The teacher is an EMA of the student (same recipe as Mean Teacher).
    On labeled data only the standard supervised loss is applied; the
    unsupervised consistency term is multiplied by a sigmoid-rampup
    ``consistency_weight`` schedule.

This implementation derives the two unlabeled views ``u_a`` and ``u_b``
from a *single* unlabeled batch via a within-batch permutation — the
exact recipe used by the official ICT codebase (see ``ict_helpers.py``
in the reference repo) so that a randomised ordering pairs each sample
with a different partner each step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update, get_current_consistency_weight,
)


class InterpolationConsistencyTraining(BaseSemiMethod):
    """Interpolation Consistency Training (ICT).

    Args:
        model: Student model.
        device: Torch device.
        ema_decay: EMA decay for the teacher (default 0.999, matching
            the paper's ``teacher_alpha``).
        consistency_weight: Maximum unsupervised loss weight (default 1.0).
        rampup_epochs: Sigmoid ramp-up length for the consistency weight.
        mixup_alpha: ``alpha`` parameter of the ``Beta(alpha, alpha)``
            distribution from which the interpolation coefficient is
            sampled.  Paper uses ``alpha=1.0`` for image classification;
            we keep that as the default.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 ema_decay: float = 0.999,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 mixup_alpha: float = 1.0,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        if mixup_alpha <= 0.0:
            raise ValueError(
                f"ICT mixup_alpha must be > 0; got {mixup_alpha}."
            )
        self.ema_decay = ema_decay
        self.mixup_alpha = float(mixup_alpha)
        self.teacher: nn.Module = None
        self._beta = None

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        a = torch.tensor(self.mixup_alpha)
        self._beta = torch.distributions.Beta(a, a)

    # ------------------------------------------------------------- helpers
    def _sample_lambda(self) -> float:
        """Sample one mixup coefficient per batch (paper recipe)."""
        return float(self._beta.sample().item())

    @staticmethod
    def _take_first(out):
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    # ---------------------------------------------------------- train_step
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
        self.teacher.eval()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        if images_u.shape[0] < 2:
            raise RuntimeError(
                "ICT requires an unlabeled batch with at least 2 samples "
                f"to form the mixup pair; got batch size {images_u.shape[0]}."
            )

        # --- Supervised loss on labeled data ---
        pred_l = self._take_first(self.model(images_l))
        sup_loss = criterion(pred_l, labels)

        # --- Build the unlabeled mixup pair (u_a, u_b) ---
        # u_a = images_u, u_b = images_u[perm]
        B = images_u.shape[0]
        perm = torch.randperm(B, device=images_u.device)
        u_a, u_b = images_u, images_u[perm]
        lam = self._sample_lambda()
        u_mix = lam * u_a + (1.0 - lam) * u_b

        # --- Teacher: compute soft predictions on u_a (reuse for u_b via perm) ---
        with torch.no_grad():
            p_a_logits = self._take_first(self.teacher(u_a)).detach()
            p_a_soft = F.softmax(p_a_logits, dim=1)
            p_b_soft = p_a_soft[perm]
            target = lam * p_a_soft + (1.0 - lam) * p_b_soft

        # --- Student on the interpolated input ---
        q_mix_logits = self._take_first(self.model(u_mix))
        q_mix_soft = F.softmax(q_mix_logits, dim=1)

        if q_mix_soft.shape[2:] != target.shape[2:]:
            q_mix_soft = F.interpolate(
                q_mix_soft, size=target.shape[2:],
                mode='bilinear', align_corners=False,
            )

        # Paper: MSE between predicted-softmax and interpolated-softmax target.
        consistency_loss = F.mse_loss(q_mix_soft, target)

        w = get_current_consistency_weight(
            epoch, self.consistency_weight, self.rampup_epochs)
        total_loss = sup_loss + w * consistency_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": consistency_loss.item(),
            "w": w,
            "lambda": lam,
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
