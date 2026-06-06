"""SSL4MIS-U: Uncertainty-Aware Semi-Supervised Learning for Medical Images.

Uncertainty-guided pseudo-label selection and consistency regularization,
inspired by the SSL4MIS framework for medical image segmentation.

Reference:
    SSL4MIS: Semi-Supervised Learning for Medical Image Segmentation
    https://github.com/HiLab-git/SSL4MIS

Key components:
    - Aleatoric + epistemic uncertainty estimation via MC sampling
    - Uncertainty-rectified pseudo-label filtering
    - Variance-dependent consistency weight (low uncertainty = high weight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight, get_strong_augmentation,
)


class SSL4MISUncertainty(BaseSemiMethod):
    """Uncertainty-Aware SSL for Medical Image Segmentation.

    Args:
        model: Student model.
        device: Torch device.
        ema_decay: EMA decay rate (default 0.99).
        consistency_weight: Max consistency weight (default 0.5).
        rampup_epochs: Ramp-up epochs (default 40).
        mc_samples: Number of MC dropout samples (default 5).
        uncertainty_alpha: Uncertainty weighting exponent (default 1.0).
        pseudo_threshold: Pseudo-label confidence threshold (default 0.85).
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 ema_decay: float = 0.99,
                 consistency_weight: float = 0.5,
                 rampup_epochs: int = 40,
                 mc_samples: int = 5,
                 uncertainty_alpha: float = 1.0,
                 pseudo_threshold: float = 0.85,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.ema_decay = ema_decay
        self.mc_samples = mc_samples
        self.uncertainty_alpha = uncertainty_alpha
        self.pseudo_threshold = pseudo_threshold
        self.teacher = None
        self.strong_aug = None

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        self.strong_aug = get_strong_augmentation(self.img_size)

    @torch.no_grad()
    def _mc_uncertainty(self, images: torch.Tensor) -> tuple:
        """MC Dropout uncertainty estimation.

        Runs multiple forward passes with dropout enabled to estimate
        predictive uncertainty. Returns mean prediction and uncertainty map.

        Args:
            images: (B, C, H, W) input images.
        Returns:
            mean_prob: Mean softmax probability (B, num_classes, H, W).
            uncertainty: Normalized entropy-based uncertainty (B, H, W) in [0, 1].
        """
        # Enable dropout for MC sampling
        self.teacher.train()
        probs = []
        for _ in range(self.mc_samples):
            pred = self.teacher(images)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            probs.append(F.softmax(pred, dim=1))

        self.teacher.eval()
        probs = torch.stack(probs, dim=0)  # (K, B, C, H, W)
        mean_prob = probs.mean(dim=0)

        # Predictive entropy as uncertainty
        entropy = -(mean_prob * torch.log(mean_prob + 1e-8)).sum(dim=1)
        # Normalize
        max_entropy = torch.log(torch.tensor(mean_prob.shape[1], dtype=torch.float32))
        uncertainty = (entropy / (max_entropy + 1e-8)).clamp(0, 1)

        return mean_prob, uncertainty

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

        # Supervised loss
        pred_l = self.model(images_l)
        if isinstance(pred_l, (list, tuple)):
            pred_l = pred_l[0]
        sup_loss = criterion(pred_l, labels)

        # MC uncertainty on unlabeled data
        with torch.no_grad():
            teacher_prob, uncertainty = self._mc_uncertainty(images_u)
            # Uncertainty-rectified pseudo-labels
            max_prob, pseudo_labels = teacher_prob.max(dim=1)
            # Filter: high confidence AND low uncertainty
            valid_mask = (
                (max_prob >= self.pseudo_threshold) &
                (uncertainty < (1.0 - self.pseudo_threshold))
            ).float()

        # Student on augmented unlabeled
        images_u_aug = self.strong_aug(images_u)
        student_pred = self.model(images_u_aug)
        if isinstance(student_pred, (list, tuple)):
            student_pred = student_pred[0]

        # Uncertainty-weighted consistency loss
        student_prob = F.softmax(student_pred, dim=1)
        diff = (student_prob - teacher_prob) ** 2

        # Weight by (1 - uncertainty)^alpha: confident regions get more weight
        unc_weight = (1.0 - uncertainty).pow(self.uncertainty_alpha).unsqueeze(1)
        consistency_loss = (diff * unc_weight * valid_mask.unsqueeze(1)).mean()

        w = get_current_consistency_weight(epoch, self.consistency_weight, self.rampup_epochs)
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
            "valid_ratio": valid_mask.mean().item(),
            "mean_uncertainty": uncertainty.mean().item(),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
