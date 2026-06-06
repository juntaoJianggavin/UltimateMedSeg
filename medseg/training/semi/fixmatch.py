"""FixMatch: confidence-thresholded pseudo-labels + strong augmentation.

Sohn et al., NeurIPS 2020.
Reference: https://github.com/HiLab-git/SSL4MIS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight,
    pseudo_label_with_threshold,
    get_strong_augmentation,
)


class FixMatch(BaseSemiMethod):
    """FixMatch: confidence-thresholded pseudo-labels + strong augmentation.

    Reference:
        Sohn et al., "FixMatch: Simplifying Semi-Supervised Learning with
        Consistency and Confidence", NeurIPS 2020.

    Args:
        confidence_threshold: Minimum softmax confidence to keep a pseudo-label.
        consistency_weight: Weight for the unsupervised loss.
        ema_decay: EMA decay for teacher model.
        rampup_epochs: Epochs to ramp up consistency weight.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 confidence_threshold: float = 0.95,
                 consistency_weight: float = 1.0,
                 ema_decay: float = 0.999,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.confidence_threshold = confidence_threshold
        self.ema_decay = ema_decay
        self.teacher = None
        self.strong_aug = None

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        self.strong_aug = get_strong_augmentation(self.img_size)

    def train_step(self, labeled_batch, unlabeled_batch, criterion, optimizer,
                   epoch, total_epochs):
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

        # Teacher: pseudo-labels with threshold
        with torch.no_grad():
            teacher_pred = self.teacher(images_u)
            if isinstance(teacher_pred, (list, tuple)):
                teacher_pred = teacher_pred[0]
            pseudo_labels, mask = pseudo_label_with_threshold(
                teacher_pred, self.confidence_threshold)

        # Student: strong augmentation
        images_u_strong = self.strong_aug(images_u)
        student_pred = self.model(images_u_strong)
        if isinstance(student_pred, (list, tuple)):
            student_pred = student_pred[0]

        # Masked cross-entropy on high-confidence pseudo-labels
        if mask.any():
            unsup_loss = F.cross_entropy(student_pred, pseudo_labels,
                                         ignore_index=-1, reduction='mean')
        else:
            unsup_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        w = get_current_consistency_weight(epoch, self.consistency_weight,
                                           self.rampup_epochs)
        total_loss = sup_loss + w * unsup_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": unsup_loss.item(),
            "w": w,
            "mask_ratio": mask.float().mean().item(),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
