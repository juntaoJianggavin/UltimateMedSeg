"""FlexMatch: FixMatch with class-wise adaptive confidence threshold.

Zhang et al., NeurIPS 2021.
Reference: https://github.com/TorchSSL/TorchSSL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight,
    get_strong_augmentation,
)


def _infer_num_classes(model: nn.Module, device: torch.device, img_size: int) -> int:
    """Infer the number of output classes from a model."""
    nc = getattr(model, 'num_classes', None)
    if nc is not None:
        return int(nc)
    # Try to find input channels from first Conv2d layer
    in_ch = 3  # default
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            in_ch = m.in_channels
            break
    with torch.no_grad():
        dummy = torch.zeros(1, in_ch, img_size, img_size, device=device)
        out = model(dummy)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.shape[1]


class FlexMatch(BaseSemiMethod):
    """FlexMatch: FixMatch with class-wise adaptive confidence threshold.

    Implements the official Convex CPL (Curriculum Pseudo Labeling) formula:
        classwise_acc[c] = selected_count[c] / max(selected_counts)
        mask = max_probs >= p_cutoff * (class_acc[max_idx] / (2 - class_acc[max_idx]))

    Reference:
        Zhang et al., "FlexMatch: Boosting Semi-Supervised Learning with
        Curriculum Pseudo Labeling", NeurIPS 2021.
        Official: https://github.com/TorchSSL/TorchSSL/blob/main/models/flexmatch/flexmatch_utils.py

    Args:
        p_cutoff: Base confidence threshold (default 0.95).
        ema_decay: EMA decay for teacher.
        consistency_weight: Weight for unsupervised loss.
        rampup_epochs: Epochs to ramp up consistency weight.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 p_cutoff: float = 0.95,
                 ema_decay: float = 0.999,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.p_cutoff = p_cutoff
        self.ema_decay = ema_decay
        self.teacher = None
        self.strong_aug = None
        self._classwise_acc = None
        self._selected_counts = None
        self._num_classes = None

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        self.strong_aug = get_strong_augmentation(self.img_size)
        self._num_classes = _infer_num_classes(self.model, self.device, self.img_size)
        # Official init: zeros (becomes ones after first batch with selected labels)
        self._classwise_acc = torch.zeros(self._num_classes, device=self.device)
        self._selected_counts = torch.zeros(self._num_classes, device=self.device)

    @torch.no_grad()
    def _update_selected_counts(self, pseudo_labels, select_mask):
        """Update per-class selected pseudo-label counts.
        Official: pseudo_counter[i] = (selected_label == i).sum()
        We accumulate across batches via EMA-like addition.
        """
        valid = pseudo_labels[select_mask]
        if valid.numel() == 0:
            return
        counts = torch.bincount(valid.long().clamp(0, self._num_classes - 1),
                                minlength=self._num_classes).float()
        self._selected_counts += counts
        max_count = self._selected_counts.max().clamp(min=1.0)
        # Official formula: classwise_acc[i] = count[i] / max(counts)
        self._classwise_acc = self._selected_counts / max_count

    def train_step(self, labeled_batch, unlabeled_batch, criterion, optimizer,
                   epoch, total_epochs):
        self.model.train()
        self.teacher.eval()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        pred_l = self.model(images_l)
        if isinstance(pred_l, (list, tuple)):
            pred_l = pred_l[0]
        sup_loss = criterion(pred_l, labels)

        with torch.no_grad():
            teacher_pred = self.teacher(images_u)
            if isinstance(teacher_pred, (list, tuple)):
                teacher_pred = teacher_pred[0]
            # Official: pseudo_label = softmax(logits_w.detach())
            pseudo_label = F.softmax(teacher_pred, dim=1)
            max_probs, max_idx = pseudo_label.max(dim=1)  # B, H, W

            # Official Convex CPL:
            # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2 - class_acc[max_idx])))
            class_acc = self._classwise_acc[max_idx.long().clamp(0, self._num_classes - 1)]
            threshold = self.p_cutoff * (class_acc / (2.0 - class_acc))
            mask = max_probs.ge(threshold).float()
            # Official: select = max_probs.ge(p_cutoff).long()  (for counter update)
            select = max_probs.ge(self.p_cutoff)

            self._update_selected_counts(max_idx, select)

        images_u_strong = self.strong_aug(images_u)
        student_pred = self.model(images_u_strong)
        if isinstance(student_pred, (list, tuple)):
            student_pred = student_pred[0]

        # Official: masked_loss = ce_loss(logits_s, max_idx, reduction='none') * mask
        if mask.sum() > 0:
            unsup_loss = (F.cross_entropy(student_pred, max_idx, reduction='none') * mask).mean()
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
            "mask_ratio": mask.mean().item(),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
