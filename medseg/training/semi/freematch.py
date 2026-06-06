"""FreeMatch: self-adaptive confidence threshold via EMA statistics.

Wang et al., ICLR 2023.
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
    in_ch = 3
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


class FreeMatch(BaseSemiMethod):
    """FreeMatch: self-adaptive confidence threshold via EMA statistics.

    Implements the official FreeMatch algorithm:
        time_p = EMA(mean(max_probs))              # global threshold
        p_model = EMA(mean(softmax(logits_w)))     # class distribution
        label_hist = EMA(bincount(argmax))          # label histogram
        threshold = time_p * (p_model / max(p_model))[max_idx]
        mask = max_probs >= threshold

    Reference:
        Wang et al., "FreeMatch: Self-Adaptive Thresholding for
        Semi-Supervised Learning", ICLR 2023.
        Official: https://github.com/TorchSSL/TorchSSL/blob/main/models/freematch/freematch_utils.py

    Args:
        ema_momentum: EMA momentum for threshold updates (default 0.999).
        ema_decay: EMA decay for teacher.
        consistency_weight: Weight for unsupervised loss.
        rampup_epochs: Epochs to ramp up.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 ema_momentum: float = 0.999,
                 ema_decay: float = 0.999,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.ema_momentum = ema_momentum
        self.ema_decay = ema_decay
        self.teacher = None
        self.strong_aug = None
        self._time_p = None
        self._p_model = None
        self._label_hist = None
        self._num_classes = None

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        self.strong_aug = get_strong_augmentation(self.img_size)
        self._num_classes = _infer_num_classes(self.model, self.device, self.img_size)
        # Official init
        self._p_model = torch.ones(self._num_classes, device=self.device) / self._num_classes
        self._label_hist = torch.ones(self._num_classes, device=self.device) / self._num_classes
        self._time_p = self._p_model.mean()

    @torch.no_grad()
    def _update_time_p_and_p_model(self, logits_w):
        """Official cal_time_p_and_p_model:
            prob_w = softmax(logits_w, dim=1)
            time_p = time_p * 0.999 + mean(max_probs) * 0.001
            p_model = p_model * 0.999 + mean(prob_w, dim=0) * 0.001
            label_hist = label_hist * 0.999 + (hist/sum) * 0.001
        """
        prob_w = F.softmax(logits_w, dim=1)
        max_probs, max_idx = prob_w.max(dim=1)  # B, H, W
        ema = self.ema_momentum

        # For segmentation: average over batch AND spatial dims
        # Official: mean(dim=0) for classification → mean(dim=(0,2,3)) for segmentation
        self._time_p = ema * self._time_p + (1 - ema) * max_probs.mean()
        p_model_cur = prob_w.mean(dim=(0, 2, 3))  # C
        self._p_model = ema * self._p_model + (1 - ema) * p_model_cur

        # Label histogram from argmax
        hist = torch.bincount(max_idx.long().flatten().clamp(0, self._num_classes - 1),
                              minlength=self._num_classes).float()
        hist = hist / hist.sum().clamp(min=1e-8)
        self._label_hist = ema * self._label_hist + (1 - ema) * hist

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

            # Official: update time_p, p_model, label_hist
            self._update_time_p_and_p_model(teacher_pred)

            # Official consistency_loss:
            pseudo_label = F.softmax(teacher_pred, dim=1)
            max_probs, max_idx = pseudo_label.max(dim=1)

            # threshold = time_p * (p_model / max(p_model))[max_idx]
            p_model_cutoff = self._p_model / self._p_model.max().clamp(min=1e-8)
            threshold = self._time_p * p_model_cutoff[max_idx.long().clamp(0, self._num_classes - 1)]
            mask = max_probs.ge(threshold).float()

        images_u_strong = self.strong_aug(images_u)
        student_pred = self.model(images_u_strong)
        if isinstance(student_pred, (list, tuple)):
            student_pred = student_pred[0]

        # Official: masked_loss = ce_loss(logits_s, max_idx, reduction='none') * mask
        if mask.sum() > 0:
            ce_loss = (F.cross_entropy(student_pred, max_idx, reduction='none') * mask).mean()
        else:
            ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        w = get_current_consistency_weight(epoch, self.consistency_weight,
                                           self.rampup_epochs)
        total_loss = sup_loss + w * ce_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": ce_loss.item(),
            "w": w,
            "mask_ratio": mask.mean().item(),
            "time_p": self._time_p.item(),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
