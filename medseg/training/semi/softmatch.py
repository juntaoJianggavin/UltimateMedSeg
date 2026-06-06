"""SoftMatch: Gaussian-weighted pseudo-labels with distribution alignment.

Chen et al., ICLR 2023.
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


class SoftMatch(BaseSemiMethod):
    """SoftMatch: Gaussian-weighted pseudo-labels with distribution alignment.

    Implements the official SoftMatch algorithm:
        mu = EMA(mean(max_probs)),  var = EMA(var(max_probs))
        weight = exp(-(clamp(max_probs - mu, max=0))^2 / (2 * var / 4))
        distribution_alignment: probs = probs * lb_prob_t / ulb_prob_t
        loss = CE(logits_s, argmax(teacher)) * weight

    Reference:
        Chen et al., "SoftMatch: Addressing the Quantity-Quality Tradeoff
        in Semi-Supervised Learning", ICLR 2023.
        Official: https://github.com/TorchSSL/TorchSSL/blob/main/models/softmatch/softmatch.py

    Args:
        dist_align: Whether to apply distribution alignment.
        ema_decay: EMA decay for teacher.
        consistency_weight: Weight for unsupervised loss.
        rampup_epochs: Epochs to ramp up.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 dist_align: bool = True,
                 ema_decay: float = 0.999,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.dist_align = dist_align
        self.ema_decay = ema_decay
        self.ema_p = 0.999  # Official: fixed EMA momentum
        self.teacher = None
        self.strong_aug = None
        self._lb_prob_t = None
        self._ulb_prob_t = None
        self._prob_max_mu_t = None
        self._prob_max_var_t = None

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        self.strong_aug = get_strong_augmentation(self.img_size)
        num_classes = _infer_num_classes(self.model, self.device, self.img_size)
        # Official init (softmatch.py train())
        self._lb_prob_t = torch.ones(num_classes, device=self.device) / num_classes
        self._ulb_prob_t = torch.ones(num_classes, device=self.device) / num_classes
        self._prob_max_mu_t = 1.0 / num_classes
        self._prob_max_var_t = 1.0

    @torch.no_grad()
    def _update_prob_t(self, lb_probs, ulb_probs):
        """Official update_prob_t from softmatch.py:
            ulb_prob_t = EMA(mean(ulb_probs, dim=0))
            lb_prob_t = EMA(mean(lb_probs, dim=0))
            prob_max_mu_t = EMA(mean(max_probs))
            prob_max_var_t = EMA(var(max_probs))
        """
        ema = self.ema_p
        # For segmentation: mean over (batch, H, W) dims
        ulb_prob_t = ulb_probs.mean(dim=(0, 2, 3))
        self._ulb_prob_t = ema * self._ulb_prob_t + (1 - ema) * ulb_prob_t

        lb_prob_t = lb_probs.mean(dim=(0, 2, 3))
        self._lb_prob_t = ema * self._lb_prob_t + (1 - ema) * lb_prob_t

        max_probs = ulb_probs.max(dim=1)[0]  # B, H, W
        max_probs_flat = max_probs.flatten()
        mu = max_probs_flat.mean().item()
        var = max_probs_flat.var(unbiased=True).item()
        self._prob_max_mu_t = ema * self._prob_max_mu_t + (1 - ema) * mu
        self._prob_max_var_t = ema * self._prob_max_var_t + (1 - ema) * var

    @torch.no_grad()
    def _calculate_mask(self, probs):
        """Official calculate_mask: truncated Gaussian weighting.
            weight = exp(-(clamp(max_probs - mu, max=0))^2 / (2 * var / 4))
        """
        max_probs, max_idx = probs.max(dim=1)  # B, H, W
        mu = self._prob_max_mu_t
        var = max(self._prob_max_var_t, 1e-8)
        # Official: only penalize below-mean confidence
        mask = torch.exp(
            -((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / 4))
        )
        return max_probs, mask, max_idx

    @torch.no_grad()
    def _distribution_alignment(self, probs):
        """Official distribution_alignment:
            probs = probs * lb_prob_t / ulb_prob_t
            probs = probs / probs.sum(dim=-1, keepdim=True)
        """
        ratio = self._lb_prob_t / self._ulb_prob_t.clamp(min=1e-8)
        probs = probs * ratio.view(1, -1, 1, 1)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return probs

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
            probs_x_ulb_w = F.softmax(teacher_pred, dim=1)

            # Official: update lb_prob_t, ulb_prob_t, mu, var using labeled probs
            probs_x_lb = F.softmax(pred_l.detach(), dim=1)
            self._update_prob_t(probs_x_lb, probs_x_ulb_w)

            # Official: distribution alignment
            if self.dist_align:
                probs_x_ulb_w = self._distribution_alignment(probs_x_ulb_w)

            # Official: calculate Gaussian weight mask
            max_probs, mask, max_idx = self._calculate_mask(probs_x_ulb_w)

        images_u_strong = self.strong_aug(images_u)
        student_pred = self.model(images_u_strong)
        if isinstance(student_pred, (list, tuple)):
            student_pred = student_pred[0]

        # Official: masked CE with hard labels (argmax)
        # ce_loss(logits_s, max_idx, reduction='none') * mask.float()
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
            "avg_max_prob": max_probs.mean().item(),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
