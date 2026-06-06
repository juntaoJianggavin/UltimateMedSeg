"""Deep Co-Training with dual networks and VAT perturbations.

Qiao et al., ECCV 2018.
Reference: https://github.com/qiaoyu1002/DeepCoTraining
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSemiMethod
from .utils import (
    get_current_consistency_weight,
    pseudo_label_with_threshold,
    get_strong_augmentation,
)


class _EnsembleModel(nn.Module):
    """Simple wrapper that averages predictions from two models."""

    def __init__(self, model1: nn.Module, model2: nn.Module):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        if isinstance(out1, (list, tuple)):
            out1 = out1[0]
        if isinstance(out2, (list, tuple)):
            out2 = out2[0]
        return (out1 + out2) / 2.0


class DeepCoTraining(BaseSemiMethod):
    """Deep Co-Training with dual networks and VAT perturbations.

    Faithful to the official DeepCoTraining algorithm:
        - Two independent networks (model1, model2) with different initializations
        - Different augmentations applied per model view
        - Cross pseudo-labeling: model1 -> PL for model2, and vice versa
        - VAT (Virtual Adversarial Training) perturbation:
            d = xi * grad(KL(f(x) || f(x + d)))
            d = epsilon * d / ||d||
            vat_loss = KL(f(x) || f(x + d))

    Reference:
        Qiao et al., "DeepCoTraining: Semi-Supervised Image Recognition
        with CNNs", ECCV 2018.
        Official: https://github.com/qiaoyu1002/DeepCoTraining

    Args:
        confidence_threshold: Minimum confidence for pseudo-labels.
        consistency_weight: Weight for co-training loss.
        vat_epsilon: Perturbation magnitude for VAT (official: 6.0).
        vat_xi: Small constant for gradient estimation (official: 1e-6).
        rampup_epochs: Epochs to ramp up.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 confidence_threshold: float = 0.9,
                 consistency_weight: float = 1.0,
                 vat_epsilon: float = 6.0,
                 vat_xi: float = 1e-6,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.confidence_threshold = confidence_threshold
        self.vat_epsilon = vat_epsilon
        self.vat_xi = vat_xi
        self.model2 = None
        self.strong_aug1 = None
        self.strong_aug2 = None
        self._model2_opt = None

    def build(self) -> None:
        # Second network with different random initialization
        self.model2 = copy.deepcopy(self.model)
        # Re-initialize weights with different seed for diversity
        for m in self.model2.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.model2.to(self.device)

        self.strong_aug1 = get_strong_augmentation(self.img_size)
        self.strong_aug2 = get_strong_augmentation(self.img_size)

        # Separate optimizer for model2
        self._model2_opt = torch.optim.AdamW(
            self.model2.parameters(), lr=1e-4, weight_decay=1e-4)

    def extra_params(self):
        return list(self.model2.parameters())

    def extra_optimizers(self, lr: float = 1e-4):
        return [(self._model2_opt, "model2")]

    def _vat_loss(self, model, x, pred_orig):
        """Compute Virtual Adversarial Training loss.

        Official VAT:
            d = random_direction * xi
            d = xi * grad_d(KL(pred_orig || f(x + d)))
            d = epsilon * d / ||d||
            vat_loss = KL(pred_orig || f(x + d))
        """
        pred_soft = F.softmax(pred_orig.detach(), dim=1)

        # Random direction
        d = torch.randn_like(x) * self.vat_xi
        d.requires_grad_(True)

        # Forward with small perturbation
        pred_perturb = model(x + d)
        if isinstance(pred_perturb, (list, tuple)):
            pred_perturb = pred_perturb[0]

        # KL divergence to get gradient direction
        kl = F.kl_div(
            F.log_softmax(pred_perturb, dim=1), pred_soft,
            reduction='batchmean')
        grad_d = torch.autograd.grad(kl, d, retain_graph=False)[0]

        # Normalize and scale
        grad_norm = grad_d.view(grad_d.size(0), -1).norm(dim=1, keepdim=True)
        grad_norm = grad_norm.view(-1, 1, 1, 1).clamp(min=1e-8)
        d_hat = self.vat_epsilon * grad_d / grad_norm

        # Final VAT loss with detached perturbation
        pred_adv = model(x + d_hat)
        if isinstance(pred_adv, (list, tuple)):
            pred_adv = pred_adv[0]
        vat_loss = F.kl_div(
            F.log_softmax(pred_adv, dim=1), pred_soft,
            reduction='batchmean')
        return vat_loss

    def train_step(self, labeled_batch, unlabeled_batch, criterion, optimizer,
                   epoch, total_epochs):
        self.model.train()
        self.model2.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # ---- Model 1 ----
        # Supervised on labeled (aug1 view)
        images_l_aug1 = self.strong_aug1(images_l)
        pred_l1 = self.model(images_l_aug1)
        if isinstance(pred_l1, (list, tuple)):
            pred_l1 = pred_l1[0]
        sup_loss1 = criterion(pred_l1, labels)

        # Cross pseudo-labels: model2 generates PL for model1
        with torch.no_grad():
            pred_u2 = self.model2(self.strong_aug2(images_u))
            if isinstance(pred_u2, (list, tuple)):
                pred_u2 = pred_u2[0]
            pseudo2, mask2 = pseudo_label_with_threshold(
                pred_u2, self.confidence_threshold)

        # Model1 on augmented unlabeled
        images_u_aug1 = self.strong_aug1(images_u)
        pred_u1 = self.model(images_u_aug1)
        if isinstance(pred_u1, (list, tuple)):
            pred_u1 = pred_u1[0]

        if mask2.any():
            co_loss1 = F.cross_entropy(pred_u1, pseudo2,
                                       ignore_index=-1, reduction='mean')
        else:
            co_loss1 = torch.tensor(0.0, device=self.device, requires_grad=True)

        # VAT for model1
        vat_loss1 = self._vat_loss(self.model, images_u, pred_u1.detach())

        w = get_current_consistency_weight(epoch, self.consistency_weight,
                                           self.rampup_epochs)
        total_loss1 = sup_loss1 + w * (co_loss1 + 0.5 * vat_loss1)

        optimizer.zero_grad()
        total_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- Model 2 ----
        # Supervised on labeled (aug2 view)
        images_l_aug2 = self.strong_aug2(images_l)
        pred_l2 = self.model2(images_l_aug2)
        if isinstance(pred_l2, (list, tuple)):
            pred_l2 = pred_l2[0]
        sup_loss2 = criterion(pred_l2, labels)

        # Cross pseudo-labels: model1 generates PL for model2
        with torch.no_grad():
            pred_u1_ref = self.model(self.strong_aug1(images_u))
            if isinstance(pred_u1_ref, (list, tuple)):
                pred_u1_ref = pred_u1_ref[0]
            pseudo1, mask1 = pseudo_label_with_threshold(
                pred_u1_ref, self.confidence_threshold)

        # Model2 on augmented unlabeled
        images_u_aug2 = self.strong_aug2(images_u)
        pred_u2_student = self.model2(images_u_aug2)
        if isinstance(pred_u2_student, (list, tuple)):
            pred_u2_student = pred_u2_student[0]

        if mask1.any():
            co_loss2 = F.cross_entropy(pred_u2_student, pseudo1,
                                       ignore_index=-1, reduction='mean')
        else:
            co_loss2 = torch.tensor(0.0, device=self.device, requires_grad=True)

        # VAT for model2
        vat_loss2 = self._vat_loss(self.model2, images_u, pred_u2_student.detach())

        total_loss2 = sup_loss2 + w * (co_loss2 + 0.5 * vat_loss2)

        self._model2_opt.zero_grad()
        total_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1.0)
        self._model2_opt.step()

        return {
            "loss": (total_loss1.item() + total_loss2.item()) / 2.0,
            "sup_loss": (sup_loss1.item() + sup_loss2.item()) / 2.0,
            "unsup_loss": (co_loss1.item() + co_loss2.item()) / 2.0,
            "vat_loss": (vat_loss1.item() + vat_loss2.item()) / 2.0,
            "w": w,
            "mask_ratio": (mask1.float().mean().item() + mask2.float().mean().item()) / 2.0,
        }

    def update(self, epoch: int) -> None:
        pass  # No EMA; both models are trained independently

    def get_eval_model(self) -> nn.Module:
        return _EnsembleModel(self.model, self.model2)
