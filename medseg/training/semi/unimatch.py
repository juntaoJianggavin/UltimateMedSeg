"""UniMatch (Yang et al., CVPR 2023).

Paper: https://arxiv.org/abs/2208.09910
Reference implementation (not copied): https://github.com/LiheYoung/UniMatch

Three unlabeled-data losses, all supervised by the EMA-teacher's hard
pseudo-label on the *weak* view (with confidence threshold):
  (1) strong-aug stream s1     -> CE(student(s1), y_hat)
  (2) strong-aug stream s2     -> CE(student(s2), y_hat)
  (3) feature-perturbation FP  -> CE(head(FP(feat)), y_hat)

The FP branch follows the official ``FeatureNoise`` block: a multiplicative
uniform perturbation ``feat * (1 + U(-r, r))`` *combined with* spatial
``Dropout2d``.  Earlier versions of this file only had the dropout half;
this change adds the noise half so the perturbation is closer to the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight, get_strong_augmentation,
    pseudo_label_with_threshold,
)


class FeaturePerturbationHead(nn.Module):
    """Feature-level perturbation = uniform-noise * Dropout2d, then 1x1 conv.

    Implements the ``FeatureNoise`` operator from the UniMatch paper
    (and SSL4MIS / CCT before it):

        feat' = feat * (1 + Uniform(-r, r))      # multiplicative noise
        feat' = Dropout2d(feat')                 # channel-wise drop
        logits = Conv1x1(feat')

    ``noise_range`` defaults to ``0.3`` per the paper's FeatureNoise default.
    """

    def __init__(self, in_channels: int, num_classes: int,
                 drop_rate: float = 0.5, noise_range: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout2d(p=drop_rate)
        self.head = nn.Conv2d(in_channels, num_classes, 1)
        self.noise_range = float(noise_range)

    def _feature_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Multiplicative uniform noise applied only in training."""
        if not self.training or self.noise_range <= 0.0:
            return x
        # noise in (-r, r), broadcast per (B, C, H, W) element
        noise = x.new_empty(x.shape).uniform_(-self.noise_range, self.noise_range)
        return x * (1.0 + noise)

    def forward(self, x):
        x = self._feature_noise(x)
        x = self.drop(x)
        return self.head(x)


class UniMatch(BaseSemiMethod):
    """UniMatch semi-supervised method.

    Employs three unsupervised branches:
      1. Strong augmentation stream 1
      2. Strong augmentation stream 2 (different random augmentation)
      3. Feature perturbation stream (dropout on features)

    All three are supervised by pseudo-labels from an EMA teacher on
    weakly augmented (original) images.

    Args:
        model: Student model.
        device: Torch device.
        ema_decay: EMA decay for teacher (default 0.999).
        confidence_threshold: Pseudo-label confidence threshold (default 0.95).
        consistency_weight: Maximum consistency weight (default 1.0).
        rampup_epochs: Ramp-up epochs (default 40).
        feat_drop_rate: ``Dropout2d`` rate inside the FP head (default 0.5).
        feat_noise_range: Range ``r`` of the multiplicative uniform noise
            ``feat * (1 + U(-r, r))`` in the FP head (default 0.3, per the
            paper's ``FeatureNoise`` block).
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 ema_decay: float = 0.999,
                 confidence_threshold: float = 0.95,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 feat_drop_rate: float = 0.5,
                 feat_noise_range: float = 0.3,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.ema_decay = ema_decay
        self.confidence_threshold = confidence_threshold
        self.feat_drop_rate = feat_drop_rate
        self.feat_noise_range = feat_noise_range
        self.teacher = None
        self.strong_aug = None
        self.feat_head = None
        self._use_feat_perturbation = False

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        self.strong_aug = get_strong_augmentation(self.img_size)

        # Try feature-level perturbation for SegmentationModel
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'conv'):
            in_ch = self.model.head.conv.in_channels
            num_cls = self.model.head.conv.out_channels
            self.feat_head = FeaturePerturbationHead(
                in_ch, num_cls,
                drop_rate=self.feat_drop_rate,
                noise_range=self.feat_noise_range,
            )
            self.feat_head.to(self.device)
            self._use_feat_perturbation = True
        else:
            # Fallback: output-level perturbation (noise + dropout on logits)
            num_cls = self._probe_num_classes()
            self.feat_head = FeaturePerturbationHead(
                num_cls, num_cls,
                drop_rate=self.feat_drop_rate,
                noise_range=self.feat_noise_range,
            )
            self.feat_head.to(self.device)
            self._use_feat_perturbation = False

    def _probe_num_classes(self) -> int:
        self.model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.img_size, self.img_size, device=self.device)
            out = self.model(dummy)
            if isinstance(out, (list, tuple)):
                out = out[0]
        self.model.train()
        return out.shape[1]

    def extra_params(self):
        if self.feat_head is not None:
            return list(self.feat_head.parameters())
        return []

    def _get_features_and_pred(self, x):
        """Get both intermediate features and final prediction."""
        if self._use_feat_perturbation:
            features = self.model.encoder(x)
            bottleneck_feat = self.model.bottleneck(features[-1])
            decoded = self.model.decoder(bottleneck_feat, features[:-1])
            pred = self.model.head(decoded)
            return decoded, pred
        else:
            pred = self.model(x)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            return pred, pred

    def _masked_ce_loss(self, pred: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss only on pixels where mask is True."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        loss = F.cross_entropy(pred, target, ignore_index=-1, reduction='none')
        return (loss * mask.float()).sum() / mask.float().sum()

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
        if self.feat_head is not None:
            self.feat_head.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # --- Supervised loss ---
        pred_l = self.model(images_l)
        if isinstance(pred_l, (list, tuple)):
            pred_l = pred_l[0]
        sup_loss = criterion(pred_l, labels)

        # --- Generate pseudo-labels from teacher (weak aug) ---
        with torch.no_grad():
            teacher_pred = self.teacher(images_u)
            if isinstance(teacher_pred, (list, tuple)):
                teacher_pred = teacher_pred[0]
            pseudo_labels, conf_mask = pseudo_label_with_threshold(
                teacher_pred, self.confidence_threshold)

        # --- Strong augmentation stream 1 ---
        images_u_s1 = self.strong_aug(images_u)
        pred_s1 = self.model(images_u_s1)
        if isinstance(pred_s1, (list, tuple)):
            pred_s1 = pred_s1[0]
        loss_s1 = self._masked_ce_loss(pred_s1, pseudo_labels, conf_mask)

        # --- Strong augmentation stream 2 (different random) ---
        images_u_s2 = self.strong_aug(images_u)
        pred_s2 = self.model(images_u_s2)
        if isinstance(pred_s2, (list, tuple)):
            pred_s2 = pred_s2[0]
        loss_s2 = self._masked_ce_loss(pred_s2, pseudo_labels, conf_mask)

        # --- Feature perturbation stream ---
        features_u, pred_u = self._get_features_and_pred(images_u)
        feat_pred = self.feat_head(features_u)
        if feat_pred.shape[2:] != pseudo_labels.shape[1:]:
            feat_pred = F.interpolate(feat_pred, size=pseudo_labels.shape[1:],
                                      mode='bilinear', align_corners=False)
        loss_feat = self._masked_ce_loss(feat_pred, pseudo_labels, conf_mask)

        # --- Total loss ---
        unsup_loss = (loss_s1 + loss_s2 + loss_feat) / 3.0
        w = get_current_consistency_weight(epoch, self.consistency_weight, self.rampup_epochs)
        total_loss = sup_loss + w * unsup_loss

        optimizer.zero_grad()
        total_loss.backward()
        params = list(self.model.parameters())
        if self.feat_head is not None:
            params += list(self.feat_head.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": unsup_loss.item(),
            "w": w,
            "conf_ratio": conf_mask.float().mean().item(),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
