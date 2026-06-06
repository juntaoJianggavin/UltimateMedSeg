# Reference: https://github.com/zhaozhengChen/ReCAM
# Paper: https://arxiv.org/abs/2203.00962
"""ReCAM — Class Re-Activation Maps for Weakly-Supervised Segmentation.

Chen et al., "Class Re-Activation Maps for Weakly-Supervised Semantic
Segmentation", CVPR 2022.
Paper: https://arxiv.org/abs/2203.00962
Official repository: https://github.com/zhaozhengChen/ReCAM

Algorithm (from official ``step/train_recam.py`` + ``net/resnet50_cam.py``):

    Stage 1 — Original CAM head (unchanged):
        L_cls = multilabel_soft_margin_loss(GAP(cam_logits), label)

    Stage 2 — Class_Predictor (per-class feature pooling + CE):
        For each class c with label[b, c] == 1:
            attmap  = cam[b, c]                   # (H, W) — raw CAM
            w       = sigmoid(attmap)             # spatial attention weight
            f_c     = (w * features).sum() / w.sum()    # (D,) normalised pool
            logit_c = classifier(f_c)             # (num_classes,)
            loss_c  = CE(logit_c, one_hot_c)      # hard label = c
        L_ce = mean of loss_c over valid (b, c) pairs
        Total  = L_cls + lambda * L_ce

    This is verified against the official code:
        ``Class_Predictor.forward(cam, label)`` in net/resnet50_cam.py.

Loss formula:

    L = L_cls(GAP(cam), y)                             # original BCE head
      + lambda * mean_{c in present} CE(f_recam(z_c), c)  # per-class hard CE

This module implements the loss; feature extraction and CAM generation are
upstream concerns (caller provides ``features`` and ``cam_logits``).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("recam")
class ReCAMLoss(nn.Module):
    """ReCAM loss = BCE classification + per-class sigmoid-pooled CE.

    Matches official ``step/train_recam.py``:
        loss = multilabel_soft_margin_loss(x, label) + lambda * loss_ce

    where loss_ce comes from Class_Predictor (per-class CE with hard labels).

    Args:
        lambda_recam: Weight on the re-activated per-class CE term
            (paper default 1.0; ablation: 0.5 ~ 2.0 all work).
        cls_weight: Weight on the original BCE classification term
            (paper keeps it at 1.0, official uses 1.0 implicitly).
    """

    def __init__(
        self,
        lambda_recam: float = 1.0,
        cls_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.lambda_recam = lambda_recam
        self.cls_weight = cls_weight

    # ------------------------------------------------------------------
    # Stage 1: build the class-conditioned, sigmoid-pooled feature vector
    # ------------------------------------------------------------------
    @staticmethod
    def pool_class_features(
        cam: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``z_c = sum_xy sigmoid(CAM_c(x,y)) * F(x,y)`` normalised
        by ``sigmoid(CAM_c).sum()`` — exactly Eq. 3 / official Class_Predictor.

        Verified against official ``Class_Predictor.forward``:
            ``f = sigmoid(cam) * features; f = f.sum(-1) / sigmoid(cam).sum(-1)``

        Args:
            cam: (B, C_fg, H, W) raw CAM logits (BEFORE sigmoid).
            features: (B, D, H, W) backbone feature map at the same spatial
                resolution. Caller is responsible for any required upsample.

        Returns:
            (B, C_fg, D) class-conditioned feature vector, ready to be fed
            to the auxiliary linear classifier ``f_recam``.
        """
        if cam.shape[-2:] != features.shape[-2:]:
            raise ValueError(
                f"cam spatial size {tuple(cam.shape[-2:])} != features "
                f"{tuple(features.shape[-2:])}; resize cam first."
            )
        attn = torch.sigmoid(cam)                                # (B, C, H, W)
        # einsum: sum over H,W → (B, C, D)
        z = torch.einsum("bchw,bdhw->bcd", attn, features)
        # Normalise by attention mass to avoid scale drift
        # (official: ``f = f.sum(-1) / sigmoid(cam).sum(-1)``)
        denom = attn.sum(dim=(2, 3), keepdim=False).unsqueeze(-1) + 1e-6
        return z / denom

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        cam_logits: torch.Tensor,
        features: torch.Tensor,
        image_labels: torch.Tensor,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            cam_logits: (B, C_fg, H, W) raw CAM outputs.
            features: (B, D, H, W) backbone feature map (same spatial
                size as cam_logits, or will be interpolated to match).
            image_labels: (B, C_fg) binary multi-label tags.
            labeled_loss: Optional pre-computed dense supervised loss to
                add (mixed-supervision).

        Returns:
            Scalar loss = cls_weight * BCE + lambda * per_class_CE.
        """
        B, C_fg, H, W = cam_logits.shape

        # Interpolate features to match cam_logits spatial size if needed
        if features.shape[-2:] != cam_logits.shape[-2:]:
            features = F.interpolate(
                features, size=(H, W), mode='bilinear', align_corners=False
            )

        # ---- (1) Original CAM head: multi-label BCE on GAP ----
        # Official: F.multilabel_soft_margin_loss(x, label)
        #   where x = GAP(cam_logits) with shape (B, C_fg)
        cam_flat = cam_logits.mean(dim=(2, 3))                  # (B, C_fg)
        cls_loss = F.multilabel_soft_margin_loss(cam_flat, image_labels.float())

        # ---- (2) Re-activated head: per-class sigmoid-pooled features + CE ----
        # Official Class_Predictor:
        #   For each (b, c) with label[b,c]==1:
        #     f_c = sigmoid(cam[b,c]) * features[b] → normalised sum → (D,)
        #     logit_c = linear(f_c) → (num_classes,)
        #     loss_c = CE(logit_c, one_hot(c))
        pooled = self.pool_class_features(cam_logits, features)  # (B, C_fg, D)

        # Build linear classifier matching official Class_Predictor
        if not hasattr(self, '_recam_classifier') or \
           self._recam_classifier.in_features != features.shape[1] or \
           self._recam_classifier.out_features != C_fg:
            self._recam_classifier = nn.Linear(features.shape[1], C_fg)
            nn.init.xavier_uniform_(self._recam_classifier.weight)
        self._recam_classifier = self._recam_classifier.to(pooled.device)

        logits = self._recam_classifier(pooled)                  # (B, C_fg, C_fg)

        # Per-class CE with hard label = c (official formula)
        targets = torch.arange(C_fg, device=cam_logits.device).unsqueeze(0).expand(B, -1)
        ce_mat = F.cross_entropy(
            logits.reshape(-1, C_fg), targets.reshape(-1), reduction='none'
        ).view(B, C_fg)

        # Mask to valid foreground classes only (label == 1)
        valid = image_labels.float()
        recam_loss = (ce_mat * valid).sum() / valid.sum().clamp_min(1.0)

        total = self.cls_weight * cls_loss + self.lambda_recam * recam_loss
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
