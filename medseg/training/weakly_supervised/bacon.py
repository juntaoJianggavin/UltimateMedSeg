# BACoN (NeurIPS 2024)
# Reference: https://github.com/rongtao-xu/BACoN
# Paper: https://arxiv.org/abs/2410.04230
# Implemented from paper formulas; not a copy of the official repo.
"""BACoN — Bias-Aware Contrastive Network for WSSS.

Bias-aware contrastive learning that mitigates the systematic
background-foreground confusion of CAM-based weakly-supervised
segmentation (NeurIPS 2024 family of WSSS contrastive methods).
Paper: https://arxiv.org/abs/2410.04230
Reference repository: https://github.com/rongtao-xu/BACoN

Why it exists.
    CAM is trained with image-level multi-label tags; the resulting
    feature map fires on a small foreground patch AND tends to leak onto
    semantically correlated background (e.g. "train" leaks onto "rail").
    BACoN tackles this by maintaining two prototype banks per class —
    foreground (fg) and *biased background* (bg) — and minimising a
    pixel-wise contrastive loss:

        L_pix = - log  exp(sim(f_i, p_fg^{c_i}) / tau)
                       -------------------------------------------------
                       exp(sim(f_i, p_fg^{c_i})/tau) + sum_bg exp(sim(f_i, p_bg^{c})/tau)

    Foreground pixels are pulled towards the fg prototype of their
    pseudo-class and pushed away from EVERY bg prototype. A symmetric
    term applies for bg pixels. Pseudo-labels come from confidence
    thresholding the (max-normalised) CAM, with a hard ``ignore`` band
    between the two thresholds.

    A "bias awareness" margin reweighs the contrastive loss by the
    correlation between a pixel's feature and the bg prototype of OTHER
    co-occurring classes, sharpening the network's response on truly
    foreground pixels.

This module implements the loss only — features and CAM come from the
caller. Prototypes are maintained as detached EMA buffers (paper Sec. 3.2
"prototype memory bank"). No code is lifted from the upstream repo.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("bacon")
class BACoNLoss(nn.Module):
    """BACoN classification + bias-aware pixel-contrastive loss.

    Args:
        num_classes: Number of foreground classes.
        feature_dim: Channel count of the feature map.
        temperature: InfoNCE temperature (paper default 0.1).
        ema_momentum: Prototype EMA coefficient (default 0.99).
        cls_weight: Weight on the multi-label BCE classification term.
        lambda_contrast: Weight on the pixel-contrastive term (default 0.4).
        high_thresh: Normalised-CAM threshold for "reliable fg" (0.7).
        low_thresh: Normalised-CAM threshold for "reliable bg" (0.25).
        bias_margin: Multiplier on the bias-awareness reweighting
            (default 1.0 — set to 0 to disable).
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        temperature: float = 0.1,
        ema_momentum: float = 0.99,
        cls_weight: float = 1.0,
        lambda_contrast: float = 0.4,
        high_thresh: float = 0.7,
        low_thresh: float = 0.25,
        bias_margin: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < low_thresh < high_thresh < 1.0):
            raise ValueError(
                f"Need 0 < low_thresh ({low_thresh}) < high_thresh "
                f"({high_thresh}) < 1"
            )
        if num_classes <= 0 or feature_dim <= 0:
            raise ValueError(
                f"num_classes/feature_dim must be positive "
                f"(got {num_classes}, {feature_dim})"
            )
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.cls_weight = cls_weight
        self.lambda_contrast = lambda_contrast
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.bias_margin = bias_margin

        fg = F.normalize(torch.randn(num_classes, feature_dim), dim=-1)
        bg = F.normalize(torch.randn(num_classes, feature_dim), dim=-1)
        self.register_buffer("fg_proto", fg)                  # (C, D)
        self.register_buffer("bg_proto", bg)                  # (C, D)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_cam(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B, C, H, W = cam.shape
        flat = cam.view(B, C, -1)
        m = flat.amin(dim=2, keepdim=True)
        M = flat.amax(dim=2, keepdim=True)
        return ((flat - m) / (M - m + eps)).view(B, C, H, W)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(
        self,
        features: torch.Tensor,
        cam_norm: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> None:
        B, D, H, W = features.shape
        feat_n = F.normalize(features, dim=1).permute(0, 2, 3, 1).reshape(-1, D)
        cam_flat = cam_norm.view(B, self.num_classes, -1)

        # bg mask: pixels below low_thresh across ALL present classes.
        # Computed per sample but pooled across batch.
        for c in range(self.num_classes):
            fg_acc = None
            fg_w = 0.0
            bg_acc = None
            bg_w = 0.0
            for b in range(B):
                if image_labels[b, c].item() <= 0:
                    continue
                conf = cam_flat[b, c]                          # (HW,)
                base = b * H * W
                fg_idx = (conf > self.high_thresh).nonzero(as_tuple=False).squeeze(1)
                if fg_idx.numel() > 0:
                    chunk = feat_n[base + fg_idx].mean(dim=0)
                    if fg_acc is None:
                        fg_acc = chunk * fg_idx.numel()
                    else:
                        fg_acc = fg_acc + chunk * fg_idx.numel()
                    fg_w = fg_w + fg_idx.numel()
                # bg pixels for class c: high confidence in NO present class.
                all_max = cam_flat[b].max(dim=0).values
                bg_idx = (all_max < self.low_thresh).nonzero(as_tuple=False).squeeze(1)
                if bg_idx.numel() > 0:
                    chunk = feat_n[base + bg_idx].mean(dim=0)
                    if bg_acc is None:
                        bg_acc = chunk * bg_idx.numel()
                    else:
                        bg_acc = bg_acc + chunk * bg_idx.numel()
                    bg_w = bg_w + bg_idx.numel()
            if fg_acc is not None and fg_w > 0:
                new_fg = F.normalize(fg_acc / fg_w, dim=0)
                self.fg_proto[c] = F.normalize(
                    self.ema_momentum * self.fg_proto[c]
                    + (1.0 - self.ema_momentum) * new_fg,
                    dim=0,
                )
            if bg_acc is not None and bg_w > 0:
                new_bg = F.normalize(bg_acc / bg_w, dim=0)
                self.bg_proto[c] = F.normalize(
                    self.ema_momentum * self.bg_proto[c]
                    + (1.0 - self.ema_momentum) * new_bg,
                    dim=0,
                )

    # ------------------------------------------------------------------
    def _pixel_contrast(
        self,
        features: torch.Tensor,
        cam_norm: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of fg-pixel and bg-pixel InfoNCE losses."""
        B, D, H, W = features.shape
        feat_n = F.normalize(features, dim=1)                  # (B, D, H, W)
        feat_flat = feat_n.permute(0, 2, 3, 1).reshape(B, H * W, D)
        cam_flat = cam_norm.view(B, self.num_classes, -1)

        fg_p = F.normalize(self.fg_proto.detach(), dim=-1)     # (C, D)
        bg_p = F.normalize(self.bg_proto.detach(), dim=-1)     # (C, D)
        # Re-attach a learned linear scale so prototypes contribute gradient
        # via the feature representation only (paper "stop-gradient" recipe).

        sim_fg = torch.einsum("bnd,cd->bcn", feat_flat, fg_p) / self.temperature
        sim_bg = torch.einsum("bnd,cd->bcn", feat_flat, bg_p) / self.temperature

        losses = []
        for b in range(B):
            present = image_labels[b].nonzero(as_tuple=False).squeeze(1)
            if present.numel() == 0:
                continue
            for c in present.tolist():
                conf = cam_flat[b, c]                          # (HW,)
                fg_idx = (conf > self.high_thresh).nonzero(as_tuple=False).squeeze(1)
                bg_idx = (cam_flat[b].max(dim=0).values < self.low_thresh) \
                    .nonzero(as_tuple=False).squeeze(1)
                if fg_idx.numel() == 0 or bg_idx.numel() == 0:
                    continue

                # ---- fg pixels of class c: pull to fg_p[c], push from bg_p[*]
                pos = sim_fg[b, c, fg_idx]                     # (Pf,)
                # Negatives: bg prototypes of ALL classes (paper Sec. 3.3).
                neg = sim_bg[b, :, fg_idx]                     # (C, Pf)
                # Bias-aware reweight: amplify gradient for pixels whose
                # current similarity to OTHER classes' bg is high.
                if self.bias_margin > 0:
                    other = neg.clone()
                    other[c] = float("-inf")
                    bias = other.amax(dim=0)                   # (Pf,)
                    w = 1.0 + self.bias_margin * torch.sigmoid(bias)
                else:
                    w = torch.ones_like(pos)
                logits = torch.cat([pos.unsqueeze(0), neg], dim=0)  # (1+C, Pf)
                log_denom = torch.logsumexp(logits, dim=0)
                fg_loss = -((pos - log_denom) * w).mean()
                losses.append(fg_loss)

                # ---- bg pixels: pull to bg_p[c], push from fg_p[*]
                pos_bg = sim_bg[b, c, bg_idx]                  # (Pb,)
                neg_bg = sim_fg[b, :, bg_idx]                  # (C, Pb)
                logits_bg = torch.cat([pos_bg.unsqueeze(0), neg_bg], dim=0)
                log_denom_bg = torch.logsumexp(logits_bg, dim=0)
                bg_loss = -(pos_bg - log_denom_bg).mean()
                losses.append(bg_loss)

        if not losses:
            return features.new_zeros(())
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        cam_logits: torch.Tensor,
        image_labels: torch.Tensor,
        cls_logits: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, D, H, W) backbone features. D must equal
                ``feature_dim``.
            cam_logits: (B, C, H, W) RAW CAM logits (fg classes only).
            image_labels: (B, C) binary multi-label tags.
            cls_logits: Optional (B, C) classifier head; defaults to
                GAP(cam_logits).
            labeled_loss: Optional dense supervised loss to add.
        """
        if features.dim() != 4 or cam_logits.dim() != 4:
            raise ValueError(
                "features and cam_logits must be 4-D (B,C,H,W)."
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features channel {features.shape[1]} != feature_dim "
                f"{self.feature_dim}"
            )
        if cam_logits.shape[1] != self.num_classes:
            raise ValueError(
                f"cam channel {cam_logits.shape[1]} != num_classes "
                f"{self.num_classes}"
            )
        if cam_logits.shape[-2:] != features.shape[-2:]:
            raise ValueError(
                f"cam {tuple(cam_logits.shape[-2:])} and features "
                f"{tuple(features.shape[-2:])} must share spatial size."
            )

        # (1) Multi-label classification.
        cls_in = cls_logits if cls_logits is not None else cam_logits.mean(dim=(2, 3))
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_in, image_labels.float()
        )

        # (2) Pixel-contrastive bias-aware loss.
        cam_norm = self._normalise_cam(cam_logits.detach())
        contrast_loss = self._pixel_contrast(features, cam_norm, image_labels)

        # (3) EMA buffer update.
        if self.training:
            self._ema_update(features.detach(), cam_norm, image_labels)

        total = (
            self.cls_weight * cls_loss
            + self.lambda_contrast * contrast_loss
        )
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
