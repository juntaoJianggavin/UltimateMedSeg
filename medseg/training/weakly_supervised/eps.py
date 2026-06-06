# Reference: https://github.com/halbielee/EPS
"""EPS — Explicit Pseudo-pixel Supervision via Saliency.

Lee et al., "Railroad is not a Train: Saliency as Pseudo-pixel Supervision
for Weakly Supervised Semantic Segmentation", CVPR 2021.
Paper: https://arxiv.org/abs/2105.08965
Official repository: https://github.com/halbielee/EPS

Algorithm (from official ``eps.py``):

    Inputs per batch:
        * cam (B, C+1, H, W) — logits with channel 0 = background
        * saliency (B, 1, H, W) — off-the-shelf saliency in [0, 1]
        * label (B, C) — multi-label binary tags

    1. Compute sal_pred = softmax(cam, dim=1)         # (B, C+1, H, W)
    2. Per-class IoU validation:
         iou_c = IoU(round(sal_pred[:, c]), round(saliency))
         valid_c = (iou_c > tau) & (label[:, c] == 1)
    3. Build fg_map / bg_map:
         fg_map[b, c] = sal_pred[b, c] if valid_c else 0    (c = 1..C)
         bg_map[b, c] = sal_pred[b, c] if ~valid_c else 0   (c = 1..C)
         bg_map[b, C+1] = sal_pred[b, 0]                      (background channel)
         fg_map = sum(fg_map, dim=1)    # (B, 1, H, W)
         bg_map = sum(bg_map, dim=1)    # (B, 1, H, W)
    4. Blend with lambda:
         blended = fg_map * lam + bg_map * (1 - lam)
    5. Loss:
         L = MSE(blended, saliency)

This module implements the loss; CAM generation and saliency extraction are
upstream concerns.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("eps")
class EPSLoss(nn.Module):
    """Saliency-as-pseudo-pixel-supervision loss (EPS, CVPR 2021).

    Matches official ``eps.py``: ``get_eps_loss(cam, saliency, num_classes,
    label, tau, lam)`` — IoU-based channel selection + λ-blending.

    Args:
        lambda_sal: Weight on the saliency MSE consistency term
            (paper default 0.5).
        cls_weight: Weight on the image-level multi-label classification term.
        seg_weight: Weight on the supervised CE term when a pixel-level
            ``target`` is also available (mixed-supervision setting).
        tau: IoU threshold for channel validity (official default 0.3).
        lam: Blending ratio λ for fg*λ + bg*(1-λ) (official default 0.5).
    """

    def __init__(
        self,
        lambda_sal: float = 0.5,
        cls_weight: float = 1.0,
        seg_weight: float = 1.0,
        tau: float = 0.3,
        lam: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.lambda_sal = lambda_sal
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.tau = tau
        self.lam = lam

    @staticmethod
    def _to_4d(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    def forward(
        self,
        predictions: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        saliency_map: Optional[torch.Tensor] = None,
        image_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) semantic logits, channel 0 = background.
            target: Optional (B, H, W) dense pixel labels (-1 = ignore).
            saliency_map: (B, 1, H, W) or (B, H, W) saliency in [0, 1].
            image_labels: (B, C_fg) binary multi-label tags.

        Returns:
            Scalar loss.
        """
        B, C, H, W = predictions.shape
        total = predictions.new_zeros(())

        prob = F.softmax(predictions, dim=1)

        # ---- (1) image-level multi-label classification ----------------
        if image_labels is not None and self.cls_weight > 0:
            # GAP over spatial dims -> (B, C)
            global_logits = predictions.mean(dim=(2, 3))
            if image_labels.dim() == 2 and image_labels.shape[1] == C - 1 and C > 1:
                # Skip bg channel of logits to align with fg-only image_labels.
                global_logits = global_logits[:, 1:]
            elif image_labels.dim() == 2 and image_labels.shape[1] != C:
                # Fallback: truncate/pad to match
                n = min(image_labels.shape[1], global_logits.shape[1])
                global_logits = global_logits[:, :n]
                image_labels = image_labels[:, :n]
            cls_loss = F.binary_cross_entropy_with_logits(
                global_logits, image_labels.float()
            )
            total = total + self.cls_weight * cls_loss

        # ---- (2) saliency consistency (official get_eps_loss) ----------
        if saliency_map is not None and self.lambda_sal > 0:
            sal = self._to_4d(saliency_map).float()
            # Normalise saliency to [0, 1] if it looks like a 0-255 map.
            if sal.max() > 1.5:
                sal = sal / 255.0
            sal = sal.clamp(0.0, 1.0)
            if sal.shape[-2:] != (H, W):
                sal = F.interpolate(sal, size=(H, W),
                                    mode='bilinear', align_corners=False)

            if C > 1 and image_labels is not None:
                # C_fg = C - 1 (exclude bg channel 0)
                C_fg = C - 1
                label_fg = image_labels.float()
                if label_fg.shape[1] != C_fg:
                    label_fg = label_fg[:, :C_fg]
                fg_prob = prob[:, 1:]                        # (B, C_fg, H, W)

                # --- IoU-based channel selection (official) ---
                # iou_c = IoU(round(sal_pred_c), round(saliency))
                pred_binary = torch.round(fg_prob.detach())  # (B, C_fg, H, W)
                sal_binary = torch.round(sal.detach())       # (B, 1, H, W)
                intersection = (pred_binary * sal_binary).view(B, C_fg, -1).sum(-1)
                pred_sum = pred_binary.view(B, C_fg, -1).sum(-1)
                iou = intersection / (pred_sum + 1e-4)       # (B, C_fg)

                # valid = (iou > tau) AND label==1
                valid = (iou > self.tau) & (label_fg.bool()) # (B, C_fg)

                # --- Build fg_map / bg_map ---
                # fg: only valid channels; bg: invalid channels + bg channel
                fg_map = torch.zeros_like(fg_prob)           # (B, C_fg, H, W)
                bg_map = torch.zeros_like(fg_prob)
                valid_exp = valid.view(B, C_fg, 1, 1).expand_as(fg_prob)
                fg_map[valid_exp] = fg_prob[valid_exp]
                bg_map[~valid_exp] = fg_prob[~valid_exp]
                fg_map = fg_map.sum(dim=1, keepdim=True)     # (B, 1, H, W)
                bg_map = bg_map.sum(dim=1, keepdim=True)     # (B, 1, H, W)
                bg_map = bg_map + prob[:, :1]                 # add bg channel prob

                # Blend: fg * lam + bg * (1 - lam)
                blended = fg_map * self.lam + bg_map * (1.0 - self.lam)
            elif C > 1:
                # No labels available — fallback: all fg channels
                fg_map = prob[:, 1:].sum(dim=1, keepdim=True)
                bg_map = prob[:, :1]
                blended = fg_map * self.lam + bg_map * (1.0 - self.lam)
            else:
                # C == 1: binary case — sigmoid-style
                fg_map = prob
                bg_map = 1.0 - prob
                blended = fg_map * self.lam + bg_map * (1.0 - self.lam)

            sal_loss = F.mse_loss(blended, sal)
            total = total + self.lambda_sal * sal_loss

        # ---- (3) optional dense CE on mixed-supervision pixels ---------
        if target is not None and self.seg_weight > 0:
            total = total + self.seg_weight * F.cross_entropy(
                predictions, target.long(), ignore_index=-1
            )

        return total
