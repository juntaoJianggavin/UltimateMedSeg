# MARS (NeurIPS 2023)
# Reference: https://github.com/shjo-april/MARS
# Paper: https://arxiv.org/abs/2304.09913
# Implemented from paper formulas; not a copy of the official repo.
"""MARS — Mask-guided Activation Reactivation Strategy for WSSS.

Jang et al., "MARS: Model-agnostic Biased Object Removal without
Additional Supervision for Weakly-Supervised Semantic Segmentation",
NeurIPS 2023.
Paper: https://arxiv.org/abs/2304.09913
Official repository: https://github.com/shjo-april/MARS

Why it exists.
    A classifier trained with image-level labels concentrates its CAM on
    the most discriminative parts of an object. MARS forces the network
    to discover the *rest* of the object via a two-pass forward:

        1.  Forward the original image -> CAM_orig and cls_logits_orig.
        2.  Build an erasing mask M from the top-K CAM_orig activations
            (the discriminative regions). Forward the masked image and
            obtain CAM_react and cls_logits_react.
        3.  Apply the standard multi-label BCE on BOTH passes — the
            classifier is forced to find evidence for the same labels
            even after the discriminative region has been removed,
            therefore the "re-activated" CAM lights up complementary
            object parts.
        4.  An L1 consistency term outside the erased region keeps the two
            CAMs aligned where they CAN agree (avoids drift).

Loss (paper Sec. 3.2):

    L = L_cls(orig)
      + lambda_react * L_cls(react)
      + lambda_cons  * mean_{(x,y) not in M} | CAM_orig(x,y) - CAM_react(x,y) |
      + lambda_spread* L_spread

  L_spread is a coverage prior that encourages the union CAM
  (max(CAM_orig, CAM_react)) to cover MORE pixels than CAM_orig alone but
  not the whole image. We implement it as a hinge on the foreground area
  ratio of the union: penalise area_ratio < a_min (under-coverage) and
  area_ratio > a_max (background bleed).

This module assumes the caller has done both forward passes and the
erasing — it is responsible only for the loss computation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("mars")
class MARSLoss(nn.Module):
    """MARS classification + reactivation + consistency + coverage loss.

    Args:
        lambda_react: Weight on the reactivated-pass classification BCE
            (paper default 1.0).
        lambda_cons: Weight on the unerased-region CAM consistency L1
            (default 0.1).
        lambda_spread: Weight on the coverage hinge (default 0.05).
        area_min: Lower hinge bound for the union CAM foreground area
            ratio (default 0.05 — at least 5% of the image must be fg).
        area_max: Upper hinge bound (default 0.6 — bleed beyond 60% is
            penalised).
        bce_thresh: CAM logits threshold (after sigmoid) above which a
            pixel counts towards the area ratio (default 0.4).
    """

    def __init__(
        self,
        lambda_react: float = 1.0,
        lambda_cons: float = 0.1,
        lambda_spread: float = 0.05,
        area_min: float = 0.05,
        area_max: float = 0.6,
        bce_thresh: float = 0.4,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < area_min < area_max < 1.0):
            raise ValueError(
                f"Need 0 < area_min ({area_min}) < area_max ({area_max}) < 1"
            )
        self.lambda_react = lambda_react
        self.lambda_cons = lambda_cons
        self.lambda_spread = lambda_spread
        self.area_min = area_min
        self.area_max = area_max
        self.bce_thresh = bce_thresh

    # ------------------------------------------------------------------
    @staticmethod
    def _cls_bce(
        cam_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if cam_logits.dim() == 4:
            cam_logits = cam_logits.mean(dim=(2, 3))
        return F.binary_cross_entropy_with_logits(cam_logits, labels.float())

    def forward(
        self,
        cam_orig: torch.Tensor,
        cam_react: torch.Tensor,
        erase_mask: torch.Tensor,
        image_labels: torch.Tensor,
        cls_logits_orig: Optional[torch.Tensor] = None,
        cls_logits_react: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            cam_orig: (B, C, H, W) RAW CAM logits from the original-image pass.
            cam_react: (B, C, H, W) RAW CAM logits from the masked-image pass.
            erase_mask: (B, 1, H, W) or (B, H, W) binary {0,1} — 1 where
                the input image was erased before the second forward. The
                consistency loss is computed on the COMPLEMENT (where the
                two passes saw the same input).
            image_labels: (B, C) binary multi-label tags.
            cls_logits_orig: Optional (B, C) classifier head logits for the
                original pass — if not supplied, GAP(cam_orig) is used.
            cls_logits_react: Same but for the reactivated pass.
            labeled_loss: Optional dense supervised loss to add.
        """
        if cam_orig.shape != cam_react.shape:
            raise ValueError(
                f"cam_orig {tuple(cam_orig.shape)} and cam_react "
                f"{tuple(cam_react.shape)} must match."
            )
        if erase_mask.dim() == 3:
            erase_mask = erase_mask.unsqueeze(1)
        if erase_mask.shape[-2:] != cam_orig.shape[-2:]:
            erase_mask = F.interpolate(
                erase_mask.float(), size=cam_orig.shape[-2:], mode="nearest"
            )
        keep_mask = 1.0 - erase_mask.clamp(0.0, 1.0)            # (B,1,H,W)

        # (1) Original-pass classification.
        cls_o = cls_logits_orig if cls_logits_orig is not None else cam_orig
        cls_r = cls_logits_react if cls_logits_react is not None else cam_react
        cls_loss = self._cls_bce(cls_o, image_labels)
        react_loss = self._cls_bce(cls_r, image_labels)

        # (2) CAM consistency on the unerased region only.
        # Mask by image_labels so absent classes don't drift.
        present = image_labels.float().view(image_labels.size(0), -1, 1, 1)
        diff = (cam_orig - cam_react).abs() * present
        diff = diff * keep_mask
        denom = (present.expand_as(diff) * keep_mask.expand_as(diff)).sum().clamp_min(1.0)
        cons_loss = diff.sum() / denom

        # (3) Coverage hinge on the union CAM (sigmoid).
        prob_o = torch.sigmoid(cam_orig) * present
        prob_r = torch.sigmoid(cam_react) * present
        union = torch.maximum(prob_o, prob_r)                  # (B,C,H,W)
        # Per-image foreground area ratio = mean over fg-channels and pixels.
        # Use a soft area via max over present channels then mean over space.
        per_image = union.amax(dim=1)                          # (B,H,W)
        area_ratio = (per_image > self.bce_thresh).float().mean(dim=(1, 2))
        # Hinge: below area_min or above area_max.
        under = F.relu(self.area_min - area_ratio)
        over = F.relu(area_ratio - self.area_max)
        spread_loss = (under + over).mean()

        total = (
            cls_loss
            + self.lambda_react * react_loss
            + self.lambda_cons * cons_loss
            + self.lambda_spread * spread_loss
        )
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
