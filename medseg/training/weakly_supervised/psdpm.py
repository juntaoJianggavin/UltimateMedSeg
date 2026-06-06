# PSDPM (CVPR 2024)
# Reference: https://github.com/xinqiaozhao/PSDPM
# Paper: https://arxiv.org/abs/2403.07630
# Implemented from paper formulas; not a copy of the official repo.
"""PSDPM — Prototype-based Secondary Discriminative Pixels Mining for
Weakly Supervised Semantic Segmentation.

Zhao et al., "PSDPM: Prototype-based Secondary Discriminative Pixels
Mining for Weakly Supervised Semantic Segmentation", CVPR 2024.
Paper: https://arxiv.org/abs/2403.07630
Reference repository: https://github.com/xinqiaozhao/PSDPM

Why it exists.
    Classical CAM activates only on the *primary* discriminative pixels
    of an object (head, wings, ...). The remaining "secondary" pixels
    that belong to the same instance are typically silent because no
    BCE gradient flows through them. PSDPM mines these secondary pixels
    by computing the cosine similarity between every pixel feature and
    a *class prototype* aggregated from the primary pixels themselves —
    pixels whose prototype similarity is high but whose CAM is low are
    declared "secondary fg" and given an additional cross-entropy
    supervision.

Loss (paper Sec. 3.3, Eq. 6-10):

    p_c = sum_{(x,y) : CAM_c(x,y) > tau_p} F(x,y) / Z       (Eq. 6 primary proto)

    s_c(x,y) = cos(F(x,y), p_c)                             (Eq. 7 similarity)

    secondary_c(x,y) = 1[ s_c > tau_s  AND  CAM_c < tau_p ] (Eq. 8 SDP set)

    L_sdp = - sum_{c in y} mean_{(x,y) in secondary_c}
                                log P(class=c | x,y)        (Eq. 9)

    L_proto = sum_{c in y} mean_{(x,y) : CAM_c > tau_p}
                                 1 - cos(F(x,y), p_c)        (Eq. 10 prototype align)

    L = cls_weight * BCE(GAP(cam), y) + lambda_sdp * L_sdp + lambda_proto * L_proto

This module implements the loss only. ``predictions`` are the dense
semantic logits (channel 0 = background, channels 1..C_fg = fg classes)
needed to compute log P. ``features`` are the backbone feature map used
to build the prototypes. CAM and image labels come from the caller.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("psdpm")
class PSDPMLoss(nn.Module):
    """PSDPM classification + secondary-pixel CE + prototype alignment.

    Args:
        lambda_sdp: Weight on the secondary-discriminative-pixel CE term
            (paper default 0.5).
        lambda_proto: Weight on the prototype alignment term
            (paper default 0.2).
        cls_weight: Weight on the multi-label BCE classification term.
        tau_primary: Normalised-CAM threshold for primary fg pixels
            (default 0.5).
        tau_secondary: Cosine-similarity threshold for accepting a pixel
            as a secondary fg of class c (default 0.7).
        fg_channel_start: Index of the first fg channel in ``predictions``;
            1 keeps channel 0 as background.
    """

    def __init__(
        self,
        lambda_sdp: float = 0.5,
        lambda_proto: float = 0.2,
        cls_weight: float = 1.0,
        tau_primary: float = 0.5,
        tau_secondary: float = 0.7,
        fg_channel_start: int = 1,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < tau_primary < 1.0):
            raise ValueError(f"tau_primary must be in (0,1); got {tau_primary}")
        if not (0.0 < tau_secondary < 1.0):
            raise ValueError(f"tau_secondary must be in (0,1); got {tau_secondary}")
        self.lambda_sdp = lambda_sdp
        self.lambda_proto = lambda_proto
        self.cls_weight = cls_weight
        self.tau_primary = tau_primary
        self.tau_secondary = tau_secondary
        self.fg_channel_start = fg_channel_start

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_cam(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B, C, H, W = cam.shape
        flat = cam.view(B, C, -1)
        m = flat.amin(dim=2, keepdim=True)
        M = flat.amax(dim=2, keepdim=True)
        return ((flat - m) / (M - m + eps)).view(B, C, H, W)

    # ------------------------------------------------------------------
    def _build_prototypes(
        self,
        features: torch.Tensor,
        cam_norm: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Per-image per-class prototype from primary pixels (Eq. 6).

        Returns (B, C, D); for absent classes the prototype is the zero
        vector and downstream consumers must mask it out.
        """
        B, D, H, W = features.shape
        _, C, _, _ = cam_norm.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
        cam_flat = cam_norm.view(B, C, H * W)

        # Soft mask = cam * 1[cam > tau_primary]
        hard = (cam_flat > self.tau_primary).float()
        w = cam_flat * hard                                       # (B,C,HW)
        denom = w.sum(dim=2, keepdim=True).clamp_min(1e-6)
        # (B, C, D) = w @ feat
        proto = torch.einsum("bcn,bnd->bcd", w, feat_flat) / denom
        # Zero out absent classes.
        proto = proto * image_labels.float().unsqueeze(-1)
        return F.normalize(proto, dim=-1)

    # ------------------------------------------------------------------
    def forward(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        cam_logits: torch.Tensor,
        image_labels: torch.Tensor,
        cls_logits: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C_total, H, W) dense semantic logits. Channel
                0 is treated as background when ``fg_channel_start`` == 1;
                the remaining channels are the fg classes.
            features: (B, D, H, W) backbone feature map at the same
                spatial resolution as ``cam_logits``.
            cam_logits: (B, C_fg, H, W) raw CAM logits (fg only).
            image_labels: (B, C_fg) multi-label tags.
            cls_logits: optional (B, C_fg) classifier head; if absent the
                BCE is taken on GAP(cam_logits).
            labeled_loss: optional dense supervised loss to add.
        """
        if predictions.dim() != 4 or features.dim() != 4 or cam_logits.dim() != 4:
            raise ValueError(
                "predictions, features and cam_logits must all be 4-D."
            )
        if cam_logits.shape[-2:] != features.shape[-2:]:
            raise ValueError(
                f"cam {tuple(cam_logits.shape[-2:])} and features "
                f"{tuple(features.shape[-2:])} must share spatial size."
            )
        if predictions.shape[-2:] != cam_logits.shape[-2:]:
            predictions_r = F.interpolate(
                predictions, size=cam_logits.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        else:
            predictions_r = predictions
        B, _, H, W = predictions_r.shape
        C_fg = cam_logits.shape[1]
        if predictions_r.shape[1] < self.fg_channel_start + C_fg:
            raise ValueError(
                f"predictions has {predictions_r.shape[1]} channels but "
                f"fg_channel_start={self.fg_channel_start} + C_fg={C_fg} "
                f"would over-run."
            )

        # (1) Multi-label classification BCE.
        cls_in = cls_logits if cls_logits is not None else cam_logits.mean(dim=(2, 3))
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_in, image_labels.float()
        )

        # (2) Build per-class prototypes from primary fg pixels.
        cam_norm = self._normalise_cam(cam_logits.detach())
        proto = self._build_prototypes(features, cam_norm, image_labels)  # (B,C,D)

        # (3) Similarity map & SDP set.
        feat_n = F.normalize(features, dim=1)                     # (B,D,H,W)
        # sim (B,C,H,W) = einsum bdhw, bcd -> bchw
        sim = torch.einsum("bdhw,bcd->bchw", feat_n, proto)

        log_p = F.log_softmax(predictions_r, dim=1)

        sdp_terms = []
        proto_terms = []
        for b in range(B):
            present = image_labels[b].nonzero(as_tuple=False).squeeze(1)
            if present.numel() == 0:
                continue
            for c in present.tolist():
                # primary mask
                prim = (cam_norm[b, c] > self.tau_primary)
                # secondary mask = high prototype similarity AND NOT primary
                sec = (sim[b, c] > self.tau_secondary) & (~prim)
                if sec.any():
                    target_chan = c + self.fg_channel_start
                    lp = log_p[b, target_chan]
                    sdp_terms.append(-lp[sec].mean())
                if prim.any():
                    proto_terms.append((1.0 - sim[b, c][prim]).mean())

        sdp_loss = (
            torch.stack(sdp_terms).mean() if sdp_terms
            else predictions.new_zeros(())
        )
        proto_loss = (
            torch.stack(proto_terms).mean() if proto_terms
            else predictions.new_zeros(())
        )

        total = (
            self.cls_weight * cls_loss
            + self.lambda_sdp * sdp_loss
            + self.lambda_proto * proto_loss
        )
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
