# LPCAM (CVPR 2023)
# Reference: https://github.com/zhaozhengChen/LPCAM
# Paper: https://arxiv.org/abs/2304.09244
# Implemented from paper formulas; not a copy of the official repo.
"""LPCAM — Local Prototype CAM for Weakly Supervised Semantic Segmentation.

Chen et al., "Extracting Class Activation Maps from Non-Discriminative
Features as well", CVPR 2023.
Paper: https://arxiv.org/abs/2304.09244
Official repository: https://github.com/zhaozhengChen/LPCAM

Why it exists.
    Standard CAM = w_c^T F uses a *single* class-specific weight vector
    (the GAP-derived linear classifier weight) per class. That weight
    captures only the most discriminative direction of the class, so its
    inner product with features fires only on a small portion of the
    object. LPCAM replaces the single ``w_c`` with a small set of K
    "local prototypes" per class — each prototype is the centroid of a
    cluster of feature vectors taken from confidently-foreground regions
    of the training set:

        LP-CAM_c(x, y) = (1/K) sum_{k=1..K} cos( F(x, y), mu_{c, k} )

    Aggregating cosine similarity over multiple prototypes lights up not
    only the discriminative head/leg of the object but also smooth body
    regions, producing a more complete localisation map (paper Fig. 2).

Loss formula used in this in-tree implementation (paper Sec. 3.3):

    L = L_BCE( GAP(LP-CAM) , y )                         # classification
      + lambda_proto * L_proto                            # prototype pull
      + lambda_div   * L_div                              # prototype diversity

  * L_proto pulls each class's prototypes towards the high-confidence
    foreground feature vectors of that class via a mean-squared-error
    (equivalent to one step of soft k-means with confidence weights).
  * L_div pushes prototypes inside a class apart (1 - mean off-diagonal
    cosine of the K-prototype Gram matrix) so they do not collapse onto a
    single direction — exactly the failure mode plain CAM suffers from.

Prototypes are stored as a non-learnable buffer and updated by EMA from
the per-batch high-confidence pull targets; this is the canonical
prototype-bank recipe in the LPCAM paper. No code is copied from the
official repository.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("lpcam")
class LPCAMLoss(nn.Module):
    """LPCAM classification + prototype pull + diversity loss.

    Args:
        num_classes: Number of foreground classes (no bg slot).
        feature_dim: Channel count of the backbone feature map fed to
            ``forward`` via ``features``. Must match.
        num_prototypes: K, the number of local prototypes per class
            (paper default 5).
        ema_momentum: EMA coefficient for prototype updates (default 0.9).
        cls_weight: Weight on the BCE(GAP(LP-CAM), y) classification term.
        lambda_proto: Weight on the prototype-pull MSE term.
        lambda_div: Weight on the intra-class prototype diversity term.
        high_thresh: Confidence above which a pixel is considered a
            reliable foreground sample of its argmax class for the pull
            target (paper default 0.7 on the [0,1]-normalised cam).
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        num_prototypes: int = 5,
        ema_momentum: float = 0.9,
        cls_weight: float = 1.0,
        lambda_proto: float = 0.1,
        lambda_div: float = 0.05,
        high_thresh: float = 0.7,
        **kwargs,
    ):
        super().__init__()
        if num_classes <= 0 or feature_dim <= 0 or num_prototypes <= 0:
            raise ValueError(
                f"num_classes/feature_dim/num_prototypes must be positive "
                f"(got {num_classes}, {feature_dim}, {num_prototypes})"
            )
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.ema_momentum = ema_momentum
        self.cls_weight = cls_weight
        self.lambda_proto = lambda_proto
        self.lambda_div = lambda_div
        self.high_thresh = high_thresh

        # (C, K, D) prototypes — random init, L2-normalised, then refined
        # in-place by EMA from confident pixels.
        proto = torch.randn(num_classes, num_prototypes, feature_dim)
        proto = F.normalize(proto, dim=-1)
        self.register_buffer("prototypes", proto)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_cam(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B, C, H, W = cam.shape
        flat = cam.view(B, C, -1)
        m = flat.amin(dim=2, keepdim=True)
        M = flat.amax(dim=2, keepdim=True)
        return ((flat - m) / (M - m + eps)).view(B, C, H, W)

    # ------------------------------------------------------------------
    def compute_lp_cam(self, features: torch.Tensor) -> torch.Tensor:
        """Build LP-CAM logits = mean_k cosine(F(x,y), mu_{c,k}).

        Args:
            features: (B, D, H, W).

        Returns:
            (B, C, H, W) LP-CAM logits in [-1, 1].
        """
        B, D, H, W = features.shape
        if D != self.feature_dim:
            raise ValueError(
                f"features channel {D} != feature_dim {self.feature_dim}"
            )
        f = F.normalize(features, dim=1)                       # (B, D, H, W)
        p = F.normalize(self.prototypes, dim=-1)               # (C, K, D)
        # Per-prototype cosine map: (B, C, K, H, W)
        # einsum: f(b, d, h, w) * p(c, k, d) -> (b, c, k, h, w)
        cam_pk = torch.einsum("bdhw,ckd->bckhw", f, p)
        return cam_pk.mean(dim=2)                              # (B, C, H, W)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update_prototypes(
        self,
        features: torch.Tensor,
        cam_norm: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> None:
        """One EMA step: pull the K prototypes of each present class towards
        the K most-confident foreground feature vectors of that class."""
        B, D, H, W = features.shape
        feat_flat = F.normalize(features, dim=1).permute(0, 2, 3, 1).reshape(-1, D)
        # (B, C, HW)
        cam_flat = cam_norm.view(B, self.num_classes, -1)

        for c in range(self.num_classes):
            # Gather all confident features across batch for class c.
            pulled = []
            for b in range(B):
                if image_labels[b, c].item() <= 0:
                    continue
                conf = cam_flat[b, c]                          # (HW,)
                top = (conf > self.high_thresh).nonzero(as_tuple=False).squeeze(1)
                if top.numel() == 0:
                    continue
                base = b * H * W
                pulled.append(feat_flat[base + top])
            if not pulled:
                continue
            pool = torch.cat(pulled, dim=0)                    # (P, D)
            if pool.size(0) < self.num_prototypes:
                # Replicate to fill K slots; safe because EMA dampens.
                rep = (self.num_prototypes + pool.size(0) - 1) // pool.size(0)
                pool = pool.repeat(rep, 1)[: self.num_prototypes]
            else:
                # Pick K via similarity to current prototypes (assignment).
                cur = F.normalize(self.prototypes[c], dim=-1)  # (K, D)
                assign = (pool @ cur.t()).argmax(dim=1)        # (P,)
                new_centroids = []
                for k in range(self.num_prototypes):
                    members = pool[assign == k]
                    if members.numel() == 0:
                        new_centroids.append(cur[k])
                    else:
                        new_centroids.append(
                            F.normalize(members.mean(dim=0), dim=-1)
                        )
                pool = torch.stack(new_centroids, dim=0)        # (K, D)

            updated = (
                self.ema_momentum * self.prototypes[c]
                + (1.0 - self.ema_momentum) * pool
            )
            self.prototypes[c] = F.normalize(updated, dim=-1)

    # ------------------------------------------------------------------
    def _diversity_loss(self) -> torch.Tensor:
        """1 - mean off-diagonal cosine of per-class prototype Gram."""
        p = F.normalize(self.prototypes, dim=-1)               # (C, K, D)
        gram = torch.einsum("ckd,cld->ckl", p, p)              # (C, K, K)
        K = gram.shape[-1]
        if K == 1:
            return p.new_zeros(())
        # Off-diagonal mask.
        eye = torch.eye(K, device=p.device, dtype=p.dtype).unsqueeze(0)
        off = gram * (1.0 - eye)
        # We want off-diagonal to be small → minimise mean |off|.
        denom = K * (K - 1)
        return off.abs().sum(dim=(1, 2)).mean() / denom

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        image_labels: torch.Tensor,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, D, H, W) backbone feature map.
            image_labels: (B, C) binary multi-label tags.
            labeled_loss: Optional pre-computed dense supervised term.
        """
        if features.dim() != 4:
            raise ValueError(
                f"features must be (B, D, H, W); got {tuple(features.shape)}"
            )
        if image_labels.shape[1] != self.num_classes:
            raise ValueError(
                f"image_labels has {image_labels.shape[1]} classes but "
                f"loss configured with {self.num_classes}"
            )

        lp_cam = self.compute_lp_cam(features)                 # (B, C, H, W)
        cam_norm = self._normalise_cam(lp_cam.detach())

        # (1) BCE(GAP(LP-CAM), y).
        gap = lp_cam.mean(dim=(2, 3))                          # (B, C)
        # LP-CAM logits live in [-1, 1] → scale ×10 (paper Sec. 3.3, fits
        # BCE saturation regime, equivalent to a learnable temperature).
        cls_loss = F.binary_cross_entropy_with_logits(
            10.0 * gap, image_labels.float()
        )

        # (2) Prototype-pull MSE: for each present class, pull prototypes
        # towards the mean of confident foreground features (differentiable
        # path even when EMA buffer is detached).
        feat_n = F.normalize(features, dim=1)
        proto_n = F.normalize(self.prototypes, dim=-1)         # (C, K, D)
        proto_loss = features.new_zeros(())
        n_classes_active = 0
        for c in range(self.num_classes):
            mask = (cam_norm[:, c] > self.high_thresh).float()  # (B, H, W)
            # Mask out classes absent from the image-level label.
            present = image_labels[:, c].float().view(-1, 1, 1)
            mask = mask * present
            weight = mask.sum()
            if weight < 1.0:
                continue
            # Confidence-weighted mean feature: (D,)
            target = (feat_n[:, :, :, :] * mask.unsqueeze(1)).sum(dim=(0, 2, 3)) / weight
            target = F.normalize(target, dim=0)
            # Pull each prototype of class c towards this mean.
            proto_loss = proto_loss + (1.0 - (proto_n[c] @ target)).mean()
            n_classes_active += 1
        if n_classes_active > 0:
            proto_loss = proto_loss / n_classes_active

        # (3) Diversity.
        div_loss = self._diversity_loss()

        # EMA buffer update (no grad).
        if self.training:
            self._ema_update_prototypes(features.detach(), cam_norm, image_labels)

        total = (
            self.cls_weight * cls_loss
            + self.lambda_proto * proto_loss
            + self.lambda_div * div_loss
        )
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
