"""DPL: Denoised Pseudo-Labeling for Source-Free Domain Adaptation (MICCAI 2021).

# Paper: https://arxiv.org/abs/2109.09735
# Reference: https://github.com/cchen-cc/SFDA-DPL

Algorithm summary (from the paper):
    DPL is a source-free domain adaptation method.  Pseudo-labels predicted
    by a fixed source-trained model on target images are noisy, so DPL
    *denoises* them at two granularities:

      * pixel-level: pixels whose prediction uncertainty
        u(x) = 1 - max_c softmax(f(x))_c is above a threshold are masked
        out of the self-training loss;
      * class-level / prototype: per-class EMA prototypes are maintained
        from the student features. Each pixel's pseudo-label is *refined*
        by combining the network softmax with a softmax over cosine
        similarities to the prototypes — this corrects spatially-incorrect
        but class-consistent confidences.

    The denoised, masked pseudo-labels are used as targets for a standard
    cross-entropy loss, optionally combined with an EMA-consistency MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("dpl")
class DPLLoss(nn.Module):
    """Denoised Pseudo-Labeling for source-free domain adaptation.

    Chen et al., MICCAI 2021.
    Reference implementation (not copied):
        https://github.com/cchen-cc/SFDA-DPL

    Components implemented from the paper formulas:
      * pixel-level uncertainty mask via 1 - max(softmax) > tau
      * class-level prototype EMA refinement of pseudo-labels
      * optional EMA-prediction consistency (MSE) for stability
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        pseudo_weight: float = 1.0,
        uncertainty_threshold: float = 0.2,
        num_classes: int = 5,
        feature_dim: int = 64,
        prototype_momentum: float = 0.99,
        prototype_weight: float = 0.5,
        consistency_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.pseudo_weight = pseudo_weight
        # Uncertainty u(x) = 1 - max(softmax). Pixels with u > tau are dropped.
        self.uncertainty_threshold = uncertainty_threshold
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        # EMA momentum for class prototypes (paper: ~0.99).
        self.prototype_momentum = prototype_momentum
        # Mixing weight between network prob and prototype-similarity prob.
        # final_prob = (1 - w) * net_prob + w * proto_prob
        self.prototype_weight = prototype_weight
        self.consistency_weight = consistency_weight

        # EMA prototypes (one per class). Lazily resized in _ensure_prototypes
        # if the first call gives a different feature dimension.
        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, feature_dim),
        )
        # Per-class init flag, so we initialise with the first batch's mean
        # feature rather than overwriting a zero vector by EMA.
        self.register_buffer(
            "proto_initialised",
            torch.zeros(num_classes, dtype=torch.bool),
        )

    # ------------------------------------------------------------------
    # Prototype machinery
    # ------------------------------------------------------------------
    def _ensure_prototypes(self, feat_dim: int, device: torch.device):
        """Reallocate the prototype buffer if feat_dim changed at runtime."""
        if self.prototypes.shape[1] != feat_dim:
            self.prototypes = torch.zeros(
                self.num_classes, feat_dim, device=device
            )
            self.proto_initialised = torch.zeros(
                self.num_classes, dtype=torch.bool, device=device
            )
            self.feature_dim = feat_dim

    @torch.no_grad()
    def _update_prototypes(
        self,
        features: torch.Tensor,      # (B, F, H, W) — student features
        pseudo_label: torch.Tensor,  # (B, H, W)   — argmax pseudo-label
        valid_mask: torch.Tensor,    # (B, H, W)   — confident-pixel mask
        confidence: torch.Tensor,    # (B, H, W)   — per-pixel max-prob
    ):
        """Confidence-weighted EMA update of per-class prototypes.

        Paper (SFDA-DPL, Sec. 3.3): prototypes are computed as the
        confidence-weighted average of student features over the *valid*
        (i.e. low-uncertainty) pixels of each class — not the unweighted mean.
        """
        B, Fdim, H, W = features.shape
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, Fdim)
        lbl_flat = pseudo_label.reshape(-1)
        valid_flat = valid_mask.reshape(-1)
        conf_flat = confidence.reshape(-1)

        for c in range(self.num_classes):
            sel = valid_flat & (lbl_flat == c)
            if not sel.any():
                continue
            # Confidence weighting: weight each valid pixel by its softmax
            # max-probability. Normalise so the sum of weights is 1.
            w = conf_flat[sel].unsqueeze(1)  # (Nc, 1)
            w = w / (w.sum() + 1e-6)
            class_feat = (feat_flat[sel] * w).sum(dim=0)
            if not bool(self.proto_initialised[c].item()):
                self.prototypes[c] = class_feat
                self.proto_initialised[c] = True
            else:
                m = self.prototype_momentum
                self.prototypes[c] = m * self.prototypes[c] + (1.0 - m) * class_feat

    def _prototype_similarity_prob(
        self,
        features: torch.Tensor,  # (B, F, H, W)
    ) -> torch.Tensor:
        """Cosine sim to each prototype -> softmax over classes."""
        B, Fdim, H, W = features.shape
        feat = F.normalize(features, dim=1)
        proto = F.normalize(self.prototypes, dim=1)  # (C, F)
        # (B, C, H, W) = einsum bfhw, cf -> bchw
        sim = torch.einsum("bfhw,cf->bchw", feat, proto)
        # Mask out un-initialised classes so they don't dominate.
        if (~self.proto_initialised).any():
            init_mask = self.proto_initialised.to(sim.dtype).view(1, -1, 1, 1)
            sim = sim * init_mask + (-1e4) * (1 - init_mask)
        return F.softmax(sim, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        target_pred: torch.Tensor,
        ema_pred: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            target_pred: student model predictions, (B, C, H, W).
            ema_pred:    EMA teacher predictions (optional), (B, C, H, W).
            target_features: student features used for prototype refinement,
                (B, F, H', W'). If None, falls back to using ``target_pred``
                as a (C-dim) pseudo-feature, which still gives a meaningful
                class-similarity refinement (now in logit space) without
                requiring the network to expose a separate feature head.
            labeled_loss: optional supervised loss to add to the total.
        """
        device = target_pred.device
        prob = F.softmax(target_pred, dim=1)
        confidence, pseudo_label = prob.max(dim=1)

        # ------------------------------------------------------------------
        # Class-level uncertainty (paper Sec. 3.2): for each class, compute
        # the mean uncertainty of pixels currently labelled to that class,
        # then keep pixels whose per-pixel uncertainty is BELOW that class's
        # mean. This is per-class, not per-pixel, and prevents an entire
        # rare class from being dropped because its absolute confidence is
        # systematically lower than the dominant class's.
        # ------------------------------------------------------------------
        uncertainty = 1.0 - confidence
        class_unc_thr = uncertainty.new_full(
            (self.num_classes,), self.uncertainty_threshold
        )
        for c in range(self.num_classes):
            sel_c = pseudo_label == c
            if sel_c.any():
                # Use the smaller of (paper class-mean) and (global threshold).
                class_unc_thr[c] = torch.minimum(
                    uncertainty[sel_c].mean(),
                    class_unc_thr[c],
                )
        per_pixel_thr = class_unc_thr[pseudo_label]      # (B, H, W)
        unc_mask = uncertainty <= per_pixel_thr
        # Also retain the absolute-confidence floor so very-low-confidence
        # pixels are dropped even if they pass the per-class threshold.
        conf_mask = confidence > self.confidence_threshold
        valid_mask = unc_mask & conf_mask

        # Prototype-based denoising — refine the pseudo-label via cosine
        # similarity to running per-class prototypes in feature space.
        feats = target_features if target_features is not None else target_pred
        # Resize features to match prediction H,W if a separate feature
        # head was passed at a different resolution.
        if feats.shape[-2:] != target_pred.shape[-2:]:
            feats = F.interpolate(
                feats, size=target_pred.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        self._ensure_prototypes(feats.shape[1], device)
        self._update_prototypes(
            feats.detach(), pseudo_label, valid_mask, confidence.detach()
        )

        proto_prob = self._prototype_similarity_prob(feats.detach())
        refined_prob = (1.0 - self.prototype_weight) * prob.detach() \
            + self.prototype_weight * proto_prob
        refined_label = refined_prob.argmax(dim=1)
        # Refresh the validity mask using the refined probabilities.
        ref_conf, _ = refined_prob.max(dim=1)
        ref_unc = 1.0 - ref_conf
        refined_valid = (ref_unc < self.uncertainty_threshold) & valid_mask

        # ----- Pseudo-label CE on denoised labels --------------------
        if refined_valid.any():
            B, C, H, W = target_pred.shape
            logits_flat = target_pred.permute(0, 2, 3, 1).reshape(-1, C)
            label_flat = refined_label.reshape(-1)
            valid_flat = refined_valid.reshape(-1)
            pseudo_loss = F.cross_entropy(
                logits_flat[valid_flat],
                label_flat[valid_flat],
            )
        else:
            pseudo_loss = torch.tensor(0.0, device=device)

        total_loss = self.pseudo_weight * pseudo_loss

        # ----- Optional EMA-consistency (back-compat with old config) ----
        if ema_pred is not None and self.consistency_weight > 0.0:
            ema_prob = F.softmax(ema_pred, dim=1)
            cons_loss = F.mse_loss(prob, ema_prob)
            total_loss = total_loss + self.consistency_weight * cons_loss

        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss

        return total_loss
