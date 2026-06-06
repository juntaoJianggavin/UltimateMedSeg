"""Pixel-wise Contrastive Loss for semantic segmentation.

Faithful reimplementation of the supervised pixel-wise contrastive loss
(SupContrast / ContrastiveSeg) from:

    Wang et al., "Exploring Cross-Image Pixel Contrast for Semantic
    Segmentation", ICCV 2021.
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.

Formula (per-anchor):
    L_anchor = - 1/|P(i)| * sum_{p in P(i)} log( exp(z_i . z_p / tau)
              / sum_{a in A(i)} exp(z_i . z_a / tau) )

with
    P(i) = positives of anchor i (same class, excluding i)
    A(i) = all other samples (positives + negatives)

Differences from the previous implementation:
  * Inputs are *features* (B, F, H, W) when available, otherwise the
    L2-normalised softmax probabilities are used as a fallback feature
    embedding so that the loss remains usable without an explicit
    projection head.
  * Anchors are sampled per class so that small classes are not
    dominated by large classes (matching ContrastiveSeg's class-balanced
    sampling policy).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("contrastive")
class ContrastiveLoss(nn.Module):
    """Supervised pixel-wise contrastive loss."""

    def __init__(
        self,
        temperature: float = 0.1,
        max_anchors_per_class: int = 50,
        max_negatives: int = 256,
        ignore_index: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.max_anchors_per_class = int(max_anchors_per_class)
        self.max_negatives = int(max_negatives)
        self.ignore_index = ignore_index

    def _sample_anchors(self, labels: torch.Tensor):
        """Pick at most ``max_anchors_per_class`` indices per class.

        labels: (N,) long.  Returns indices into [0, N).
        """
        idxs = []
        unique = torch.unique(labels)
        for c in unique.tolist():
            if self.ignore_index is not None and c == self.ignore_index:
                continue
            mask_c = (labels == c).nonzero(as_tuple=False).flatten()
            if mask_c.numel() == 0:
                continue
            if mask_c.numel() > self.max_anchors_per_class:
                perm = torch.randperm(mask_c.numel(), device=labels.device)
                mask_c = mask_c[perm[: self.max_anchors_per_class]]
            idxs.append(mask_c)
        return torch.cat(idxs) if idxs else labels.new_empty(0, dtype=torch.long)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            pred: (B, C, H, W) logits — only used as a fallback feature source
                  when ``features`` is None.
            target: (B, H, W) long.
            features: optional (B, F, H_f, W_f) feature map; will be resized
                  to the target's spatial size if needed.
        """
        target = target.long()
        B, C, H, W = pred.shape

        if features is None:
            feats = F.softmax(pred, dim=1)
        else:
            feats = features
            if feats.shape[-2:] != (H, W):
                feats = F.interpolate(feats, size=(H, W), mode="bilinear",
                                      align_corners=False)

        F_dim = feats.shape[1]
        feats = feats.permute(0, 2, 3, 1).reshape(-1, F_dim)
        feats = F.normalize(feats, dim=1)
        labels = target.reshape(-1)

        if self.ignore_index is not None:
            valid = labels != self.ignore_index
            feats = feats[valid]
            labels = labels[valid]
        if labels.numel() < 2:
            return pred.new_zeros(())

        # Class-balanced anchor sampling.
        anchor_idx = self._sample_anchors(labels)
        if anchor_idx.numel() == 0:
            return pred.new_zeros(())

        # Cap the negative pool size per anchor for memory safety.
        if labels.numel() > self.max_negatives:
            n_full = labels.numel()
            perm = torch.randperm(n_full, device=labels.device)
            keep = perm[: self.max_negatives]
            # Always include the anchors themselves.
            keep = torch.unique(torch.cat([keep, anchor_idx]))
            # Build a remap on the *original* (pre-trim) coordinate system.
            new_pos = torch.full(
                (n_full,), -1, device=labels.device, dtype=torch.long
            )
            new_pos[keep] = torch.arange(keep.numel(), device=labels.device)
            feats = feats[keep]
            labels = labels[keep]
            anchor_idx = new_pos[anchor_idx]
            anchor_idx = anchor_idx[anchor_idx >= 0]
            if anchor_idx.numel() == 0:
                return pred.new_zeros(())

        # Similarity matrix between anchors and the full pool.
        anchor_feats = feats[anchor_idx]                              # (A, F)
        anchor_labels = labels[anchor_idx]                            # (A,)
        logits = anchor_feats @ feats.t() / self.temperature           # (A, N)

        # Positive mask (same class, exclude self).
        pos_mask = (anchor_labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        # Zero-out self-similarity.
        N = labels.numel()
        all_idx = torch.arange(N, device=labels.device).unsqueeze(0).expand(
            anchor_idx.numel(), -1
        )
        pos_mask = pos_mask * (all_idx != anchor_idx.unsqueeze(1)).float()

        # Numerical stability (subtract max).
        logits_max = logits.max(dim=1, keepdim=True).values.detach()
        logits = logits - logits_max

        exp_logits = logits.exp()
        # Denominator excludes the anchor itself.
        denom_mask = (all_idx != anchor_idx.unsqueeze(1)).float()
        denom = (exp_logits * denom_mask).sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = logits - denom.log()

        pos_count = pos_mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count

        # Drop anchors that have zero positives in the pool.
        valid_anchor = (pos_mask.sum(dim=1) > 0)
        if valid_anchor.sum() == 0:
            return pred.new_zeros(())
        loss = -mean_log_prob_pos[valid_anchor].mean()
        return loss
