# ToCo (CVPR 2023)
# Reference: https://github.com/rulixiang/ToCo
# Paper: https://arxiv.org/abs/2303.01267
# Implemented from paper formulas; not a copy of the official repo.
"""ToCo — Token Contrast for Weakly Supervised Semantic Segmentation.

Ru et al., "Token Contrast for Weakly-Supervised Semantic Segmentation",
CVPR 2023.
Paper: https://arxiv.org/abs/2303.01267
Official repository: https://github.com/rulixiang/ToCo

Why it exists.
    Vanilla ViT-based CAM suffers two problems: (1) the global class token
    over-smooths so its CAM activates only on the most discriminative
    patches (PTC problem); (2) the class token of the original image is
    not robust to occlusion (CTC problem). ToCo introduces two contrastive
    objectives:

    (1) Patch Token Contrast (PTC, Sec. 3.2 / Eq. 4-5).
        Reliable foreground / background patch tokens are identified from
        a confidence-thresholded CAM. The remaining "uncertain" tokens are
        pushed towards their most-similar reliable token via an InfoNCE
        contrastive loss in feature space:

            L_PTC = - (1/|U|) sum_{i in U} log
                          exp(sim(p_i, p_pos)/tau)
                        ----------------------------------------------------
                        sum_{j in R} exp(sim(p_i, p_j)/tau)

        where R is the set of reliable tokens and p_pos is the reliable
        token with maximum cosine similarity to p_i within the SAME class
        bucket as p_i's pseudo-label.

    (2) Class Token Contrast (CTC, Sec. 3.3 / Eq. 6).
        ToCo masks the image at the patches NOT covered by the foreground
        CAM (so only foreground patches survive) and pushes the class token
        of the masked image towards the class token of the full image
        with a symmetric InfoNCE loss. A simpler symmetric L2 on the
        normalised vectors is the form actually optimised in their code,
        equivalent to maximising cosine similarity:

            L_CTC = 1 - cos(cls_full, cls_fg_masked)

This module implements just the two loss terms; the ViT, CAM generation,
masking strategy and dataloader live in the user's training script.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("toco")
class ToCoLoss(nn.Module):
    """ToCo PTC + CTC + classification loss.

    Args:
        lambda_ptc: Weight on Patch Token Contrast (paper default 0.2).
        lambda_ctc: Weight on Class Token Contrast (paper default 0.5).
        cls_weight: Weight on the standard multi-label BCE classification
            term on GAP(CAM).
        temperature: InfoNCE temperature for PTC (paper default 0.5).
        high_thresh: CAM normalised score above which a patch token is
            treated as a *reliable foreground* of its argmax class
            (paper default 0.7).
        low_thresh: CAM normalised score below which a patch is treated as
            a *reliable background* token (paper default 0.25).
    """

    def __init__(
        self,
        lambda_ptc: float = 0.2,
        lambda_ctc: float = 0.5,
        cls_weight: float = 1.0,
        temperature: float = 0.5,
        high_thresh: float = 0.7,
        low_thresh: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < low_thresh < high_thresh < 1.0):
            raise ValueError(
                f"Need 0 < low_thresh ({low_thresh}) < high_thresh "
                f"({high_thresh}) < 1."
            )
        self.lambda_ptc = lambda_ptc
        self.lambda_ctc = lambda_ctc
        self.cls_weight = cls_weight
        self.temperature = temperature
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_cam(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-sample per-class min-max normalisation to [0, 1]."""
        B, C, N = cam.shape
        m = cam.amin(dim=2, keepdim=True)
        M = cam.amax(dim=2, keepdim=True)
        return (cam - m) / (M - m + eps)

    # ------------------------------------------------------------------
    # Patch Token Contrast (PTC)
    # ------------------------------------------------------------------
    def _patch_token_contrast(
        self,
        patch_tokens: torch.Tensor,
        cam_tokens: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (4-5) of the paper, per sample.

        Args:
            patch_tokens: (B, N, D) ViT patch tokens.
            cam_tokens: (B, C, N) CAM logits in token-space, fg only.
            image_labels: (B, C) binary multi-label.
        """
        B, N, D = patch_tokens.shape
        C = cam_tokens.shape[1]
        device = patch_tokens.device

        # Mask CAMs by image-level tags so only present classes contribute.
        present = image_labels.float().unsqueeze(-1)            # (B, C, 1)
        cam = cam_tokens * present
        cam_norm = self._normalise_cam(cam)                     # (B, C, N)

        # Pseudo-label per token: argmax class score, plus a confidence.
        conf, cls_id = cam_norm.max(dim=1)                      # (B, N) each
        # Reliable foreground = high confidence AND its class is present.
        is_fg = (conf > self.high_thresh)
        # Reliable background = below low threshold across ALL present classes.
        is_bg = (cam_norm.max(dim=1).values < self.low_thresh)

        feat = F.normalize(patch_tokens, dim=-1)                # (B, N, D)

        losses = []
        for b in range(B):
            f = feat[b]                                          # (N, D)
            fg_mask = is_fg[b]
            bg_mask = is_bg[b]
            unc_mask = ~(fg_mask | bg_mask)

            if (not fg_mask.any()) or (not bg_mask.any()) or (not unc_mask.any()):
                continue

            unc_idx = unc_mask.nonzero(as_tuple=False).squeeze(1)
            fg_idx = fg_mask.nonzero(as_tuple=False).squeeze(1)
            bg_idx = bg_mask.nonzero(as_tuple=False).squeeze(1)

            unc_feat = f[unc_idx]                                # (U, D)
            fg_feat = f[fg_idx]                                  # (Pf, D)
            bg_feat = f[bg_idx]                                  # (Pb, D)
            unc_cls = cls_id[b, unc_idx]                         # (U,)
            fg_cls = cls_id[b, fg_idx]                           # (Pf,)

            # Similarity uncertain → all reliable (concat fg + bg).
            rel_feat = torch.cat([fg_feat, bg_feat], dim=0)      # (R, D)
            rel_is_fg = torch.cat([
                torch.ones(fg_feat.size(0), dtype=torch.bool, device=device),
                torch.zeros(bg_feat.size(0), dtype=torch.bool, device=device),
            ])
            rel_cls = torch.cat([
                fg_cls,
                torch.full((bg_feat.size(0),), -1, dtype=cls_id.dtype, device=device),
            ])

            sim = unc_feat @ rel_feat.t() / self.temperature     # (U, R)

            # Positive = reliable token of the same pseudo-class (or background
            # if the uncertain token's max activation is also low).
            # Take the most similar reliable token of the matching label as
            # the single positive (InfoNCE-1 form).
            U = unc_feat.shape[0]
            # Per row, mask of allowed positives.
            cls_eq = (rel_cls.unsqueeze(0) == unc_cls.unsqueeze(1))
            # If no class match exists, fall back to background tokens.
            no_match = ~cls_eq.any(dim=1)
            if no_match.any():
                cls_eq[no_match] = (~rel_is_fg).unsqueeze(0).expand_as(cls_eq)[no_match]

            # log-sum-exp denominator over all reliable tokens.
            log_denom = torch.logsumexp(sim, dim=1)              # (U,)
            # Numerator: max similarity inside the allowed positive set.
            neg_inf = torch.finfo(sim.dtype).min
            sim_pos = sim.masked_fill(~cls_eq, neg_inf)
            log_num = sim_pos.max(dim=1).values                  # (U,)

            losses.append(-(log_num - log_denom).mean())

        if not losses:
            return patch_tokens.new_zeros(())
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Class Token Contrast (CTC)
    # ------------------------------------------------------------------
    @staticmethod
    def _class_token_contrast(
        cls_full: torch.Tensor,
        cls_masked: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric cosine-distance loss between full and fg-masked class
        tokens (Eq. 6 reduction)."""
        a = F.normalize(cls_full, dim=-1)
        b = F.normalize(cls_masked, dim=-1)
        return (1.0 - (a * b).sum(dim=-1)).mean()

    # ------------------------------------------------------------------
    def forward(
        self,
        patch_tokens: torch.Tensor,
        cam_tokens: torch.Tensor,
        cls_token_full: torch.Tensor,
        cls_token_masked: torch.Tensor,
        image_labels: torch.Tensor,
        cls_logits: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, D) ViT patch tokens.
            cam_tokens: (B, C_fg, N) CAM scores in token space (no bg). If
                given as (B, C_fg, H, W) it is reshaped to (B, C_fg, H*W).
            cls_token_full: (B, D) class token of the original image.
            cls_token_masked: (B, D) class token from a forward pass on the
                foreground-only masked image (background patches zeroed out
                per the CAM mask).
            image_labels: (B, C_fg) binary multi-label tags.
            cls_logits: Optional (B, C_fg) classification logits. If not
                supplied, GAP(cam_tokens) is used.
            labeled_loss: Optional pre-computed dense supervised term.
        """
        if cam_tokens.dim() == 4:
            B, C, H, W = cam_tokens.shape
            cam_tokens = cam_tokens.view(B, C, H * W)
        if cam_tokens.dim() != 3:
            raise ValueError(
                f"cam_tokens must be (B,C,N) or (B,C,H,W); got "
                f"{tuple(cam_tokens.shape)}"
            )
        if patch_tokens.shape[1] != cam_tokens.shape[2]:
            raise ValueError(
                f"patch_tokens N={patch_tokens.shape[1]} must match cam_tokens "
                f"N={cam_tokens.shape[2]}"
            )

        # (1) Multi-label classification on GAP(CAM) or supplied logits.
        if cls_logits is None:
            cls_logits = cam_tokens.mean(dim=2)
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits, image_labels.float()
        )

        # (2) PTC.
        ptc = self._patch_token_contrast(patch_tokens, cam_tokens, image_labels)
        # (3) CTC.
        ctc = self._class_token_contrast(cls_token_full, cls_token_masked)

        total = (
            self.cls_weight * cls_loss
            + self.lambda_ptc * ptc
            + self.lambda_ctc * ctc
        )
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
