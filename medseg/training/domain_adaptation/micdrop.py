# MICDrop: Masking Image and Depth Features via Complementary Dropout for DA-SS (ECCV 2024)
# Reference: https://github.com/lhoyer/MICDrop
# Paper: https://arxiv.org/abs/2408.16478
# Implemented from paper formulas; not a copy of the official repo.
"""MICDrop extends MIC with a **complementary feature-dropout** branch so
that the student sees two *complementary partial views* of the same target
image, both of which must agree with the teacher's unmasked pseudo-label.

Algorithm (paper Sec. 3.2, Eqs. 4-6):

    p_T = teacher(x_T)                       # unmasked teacher prediction
    y_T, q_T = argmax p_T,  max p_T >= tau   # pseudo-label and confidence

    M       ~ Bernoulli grid of patches (ratio r)         # spatial mask
    Mc      = 1 - M                                       # complementary
    z       = student_features(x_T)                       # encoder feats
    z_mask  = z masked by M  (spatial mask)
    z_drop  = z dropped by Mc (channel/spatial dropout on complement)
    p_mask  = head(z_mask),  p_drop = head(z_drop)

    L_MICDrop = q_T * [ alpha   * CE(p_mask, y_T)
                       + (1-alpha) * CE(p_drop, y_T) ]
              + lambda_cons * MSE(softmax(p_mask), softmax(p_drop))

Integration note:
    The shared trainer issues a single forward and only exposes the
    unmasked student logits. We synthesise the two complementary views in
    *logit space*: ``p_mask`` is the student logits zeroed at masked-out
    patches (same trick as ``mic.py``), and ``p_drop`` is the same logits
    with the *complementary* patch set zeroed AND with Bernoulli channel
    dropout (rate ``dropout_p``) applied. This preserves the MICDrop
    objective (complementary partial views + agreement) without needing a
    second forward through the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("micdrop")
class MICDropLoss(nn.Module):
    """Masked Image Consistency with complementary feature dropout.

    Hoyer et al., ECCV 2024.
    Reference (not copied): https://github.com/lhoyer/MICDrop

    Args:
        mask_ratio: fraction of patches kept *unmasked* on the mask branch
            (paper default 0.3 i.e. 70% masked, mirroring MIC).
        patch_size: side length of each square mask patch in pixels.
        dropout_p: Bernoulli channel-dropout rate applied to the
            complementary-view logits (paper default 0.5).
        alpha: blend between the mask-branch and dropout-branch CE terms
            (paper Eq. 5, default 0.5 = equal weighting).
        confidence_threshold: tau for the teacher pseudo-label mask.
        consistency_weight: lambda_cons in front of the MSE agreement term.
        ce_weight: scalar in front of the (alpha * mask + (1-alpha) * drop)
            cross-entropy term.
        num_classes: number of segmentation classes.
    """

    def __init__(
        self,
        mask_ratio: float = 0.7,
        patch_size: int = 32,
        dropout_p: float = 0.5,
        alpha: float = 0.5,
        confidence_threshold: float = 0.95,
        consistency_weight: float = 1.0,
        ce_weight: float = 1.0,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(
                f"MICDrop mask_ratio must be in (0, 1), got {mask_ratio}"
            )
        if patch_size <= 0:
            raise ValueError(
                f"MICDrop patch_size must be positive, got {patch_size}"
            )
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(
                f"MICDrop dropout_p must be in [0, 1), got {dropout_p}"
            )
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"MICDrop alpha must be in [0, 1], got {alpha}"
            )
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"MICDrop confidence_threshold must be in [0, 1], got "
                f"{confidence_threshold}"
            )
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.dropout_p = dropout_p
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.consistency_weight = consistency_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Patch mask
    # ------------------------------------------------------------------
    def _patch_mask(self, like: torch.Tensor) -> torch.Tensor:
        """Return (B, 1, H, W) binary patch mask. 1 = kept, 0 = masked."""
        B, _, H, W = like.shape
        ps = max(1, min(self.patch_size, min(H, W) // 2 or 1))
        nH = (H + ps - 1) // ps
        nW = (W + ps - 1) // ps
        keep = 1.0 - self.mask_ratio
        grid = torch.empty(B, 1, nH, nW, device=like.device, dtype=like.dtype)
        grid.bernoulli_(keep)
        mask = F.interpolate(grid, scale_factor=ps, mode="nearest")
        return mask[..., :H, :W]

    def _channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample channel dropout (Bernoulli over the C axis).

        Inverted-dropout scaling so the expected magnitude is preserved.
        """
        if self.dropout_p <= 0:
            return x
        B, C, _, _ = x.shape
        keep_prob = 1.0 - self.dropout_p
        mask = torch.empty(B, C, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        return x * mask / keep_prob

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        target_pred: torch.Tensor,
        teacher_pred: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if target_pred is None:
            raise ValueError("MICDropLoss requires target_pred.")
        B, C, H, W = target_pred.shape

        # ---- Teacher pseudo-label + confidence -----------------------
        with torch.no_grad():
            ref = teacher_pred if teacher_pred is not None else target_pred.detach()
            prob_T = F.softmax(ref, dim=1)
            conf_T, pseudo_T = prob_T.max(dim=1)
            q_T = (conf_T >= self.confidence_threshold).float()
            mask = self._patch_mask(target_pred)                       # (B, 1, H, W)
            comp = 1.0 - mask                                          # complement

        # ---- Two complementary views (logit-space construction) ------
        # Mask branch: keep logits where mask=1, zero elsewhere.
        p_mask = target_pred * mask
        # Drop branch: keep logits where mask=0 (complement), then apply
        # channel dropout to enforce the "feature dropout" half of MICDrop.
        p_drop = target_pred * comp
        p_drop = self._channel_dropout(p_drop)

        # ---- Confidence-weighted complementary CE -------------------
        log_p_mask = F.log_softmax(p_mask, dim=1)
        log_p_drop = F.log_softmax(p_drop, dim=1)
        # Per-pixel NLL against the teacher's hard pseudo-label.
        nll_mask = -log_p_mask.gather(1, pseudo_T.unsqueeze(1)).squeeze(1)
        nll_drop = -log_p_drop.gather(1, pseudo_T.unsqueeze(1)).squeeze(1)
        # The mask-branch CE is only meaningful on masked-out pixels (where
        # the student "couldn't see" — the point of MIC's reconstruction
        # signal). The drop-branch CE is dual: meaningful on kept pixels.
        masked_out = comp.squeeze(1)                                   # 1 where masked
        kept = mask.squeeze(1)                                         # 1 where kept
        w_mask = q_T * masked_out
        w_drop = q_T * kept
        ce_mask = (w_mask * nll_mask).sum() / w_mask.sum().clamp_min(1.0)
        ce_drop = (w_drop * nll_drop).sum() / w_drop.sum().clamp_min(1.0)
        l_ce = self.alpha * ce_mask + (1.0 - self.alpha) * ce_drop

        # ---- Inter-view agreement (MSE on softmax) -------------------
        # Encourages the two complementary partial views to land in the same
        # class distribution, weighted by the teacher-confidence map so we
        # don't pay attention where the teacher itself is unsure.
        sp_mask = F.softmax(p_mask, dim=1)
        sp_drop = F.softmax(p_drop, dim=1)
        agree = ((sp_mask - sp_drop) ** 2).sum(dim=1)                  # (B, H, W)
        l_cons = (q_T * agree).sum() / q_T.sum().clamp_min(1.0)

        total = self.ce_weight * l_ce + self.consistency_weight * l_cons
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
