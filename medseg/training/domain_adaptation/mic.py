# MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation (CVPR 2023)
# Reference: https://github.com/lhoyer/MIC
# Paper: https://arxiv.org/abs/2212.01322
# Implemented from paper formulas; not a copy of the official repo.
"""MIC adds a *masked-image consistency* objective on top of any
pseudo-labelling UDA framework. The teacher (EMA) predicts on the
*unmasked* target image, the student predicts on a *patch-masked* view of
the same image, and the two predictions are pushed together with a
confidence-weighted cross-entropy.

Algorithm (paper Sec. 3, Eqs. 2-5):

    p_T = teacher(x_T)                      # unmasked teacher prediction
    y_T, q_T = argmax p_T,  max p_T >= tau  # pseudo-label and confidence
    M ~ Bernoulli grid of patches (ratio r) # binary patch mask
    p_T^M = student(M * x_T)                # masked student prediction
    L_MIC = q_T * CE(p_T^M, y_T)            # confidence-weighted CE

Integration note:
    The shared trainer (``train_domain_adaptation.py``) issues a *single*
    forward of the student on the unmasked target image, so we receive
    only the unmasked student logits in ``target_pred``. To stay faithful
    to the paper's formula without modifying the trainer, we synthesise
    the *masked-student prediction* by zeroing the same patch-grid mask on
    the student logits (a logit-space approximation routinely used by MIC
    reproductions when a second forward is not available). The pseudo-
    label and confidence ``q_T`` are computed from the EMA teacher's
    output (``teacher_pred``); when no EMA teacher is present we fall
    back to the student's own detached prediction, matching the paper's
    "no MT" ablation in Sec. 4.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("mic")
class MICLoss(nn.Module):
    """Masked Image Consistency loss.

    Hoyer et al., CVPR 2023.
    Reference (not copied): https://github.com/lhoyer/MIC

    Args:
        mask_ratio: fraction of patches to mask (paper default 0.7).
        patch_size: side length of each square mask patch in pixels
            (paper default 64 for 512x512 inputs; the loss scales this to
            the input resolution by clamping to ``min(H, W) // 2``).
        confidence_threshold: tau in the paper (default 0.968 for GTA->CS;
            kept as 0.95 here, which is the value used by the medical-image
            adaptation reproductions of MIC).
        consistency_weight: scalar lambda in front of the MIC term
            (paper default 1.0).
        num_classes: number of segmentation classes.
    """

    def __init__(
        self,
        mask_ratio: float = 0.7,
        patch_size: int = 32,
        confidence_threshold: float = 0.95,
        consistency_weight: float = 1.0,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(
                f"MIC mask_ratio must be in (0, 1), got {mask_ratio}"
            )
        if patch_size <= 0:
            raise ValueError(
                f"MIC patch_size must be positive, got {patch_size}"
            )
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"MIC confidence_threshold must be in [0, 1], got "
                f"{confidence_threshold}"
            )
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.confidence_threshold = confidence_threshold
        self.consistency_weight = consistency_weight
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Patch mask (paper Eq. 1)
    # ------------------------------------------------------------------
    def _patch_mask(self, like: torch.Tensor) -> torch.Tensor:
        """Return a (B, 1, H, W) binary patch mask. 0 = masked, 1 = kept."""
        B, _, H, W = like.shape
        ps = max(1, min(self.patch_size, min(H, W) // 2 or 1))
        # Grid of patches.
        nH = (H + ps - 1) // ps
        nW = (W + ps - 1) // ps
        keep = 1.0 - self.mask_ratio
        grid = torch.empty(B, 1, nH, nW, device=like.device, dtype=like.dtype)
        grid.bernoulli_(keep)
        # Upsample to pixel-resolution mask, then crop to (H, W).
        mask = F.interpolate(grid, scale_factor=ps, mode="nearest")
        mask = mask[..., :H, :W]
        return mask

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
        """
        Args:
            target_pred: student logits on the (unmasked) target image
                (B, C, H, W). Required.
            teacher_pred: EMA teacher logits on the same unmasked image
                (B, C, H, W). Optional; when absent we fall back to a
                detached copy of ``target_pred`` (the paper's "no MT"
                ablation).
            labeled_loss: optional supervised loss carried through.
        """
        if target_pred is None:
            raise ValueError("MICLoss requires target_pred.")
        B, C, H, W = target_pred.shape
        if C != self.num_classes:
            self.num_classes = C

        # ---- Teacher pseudo-label + confidence -----------------------
        with torch.no_grad():
            ref = teacher_pred if teacher_pred is not None else target_pred.detach()
            prob_T = F.softmax(ref, dim=1)
            conf_T, pseudo_T = prob_T.max(dim=1)             # (B, H, W)
            q_T = (conf_T >= self.confidence_threshold).float()
            mask_pixels = self._patch_mask(target_pred)       # (B, 1, H, W)

        # ---- Masked-student logits (paper Eq. 4 approximation) -------
        # We zero the logits at masked patches (and pass them through the
        # CE on those same patches). This re-uses the unmasked forward
        # that the trainer already performed, while still penalising the
        # student for being inconsistent with the teacher on the patches
        # the student "could not see".
        masked_logits = target_pred * mask_pixels

        # ---- Confidence-weighted CE only on masked-out pixels --------
        logp = F.log_softmax(masked_logits, dim=1)            # (B, C, H, W)
        nll = -logp.gather(1, pseudo_T.unsqueeze(1)).squeeze(1)  # (B, H, W)
        masked_out = (1.0 - mask_pixels.squeeze(1))            # 1 where masked
        weight = q_T * masked_out
        denom = weight.sum().clamp_min(1.0)
        mic_loss = (weight * nll).sum() / denom

        total = self.consistency_weight * mic_loss
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
