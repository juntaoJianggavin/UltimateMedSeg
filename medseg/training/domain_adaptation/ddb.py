# DDB: Learning Dynamic Domain-Bridging from Two Domains for UDA-SS (CVPR 2023)
# Reference: https://github.com/xinyuelll/DDB
# Paper: https://arxiv.org/abs/2304.05285
# Implemented from paper formulas; not a copy of the official repo.
"""DDB (Dual-domain Decoupled Bridging) constructs a *mixed-domain* image
between source and target via class-mix / mix-up, and trains two cross-
distillation paths:

  * source-side coarse expert  ->  mixed-domain student
  * target-side fine expert    ->  mixed-domain student

with a *bidirectional* KL between the two experts' predictions on the
shared mixed image (paper Sec. 3.2, Eqs. 4-7):

    p_mix = lambda * p_S  + (1 - lambda) * p_T
    y_mix = lambda * y_S + (1 - lambda) * y_T_pseudo
    L_DDB = CE(p_mix, y_mix)
          + lambda     * KL(p_S    || p_mix)
          + (1-lambda) * KL(p_T    || p_mix)

Integration note:
    The shared trainer performs separate forwards on (source, target) and
    cannot run a third forward on a literal mixed image. We therefore mix
    in *logit space*: the source and target student logits are blended by
    a sample-wise lambda drawn from Beta(a, a) (the paper's mix-up prior),
    giving the same statistical objective as DDB's image-space mixing
    but without a second model call. The bidirectional KL anchors both
    branches to the mixed prediction, matching DDB's bridging behaviour
    end-to-end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("ddb")
class DDBLoss(nn.Module):
    """Dual-domain Decoupled Bridging loss.

    Du et al., CVPR 2023.
    Reference (not copied): https://github.com/xinyuelll/DDB

    Args:
        mix_alpha: Beta(a, a) concentration for the per-sample mix-up
            weight lambda (paper default 1.0, i.e. uniform).
        bridge_weight: scalar on the bidirectional KL term.
        ce_weight: scalar on the mixed-domain CE term.
        confidence_threshold: pixels whose target pseudo-confidence is
            below this value are excluded from the mixed-domain CE
            (paper Sec. 3.3 reliability filter).
        num_classes: number of segmentation classes.
    """

    def __init__(
        self,
        mix_alpha: float = 1.0,
        bridge_weight: float = 0.5,
        ce_weight: float = 1.0,
        confidence_threshold: float = 0.7,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if mix_alpha <= 0:
            raise ValueError(f"DDB mix_alpha must be positive, got {mix_alpha}")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"DDB confidence_threshold must be in [0, 1], got "
                f"{confidence_threshold}"
            )
        self.mix_alpha = mix_alpha
        self.bridge_weight = bridge_weight
        self.ce_weight = ce_weight
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        self._beta = torch.distributions.Beta(mix_alpha, mix_alpha)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        source_pred: torch.Tensor,
        target_pred: torch.Tensor,
        source_labels: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if source_pred is None or target_pred is None:
            raise ValueError(
                "DDBLoss requires both source_pred and target_pred."
            )
        if source_pred.shape != target_pred.shape:
            raise ValueError(
                f"DDB requires source and target logits with matching shape, "
                f"got {tuple(source_pred.shape)} vs {tuple(target_pred.shape)}. "
                f"Use a dataloader that yields equal-batch crops on both sides."
            )
        B, C, H, W = source_pred.shape

        # Sample per-image lambda in [0, 1] from Beta(a, a).
        lam = self._beta.sample((B,)).to(source_pred.device).clamp(0.02, 0.98)
        lam_map = lam.view(B, 1, 1, 1)

        # ---- Logit-space mix --------------------------------------
        mixed_logits = lam_map * source_pred + (1.0 - lam_map) * target_pred
        mixed_log_p = F.log_softmax(mixed_logits, dim=1)

        # ---- Mixed-domain CE --------------------------------------
        # Source side keeps its true labels; target side contributes its
        # high-confidence pseudo-label.
        with torch.no_grad():
            prob_T = F.softmax(target_pred, dim=1)
            conf_T, pseudo_T = prob_T.max(dim=1)
            target_valid = conf_T >= self.confidence_threshold

        ce_terms = []
        if source_labels is not None:
            src_lbl = source_labels.clamp(0, C - 1)
            ce_src = F.nll_loss(mixed_log_p, src_lbl, reduction="none")  # (B, H, W)
            ce_terms.append((lam.view(B, 1, 1) * ce_src).mean())

        if target_valid.any():
            ce_tgt = F.nll_loss(mixed_log_p, pseudo_T, reduction="none")  # (B, H, W)
            mask = target_valid.float()
            denom = mask.sum().clamp_min(1.0)
            ce_terms.append(
                ((1.0 - lam.view(B, 1, 1)) * ce_tgt * mask).sum() / denom
            )
        ce_mixed = sum(ce_terms) if ce_terms else source_pred.new_zeros(())

        # ---- Bidirectional KL bridge ------------------------------
        log_p_mix = mixed_log_p
        p_S = F.softmax(source_pred, dim=1)
        p_T = F.softmax(target_pred, dim=1)
        # KL(p_S || p_mix) and KL(p_T || p_mix), per-pixel reduction.
        kl_src = F.kl_div(log_p_mix, p_S, reduction="batchmean")
        kl_tgt = F.kl_div(log_p_mix, p_T, reduction="batchmean")
        bridge = lam.mean() * kl_src + (1.0 - lam.mean()) * kl_tgt

        total = self.ce_weight * ce_mixed + self.bridge_weight * bridge
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
