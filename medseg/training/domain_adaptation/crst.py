"""CRST: Confidence Regularized Self-Training (ICCV 2019).

# Paper: https://arxiv.org/abs/1908.09822
# Reference: https://github.com/yzou2/CRST

Algorithm summary (from the paper):
    CRST is a source-free, self-training approach. At each round:

      1.  *Class-balanced pseudo-label selection (CBST)*: for every class c,
          select the pixels whose softmax confidence for c is above a
          per-class threshold k_c, where k_c is chosen so that the top
          ``portion`` fraction of class-c pixels survive. This balances
          rare and common classes.
      2.  *Confidence regularisation*: a regulariser is added to the
          self-training cross-entropy so that the network's predictions do
          not collapse to one-hot vectors. The paper studies several forms;
          the default ``MRKLD`` regulariser is the KL divergence from the
          uniform distribution to the network's softmax,
              R_KLD(x) = KL( u || softmax(f(x)) )
                       = log(C) + (1/C) * sum_c log softmax(f(x))_c   (× -1)
          We use this default. ``LRENT`` (negative entropy) is also
          supported via ``regularizer="LRENT"``.

    The combined loss is

        L = CE_pseudo (on selected pixels)  -  alpha * R(x)

    where alpha = ``reg_weight``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("crst")
class CRSTLoss(nn.Module):
    """Confidence-Regularised Self-Training for source-free DA.

    Zou et al., ICCV 2019.
    Reference (not copied): https://github.com/yzou2/CRST
    """

    SUPPORTED_REGULARIZERS = ("MRKLD", "LRENT")

    def __init__(
        self,
        portion: float = 0.5,
        reg_weight: float = 0.1,
        regularizer: str = "MRKLD",
        num_classes: int = 5,
        confidence_floor: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < portion <= 1.0):
            raise ValueError(f"CRST 'portion' must be in (0, 1], got {portion}")
        if regularizer not in self.SUPPORTED_REGULARIZERS:
            raise ValueError(
                f"CRST regularizer must be one of "
                f"{self.SUPPORTED_REGULARIZERS}, got {regularizer!r}"
            )
        # Top fraction of per-class confidences to keep as pseudo-labels.
        self.portion = portion
        # Weight on the confidence regularisation term.
        self.reg_weight = reg_weight
        self.regularizer = regularizer
        self.num_classes = num_classes
        # Optional global confidence floor: pixels below this are always
        # dropped even if they survive the per-class CBST cutoff.
        self.confidence_floor = confidence_floor

    # ------------------------------------------------------------------
    # Class-balanced pseudo-label selection (paper Sec. 3.2)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _cbst_select(
        self,
        prob: torch.Tensor,
    ):
        """Return (pseudo_label, valid_mask) using per-class top-``portion``."""
        B, C, H, W = prob.shape
        confidence, pseudo = prob.max(dim=1)         # (B, H, W) each
        valid = confidence.new_zeros(confidence.shape, dtype=torch.bool)

        for c in range(C):
            sel_c = pseudo == c
            n_c = int(sel_c.sum().item())
            if n_c == 0:
                continue
            conf_c = confidence[sel_c]
            k = max(1, int(self.portion * n_c))
            # Per-class threshold = top-k confidence within class c.
            thr = torch.topk(conf_c, k, largest=True, sorted=False).values.min()
            cls_valid = sel_c & (confidence >= thr)
            valid = valid | cls_valid

        if self.confidence_floor > 0.0:
            valid = valid & (confidence >= self.confidence_floor)
        return pseudo, valid

    # ------------------------------------------------------------------
    # Confidence regularisers (paper Eqs. 11 / 12)
    # ------------------------------------------------------------------
    @staticmethod
    def _mrkld(prob: torch.Tensor) -> torch.Tensor:
        """KL( uniform || prob ) per pixel, averaged over pixels.

        KL(u||p) = log(C) + (1/C) * sum_c log(1/p_c)
                 = log(C) - (1/C) * sum_c log p_c
        Returning the *positive* KL; the sign of its contribution to the
        total loss is controlled by ``reg_weight`` in ``forward``.
        """
        C = prob.shape[1]
        log_p = torch.log(prob.clamp_min(1e-30))
        kl = -log_p.mean(dim=1) - torch.log(
            torch.tensor(C, dtype=prob.dtype, device=prob.device)
        )
        # The +log(C) constant makes the "ideal" (uniform p) value zero.
        # Mathematically: KL(u||p) = -mean_c log p_c - log(C)
        # We negate so a *larger* return means "p is sharper", matching
        # the convention that we want to MINIMISE sharpness => SUBTRACT.
        return -kl.mean()

    @staticmethod
    def _lrent(prob: torch.Tensor) -> torch.Tensor:
        """Negative entropy: -H(p) = sum_c p_c log p_c (per pixel, averaged).

        Minimising -H(p) over the network is equivalent to maximising H(p),
        which is the LRENT regulariser in the paper (drives p away from
        one-hot toward uniform).
        """
        log_p = torch.log(prob.clamp_min(1e-30))
        neg_ent = (prob * log_p).sum(dim=1)   # = -H(p), already negative
        return neg_ent.mean()

    def _regularizer_term(self, prob: torch.Tensor) -> torch.Tensor:
        if self.regularizer == "MRKLD":
            return self._mrkld(prob)
        return self._lrent(prob)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        target_pred: torch.Tensor,
        labeled_loss: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """
        Args:
            target_pred: target logits (B, C, H, W). Required.
            labeled_loss: optional supervised loss carried through unchanged
                (CRST is normally source-free, so the trainer will not pass
                this when ``source_free: true``).
        """
        if target_pred is None:
            raise ValueError("CRSTLoss requires target_pred.")
        B, C, H, W = target_pred.shape
        if C != self.num_classes:
            # The official repo derives num_classes from the model, so we
            # do the same here rather than mismatching silently.
            self.num_classes = C

        prob = F.softmax(target_pred, dim=1)
        pseudo, valid_mask = self._cbst_select(prob)

        # Self-training cross-entropy on the surviving pixels only.
        if valid_mask.any():
            logits_flat = target_pred.permute(0, 2, 3, 1).reshape(-1, C)
            pseudo_flat = pseudo.reshape(-1)
            valid_flat = valid_mask.reshape(-1)
            ce_loss = F.cross_entropy(
                logits_flat[valid_flat],
                pseudo_flat[valid_flat],
            )
        else:
            ce_loss = target_pred.new_zeros(())

        reg = self._regularizer_term(prob)
        # Paper sign convention: subtract the regulariser (we want to keep
        # the prediction *spread out* and prevent collapse).
        total = ce_loss - self.reg_weight * reg
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
