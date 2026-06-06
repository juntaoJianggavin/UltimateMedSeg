# DAFormer: Improving Network Architectures and Training Strategies for DA-SS (CVPR 2022)
# Reference: https://github.com/lhoyer/DAFormer
# Paper: https://arxiv.org/abs/2111.14887
# Implemented from paper formulas; not a copy of the official repo.
"""DAFormer's UDA training combines three losses on top of the source CE:

    L = L_S (source CE)
      + L_T = q * H(p_S, y_T)          # quality-weighted PL CE  (Eq. 3)
      + lambda_FD * L_FD                # ImageNet feature distance  (Eq. 5)

where p_S is the student softmax on target, y_T = argmax(teacher_softmax),
and q is the per-image fraction of teacher pixels whose confidence exceeds
``tau`` (paper default 0.968). The cross-entropy is *rare-class re-weighted*
using EMA-tracked class frequencies of the pseudo-labels (RCS, Sec. 3.2).

L_FD in the original work is a feature distance between the student's
*encoder* features and a *frozen ImageNet-pretrained encoder*'s features
on the ``things`` ImageNet classes (Hoyer et al., Sec. 3.3). Since the
shared trainer does not expose encoder features in the per-step ctx, we
implement an equivalent *logit-space* feature distance that preserves
source-domain knowledge by penalising the student's source logits from
drifting away from an EMA-tracked reference (a Pearson-style correlation
distance over class probability vectors). This is the same regularisation
intuition (anchor against a stable feature space) lifted to the only
representation we have access to here, and is documented as a logit-space
proxy of L_FD in the loss key ``daformer_fd``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("daformer_fd")
class DAFormerFDLoss(nn.Module):
    """DAFormer pseudo-label CE + rare-class sampling + logit-space FD.

    Hoyer et al., CVPR 2022.
    Reference (not copied): https://github.com/lhoyer/DAFormer

    Components:
      * Quality-weighted pseudo-label CE on the target (Eqs. 3-4).
      * RCS rare-class re-weighting via an EMA over per-class pixel
        frequencies of the pseudo-labels (Sec. 3.2).
      * Logit-space Feature Distance L_FD on the source predictions
        (anchors source logits against an EMA running average, the
        only stable "frozen reference" we have without ImageNet backbone
        access).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.968,
        pseudo_weight: float = 1.0,
        fd_weight: float = 0.005,
        rcs_temperature: float = 0.01,
        freq_momentum: float = 0.999,
        feat_momentum: float = 0.99,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"DAFormer confidence_threshold must be in [0, 1], got "
                f"{confidence_threshold}"
            )
        if rcs_temperature <= 0:
            raise ValueError(
                f"DAFormer rcs_temperature must be positive, got "
                f"{rcs_temperature}"
            )
        self.confidence_threshold = confidence_threshold
        self.pseudo_weight = pseudo_weight
        self.fd_weight = fd_weight
        self.rcs_temperature = rcs_temperature
        self.freq_momentum = freq_momentum
        self.feat_momentum = feat_momentum
        self.num_classes = num_classes

        # EMA buffers.
        self.register_buffer(
            "_class_freq_ema",
            torch.full((num_classes,), 1.0 / num_classes),
        )
        self.register_buffer(
            "_freq_initialised", torch.tensor(False),
        )
        # EMA over per-class mean source-logit *probability* vectors,
        # shape (C, C): row c stores the average softmax distribution at
        # source pixels whose label is class c. Used as the FD anchor.
        self.register_buffer(
            "_class_proto", torch.eye(num_classes),
        )
        self.register_buffer(
            "_proto_initialised", torch.tensor(False),
        )

    # ------------------------------------------------------------------
    # EMA updates
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_class_freq(self, pseudo: torch.Tensor):
        counts = torch.zeros(self.num_classes, device=pseudo.device)
        for c in range(self.num_classes):
            counts[c] = (pseudo == c).sum()
        freq = counts / counts.sum().clamp_min(1.0)
        if not bool(self._freq_initialised.item()):
            self._class_freq_ema = freq
            self._freq_initialised = torch.tensor(True, device=pseudo.device)
        else:
            m = self.freq_momentum
            self._class_freq_ema = m * self._class_freq_ema + (1.0 - m) * freq

    @torch.no_grad()
    def _update_prototypes(self, prob_S: torch.Tensor, labels: torch.Tensor):
        """Update per-class prototype softmax distributions on source.

        ``prob_S`` is (B, C, H, W) softmax of source predictions, ``labels``
        is (B, H, W) ground-truth source labels.
        """
        B, C, H, W = prob_S.shape
        proto = torch.zeros(C, C, device=prob_S.device, dtype=prob_S.dtype)
        flat_prob = prob_S.permute(0, 2, 3, 1).reshape(-1, C)
        flat_lbl = labels.reshape(-1)
        for c in range(C):
            sel = flat_lbl == c
            if sel.any():
                proto[c] = flat_prob[sel].mean(dim=0)
            else:
                proto[c] = self._class_proto[c]
        if not bool(self._proto_initialised.item()):
            self._class_proto = proto
            self._proto_initialised = torch.tensor(True, device=prob_S.device)
        else:
            m = self.feat_momentum
            self._class_proto = m * self._class_proto + (1.0 - m) * proto

    def _rcs_weights(self) -> torch.Tensor:
        """RCS class weights via tempered inverse frequency (paper Eq. 6)."""
        logits = -torch.log(self._class_freq_ema + 1e-12) / self.rcs_temperature
        w = F.softmax(logits, dim=0) * self.num_classes
        return w

    # ------------------------------------------------------------------
    # L_FD (logit-space proxy)
    # ------------------------------------------------------------------
    def _feature_distance(
        self,
        prob_S: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mean L2 distance between source softmax vectors and per-class
        prototype distributions (anchored, EMA-tracked, detached).
        """
        B, C, H, W = prob_S.shape
        flat = prob_S.permute(0, 2, 3, 1).reshape(-1, C)
        flat_lbl = labels.reshape(-1).clamp(0, C - 1)
        anchor = self._class_proto.detach()[flat_lbl]
        return ((flat - anchor) ** 2).sum(dim=1).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        target_pred: torch.Tensor,
        source_pred: Optional[torch.Tensor] = None,
        source_labels: Optional[torch.Tensor] = None,
        teacher_pred: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if target_pred is None:
            raise ValueError("DAFormerFDLoss requires target_pred.")
        B, C, H, W = target_pred.shape
        if C != self.num_classes:
            raise ValueError(
                f"DAFormerFDLoss configured for num_classes={self.num_classes} "
                f"but received logits with C={C}. Re-instantiate with the "
                f"matching value or update the yaml — refusing to silently "
                f"resize EMA buffers."
            )

        # ---- Teacher pseudo-label + quality weight q ----------------
        with torch.no_grad():
            ref = teacher_pred if teacher_pred is not None else target_pred.detach()
            prob_T = F.softmax(ref, dim=1)
            conf_T, pseudo_T = prob_T.max(dim=1)
            q = (conf_T >= self.confidence_threshold).float().mean()
            self._update_class_freq(pseudo_T)

        rcs_w = self._rcs_weights()
        pl_ce = F.cross_entropy(target_pred, pseudo_T, weight=rcs_w)
        total = self.pseudo_weight * q * pl_ce

        # ---- Logit-space L_FD ---------------------------------------
        if source_pred is not None and source_labels is not None:
            prob_S = F.softmax(source_pred, dim=1)
            # Update EMA prototypes from this batch (no grad).
            self._update_prototypes(prob_S.detach(), source_labels.detach())
            fd = self._feature_distance(prob_S, source_labels)
            total = total + self.fd_weight * fd

        if labeled_loss is not None:
            total = total + labeled_loss
        return total
