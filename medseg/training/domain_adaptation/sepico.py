# SePiCo: Semantic-Guided Pixel Contrast for Domain Adaptive Semantic Segmentation (TPAMI 2023)
# Reference: https://github.com/BIT-DA/SePiCo
# Paper: https://arxiv.org/abs/2204.08808
# Implemented from paper formulas; not a copy of the official repo.
"""SePiCo augments any UDA pseudo-labelling pipeline with a *semantic-guided
pixel contrast* objective using **persistent class prototypes** maintained
across the whole training run (not just a single batch).

Two complementary variants are described in the paper (Sec. 3 & 4):

  * **DistCL (distributional contrast, Eq. 7)** — each class is represented by
    a Gaussian whose mean ``mu_c`` and (diagonal) covariance ``sigma_c`` are
    EMA-tracked from confident pixels of that class. The per-pixel loss is
    the Mahalanobis-style KL of the pixel feature to its own class's Gaussian
    against the contrast with every other class.

  * **BankCL / ProtoCL (Eq. 4)** — InfoNCE between pixel features and the
    *running prototype bank* ``{mu_c}_{c=1..C}`` with the same-class
    prototype as the positive.

We provide the lighter **ProtoCL** form by default (it is what the official
"SePiCo-PC" config uses for medical reproductions) and an optional
distributional term gated by ``dist_weight``. Features used are the
L2-normalised softmax vectors of the student (the "projection-free"
variant; identical to what PiPa uses in this codebase), so we never need
the model to expose a separate feature head.

Integration:
    A single forward through the trainer yields per-step ``target_pred``
    and a detached EMA teacher prediction (``teacher_pred``). The teacher's
    high-confidence pseudo-labels drive prototype updates; the student's
    features pull toward their class prototype and push from the others.

Algorithm summary (per minibatch):
    1. p_T  = softmax(teacher_pred)
       y_T  = argmax p_T,  q_T = (max p_T >= tau).
    2. Update EMA prototypes:
           mu_c <- m * mu_c + (1 - m) * mean_{i: y_i = c, q_i = 1} z_i
           sigma_c analogously over (z - mu_c)^2.
    3. ProtoCL (InfoNCE):
           L_proto = -mean log [ exp(<z, mu_{y}>/tau)
                                / sum_c exp(<z, mu_c>/tau) ]
    4. DistCL (optional, Eq. 7):
           L_dist  = mean_i  [  KL( N(z_i; mu_{y_i}, sigma_{y_i})
                              || avg_{c != y_i} N(z_i; mu_c, sigma_c) ) ]
       (we use the simpler -log p / sum_c p form which is mathematically
       equivalent up to a constant for diagonal Gaussians.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("sepico")
class SePiCoLoss(nn.Module):
    """Semantic-Guided Pixel Contrast for UDA segmentation.

    Xie et al., TPAMI 2023.
    Reference (not copied): https://github.com/BIT-DA/SePiCo

    Args:
        proto_weight: scalar on the InfoNCE prototype-contrast term L_proto.
        dist_weight: scalar on the distributional-contrast term L_dist
            (paper Eq. 7); set to 0 to disable.
        temperature: tau in the InfoNCE denominator (paper default 0.1).
        confidence_threshold: pixels whose teacher confidence is below this
            value do not contribute to prototype updates or to the loss.
        proto_momentum: EMA momentum for the running prototypes (paper 0.999).
        pixels_per_image: per-image cap on the number of anchor pixels (for
            tractable contrast — paper recommends ~256-1024).
        num_classes: number of segmentation classes.
    """

    def __init__(
        self,
        proto_weight: float = 0.1,
        dist_weight: float = 0.0,
        temperature: float = 0.1,
        confidence_threshold: float = 0.7,
        proto_momentum: float = 0.999,
        pixels_per_image: int = 512,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"SePiCo temperature must be positive, got {temperature}")
        if not (0.0 < proto_momentum < 1.0):
            raise ValueError(
                f"SePiCo proto_momentum must be in (0, 1), got {proto_momentum}"
            )
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"SePiCo confidence_threshold must be in [0, 1], got "
                f"{confidence_threshold}"
            )
        if pixels_per_image <= 0:
            raise ValueError(
                f"SePiCo pixels_per_image must be positive, got {pixels_per_image}"
            )
        self.proto_weight = proto_weight
        self.dist_weight = dist_weight
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.proto_momentum = proto_momentum
        self.pixels_per_image = pixels_per_image
        self.num_classes = num_classes

        # Feature dim equals num_classes (we use softmax vectors as features).
        # Prototype mean / variance EMA buffers.
        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, num_classes),
        )
        self.register_buffer(
            "proto_variance",
            torch.ones(num_classes, num_classes),
        )
        self.register_buffer(
            "proto_initialised",
            torch.zeros(num_classes, dtype=torch.bool),
        )

    # ------------------------------------------------------------------
    # Prototype maintenance (Eq. 5 of the paper)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_prototypes(
        self,
        feats: torch.Tensor,         # (N, D) normalised features
        labels: torch.Tensor,        # (N,) class ids
    ):
        m = self.proto_momentum
        for c in range(self.num_classes):
            sel = labels == c
            if not sel.any():
                continue
            mean_c = feats[sel].mean(dim=0)
            var_c = feats[sel].var(dim=0, unbiased=False) + 1e-6
            if not bool(self.proto_initialised[c].item()):
                self.prototypes[c] = mean_c
                self.proto_variance[c] = var_c
                self.proto_initialised[c] = True
            else:
                self.prototypes[c] = m * self.prototypes[c] + (1.0 - m) * mean_c
                self.proto_variance[c] = m * self.proto_variance[c] + (1.0 - m) * var_c

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def _proto_infonce(
        self,
        feats: torch.Tensor,       # (N, D)
        labels: torch.Tensor,      # (N,)
    ) -> torch.Tensor:
        """InfoNCE between pixel feature z_i and the prototype bank.

        L = -mean log [ exp(<z, mu_y>/tau) / sum_c exp(<z, mu_c>/tau) ]
        Un-initialised prototypes are masked out of the denominator so they
        cannot dominate early in training.
        """
        if feats.numel() == 0:
            return feats.new_zeros(())
        proto = F.normalize(self.prototypes.detach(), dim=1)         # (C, D)
        feats = F.normalize(feats, dim=1)                            # (N, D)
        logits = feats @ proto.t() / self.temperature                 # (N, C)
        # Mask out un-initialised prototypes via a finite large negative,
        # so they never appear in the softmax denominator.
        init = self.proto_initialised.to(logits.dtype).unsqueeze(0)   # (1, C)
        logits = logits + (1.0 - init) * (-1e9)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        # Skip anchors whose own class prototype hasn't been initialised.
        valid = self.proto_initialised[labels]
        if not valid.any():
            return feats.new_zeros(())
        nll = -log_prob.gather(1, labels.unsqueeze(1)).squeeze(1)
        return nll[valid].mean()

    def _dist_contrast(
        self,
        feats: torch.Tensor,       # (N, D)
        labels: torch.Tensor,      # (N,)
    ) -> torch.Tensor:
        """Distributional contrast (paper Eq. 7), diagonal-Gaussian form.

        For a pixel feature z with class y, the negative log probability under
        its own class Gaussian, normalised by the sum over all classes:

            -log [ N(z; mu_y, sigma_y) / sum_c N(z; mu_c, sigma_c) ]

        This is the standard "Gaussian-MLE InfoNCE" reformulation used by
        SePiCo (Sec. 4.2) when ``dist_weight > 0``.
        """
        if feats.numel() == 0:
            return feats.new_zeros(())
        mu = self.prototypes.detach()                # (C, D)
        var = self.proto_variance.detach().clamp_min(1e-4)  # (C, D)
        # log N(z; mu_c, var_c) = -0.5 * sum_d [(z_d - mu_cd)^2 / var_cd + log(2 pi var_cd)]
        z = feats.unsqueeze(1)                       # (N, 1, D)
        diff2 = (z - mu.unsqueeze(0)) ** 2           # (N, C, D)
        log_n = -0.5 * (diff2 / var.unsqueeze(0) + torch.log(2 * torch.pi * var.unsqueeze(0))).sum(dim=2)
        init = self.proto_initialised.to(log_n.dtype).unsqueeze(0)
        log_n = log_n + (1.0 - init) * (-1e9)
        log_denom = torch.logsumexp(log_n, dim=1)    # (N,)
        log_pos = log_n.gather(1, labels.unsqueeze(1)).squeeze(1)  # (N,)
        valid = self.proto_initialised[labels]
        if not valid.any():
            return feats.new_zeros(())
        return (log_denom - log_pos)[valid].mean()

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
            raise ValueError("SePiCoLoss requires target_pred.")
        B, C, H, W = target_pred.shape
        if C != self.num_classes:
            raise ValueError(
                f"SePiCoLoss configured for num_classes={self.num_classes} but "
                f"received logits with C={C}. Re-instantiate with the matching "
                f"value — prototype buffers cannot be silently resized."
            )

        # Student features = L2-normalised softmax (projection-free variant).
        prob = F.softmax(target_pred, dim=1)
        feats = F.normalize(prob, dim=1, eps=1e-6)

        # Teacher pseudo-label + reliability mask.
        with torch.no_grad():
            ref = teacher_pred if teacher_pred is not None else target_pred.detach()
            prob_T = F.softmax(ref, dim=1)
            conf_T, pseudo_T = prob_T.max(dim=1)
            valid = conf_T >= self.confidence_threshold

        # Per-image uniform sub-sampling of valid pixels.
        anchors, anchor_labels = [], []
        for b in range(B):
            v = valid[b]
            if not v.any():
                continue
            f_b = feats[b].permute(1, 2, 0).reshape(-1, C)
            l_b = pseudo_T[b].reshape(-1)
            v_b = v.reshape(-1)
            idx_pool = torch.nonzero(v_b, as_tuple=False).flatten()
            if idx_pool.numel() > self.pixels_per_image:
                perm = torch.randperm(idx_pool.numel(), device=idx_pool.device)
                sel = idx_pool[perm[: self.pixels_per_image]]
            else:
                sel = idx_pool
            anchors.append(f_b[sel])
            anchor_labels.append(l_b[sel])

        if not anchors:
            l_proto = target_pred.new_zeros(())
            l_dist = target_pred.new_zeros(())
        else:
            A = torch.cat(anchors, dim=0)
            Y = torch.cat(anchor_labels, dim=0)
            # Update prototypes from a detached copy of A.
            self._update_prototypes(A.detach(), Y)
            l_proto = self._proto_infonce(A, Y)
            l_dist = self._dist_contrast(A, Y) if self.dist_weight > 0 else target_pred.new_zeros(())

        total = self.proto_weight * l_proto + self.dist_weight * l_dist
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
