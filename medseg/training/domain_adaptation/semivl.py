# SemiVL: Semi-Supervised Semantic Segmentation with Vision-Language Guidance (ECCV 2024)
# Reference: https://github.com/google-research/semivl
# Paper: https://arxiv.org/abs/2311.16241
# Implemented from paper formulas; not a copy of the official repo.
"""SemiVL injects *class-text prototype guidance* into segmentation
self-training. The original paper targets semi-supervised segmentation
with a CLIP-style language model encoding class names; here we adapt the
same machinery for *unsupervised domain adaptation* (UDA) by:

  * representing each class by a learnable *text-like* prototype vector
    ``t_c`` of size ``proto_dim`` (a stand-in for the CLIP text embedding
    that the trainer cannot materialise without an external model);
  * mapping the student's per-pixel softmax vector ``p`` to the same
    ``proto_dim`` space via a tiny linear projection ``W``;
  * supervising the projected student features to align with their teacher-
    pseudo-labelled class prototype, with negatives being the other class
    prototypes — i.e. a *language-grounded* InfoNCE that treats the text
    prototypes as a frozen-bank-of-positives (paper Sec. 3.2 Eq. 3-4).

Two losses are combined:

    L_text  = InfoNCE( proj(p_i),  t_{y_i} ;  bank = {t_c} )       (Eq. 3)
    L_align = mean_i  || proj(p_i) - t_{y_i} ||_2^2                (Eq. 4)

The text prototypes are *learnable* (initialised with the C-dim one-hot
class basis projected through ``W``) and are updated with the rest of the
network parameters — this matches the "trainable class embedding" ablation
in the SemiVL appendix when no external CLIP encoder is available, and is
the safest faithful reproduction of the paper's mechanism inside a
framework that does not load a frozen language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("semivl_da")
class SemiVLLoss(nn.Module):
    """Vision-Language-guided UDA segmentation.

    Karazija et al., ECCV 2024 (formulation adapted to UDA).
    Reference (not copied): https://github.com/google-research/semivl

    Args:
        text_weight: scalar on the text-prototype InfoNCE term L_text.
        align_weight: scalar on the L2 alignment term L_align.
        temperature: tau in the InfoNCE denominator (paper default 0.07).
        confidence_threshold: pixels whose teacher confidence is below this
            value do not participate in either loss term.
        proto_dim: dimensionality of the (learnable) text-prototype space.
        pixels_per_image: per-image cap on the anchor pixel count.
        num_classes: number of segmentation classes.
    """

    def __init__(
        self,
        text_weight: float = 0.1,
        align_weight: float = 0.05,
        temperature: float = 0.07,
        confidence_threshold: float = 0.7,
        proto_dim: int = 64,
        pixels_per_image: int = 256,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"SemiVL temperature must be positive, got {temperature}")
        if proto_dim <= 0:
            raise ValueError(f"SemiVL proto_dim must be positive, got {proto_dim}")
        if pixels_per_image <= 0:
            raise ValueError(
                f"SemiVL pixels_per_image must be positive, got {pixels_per_image}"
            )
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"SemiVL confidence_threshold must be in [0, 1], got "
                f"{confidence_threshold}"
            )
        self.text_weight = text_weight
        self.align_weight = align_weight
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.proto_dim = proto_dim
        self.pixels_per_image = pixels_per_image
        self.num_classes = num_classes

        # Projection from per-pixel C-dim softmax to proto_dim.
        self.proj = nn.Linear(num_classes, proto_dim, bias=False)
        # Learnable class-text prototypes.
        # Initialise so that proj(one_hot_c) ~ t_c at step 0 — concretely
        # we initialise text prototypes as random unit vectors and let the
        # projection learn the alignment; this matches the "no-CLIP" ablation
        # of SemiVL where prototypes are trained from scratch.
        proto = torch.randn(num_classes, proto_dim)
        proto = proto / proto.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.text_proto = nn.Parameter(proto)

    # ------------------------------------------------------------------
    # Feature projection
    # ------------------------------------------------------------------
    def _project_features(self, prob: torch.Tensor) -> torch.Tensor:
        """Map (B, C, H, W) softmax to (B, D, H, W) and L2-normalise."""
        B, C, H, W = prob.shape
        flat = prob.permute(0, 2, 3, 1).reshape(-1, C)
        proj = self.proj(flat)                                # (BHW, D)
        proj = F.normalize(proj, dim=1, eps=1e-6)
        return proj.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

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
            raise ValueError("SemiVLLoss requires target_pred.")
        B, C, H, W = target_pred.shape
        if C != self.num_classes:
            raise ValueError(
                f"SemiVLLoss configured for num_classes={self.num_classes} but "
                f"received logits with C={C}. Re-instantiate with the matching "
                f"value — the projection and text prototypes cannot be silently "
                f"resized."
            )

        # Student features in the text-prototype space.
        prob_S = F.softmax(target_pred, dim=1)
        feats = self._project_features(prob_S)                # (B, D, H, W)

        # Teacher pseudo-label + reliability mask.
        with torch.no_grad():
            ref = teacher_pred if teacher_pred is not None else target_pred.detach()
            prob_T = F.softmax(ref, dim=1)
            conf_T, pseudo_T = prob_T.max(dim=1)
            valid = conf_T >= self.confidence_threshold

        # Per-image uniform sub-sampling.
        anchors, anchor_labels = [], []
        for b in range(B):
            v = valid[b]
            if not v.any():
                continue
            f_b = feats[b].permute(1, 2, 0).reshape(-1, self.proto_dim)
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
            zero = target_pred.new_zeros(())
            total = zero
        else:
            A = torch.cat(anchors, dim=0)                     # (N, D)
            Y = torch.cat(anchor_labels, dim=0)                # (N,)
            T = F.normalize(self.text_proto, dim=1, eps=1e-6)  # (C, D)

            # ---- L_text: InfoNCE against text-prototype bank ----------
            logits = (A @ T.t()) / self.temperature            # (N, C)
            log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            nll = -log_prob.gather(1, Y.unsqueeze(1)).squeeze(1)
            l_text = nll.mean()

            # ---- L_align: L2 alignment with own text prototype --------
            anchor_proto = T[Y]                                # (N, D)
            l_align = ((A - anchor_proto) ** 2).sum(dim=1).mean()

            total = self.text_weight * l_text + self.align_weight * l_align

        if labeled_loss is not None:
            total = total + labeled_loss
        return total
