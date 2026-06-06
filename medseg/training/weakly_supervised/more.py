# MoRe (AAAI 2025)
# Reference: https://github.com/zwyang6/MoRe
# Paper: https://arxiv.org/abs/2412.11076
# Implemented from paper formulas; not a copy of the official repo.
"""MoRe — Class Patch Attention Needs Regularization for Weakly Supervised
Semantic Segmentation.

Yang et al., "MoRe: Class Patch Attention Needs Regularization for Weakly
Supervised Semantic Segmentation", AAAI 2025.
Paper: https://arxiv.org/abs/2412.11076
Reference repository: https://github.com/zwyang6/MoRe

Why it exists.
    ViT-based WSSS (MCTformer, ToCo) reads CAMs from the class-to-patch
    attention block of the last few layers. The authors observe two
    pathologies in these attention maps:

      (a)  *Localization-uninformed activation* — attention from a class
           token leaks onto patches whose category is NOT among the image
           labels. Vanilla BCE on GAP(attn) is not enough to suppress this.
      (b)  *Graph-incoherent attention* — patches that are semantically
           similar (high feature-space affinity) often receive divergent
           class-attention scores, producing speckled CAMs.

    MoRe adds two cheap regularizers on top of the standard multi-label
    BCE used for CAM training:

        (i)  Localization-informed Regularization (LIR, Eq. 5):
                L_LIR = sum_{c not in y}   ||attn_{c -> *}||_1 / N
             — drives attention from ABSENT-class tokens to zero.

        (ii) Graph Category Attention Regularization (GCA, Eq. 7):
                A_ij = sim(p_i, p_j)         (patch-patch affinity)
                L_GCA = sum_{c in y} sum_ij  A_ij * (attn_{c,i} - attn_{c,j})^2 / N
             — a graph-Laplacian smoothness term that pulls the class
             attention of similar patches together.

Loss:

    L = cls_weight * BCE(GAP(cam), y)
      + lambda_lir * L_LIR
      + lambda_gca * L_GCA

The class-to-patch attention tensor and the patch tokens are MoRe's only
inputs from the ViT — both are supplied by the caller. No code is copied
from the upstream repo.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("more")
class MoReLoss(nn.Module):
    """MoRe classification + LIR + GCA loss.

    Args:
        lambda_lir: Weight on Localization-informed Regularization
            (paper default 0.1).
        lambda_gca: Weight on Graph Category Attention Regularization
            (paper default 0.05).
        cls_weight: Weight on the multi-label BCE classification term.
        gca_topk: For each patch keep only the top-K nearest-neighbour
            patches when building A_ij — controls O(N^2) memory.
            None disables the truncation (paper uses K=10).
        affinity_temp: Temperature applied to the cosine affinity before
            row-softmax (paper default 0.5).
    """

    def __init__(
        self,
        lambda_lir: float = 0.1,
        lambda_gca: float = 0.05,
        cls_weight: float = 1.0,
        gca_topk: Optional[int] = 10,
        affinity_temp: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        if affinity_temp <= 0:
            raise ValueError(f"affinity_temp must be > 0 (got {affinity_temp})")
        self.lambda_lir = lambda_lir
        self.lambda_gca = lambda_gca
        self.cls_weight = cls_weight
        self.gca_topk = gca_topk
        self.affinity_temp = affinity_temp

    # ------------------------------------------------------------------
    @staticmethod
    def _lir(class_attn: torch.Tensor, image_labels: torch.Tensor) -> torch.Tensor:
        """L1 of attention rows for ABSENT classes.

        Args:
            class_attn: (B, C, N) class-to-patch attention probabilities
                (already softmax-normalised by the caller — if logits are
                supplied they are softmax'd here for safety).
            image_labels: (B, C) binary multi-label tags.
        """
        if class_attn.dim() != 3:
            raise ValueError(
                f"class_attn must be (B,C,N); got {tuple(class_attn.shape)}"
            )
        # If the row mass departs from 1 by a wide margin, normalise.
        row_sum = class_attn.sum(dim=2, keepdim=True)
        if (row_sum - 1.0).abs().max().item() > 0.1:
            attn = F.softmax(class_attn, dim=2)
        else:
            attn = class_attn
        absent = (1.0 - image_labels.float()).unsqueeze(-1)        # (B,C,1)
        # Sum L1 of attention rows for absent classes, normalised by N.
        return (attn.abs() * absent).sum(dim=2).mean()

    # ------------------------------------------------------------------
    def _gca(
        self,
        class_attn: torch.Tensor,
        patch_tokens: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-Laplacian smoothness on class attention (Eq. 7).

        L_GCA = sum_c sum_{(i,j) in topK}  A_ij (a_{c,i} - a_{c,j})^2
        where A_ij = softmax_j(sim(p_i, p_j) / tau).
        """
        if patch_tokens.dim() != 3:
            raise ValueError(
                f"patch_tokens must be (B,N,D); got {tuple(patch_tokens.shape)}"
            )
        B, N, D = patch_tokens.shape
        if class_attn.shape[0] != B or class_attn.shape[2] != N:
            raise ValueError(
                f"class_attn (B,C,N)={tuple(class_attn.shape)} incompatible "
                f"with patch_tokens (B,N,D)={tuple(patch_tokens.shape)}"
            )
        C = class_attn.shape[1]

        feat = F.normalize(patch_tokens, dim=-1)
        # Affinity (B, N, N) (caller-friendly: small N for ViT-S/16 @ 224 = 196).
        sim = torch.einsum("bnd,bmd->bnm", feat, feat) / self.affinity_temp
        if self.gca_topk is not None and self.gca_topk < N:
            top_vals, top_idx = sim.topk(self.gca_topk, dim=-1)
            # Build a sparse-style normalised affinity over the top-K only.
            top_aff = F.softmax(top_vals, dim=-1)                  # (B,N,K)
            # Gather neighbour attention values: (B, C, N, K).
            idx_expand = top_idx.unsqueeze(1).expand(B, C, N, self.gca_topk)
            attn_self = class_attn.unsqueeze(-1)                   # (B,C,N,1)
            attn_neig = torch.gather(
                class_attn.unsqueeze(2).expand(B, C, N, N), 3, idx_expand
            )
            diff_sq = (attn_self - attn_neig).pow(2)               # (B,C,N,K)
            weighted = diff_sq * top_aff.unsqueeze(1)              # (B,C,N,K)
            per_class = weighted.sum(dim=(2, 3))                   # (B,C)
        else:
            aff = F.softmax(sim, dim=-1)                           # (B,N,N)
            # (a_{c,i} - a_{c,j})^2 broadcast: (B,C,N,N)
            attn_i = class_attn.unsqueeze(3)                       # (B,C,N,1)
            attn_j = class_attn.unsqueeze(2)                       # (B,C,1,N)
            diff_sq = (attn_i - attn_j).pow(2)
            weighted = diff_sq * aff.unsqueeze(1)
            per_class = weighted.sum(dim=(2, 3))

        # Mask out absent classes (no penalty on classes whose attention
        # the LIR term has already suppressed).
        present = image_labels.float()
        per_class = per_class * present
        denom = present.sum().clamp_min(1.0)
        # Normalise by N so the term is scale-invariant.
        return per_class.sum() / denom / float(N)

    # ------------------------------------------------------------------
    def forward(
        self,
        class_attn: torch.Tensor,
        patch_tokens: torch.Tensor,
        image_labels: torch.Tensor,
        cam_logits: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            class_attn: (B, C, N) class-to-patch attention — softmax over
                the patch dim. If raw logits are supplied they are softmax'd
                inside ``_lir``.
            patch_tokens: (B, N, D) ViT patch tokens used to build the
                patch-patch affinity for GCA.
            image_labels: (B, C) binary multi-label tags.
            cam_logits: optional (B, C, H, W) raw CAM logits — if absent
                the classification BCE is computed on the GAP of attention
                rows (summing each class' attention mass).
            labeled_loss: optional dense supervised loss to add.
        """
        if image_labels.dim() != 2 or image_labels.shape[1] != class_attn.shape[1]:
            raise ValueError(
                f"image_labels shape {tuple(image_labels.shape)} must be "
                f"(B, C={class_attn.shape[1]})"
            )

        # (1) Classification BCE.
        if cam_logits is not None:
            if cam_logits.dim() == 4:
                cls_logits = cam_logits.mean(dim=(2, 3))
            elif cam_logits.dim() == 2:
                cls_logits = cam_logits
            else:
                raise ValueError(
                    f"cam_logits must be (B,C) or (B,C,H,W); got "
                    f"{tuple(cam_logits.shape)}"
                )
        else:
            # GAP across patches of the attention rows → soft class score.
            # Logit-ify with a stable inverse-sigmoid so BCE-with-logits behaves.
            attn_score = class_attn.mean(dim=2).clamp(1e-6, 1 - 1e-6)
            cls_logits = torch.log(attn_score / (1.0 - attn_score))

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits, image_labels.float()
        )

        # (2) LIR and (3) GCA.
        lir = self._lir(class_attn, image_labels)
        gca = self._gca(class_attn, patch_tokens, image_labels) \
            if self.lambda_gca > 0 else class_attn.new_zeros(())

        total = (
            self.cls_weight * cls_loss
            + self.lambda_lir * lir
            + self.lambda_gca * gca
        )
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
