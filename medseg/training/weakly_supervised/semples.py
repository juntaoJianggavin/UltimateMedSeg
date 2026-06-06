# SemPLeS (WACV 2025)
# Reference: https://github.com/CIAI-NCKU/SemPLeS
# Paper: https://arxiv.org/abs/2401.11791
# Implemented from paper formulas; not a copy of the official repo.
"""SemPLeS — Semantic Prompt Learning for Weakly-Supervised Semantic
Segmentation.

Lin et al., "SemPLeS: Semantic Prompt Learning for Weakly-Supervised
Semantic Segmentation", WACV 2025.
Paper: https://arxiv.org/abs/2401.11791
Reference repository: https://github.com/CIAI-NCKU/SemPLeS

Why it exists.
    CLIP-driven WSSS systems (CLIP-ES, CLIMS, WeCLIP) use a hand-crafted
    class prompt ("a photo of {class}") and read CAM from the cross
    modality attention. The fixed prompt is suboptimal because the
    *visual* concept of "table" inside a cluttered medical scene is not
    described by the textual concept of "table" alone. SemPLeS introduces
    two LEARNABLE token banks:

      - p_c^cls : a per-class semantic prompt (D dims).
      - p_c^bg  : a per-class background-distractor prompt that absorbs
                  the categorical concepts CO-occurring with class c
                  (e.g. for "boat" the "water" distractor).

    Two losses optimise them jointly with the standard WSSS BCE.

Loss (paper Sec. 3.3, Eq. 4-9):

    (1) Contrastive Prompt Learning (CPL, Eq. 5).
        Class image embeddings (extracted from the CLIP visual encoder
        with the class CAM as a soft mask) are pulled towards p_c^cls
        and pushed from p_c^bg and from p_{c'}^cls for c' != c:

            L_CPL = -log  exp(sim(v_c, p_c^cls) / tau)
                          ------------------------------------------------
                          sum_{q in {p_*^cls, p_c^bg}} exp(sim(v_c, q)/tau)

    (2) Prompt-guided Semantic Refinement (PSR, Eq. 7).
        The CAM generated under the LEARNED prompt should agree with the
        CAM generated under a frozen TEACHER prompt (zero-shot CLIP) on
        confident regions and the boundaries:

            L_PSR = || CAM^learn  -  CAM^teacher ||_1  (per-class mean)

This module implements the loss only. The CLIP backbone, prompt token
bank, image embeddings, teacher CAMs and student CAMs are produced by
the user's model and passed in as forward kwargs.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("semples")
class SemPLeSLoss(nn.Module):
    """SemPLeS classification + contrastive prompt + prompt-guided refinement.

    Args:
        lambda_cpl: Weight on contrastive prompt learning (default 0.5).
        lambda_psr: Weight on prompt-guided CAM refinement (default 0.2).
        cls_weight: Weight on the multi-label BCE classification term.
        temperature: InfoNCE temperature for CPL (default 0.1).
        include_bg_prompt: If True the per-class background prompts
            participate in the CPL denominator (paper default True).
    """

    def __init__(
        self,
        lambda_cpl: float = 0.5,
        lambda_psr: float = 0.2,
        cls_weight: float = 1.0,
        temperature: float = 0.1,
        include_bg_prompt: bool = True,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 (got {temperature})")
        self.lambda_cpl = lambda_cpl
        self.lambda_psr = lambda_psr
        self.cls_weight = cls_weight
        self.temperature = temperature
        self.include_bg_prompt = include_bg_prompt

    # ------------------------------------------------------------------
    def _cpl(
        self,
        class_image_emb: torch.Tensor,
        cls_prompts: torch.Tensor,
        bg_prompts: Optional[torch.Tensor],
        image_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. 5 — InfoNCE over learned prompt banks.

        Args:
            class_image_emb: (B, C, D) per-class image embeddings, where
                v_{b,c} is the CLIP visual embedding of image b masked by
                CAM_c.
            cls_prompts: (C, D) learnable class prompts p_c^cls.
            bg_prompts: (C, D) learnable bg distractor prompts p_c^bg
                (or None to disable).
            image_labels: (B, C) multi-label tags — only PRESENT classes
                receive a CPL gradient.
        """
        if class_image_emb.dim() != 3:
            raise ValueError(
                f"class_image_emb must be (B,C,D); got "
                f"{tuple(class_image_emb.shape)}"
            )
        if cls_prompts.dim() != 2:
            raise ValueError(
                f"cls_prompts must be (C,D); got {tuple(cls_prompts.shape)}"
            )
        B, C, D = class_image_emb.shape
        if cls_prompts.shape[0] != C or cls_prompts.shape[1] != D:
            raise ValueError(
                f"cls_prompts shape {tuple(cls_prompts.shape)} incompatible "
                f"with class_image_emb {tuple(class_image_emb.shape)}"
            )

        v = F.normalize(class_image_emb, dim=-1)                  # (B,C,D)
        p_cls = F.normalize(cls_prompts, dim=-1)                  # (C,D)
        # sim (B, C, C) — row c is v_{b,c} vs all cls prompts.
        sim_cls = torch.einsum("bcd,kd->bck", v, p_cls) / self.temperature

        if self.include_bg_prompt and bg_prompts is not None:
            if bg_prompts.shape != cls_prompts.shape:
                raise ValueError(
                    f"bg_prompts shape {tuple(bg_prompts.shape)} must match "
                    f"cls_prompts {tuple(cls_prompts.shape)}"
                )
            p_bg = F.normalize(bg_prompts, dim=-1)
            # Match v_{b,c} only with its OWN bg prompt (per paper Eq. 5).
            sim_bg = (v * p_bg.unsqueeze(0)).sum(dim=-1, keepdim=True) \
                / self.temperature                                # (B,C,1)
            logits = torch.cat([sim_cls, sim_bg], dim=-1)         # (B,C,C+1)
        else:
            logits = sim_cls

        # Positive index = c, which is row c's own column.
        log_softmax = F.log_softmax(logits, dim=-1)
        # Gather diagonal per row.
        diag_idx = torch.arange(C, device=v.device).view(1, C, 1).expand(B, C, 1)
        pos = log_softmax.gather(-1, diag_idx).squeeze(-1)        # (B,C)

        mask = image_labels.float()
        loss = -(pos * mask).sum() / mask.sum().clamp_min(1.0)
        return loss

    # ------------------------------------------------------------------
    @staticmethod
    def _psr(
        cam_learn: torch.Tensor,
        cam_teacher: torch.Tensor,
        image_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. 7 — L1 between student and teacher CAMs on present classes.

        Both CAMs are min-max normalised per (sample, class) before the L1
        so the teacher's calibration does not dominate the gradient.
        """
        if cam_learn.shape != cam_teacher.shape:
            raise ValueError(
                f"cam_learn {tuple(cam_learn.shape)} != cam_teacher "
                f"{tuple(cam_teacher.shape)}"
            )
        if cam_learn.dim() != 4:
            raise ValueError(
                f"CAMs must be (B,C,H,W); got {tuple(cam_learn.shape)}"
            )

        def _norm(x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            f = x.view(B, C, -1)
            m = f.amin(dim=2, keepdim=True)
            M = f.amax(dim=2, keepdim=True)
            return ((f - m) / (M - m + 1e-6)).view(B, C, H, W)

        a = _norm(cam_learn)
        b = _norm(cam_teacher.detach())
        present = image_labels.float().view(image_labels.size(0), -1, 1, 1)
        diff = (a - b).abs() * present
        denom = (present.expand_as(diff)).sum().clamp_min(1.0) * a.shape[-1] * a.shape[-2]
        return diff.sum() / denom

    # ------------------------------------------------------------------
    def forward(
        self,
        class_image_emb: torch.Tensor,
        cls_prompts: torch.Tensor,
        cam_learn: torch.Tensor,
        cam_teacher: torch.Tensor,
        image_labels: torch.Tensor,
        bg_prompts: Optional[torch.Tensor] = None,
        cls_logits: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            class_image_emb: (B, C, D) per-class CLIP visual embeddings.
            cls_prompts: (C, D) learnable class prompt bank.
            cam_learn: (B, C, H, W) CAM from the learned prompt branch.
            cam_teacher: (B, C, H, W) CAM from the frozen-prompt teacher.
            image_labels: (B, C) multi-label tags.
            bg_prompts: (C, D) learnable bg distractor prompt bank.
                Required when ``include_bg_prompt=True`` and lambda_cpl>0.
            cls_logits: optional (B, C) classifier logits — if absent the
                BCE is computed on GAP(cam_learn).
            labeled_loss: optional dense supervised loss to add.
        """
        if image_labels.dim() != 2:
            raise ValueError(
                f"image_labels must be (B,C); got {tuple(image_labels.shape)}"
            )

        # (1) Multi-label classification BCE on the learned CAM.
        cls_in = cls_logits if cls_logits is not None else cam_learn.mean(dim=(2, 3))
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_in, image_labels.float()
        )

        # (2) CPL.
        if self.lambda_cpl > 0:
            if self.include_bg_prompt and bg_prompts is None:
                raise ValueError(
                    "include_bg_prompt=True but bg_prompts was not provided "
                    "to SemPLeSLoss.forward()."
                )
            cpl = self._cpl(class_image_emb, cls_prompts, bg_prompts, image_labels)
        else:
            cpl = cam_learn.new_zeros(())

        # (3) PSR.
        psr = self._psr(cam_learn, cam_teacher, image_labels) \
            if self.lambda_psr > 0 else cam_learn.new_zeros(())

        total = (
            self.cls_weight * cls_loss
            + self.lambda_cpl * cpl
            + self.lambda_psr * psr
        )
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
