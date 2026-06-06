# DuPL (CVPR 2024)
# Reference: https://github.com/Wu0409/DuPL
# Paper: https://arxiv.org/abs/2403.11184
# Implemented from paper formulas; not a copy of the official repo.
"""DuPL — Dual Student with Trustworthy Progressive Learning for Robust
End-to-End Single-Stage Weakly-Supervised Semantic Segmentation.

Wu et al., "DuPL: Dual Student with Trustworthy Progressive Learning for
Robust End-to-End Weakly-Supervised Semantic Segmentation", CVPR 2024.
Paper: https://arxiv.org/abs/2403.11184
Reference repository: https://github.com/Wu0409/DuPL

Why it exists.
    Single-stage end-to-end WSSS uses CAMs as on-the-fly pseudo-masks for
    a segmentation head; this couples noise in CAM to the segmentation
    head and quickly accumulates confirmation bias. DuPL maintains TWO
    student branches with different parameter initialisations / dropout
    so they generate (nearly) independent CAMs and segmentation logits;
    the two branches then cross-supervise each other on the regions where
    both agree (the "trustworthy" set). A progressive thresholding schedule
    grows the trustworthy region as training proceeds.

Loss (paper Sec. 3.3, Eq. 4-9):

    L = L_cls^A + L_cls^B                       (multi-label BCE on each branch)
      + lambda_seg * ( L_seg^A->B + L_seg^B->A )
      + lambda_disc * L_discrepancy
      + lambda_cam_eq * L_cam_eq                (optional CAM equivariance)

    where, for the cross-pseudo segmentation term:

        y^B_hat(x,y) = argmax_c softmax(seg^B(x,y))_c        # detached
        conf^B(x,y)  = max_c softmax(seg^B(x,y))_c           # detached
        M(x,y)       = 1[ conf^A > tau_t AND conf^B > tau_t
                          AND y^A_hat == y^B_hat ]           # trustworthy mask

        L_seg^A->B = mean_{(x,y) in M}  CE( seg^A(x,y) , y^B_hat(x,y) )

    Discrepancy regulariser keeps the two branches diverse so they keep
    providing complementary signal (official code, asymmetric detach):

        L_discrepancy = (1 + cos(feat^A.detach(), feat^B))
                      + (1 + cos(feat^B.detach(), feat^A))

    Progressive threshold (Eq. 9):

        tau_t = tau_min + (tau_max - tau_min) * (t / T_total)

    growing from tau_min ~ 0.5 to tau_max ~ 0.9 across training.

This module implements only the loss; the two branches, CAM heads and
segmentation heads live in the user's model. The progressive ``tau`` is
exposed as a forward kwarg so the training loop can step it.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("dupl")
class DuPLLoss(nn.Module):
    """DuPL dual-branch classification + trustworthy cross-pseudo segmentation
    + branch-discrepancy regulariser.

    Args:
        lambda_seg: Weight on the symmetric cross-pseudo segmentation term
            (paper default 1.0).
        lambda_disc: Weight on the branch-discrepancy regulariser
            (paper default 0.1).
        lambda_cam_eq: Weight on optional CAM-equivariance term between the
            two branches' raw CAM logits. The paper enables it only in some
            ablations; default 0.0.
        cls_weight: Weight on each branch's classification BCE
            (paper keeps each at 1.0).
        tau_min: Progressive-threshold lower bound (early-training floor).
        tau_max: Progressive-threshold upper bound (late-training cap).
        ignore_index: Pixel label treated as no-supervision (default 255).
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_disc: float = 0.1,
        lambda_cam_eq: float = 0.0,
        cls_weight: float = 1.0,
        tau_min: float = 0.5,
        tau_max: float = 0.9,
        ignore_index: int = 255,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < tau_min < tau_max < 1.0):
            raise ValueError(
                f"Need 0 < tau_min ({tau_min}) < tau_max ({tau_max}) < 1"
            )
        self.lambda_seg = lambda_seg
        self.lambda_disc = lambda_disc
        self.lambda_cam_eq = lambda_cam_eq
        self.cls_weight = cls_weight
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.ignore_index = ignore_index

    # ------------------------------------------------------------------
    @staticmethod
    def _cls_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 4:
            logits = logits.mean(dim=(2, 3))
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    # ------------------------------------------------------------------
    def _cross_pseudo(
        self,
        seg_student: torch.Tensor,
        seg_teacher: torch.Tensor,
        tau: float,
        agree_with: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """L_seg^teacher->student per paper Eq. 6.

        Args:
            seg_student: (B, C, H, W) student logits being supervised.
            seg_teacher: (B, C, H, W) teacher logits (detached) producing
                the pseudo-label.
            tau: per-iteration trustworthy-confidence threshold.
            agree_with: optional (B, H, W) integer argmax of the OTHER
                branch — used to build the AGREEMENT mask. If None the
                pseudo is accepted on confidence alone.
        """
        teacher_p = F.softmax(seg_teacher.detach(), dim=1)
        conf, pseudo = teacher_p.max(dim=1)                       # (B,H,W)
        mask = conf > tau
        if agree_with is not None:
            mask = mask & (pseudo == agree_with)
        if not mask.any():
            return seg_student.new_zeros(())

        # Pixel-wise CE on the mask only.
        log_p = F.log_softmax(seg_student, dim=1)
        gathered = log_p.gather(1, pseudo.unsqueeze(1)).squeeze(1)
        loss = -(gathered * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        return loss

    # ------------------------------------------------------------------
    @staticmethod
    def _discrepancy(
        feat_a: torch.Tensor, feat_b: torch.Tensor
    ) -> torch.Tensor:
        """Asymmetric cosine discrepancy: (1 + cos(f_a.detach(), f_b)) +
        (1 + cos(f_b.detach(), f_a)), matching official DuPL code
        (train_final_voc.py lines 247-254 / 440-447).

        One branch is detached so that the discrepancy loss pushes the
        OTHER branch away, keeping gradients asymmetric as intended by
        the paper.

        Either flat (B, D) or spatial (B, D, H, W) features are accepted;
        spatial features are GAP'd first.
        """
        if feat_a.dim() == 4:
            feat_a = feat_a.mean(dim=(2, 3))
        if feat_b.dim() == 4:
            feat_b = feat_b.mean(dim=(2, 3))
        cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        sim_loss_1 = 1.0 + cos_sim(feat_a.detach(), feat_b).mean()
        sim_loss_2 = 1.0 + cos_sim(feat_b.detach(), feat_a).mean()
        return sim_loss_1 + sim_loss_2

    # ------------------------------------------------------------------
    @staticmethod
    def progressive_tau_default(
        step: int, total_steps: int, tau_min: float, tau_max: float
    ) -> float:
        """Default schedule (Eq. 9). Exposed as a helper so callers can
        pass the same formula explicitly via the `tau` forward arg."""
        if total_steps <= 0:
            return tau_max
        frac = max(0.0, min(1.0, step / total_steps))
        return tau_min + (tau_max - tau_min) * frac

    # ------------------------------------------------------------------
    def forward(
        self,
        seg_a: torch.Tensor,
        seg_b: torch.Tensor,
        cam_a: torch.Tensor,
        cam_b: torch.Tensor,
        image_labels: torch.Tensor,
        feat_a: Optional[torch.Tensor] = None,
        feat_b: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            seg_a, seg_b: (B, C, H, W) per-branch segmentation logits (C
                includes a background channel as channel 0).
            cam_a, cam_b: (B, C_fg, H, W) per-branch raw CAM logits used
                ONLY for the multi-label BCE and the optional equivariance.
            image_labels: (B, C_fg) multi-label tags.
            feat_a, feat_b: optional (B, D, H, W) or (B, D) features for
                the discrepancy regulariser. If both None, the term is 0.
            tau: explicit trustworthy threshold for THIS step. If None,
                computed via ``progressive_tau_default(step, total_steps)``;
                if those are also missing falls back to ``tau_max``.
            step / total_steps: optional training progress for tau.
            labeled_loss: optional dense supervised loss to add.
        """
        if seg_a.shape != seg_b.shape:
            raise ValueError(
                f"seg_a {tuple(seg_a.shape)} != seg_b {tuple(seg_b.shape)}"
            )
        if cam_a.shape != cam_b.shape:
            raise ValueError(
                f"cam_a {tuple(cam_a.shape)} != cam_b {tuple(cam_b.shape)}"
            )
        if cam_a.shape[1] != image_labels.shape[1]:
            raise ValueError(
                f"cam channel {cam_a.shape[1]} != image_labels channel "
                f"{image_labels.shape[1]}"
            )

        # (1) Per-branch classification BCE.
        cls_a = self._cls_bce(cam_a, image_labels)
        cls_b = self._cls_bce(cam_b, image_labels)

        # Resolve tau.
        if tau is None:
            if step is not None and total_steps is not None:
                tau = self.progressive_tau_default(
                    step, total_steps, self.tau_min, self.tau_max
                )
            else:
                tau = self.tau_max
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0,1); got {tau}")

        # Both branches' argmax pseudo (detached) — needed for the agreement.
        with torch.no_grad():
            arg_a = F.softmax(seg_a, dim=1).argmax(dim=1)
            arg_b = F.softmax(seg_b, dim=1).argmax(dim=1)

        # (2) Symmetric cross-pseudo segmentation on the agreement mask.
        loss_ab = self._cross_pseudo(seg_a, seg_b, tau, agree_with=arg_a)
        loss_ba = self._cross_pseudo(seg_b, seg_a, tau, agree_with=arg_b)
        seg_loss = 0.5 * (loss_ab + loss_ba)

        # (3) Branch discrepancy regulariser.
        if feat_a is not None and feat_b is not None and self.lambda_disc > 0:
            disc = self._discrepancy(feat_a, feat_b)
        else:
            disc = seg_a.new_zeros(())

        # (4) Optional CAM equivariance — L1 between sigmoid CAMs of the
        # two branches on present classes (paper Sec. 3.3 ablation).
        if self.lambda_cam_eq > 0:
            present = image_labels.float().view(image_labels.size(0), -1, 1, 1)
            eq = ((torch.sigmoid(cam_a) - torch.sigmoid(cam_b)).abs()
                  * present).sum() / present.sum().clamp_min(1.0) / (
                      cam_a.size(-1) * cam_a.size(-2)
                  )
        else:
            eq = seg_a.new_zeros(())

        total = (
            self.cls_weight * (cls_a + cls_b)
            + self.lambda_seg * seg_loss
            + self.lambda_disc * disc
            + self.lambda_cam_eq * eq
        )
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
