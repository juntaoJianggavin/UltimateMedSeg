# DiGA: Distillation-Guided Adaptation for Domain Adaptive Semantic Segmentation (CVPR 2023)
# Reference: https://github.com/BIT-DA/DiGA
# Paper: https://arxiv.org/abs/2304.02222
# Implemented from paper formulas; not a copy of the official repo.
"""DiGA replaces the brittle "argmax pseudo-label CE" of vanilla
self-training with a **symmetric distillation** between the student and a
slow EMA teacher, gated by per-class quality weights derived from the
teacher's class-wise confidence histogram (paper Sec. 3.3).

Two stages, both implemented inside this single loss for compatibility
with the shared trainer:

  Stage A (warm-up): bidirectional KL between student softmax p_S and
                     teacher softmax p_T, weighted by per-pixel teacher
                     confidence. No hard argmax is taken, which keeps the
                     gradient signal smooth (paper Eq. 3 & 4).

        L_distill = 0.5 * KL(p_T || p_S) + 0.5 * KL(p_S || p_T)
                  weighted by q = max p_T

  Stage B (refinement): a class-balanced soft-CE between the *temperature-
                     scaled* teacher softmax and the student logits, with
                     per-class weight ``w_c = 1 - acc_c`` (paper Eq. 6),
                     where ``acc_c`` is an EMA over the fraction of teacher
                     pixels whose predicted class is ``c``. Rare classes
                     therefore receive a larger weight, attenuating the
                     long-tail collapse the authors document in Sec. 4.3.

The two stages are blended by ``stage_blend`` (default 1.0, i.e. always run
both jointly). ``update_epoch`` is honoured so an outer training loop can
ramp ``stage_blend`` from 0 → 1 over ``rampup_epochs`` if desired.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("diga")
class DiGALoss(nn.Module):
    """Distillation-Guided Adaptation.

    Shen et al., CVPR 2023.
    Reference (not copied): https://github.com/BIT-DA/DiGA

    Args:
        distill_weight: scalar on the symmetric-KL distillation term.
        ce_weight: scalar on the class-balanced soft-CE refinement term.
        temperature: distillation temperature T (paper default 2.0).
        confidence_threshold: pixels whose teacher confidence is below this
            value are excluded from both terms.
        rampup_epochs: number of epochs to linearly ramp ``stage_blend``
            from 0 to 1 (use 0 to disable rampup).
        class_acc_momentum: EMA momentum for the per-class accuracy buffer.
        num_classes: number of segmentation classes.
    """

    def __init__(
        self,
        distill_weight: float = 1.0,
        ce_weight: float = 1.0,
        temperature: float = 2.0,
        confidence_threshold: float = 0.5,
        rampup_epochs: int = 0,
        class_acc_momentum: float = 0.99,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"DiGA temperature must be positive, got {temperature}")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"DiGA confidence_threshold must be in [0, 1], got "
                f"{confidence_threshold}"
            )
        if not (0.0 < class_acc_momentum < 1.0):
            raise ValueError(
                f"DiGA class_acc_momentum must be in (0, 1), got "
                f"{class_acc_momentum}"
            )
        self.distill_weight = distill_weight
        self.ce_weight = ce_weight
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.rampup_epochs = rampup_epochs
        self.class_acc_momentum = class_acc_momentum
        self.num_classes = num_classes
        self.current_epoch = 0

        self.register_buffer(
            "_class_acc_ema",
            torch.full((num_classes,), 1.0 / num_classes),
        )
        self.register_buffer(
            "_acc_initialised", torch.tensor(False),
        )

    # ------------------------------------------------------------------
    # Rampup hook (called by the trainer per epoch)
    # ------------------------------------------------------------------
    def update_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _stage_blend(self) -> float:
        if self.rampup_epochs <= 0:
            return 1.0
        return float(min(1.0, self.current_epoch / float(self.rampup_epochs)))

    # ------------------------------------------------------------------
    # Class-accuracy EMA (drives the per-class weights in Stage B)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_class_acc(self, pseudo_T: torch.Tensor, valid: torch.Tensor):
        counts = torch.zeros(self.num_classes, device=pseudo_T.device)
        for c in range(self.num_classes):
            counts[c] = ((pseudo_T == c) & valid).sum()
        total = counts.sum().clamp_min(1.0)
        acc = counts / total
        if not bool(self._acc_initialised.item()):
            self._class_acc_ema = acc
            self._acc_initialised = torch.tensor(True, device=pseudo_T.device)
        else:
            m = self.class_acc_momentum
            self._class_acc_ema = m * self._class_acc_ema + (1.0 - m) * acc

    def _class_weights(self) -> torch.Tensor:
        # w_c = 1 - acc_c, normalised to mean 1 across classes.
        w = (1.0 - self._class_acc_ema).clamp_min(1e-6)
        return w / w.mean().clamp_min(1e-6)

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
            raise ValueError("DiGALoss requires target_pred.")
        B, C, H, W = target_pred.shape
        if C != self.num_classes:
            raise ValueError(
                f"DiGALoss configured for num_classes={self.num_classes} but "
                f"received logits with C={C}. Re-instantiate with the matching "
                f"value — EMA accuracy buffer cannot be silently resized."
            )

        # ---- Teacher reference (EMA preferred) ---------------------
        with torch.no_grad():
            ref = teacher_pred if teacher_pred is not None else target_pred.detach()
            T = self.temperature
            prob_T = F.softmax(ref / T, dim=1)
            conf_T, pseudo_T = F.softmax(ref, dim=1).max(dim=1)
            valid = conf_T >= self.confidence_threshold
            self._update_class_acc(pseudo_T, valid)

        # ---- Symmetric KL (Stage A) --------------------------------
        # Use temperature-scaled distributions on both sides — standard
        # distillation form (Hinton et al.). KL is over the class dimension.
        log_p_S = F.log_softmax(target_pred / T, dim=1)
        p_S = F.softmax(target_pred / T, dim=1)
        log_p_T = torch.log(prob_T.clamp_min(1e-8))

        # KL(p_T || p_S) and KL(p_S || p_T), per-pixel (sum over classes).
        kl_ts = (prob_T * (log_p_T - log_p_S)).sum(dim=1)   # (B, H, W)
        kl_st = (p_S * (log_p_S - log_p_T)).sum(dim=1)      # (B, H, W)
        sym_kl = 0.5 * (kl_ts + kl_st)
        # Per-pixel weight = teacher confidence on valid pixels, else 0.
        w_pix = conf_T * valid.float()
        denom = w_pix.sum().clamp_min(1.0)
        # T^2 scaling is the standard Hinton-distillation correction so
        # gradients are not divided by T^2.
        l_distill = (T * T) * (w_pix * sym_kl).sum() / denom

        # ---- Class-balanced soft-CE (Stage B) -----------------------
        # Targets are the *un-temperature-scaled* teacher softmax on valid
        # pixels, weighted per-class by w_c = 1 - acc_c (rare-class boost).
        cw = self._class_weights().to(target_pred.dtype)             # (C,)
        # Soft-CE = -sum_c w_c * p_T_c * log p_S_c, averaged over valid pixels.
        log_p_S_full = F.log_softmax(target_pred, dim=1)
        p_T_full = F.softmax(ref, dim=1).detach()
        per_class = -(p_T_full * log_p_S_full)                        # (B, C, H, W)
        per_class = per_class * cw.view(1, -1, 1, 1)
        per_pixel = per_class.sum(dim=1)                              # (B, H, W)
        l_ce = (valid.float() * per_pixel).sum() / valid.float().sum().clamp_min(1.0)

        blend = self._stage_blend()
        total = (
            self.distill_weight * l_distill
            + blend * self.ce_weight * l_ce
        )
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
