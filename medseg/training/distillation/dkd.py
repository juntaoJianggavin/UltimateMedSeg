# Reference: https://github.com/megvii-research/mdistiller
# Paper: https://arxiv.org/abs/2203.08679
"""DKD: Decoupled Knowledge Distillation (CVPR 2022).

将传统KD损失解耦为TCKD（target class KD, weight alpha）+ NCKD
（non-target class KD, weight beta），对应论文公式 (8)。适用于分割
任务：在逐像素上计算DKD。

Per Zhao et al. Eq. 9, warmup uses ``min(1, epoch / warmup_epochs)``.
The host trainer must call ``update_epoch(epoch)`` if warmup > 0;
the default ``warmup=0`` disables warmup so the loss is active from
the first step even when the trainer does not push epoch state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("dkd")
class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation (CVPR 2022).

    Official source: megvii-research/mdistiller.
    """

    def __init__(
        self,
        ce_weight: float = 0.0,
        alpha: float = 1.0,
        beta: float = 8.0,
        temperature: float = 4.0,
        warmup: int = 0,
        **kwargs,
    ):
        # NOTE: ce_weight defaults to 0.0 because train_distillation.py
        # already adds the supervised task loss outside this module
        # (loss = task_loss + distill_weight * kd_loss). Set ce_weight>0
        # only when invoking DKD as the sole training criterion.
        #
        # warmup defaults to 0 (paper Eq. 9 with no ramp). The official
        # repo uses 20 epochs of warmup, but that requires the trainer to
        # forward the current epoch to this module via update_epoch().
        # Set warmup>0 only when that hook is wired up; otherwise the KD
        # term would be zero for the entire run.
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"DKD temperature must be > 0, got {temperature}.")
        self.ce_weight = float(ce_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.warmup = int(warmup)
        # Start at the end of warmup by default so a trainer that never
        # calls update_epoch() still gets the full (un-ramped) DKD loss.
        self.current_epoch = self.warmup

    def update_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    @staticmethod
    def _get_gt_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    @staticmethod
    def _get_other_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    @staticmethod
    def _cat_mask(t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def _dkd_loss(self, logits_student, logits_teacher, target):
        """Strictly reproduce official dkd_loss function."""
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        T = self.temperature

        pred_student = F.softmax(logits_student / T, dim=1)
        pred_teacher = F.softmax(logits_teacher / T, dim=1)
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student.clamp(min=1e-8))

        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (T * T)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / T - 1000.0 * gt_mask.float(), dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / T - 1000.0 * gt_mask.float(), dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
            * (T * T)
            / target.shape[0]
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_output: (B, C, H, W)
            teacher_output: (B, C, H, W)
            target: (B, H, W) long
        """
        loss_ce = self.ce_weight * F.cross_entropy(student_output, target.long())

        B, C, H, W = student_output.shape
        s = student_output.permute(0, 2, 3, 1).reshape(-1, C)
        t = teacher_output.permute(0, 2, 3, 1).reshape(-1, C).detach()
        tgt = target.long().reshape(-1)

        valid = (tgt >= 0) & (tgt < C)
        if valid.sum() > 0:
            loss_dkd = self._dkd_loss(s[valid], t[valid], tgt[valid])
        else:
            loss_dkd = torch.tensor(0.0, device=student_output.device)

        warmup_factor = min(self.current_epoch / max(self.warmup, 1), 1.0) if self.warmup > 0 else 1.0
        return loss_ce + warmup_factor * loss_dkd
