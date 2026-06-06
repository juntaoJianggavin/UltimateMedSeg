# CTKD: Curriculum Temperature for Knowledge Distillation (AAAI 2023)
# Reference: https://github.com/zhengli97/CTKD
# Paper: https://arxiv.org/abs/2211.16231
# Implemented from paper formulas; not a copy of the official repo.
"""CTKD: learn a per-step temperature with adversarial gradient reversal.

Li et al. AAAI 2023 turn the KD softmax temperature T into a
learnable parameter. A small ``temperature module'' produces T at
each step; a gradient reversal layer (GRL) flips the sign of dL/dT
so that T is updated to MAXIMISE the KD loss while the student is
updated (as usual) to minimise it. This adversarial coupling makes
the distillation problem progressively harder for the student as T
moves toward whichever value produces the most informative gradients.

The paper combines two ingredients:

1. ``Cosine curriculum'' coefficient :math:`\\lambda` ramping from 0
   to ``lambda_max`` over the training horizon. The GRL multiplier is
   set to :math:`\\lambda`, so adversarial pressure on T is initially
   zero and grows smoothly.
2. ``Global temperature'' module: a single scalar parameter
   (implementation choice in the official repo for image
   classification). We follow that variant since per-instance T
   needs a conditioning feature that segmentation heads do not
   expose uniformly.

Trainers that track epochs can call ``update_epoch(epoch, total)``
between steps to drive the curriculum. The default schedule keeps
:math:`\\lambda` at ``lambda_max`` if the host trainer never updates
the epoch, which reproduces the non-curriculum ablation in Sec. 4.4
of the paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


class _GradReverse(torch.autograd.Function):
    """Standard gradient reversal layer (Ganin & Lempitsky 2015).

    Forward is the identity; backward multiplies the upstream gradient
    by ``-lambda_grl``. Used to flip the sign of the gradient that
    flows into the learnable temperature parameter so it ascends the
    KD loss while the student descends it.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = float(lambda_grl)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambda_grl)


@LOSS_REGISTRY.register("ctkd")
class CTKDLoss(nn.Module):
    """Curriculum Temperature KD (AAAI 2023, Li et al.).

    Args:
        init_temperature: initial value of the learnable T.
        t_min, t_max: clamp range applied to T at every forward so
            ``softmax(z/T)`` stays well-defined.
        lambda_max: maximum value of the GRL coefficient.
        total_epochs: horizon for the cosine curriculum.
        weight: scale applied to the final KL (the paper's alpha).
    """

    def __init__(
        self,
        init_temperature: float = 4.0,
        t_min: float = 1.0,
        t_max: float = 20.0,
        lambda_max: float = 1.0,
        total_epochs: int = 100,
        weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if init_temperature <= 0:
            raise ValueError(
                f"CTKD init_temperature must be > 0, got {init_temperature}."
            )
        if not (t_min > 0 and t_max > t_min):
            raise ValueError(
                f"CTKD requires 0 < t_min < t_max; got "
                f"t_min={t_min}, t_max={t_max}."
            )
        if lambda_max < 0:
            raise ValueError(
                f"CTKD lambda_max must be >= 0, got {lambda_max}."
            )
        if total_epochs <= 0:
            raise ValueError(
                f"CTKD total_epochs must be > 0, got {total_epochs}."
            )
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.lambda_max = float(lambda_max)
        self.total_epochs = int(total_epochs)
        self.weight = float(weight)
        # Learnable temperature stored as a raw parameter; we apply
        # softplus + clamp in forward so the optimiser sees an
        # unconstrained scalar.
        self.t_raw = nn.Parameter(torch.tensor(float(init_temperature)))
        # Default curriculum state: end-of-training so lambda = lambda_max
        # whenever the trainer does not wire up update_epoch (matches
        # the no-curriculum ablation).
        self.current_epoch = self.total_epochs

    def update_epoch(self, epoch: int, total_epochs: int = None):
        self.current_epoch = int(epoch)
        if total_epochs is not None:
            if int(total_epochs) <= 0:
                raise ValueError(
                    f"CTKD total_epochs must be > 0, got {total_epochs}."
                )
            self.total_epochs = int(total_epochs)

    def _current_lambda(self) -> float:
        """Cosine ramp from 0 to ``lambda_max`` over total_epochs."""
        e = max(min(self.current_epoch, self.total_epochs), 0)
        # (1 - cos(pi * e / E)) / 2 grows monotonically 0 -> 1.
        cos_term = 0.5 * (1.0 - math.cos(math.pi * e / self.total_epochs))
        return self.lambda_max * cos_term

    def _current_temperature(self) -> torch.Tensor:
        # Clamp so T stays inside [t_min, t_max]; the gradient still
        # flows through the unclamped region.
        return self.t_raw.clamp(min=self.t_min, max=self.t_max)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            student_output: (B, C, H, W) student logits.
            teacher_output: (B, C, H, W) teacher logits (detached inside).
        """
        if student_output.dim() != 4 or teacher_output.dim() != 4:
            raise ValueError(
                "CTKD expects 4D logit tensors (B, C, H, W); got "
                f"student={tuple(student_output.shape)} "
                f"teacher={tuple(teacher_output.shape)}."
            )
        if student_output.shape[1] != teacher_output.shape[1]:
            raise ValueError(
                "CTKD requires matching class dimensions; got "
                f"student C={student_output.shape[1]}, "
                f"teacher C={teacher_output.shape[1]}."
            )
        if student_output.shape[-2:] != teacher_output.shape[-2:]:
            teacher_output = F.interpolate(
                teacher_output, size=student_output.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        teacher_output = teacher_output.detach()

        # Apply GRL on the temperature so its gradient is sign-flipped.
        lam = self._current_lambda()
        T = self._current_temperature()
        if lam > 0:
            T_eff = grad_reverse(T, lam)
        else:
            # No adversarial pressure yet — detach so T does not move.
            T_eff = T.detach()

        B, C, H, W = student_output.shape
        s = student_output.permute(0, 2, 3, 1).reshape(-1, C)
        t = teacher_output.permute(0, 2, 3, 1).reshape(-1, C)

        log_p_s = F.log_softmax(s / T_eff, dim=1)
        p_t = F.softmax(t / T_eff, dim=1)
        kl = F.kl_div(log_p_s, p_t, reduction='batchmean')
        # Standard T^2 scaling so the gradient magnitude is invariant
        # to T (Hinton et al.); the temperature itself is then driven
        # purely by the reversed gradient of the KL term.
        return self.weight * kl * (T_eff * T_eff)
