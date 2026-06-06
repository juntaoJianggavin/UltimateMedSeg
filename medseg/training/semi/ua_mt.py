"""UA-MT: Uncertainty-Aware Mean Teacher (Yu et al., MICCAI 2019).

Paper: https://arxiv.org/abs/1907.07034
Reference implementation (not copied): https://github.com/yulequan/UA-MT

The original paper estimates teacher uncertainty by running ``T`` stochastic
forward passes through the teacher with **MC-Dropout** enabled (only the
``Dropout`` modules are kept in training mode; BN/etc stay in eval) and
computing the **predictive entropy** of the average softmax::

    p_bar = (1/T) * sum_t softmax(f_teacher^t(x))
    H     = - sum_c p_bar_c * log(p_bar_c)               (per-pixel entropy)
    reliable_mask = (H < H_threshold)
    L_cons = sum_pixels  reliable_mask * (p_student - p_teacher)^2

If the model has no ``Dropout`` modules MC-Dropout collapses to a single
deterministic forward — we emit a one-time warning and fall back to entropy
on a single teacher prediction.  A ``"tta"`` estimator is preserved as an
opt-in for ablations (random flip + rot variance, identical to the previous
implementation).
"""

import copy
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight, get_strong_augmentation,
)


# Modules that should remain in training mode for MC-Dropout while everything
# else (BatchNorm, etc.) stays in eval mode.
_DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


def _enable_dropout(model: nn.Module) -> int:
    """Put only the dropout modules into ``train()`` mode. Returns count."""
    n = 0
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()
            n += 1
    return n


class UncertaintyAwareMeanTeacher(BaseSemiMethod):
    """Uncertainty-Aware Mean Teacher for medical semi-supervised segmentation.

    Args:
        model: Student model.
        device: Torch device.
        ema_decay: EMA decay rate (default 0.999).
        consistency_weight: Max consistency weight (default 1.0).
        rampup_epochs: Consistency weight ramp-up epochs (default 40).
        uncertainty_threshold: Max entropy (in nats) to include in consistency.
            Following the paper, entropy is normalised to ``[0, 1]`` by
            ``log(C)`` before thresholding, so the default ``0.3`` means
            "keep pixels whose normalised predictive entropy is below 0.3".
        uncertainty_estimator: ``"mc_dropout"`` (paper, default) or ``"tta"``
            (ablation: random flip + rot variance).
        num_tta: Number of MC samples (also reused as TTA samples).
        img_size: Image spatial size.

    Backward compat:
        Old configs that pass ``num_tta=K`` and rely on the deprecated TTA
        path still work; set ``uncertainty_estimator: "tta"`` to restore it.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 ema_decay: float = 0.999,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 uncertainty_threshold: float = 0.3,
                 num_tta: int = 8,
                 uncertainty_estimator: str = "mc_dropout",
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.ema_decay = ema_decay
        self.uncertainty_threshold = uncertainty_threshold
        self.num_tta = num_tta
        estimator = str(uncertainty_estimator).lower()
        if estimator not in {"mc_dropout", "tta"}:
            raise ValueError(
                f"uncertainty_estimator must be 'mc_dropout' or 'tta', "
                f"got '{uncertainty_estimator}'")
        self.uncertainty_estimator = estimator
        self.teacher = None
        self.strong_aug = None
        self._warned_no_dropout = False

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        self.strong_aug = get_strong_augmentation(self.img_size)

    # ------------------------------------------------------------------ #
    # Uncertainty estimators                                              #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _teacher_forward(self, x):
        out = self.teacher(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    @torch.no_grad()
    def _mc_dropout_uncertainty(self, images: torch.Tensor):
        """Predictive entropy of the mean softmax over T MC-Dropout passes.

        Returns
        -------
        mean_softmax : (B, C, H, W) — used as the consistency target.
        entropy      : (B, H, W)    — normalised to ``[0, 1]`` by log(C).
        """
        # Put teacher fully in eval, then re-enable only Dropout modules.
        self.teacher.eval()
        n_drop = _enable_dropout(self.teacher)
        if n_drop == 0 and not self._warned_no_dropout:
            warnings.warn(
                "UA-MT: teacher contains no Dropout modules; MC-Dropout "
                "degenerates to a single deterministic forward pass. "
                "Use a backbone with Dropout or set "
                "uncertainty_estimator='tta' for stochastic uncertainty.",
                RuntimeWarning, stacklevel=2)
            self._warned_no_dropout = True

        T = max(1, int(self.num_tta)) if n_drop > 0 else 1
        soft_sum = None
        for _ in range(T):
            logits = self._teacher_forward(images)
            p = F.softmax(logits, dim=1)
            soft_sum = p if soft_sum is None else soft_sum + p
        mean_soft = soft_sum / float(T)

        C = mean_soft.shape[1]
        entropy = -(mean_soft * (mean_soft.clamp(min=1e-8)).log()).sum(dim=1)
        # Normalise to [0, 1]: divide by log(C) (the maximum possible entropy)
        norm = float(torch.log(torch.tensor(max(C, 2), dtype=torch.float32)))
        entropy = entropy / max(norm, 1e-8)
        # Restore eval mode on the dropout layers we enabled
        self.teacher.eval()
        return mean_soft, entropy

    @torch.no_grad()
    def _tta_uncertainty(self, images: torch.Tensor):
        """Variance of the softmax across random flip+rot augmentations.

        Kept as an opt-in ablation path. Returns ``(mean_softmax, entropy)``
        where the "entropy" channel is the normalised variance to keep the
        downstream thresholding identical.
        """
        self.teacher.eval()
        preds = []
        T = max(1, int(self.num_tta))
        for _ in range(T):
            aug_img = images.clone()
            do_flip = torch.rand(1).item() > 0.5
            if do_flip:
                aug_img = torch.flip(aug_img, dims=[3])
            k = int(torch.randint(0, 4, (1,)).item())
            aug_img = torch.rot90(aug_img, k=k, dims=[2, 3])

            logits = self._teacher_forward(aug_img)
            # Undo augmentation in prediction space
            logits = torch.rot90(logits, k=-k, dims=[2, 3])
            if do_flip:
                logits = torch.flip(logits, dims=[3])
            preds.append(F.softmax(logits, dim=1))

        preds = torch.stack(preds, dim=0)  # (T, B, C, H, W)
        mean_soft = preds.mean(dim=0)
        variance = preds.var(dim=0).mean(dim=1)  # (B, H, W)
        uncertainty = variance / (variance.max() + 1e-8)
        return mean_soft, uncertainty

    @torch.no_grad()
    def _estimate(self, images: torch.Tensor):
        if self.uncertainty_estimator == "mc_dropout":
            return self._mc_dropout_uncertainty(images)
        return self._tta_uncertainty(images)

    # ------------------------------------------------------------------ #
    # Train step                                                          #
    # ------------------------------------------------------------------ #
    def train_step(
        self,
        labeled_batch: Dict[str, Any],
        unlabeled_batch: Dict[str, Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # Supervised loss
        pred_l = self.model(images_l)
        if isinstance(pred_l, (list, tuple)):
            pred_l = pred_l[0]
        sup_loss = criterion(pred_l, labels)

        # Teacher mean prediction + uncertainty on unlabeled
        teacher_soft, uncertainty = self._estimate(images_u)
        reliable_mask = (uncertainty < self.uncertainty_threshold).float()

        # Student on strong-augmented unlabeled
        images_u_strong = self.strong_aug(images_u)
        student_pred = self.model(images_u_strong)
        if isinstance(student_pred, (list, tuple)):
            student_pred = student_pred[0]

        # Uncertainty-masked MSE consistency (paper Eq. 3)
        diff = (F.softmax(student_pred, dim=1) - teacher_soft) ** 2
        masked = diff * reliable_mask.unsqueeze(1)
        denom = reliable_mask.sum().clamp(min=1.0) * float(teacher_soft.shape[1])
        consistency_loss = masked.sum() / denom

        w = get_current_consistency_weight(epoch, self.consistency_weight, self.rampup_epochs)
        total_loss = sup_loss + w * consistency_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": consistency_loss.item(),
            "w": w,
            "reliable_ratio": reliable_mask.mean().item(),
            "estimator": self.uncertainty_estimator,
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
