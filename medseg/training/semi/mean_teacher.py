# Reference: https://github.com/CuriousAI/mean-teacher
# Paper:     https://arxiv.org/abs/1703.01780
"""Mean Teacher semi-supervised segmentation.

Tarvainen & Valpola, "Mean teachers are better role models", NeurIPS 2017.

Student model is trained with supervised loss on labeled data and MSE
consistency loss against an EMA teacher on unlabeled data.

Faithful-to-paper notes:
  * The paper's perturbation is "Gaussian input noise + dropout" (Sec. 3.1
    and Algo. 1 of the NeurIPS 2017 paper / official CuriousAI repo).
    The teacher sees one noised draw of ``x``; the student sees a second,
    independent noised draw of the same ``x``.  We default to this recipe
    via ``consistency_noise="gaussian"``.  Setting it to ``"strong_aug"``
    swaps in the heavier (color jitter / blur / cutout) pipeline used by
    later FixMatch-style follow-ups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight, get_input_noise,
)


class MeanTeacher(BaseSemiMethod):
    """Mean Teacher semi-supervised method.

    Args:
        model: Student model.
        device: Torch device.
        ema_decay: EMA decay rate for teacher update (default 0.999).
        consistency_weight: Maximum consistency loss weight (default 1.0).
        rampup_epochs: Epochs to ramp up consistency weight (default 40).
        consistency_noise: ``"gaussian"`` (paper default — additive
            Gaussian input noise + internal dropout) or ``"strong_aug"``
            (heavier color/blur/cutout pipeline used by later work).
        gaussian_std: Std-dev for the Gaussian input noise (paper uses
            small values around 0.15 in input space).
        img_size: Image spatial size (only used by ``"strong_aug"``).
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 ema_decay: float = 0.999,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 consistency_noise: str = "gaussian",
                 gaussian_std: float = 0.15,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.ema_decay = ema_decay
        self.consistency_noise = consistency_noise
        self.gaussian_std = gaussian_std
        self.teacher = None
        self.noise = None

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        # Raises ValueError for unknown noise types — no silent fallback.
        self.noise = get_input_noise(
            self.consistency_noise,
            img_size=self.img_size,
            std=self.gaussian_std,
        )

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
        self.teacher.eval()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # --- Supervised loss on labeled data ---
        pred_l = self.model(images_l)
        if isinstance(pred_l, (list, tuple)):
            pred_l = pred_l[0]
        sup_loss = criterion(pred_l, labels)

        # --- Consistency on unlabeled data ---
        # Paper: teacher sees x + n_t (Gaussian), student sees x + n_s,
        # n_t and n_s are independent draws.  With "strong_aug" the teacher
        # path uses the *original* image (more in line with FixMatch / SSL4MIS
        # semi-seg follow-ups) so we keep that branch unchanged.
        if self.consistency_noise == "gaussian":
            images_u_teacher = self.noise(images_u)
        else:
            images_u_teacher = images_u

        with torch.no_grad():
            teacher_pred = self.teacher(images_u_teacher)
            if isinstance(teacher_pred, (list, tuple)):
                teacher_pred = teacher_pred[0]
            teacher_pred = teacher_pred.detach()

        images_u_student = self.noise(images_u)
        student_pred = self.model(images_u_student)
        if isinstance(student_pred, (list, tuple)):
            student_pred = student_pred[0]

        # MSE consistency loss on softmax outputs (paper Sec. 3, Eq. 2).
        consistency_loss = F.mse_loss(
            F.softmax(student_pred, dim=1),
            F.softmax(teacher_pred, dim=1),
        )

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
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
