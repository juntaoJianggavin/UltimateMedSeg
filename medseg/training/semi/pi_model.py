"""Pi-Model (Laine & Aila, ICLR 2017).

Paper: https://arxiv.org/abs/1610.02242
Reference implementation (not copied): https://github.com/smlaine2/tempens

Algorithm (paper Section 2.1, "Π-model"):
    For every unlabeled (and labeled) sample, run the network *twice* with
    independent stochastic perturbations -- independent noise and independent
    dropout masks -- producing predictions ``z`` and ``z_tilde``.  The
    unsupervised loss is the mean squared error between the two softmax
    outputs.  The total loss is

        L = CE(f(x_l, eta_l), y_l)  +  w(t) * MSE(softmax(f(x, eta1)),
                                                  softmax(f(x, eta2)))

    where ``w(t)`` follows the standard sigmoid ramp-up
    ``w(t) = w_max * exp(-5 * (1 - t/T)^2)``.  No teacher, no EMA --
    consistency is enforced between two stochastic forwards of the same net.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import (
    get_current_consistency_weight, get_strong_augmentation,
)


# Modules that we toggle ON to ensure two forwards see different stochastic
# realisations even when the surrounding caller has put the model in eval.
_DROPOUT_TYPES = (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


def _has_dropout(model: nn.Module) -> bool:
    return any(isinstance(m, _DROPOUT_TYPES) for m in model.modules())


class PiModel(BaseSemiMethod):
    """Π-model: two stochastic passes, MSE-on-softmax consistency.

    Args:
        model: Student network (same network is run twice).
        device: Torch device.
        consistency_weight: Maximum unsupervised weight ``w_max`` (default 1.0;
            the paper uses 100 for CIFAR-10 measured on the *log* scale, but
            for sigmoid ramp-up on segmentation we keep 1.0 as a safe default).
        rampup_epochs: Sigmoid ramp-up length, in epochs (default 40).
        use_strong_aug: If True (default) the *second* forward additionally
            sees a strong augmentation, which is a common segmentation
            adaptation of Π-model.  Set False to match the original paper
            (only dropout + Gaussian noise as the perturbation).
        input_noise_std: Std of additive Gaussian input noise injected on
            both forwards (paper used 0.15).
        img_size: Image spatial size for strong augmentation.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 use_strong_aug: bool = True,
                 input_noise_std: float = 0.15,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.use_strong_aug = bool(use_strong_aug)
        self.input_noise_std = float(input_noise_std)
        self.strong_aug = None
        self._warned_no_dropout = False

    def build(self) -> None:
        self.strong_aug = get_strong_augmentation(self.img_size)
        if not _has_dropout(self.model) and self.input_noise_std <= 0.0 \
                and not self.use_strong_aug:
            warnings.warn(
                "PiModel: model has no Dropout, input_noise_std=0, and "
                "use_strong_aug=False -- the two forwards will be identical "
                "and the consistency loss will be zero.",
                RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------ #
    def _perturb(self, x: torch.Tensor, use_strong: bool) -> torch.Tensor:
        if use_strong and self.strong_aug is not None:
            x = self.strong_aug(x)
        if self.input_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.input_noise_std
        return x

    def _student_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

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
        # train() ensures dropout is active for both stochastic passes
        self.model.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # --- supervised CE/Dice on labeled data ---
        pred_l = self._student_forward(images_l)
        sup_loss = criterion(pred_l, labels)

        # --- two stochastic forwards on unlabeled data ---
        # First pass: weak / noise-only perturbation
        x1 = self._perturb(images_u, use_strong=False)
        # Second pass: strong perturbation (or another noise draw if disabled)
        x2 = self._perturb(images_u, use_strong=self.use_strong_aug)

        pred1 = self._student_forward(x1)
        pred2 = self._student_forward(x2)

        # paper's unsupervised loss: MSE on softmax outputs (Eq. 4)
        # detach() on one side breaks symmetry and matches the paper's
        # ``stop_grad`` shorthand for the "target" branch in segmentation.
        cons_loss = F.mse_loss(
            F.softmax(pred1, dim=1),
            F.softmax(pred2, dim=1).detach(),
        )

        w = get_current_consistency_weight(epoch, self.consistency_weight,
                                           self.rampup_epochs)
        total_loss = sup_loss + w * cons_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": cons_loss.item(),
            "w": w,
        }

    def get_eval_model(self) -> nn.Module:
        return self.model
