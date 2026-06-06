# Reference: https://github.com/dropreg/R-Drop
# Paper:     https://arxiv.org/abs/2106.14448
"""R-Drop: Regularized Dropout for Neural Networks (Wu et al., NeurIPS 2021).

Algorithm (paper Sec. 2):

    For each input x, perform two stochastic forward passes through the
    model.  Because dropout is active during training, the two outputs
    p1, p2 differ.  Add a *symmetric KL* regulariser between the two
    output distributions:

        L_KL(x) = 0.5 * (KL(p1 || p2) + KL(p2 || p1))

    On labeled data the total loss is

        L = 0.5 * (L_sup(p1) + L_sup(p2)) + alpha * L_KL(x_labeled)

    The semi-supervised extension applies the *same* L_KL term to
    unlabeled inputs as a consistency regulariser, with a sigmoid
    ramp-up on the unlabeled weight.

Notes:
    * R-Drop is meaningless without stochastic forward variation, so this
      class raises if the wrapped model contains no ``Dropout`` /
      ``Dropout2d`` / ``Dropout3d`` modules.  No silent fallback.
    * KL is computed per-pixel and reduced via mean — appropriate for
      dense prediction (segmentation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import get_current_consistency_weight


def _has_dropout(model: nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            if getattr(m, 'p', 0.0) > 0.0:
                return True
    return False


def _sym_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two logit tensors (segmentation: (B,C,H,W))."""
    p_log = F.log_softmax(p_logits, dim=1)
    q_log = F.log_softmax(q_logits, dim=1)
    p = p_log.exp()
    q = q_log.exp()
    # KL(p || q) and KL(q || p), summed over classes, averaged over BHW.
    kl_pq = (p * (p_log - q_log)).sum(dim=1).mean()
    kl_qp = (q * (q_log - p_log)).sum(dim=1).mean()
    return 0.5 * (kl_pq + kl_qp)


class RDrop(BaseSemiMethod):
    """R-Drop: symmetric KL between two stochastic forward passes.

    Args:
        model: Segmentation model.  Must contain at least one active
            Dropout / Dropout2d / Dropout3d module (otherwise R-Drop has
            no signal and this class raises).
        device: Torch device.
        alpha: Weight of the labeled R-Drop term (paper uses 5.0 for
            image classification, 1.0 for translation; 1.0 is a safe
            default for medical 2D seg).
        consistency_weight: Weight of the unlabeled R-Drop term (default
            1.0, ramped via sigmoid_rampup over ``rampup_epochs``).
        rampup_epochs: Sigmoid ramp-up length.
        img_size: Image spatial size (unused by R-Drop itself).

    Raises:
        ValueError: If the wrapped model has no active Dropout layers.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 alpha: float = 1.0,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.alpha = float(alpha)

    def build(self) -> None:
        if not _has_dropout(self.model):
            raise ValueError(
                "RDrop requires at least one Dropout/Dropout2d/Dropout3d "
                "module with p>0 in the model — two forward passes would "
                "otherwise be deterministic and the regulariser would "
                "collapse to zero.  Enable dropout in your model config "
                "(e.g. add a Dropout2d in the decoder/head)."
            )

    @staticmethod
    def _take_first(out):
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

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

        # --- Two stochastic forward passes on labeled data ---
        pred_l1 = self._take_first(self.model(images_l))
        pred_l2 = self._take_first(self.model(images_l))

        sup_loss = 0.5 * (criterion(pred_l1, labels) + criterion(pred_l2, labels))
        rdrop_l = _sym_kl(pred_l1, pred_l2)

        # --- Two stochastic forward passes on unlabeled data ---
        pred_u1 = self._take_first(self.model(images_u))
        pred_u2 = self._take_first(self.model(images_u))
        rdrop_u = _sym_kl(pred_u1, pred_u2)

        w = get_current_consistency_weight(
            epoch, self.consistency_weight, self.rampup_epochs)
        total_loss = sup_loss + self.alpha * rdrop_l + w * rdrop_u

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            # Report the unlabeled R-Drop term as "unsup_loss" for the
            # training loop's standard accounting; the labeled R-Drop term
            # is broken out separately.
            "unsup_loss": rdrop_u.item(),
            "rdrop_labeled": rdrop_l.item(),
            "w": w,
        }

    def get_eval_model(self) -> nn.Module:
        return self.model
