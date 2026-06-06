# Reference: https://github.com/charlesCXK/TorchSemiSeg
# Paper:     https://arxiv.org/abs/2106.01226
"""Cross Pseudo Supervision (Chen et al., CVPR 2021).

Two models with the same architecture but different random initialisations
are trained jointly.  Each model's argmax pseudo-label supervises the other:

    L_total = (L_sup^1 + L_sup^2) + lambda * (L_cps^{1->2} + L_cps^{2->1})

Faithful-to-paper notes (read alongside Sec. 3.3 of the paper and
``TorchSemiSeg/exp.voc/voc8.res50v3+.CPS/train.py``):

* The two networks have **independent optimizers** (the official repo
  literally creates ``optimizer_l`` and ``optimizer_r`` and calls
  ``step()`` on both per iteration).  We expose ``model2``'s optimizer
  via :py:meth:`extra_optimizers` so the training loop drives it
  separately rather than fusing both parameter sets into one optimizer.
* The two networks are initialised with **different schemes** so that
  their pseudo-labels disagree early on (the paper calls this "different
  initialisation").  We follow the common practice of seeding ``model1``
  with Kaiming-normal and ``model2`` with Xavier-normal.  Pretrained
  encoder weights are preserved by default (controlled by
  ``reinit_encoder``).
* The supervised term is NOT multiplied by lambda.
* ``cps_weight`` is the *target* lambda; the effective weight at epoch
  ``e`` is ``lambda(e) = cps_weight * sigmoid_rampup(e, rampup_epochs)``.
* Paper-recommended targets: 1.5 (PASCAL VOC) / 6.0 (Cityscapes).  The
  default 1.0 here is a conservative starting point for medical 2D seg.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import get_current_consistency_weight, pseudo_label_hard


def _kaiming_init_(module: nn.Module) -> None:
    """Kaiming-normal init for Conv/Linear, ones/zeros for norm layers."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _xavier_init_(module: nn.Module) -> None:
    """Xavier-normal init for Conv/Linear, ones/zeros for norm layers."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class CrossPseudoSupervision(BaseSemiMethod):
    """Cross Pseudo Supervision (CPS).

    Args:
        model: Primary model (model1).
        device: Torch device.
        cps_weight: Target lambda for the CPS term (default 1.0).
        rampup_epochs: Epochs to ramp CPS weight from 0 (default 40).
        reinit_encoder: If True the encoder of each model is also re-initialised
            (destroys pretrained weights).  Default False to preserve any
            pretrained encoder; only the bottleneck / decoder / head are
            re-init'd, giving the two models diverging predictions while
            keeping a good feature extractor.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 cps_weight: float = 1.0,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 reinit_encoder: bool = False,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.cps_weight = cps_weight
        self.reinit_encoder = reinit_encoder
        self.model2 = None
        self._model2_opt = None

    # ------------------------------------------------------------------ build
    def _reinit(self, model: nn.Module, init_fn) -> None:
        """Re-init non-encoder parts of ``model`` (or everything if requested)."""
        if self.reinit_encoder or not hasattr(model, "encoder"):
            init_fn(model)
            return
        # Re-init bottleneck + decoder + head while preserving encoder.
        for name in ("bottleneck", "decoder", "head"):
            sub = getattr(model, name, None)
            if sub is not None:
                init_fn(sub)

    def build(self) -> None:
        # Independent random init for model1 (Kaiming).  Pretrained encoder
        # weights are preserved unless ``reinit_encoder=True``.
        self._reinit(self.model, _kaiming_init_)

        # Second model: same architecture, different init scheme (Xavier).
        self.model2 = copy.deepcopy(self.model)
        self._reinit(self.model2, _xavier_init_)
        self.model2.to(self.device)

    # --------------------------------------------------- optimizer plumbing
    def extra_params(self):
        # IMPORTANT: do NOT return model2 params here, otherwise the main
        # optimizer also owns them and we'd be double-stepping.  The paper /
        # TorchSemiSeg uses TWO independent optimizers — model2 gets its own,
        # exposed via :py:meth:`extra_optimizers`.
        return []

    def extra_optimizers(self, lr: float = 1e-4):
        """Build model2's optimizer lazily so it can pick up the actual LR.

        We mirror the project's AdamW + weight_decay=1e-4 default; that
        matches the optimiser type used by the main optimiser in
        ``semi_train.py`` and so keeps both branches symmetric (TorchSemiSeg
        uses SGD+poly externally; we deliberately follow the project's
        optimiser convention rather than hard-coding a different one).
        """
        if self._model2_opt is None:
            if self.model2 is None:
                raise RuntimeError(
                    "CPS.extra_optimizers() called before build() — model2 "
                    "has not been constructed yet."
                )
            self._model2_opt = torch.optim.AdamW(
                self.model2.parameters(), lr=lr, weight_decay=1e-4,
            )
        return [(self._model2_opt, "model2_opt")]

    # ------------------------------------------------------------- training
    def train_step(
        self,
        labeled_batch: Dict[str, Any],
        unlabeled_batch: Dict[str, Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        if self._model2_opt is None:
            raise RuntimeError(
                "CPS.train_step() called but model2's optimizer has not "
                "been built — the training loop must call "
                "`semi_method.extra_optimizers(lr=...)` first."
            )
        self.model.train()
        self.model2.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # --- Supervised losses ---
        pred1_l = self.model(images_l)
        pred2_l = self.model2(images_l)
        if isinstance(pred1_l, (list, tuple)):
            pred1_l = pred1_l[0]
        if isinstance(pred2_l, (list, tuple)):
            pred2_l = pred2_l[0]

        sup_loss1 = criterion(pred1_l, labels)
        sup_loss2 = criterion(pred2_l, labels)

        # --- Cross pseudo supervision on unlabeled data ---
        pred1_u = self.model(images_u)
        pred2_u = self.model2(images_u)
        if isinstance(pred1_u, (list, tuple)):
            pred1_u = pred1_u[0]
        if isinstance(pred2_u, (list, tuple)):
            pred2_u = pred2_u[0]

        # Generate hard pseudo-labels (detach to stop gradient flow into the
        # "teacher" branch within each direction).
        with torch.no_grad():
            pseudo1 = pseudo_label_hard(pred1_u.detach())
            pseudo2 = pseudo_label_hard(pred2_u.detach())

        cps_loss1 = F.cross_entropy(pred1_u, pseudo2)
        cps_loss2 = F.cross_entropy(pred2_u, pseudo1)

        # CPS Eq. (6): lambda is applied ONLY to the CPS term.
        w = get_current_consistency_weight(epoch, self.cps_weight, self.rampup_epochs)
        total_loss = (sup_loss1 + sup_loss2) + w * (cps_loss1 + cps_loss2)

        # Both optimisers see the same combined loss but each only steps
        # over *its own* parameters (the gradients flow into both via
        # autograd because both models contributed to the loss).
        optimizer.zero_grad()
        self._model2_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1.0)
        optimizer.step()
        # model2's optimizer.step() is called by the training loop
        # (semi_train.py: ``for extra_opt, _ in extra_optimizers: step()
        # then zero_grad()``).  We leave the grads in place for the loop.

        return {
            "loss": total_loss.item(),
            "sup_loss": (sup_loss1.item() + sup_loss2.item()) / 2,
            "unsup_loss": (cps_loss1.item() + cps_loss2.item()) / 2,
            "w": w,
        }

    def get_eval_model(self) -> nn.Module:
        return self.model
