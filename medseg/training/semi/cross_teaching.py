# Reference: https://github.com/HiLab-git/SSL4MIS
# Paper:     https://arxiv.org/abs/2112.04894
"""Cross-Teaching between CNN and Transformer (Luo et al., MIDL 2022).

Algorithm:
    Train two heterogeneous segmentation models (typically a CNN and a
    Transformer) in parallel.  On labeled data each model is trained
    with the standard supervised loss.  On unlabeled data each model is
    supervised by the *argmax pseudo-label* produced by the other model
    — there is no EMA teacher and no confidence threshold; the diversity
    of the two architectures is the regulariser.

        L_total = L_sup(model_cnn) + L_sup(model_tf)
                  + lambda(t) * ( CE(model_cnn(x_u), argmax(model_tf(x_u)).detach())
                                 + CE(model_tf(x_u),  argmax(model_cnn(x_u)).detach()) )

The two models are heterogeneous, so they need independent optimizers.
Following the project's existing convention (see CPS), the second model's
optimizer is exposed via :py:meth:`extra_optimizers` and driven by the
training loop.

The second model is constructed from a user-supplied ``second_model``
config block (encoder / decoder / bottleneck / num_classes / img_size /
architecture).  The class internally calls :func:`build_model` to
materialise it — this keeps :class:`BaseSemiMethod` decoupled from
specific architectures while still letting the user wire up any
combination the project's model builder supports (e.g. ResNet-UNet +
SwinUNet, ConvNeXt + MissFormer, etc.).
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .base import BaseSemiMethod
from .utils import get_current_consistency_weight, pseudo_label_hard


class CrossTeaching(BaseSemiMethod):
    """Cross-Teaching between CNN and Transformer (CTBCT).

    Args:
        model: Primary model (e.g. the CNN branch).
        device: Torch device.
        second_model: Configuration dict for the second branch.  Must be
            a valid ``model`` config consumable by
            :func:`medseg.model_builder.build_model`.  Required — no
            silent default; the whole point of CTBCT is heterogeneity.
        consistency_weight: Maximum cross-teaching loss weight (default 1.0).
        rampup_epochs: Sigmoid ramp-up length for the cross term.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 second_model: Optional[Dict[str, Any]] = None,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        if second_model is None:
            raise ValueError(
                "CrossTeaching requires `semi.params.second_model` to be "
                "specified — it is the config dict for the second branch "
                "and is consumed by medseg.model_builder.build_model.  No "
                "silent default is provided because the heterogeneity of "
                "the two branches is essential to the method."
            )
        if not isinstance(second_model, dict):
            raise TypeError(
                "`second_model` must be a dict (model config), got "
                f"{type(second_model).__name__}."
            )
        self.second_model_cfg = copy.deepcopy(second_model)
        self.model2: nn.Module = None
        self._model2_opt: torch.optim.Optimizer = None

    # ------------------------------------------------------------------ build
    def build(self) -> None:
        # Local import to avoid an import cycle (model_builder imports many
        # submodules that may pull in this one transitively in the future).
        from medseg.model_builder import build_model

        # build_model accepts both {"model": {...}} and a bare model dict.
        cfg = {"model": self.second_model_cfg}
        self.model2 = build_model(cfg)
        self.model2.to(self.device)

    # --------------------------------------------------- optimizer plumbing
    def extra_params(self):
        # model2 has its own optimizer; do not also fuse into the main one.
        return []

    def extra_optimizers(self, lr: float = 1e-4):
        if self._model2_opt is None:
            if self.model2 is None:
                raise RuntimeError(
                    "CrossTeaching.extra_optimizers() called before build()."
                )
            self._model2_opt = torch.optim.AdamW(
                self.model2.parameters(), lr=lr, weight_decay=1e-4,
            )
        return [(self._model2_opt, "model2_opt")]

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _take_first(out):
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    # ---------------------------------------------------------- train_step
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
                "CrossTeaching.train_step() called but model2's optimizer "
                "is not built — the training loop must call "
                "`semi_method.extra_optimizers(lr=...)` first."
            )
        self.model.train()
        self.model2.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # --- Supervised on labeled (each branch on its own) ---
        pred1_l = self._take_first(self.model(images_l))
        pred2_l = self._take_first(self.model2(images_l))
        sup_loss1 = criterion(pred1_l, labels)
        sup_loss2 = criterion(pred2_l, labels)

        # --- Cross-teaching on unlabeled ---
        pred1_u = self._take_first(self.model(images_u))
        pred2_u = self._take_first(self.model2(images_u))

        with torch.no_grad():
            pseudo_from_1 = pseudo_label_hard(pred1_u.detach())
            pseudo_from_2 = pseudo_label_hard(pred2_u.detach())

        # Each model is trained on the OTHER's pseudo-label.
        ct_loss1 = F.cross_entropy(pred1_u, pseudo_from_2)
        ct_loss2 = F.cross_entropy(pred2_u, pseudo_from_1)

        w = get_current_consistency_weight(
            epoch, self.consistency_weight, self.rampup_epochs)
        total_loss = (sup_loss1 + sup_loss2) + w * (ct_loss1 + ct_loss2)

        optimizer.zero_grad()
        self._model2_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1.0)
        optimizer.step()
        # model2's optimizer.step() is invoked by the training loop via
        # extra_optimizers (same contract as CPS).

        return {
            "loss": total_loss.item(),
            "sup_loss": (sup_loss1.item() + sup_loss2.item()) / 2.0,
            "unsup_loss": (ct_loss1.item() + ct_loss2.item()) / 2.0,
            "w": w,
        }

    def get_eval_model(self) -> nn.Module:
        # Evaluate the primary (CNN) branch by default — symmetric with CPS.
        return self.model
