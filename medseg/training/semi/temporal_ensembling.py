"""Temporal Ensembling (Laine & Aila, ICLR 2017).

Paper: https://arxiv.org/abs/1610.02242
Reference implementation (not copied): https://github.com/smlaine2/tempens

Algorithm (paper Section 2.2, "Temporal Ensembling"):
    Maintain, per-sample, a running ensemble of past softmax predictions::

        Z_i  = alpha * Z_i + (1 - alpha) * softmax(f(x_i))            # raw EMA
        z_i  = Z_i / (1 - alpha**t)                                    # bias-correction

    where ``t`` is the number of updates applied to ``Z_i``.  The consistency
    target is the bias-corrected ``z_i`` (detached), and the unsupervised
    loss is the MSE between the *current* softmax output and ``z_i``::

        L_us = w(t) * mean_i  ||softmax(f(x_i)) - z_i||^2

    Compared to Mean Teacher, the EMA is over *predictions* (indexed by
    sample), not over *model weights*.

Implementation notes:
    * We index ``Z`` by the dataset-provided ``case_name`` string when it is
      present in the batch dict.  If not available we fall back to a pure
      *batch-local* temporal EMA that still produces a valid target but
      cannot benefit from cross-epoch accumulation.
    * Since segmentation predictions are large 4-D tensors, ``Z`` is stored
      on CPU and only the rows for the current batch are moved to GPU each
      step -- this keeps memory bounded for large unlabeled pools.
    * Spatial size is taken from the *first* time a sample is seen; if the
      data loader returns a different size on a later epoch (e.g. random
      crop), the stored ensemble is reset for that sample.
"""

import warnings
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSemiMethod
from .utils import (
    get_current_consistency_weight, get_strong_augmentation,
)


class _PerSampleEMABank:
    """Per-sample EMA of softmax predictions with bias correction.

    Stores ``Z[id] = sum_k alpha^k * (1-alpha) * p_k`` and the visit count
    so the unbiased target is ``Z[id] / (1 - alpha**count[id])``.
    """

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._z: Dict[str, torch.Tensor] = {}
        self._n: Dict[str, int] = {}

    def get_target(self, sample_id: str, current_soft: torch.Tensor) -> torch.Tensor:
        """Return bias-corrected target tensor on the same device as ``current_soft``.

        If the sample has never been seen (or shape changed), the *current*
        softmax is used as the target -- this makes the first-pass
        consistency loss exactly zero, matching the paper's "Z=0 init" trick
        plus the ``1 - alpha**t`` correction in the limit ``t=1``.
        """
        prev = self._z.get(sample_id)
        if prev is None or prev.shape != current_soft.shape[1:]:
            return current_soft.detach()
        n = self._n[sample_id]
        # Bias-correct so the target is a proper running mean.
        correction = 1.0 - self.alpha ** max(n, 1)
        target = prev.to(current_soft.device, non_blocking=True) / max(correction, 1e-8)
        return target.detach()

    def update(self, sample_id: str, current_soft: torch.Tensor) -> None:
        """In-place EMA update with the just-observed (detached) softmax."""
        new = current_soft.detach().to('cpu')
        prev = self._z.get(sample_id)
        if prev is None or prev.shape != new.shape:
            # First time seen -> initialise with (1 - alpha) * p (count := 1)
            self._z[sample_id] = (1.0 - self.alpha) * new
            self._n[sample_id] = 1
        else:
            self._z[sample_id] = self.alpha * prev + (1.0 - self.alpha) * new
            self._n[sample_id] += 1


class TemporalEnsembling(BaseSemiMethod):
    """Per-sample EMA of predictions; MSE consistency vs bias-corrected target.

    Args:
        model: Student network.
        device: Torch device.
        consistency_weight: Maximum ``w_max`` for the unsupervised MSE.
        rampup_epochs: Sigmoid ramp-up length, in epochs.
        ensemble_alpha: EMA coefficient for the per-sample prediction bank
            (paper: 0.6).  Larger -> slower-moving target.
        use_strong_aug: Apply strong augmentation to the student before its
            forward (the EMA is built over the *clean* prediction).
        img_size: Image spatial size for strong augmentation.

    Batch key for sample ids: ``case_name`` (list[str] after collate) is
    used when present.  Fall back to a batch-local EMA otherwise.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 ensemble_alpha: float = 0.6,
                 use_strong_aug: bool = True,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.ensemble_alpha = float(ensemble_alpha)
        self.use_strong_aug = bool(use_strong_aug)
        self.strong_aug = None
        self._bank = _PerSampleEMABank(self.ensemble_alpha)
        # Lazy batch-local fallback target, when no sample ids are provided.
        self._batch_local_target: Optional[torch.Tensor] = None
        self._warned_no_ids = False

    def build(self) -> None:
        self.strong_aug = get_strong_augmentation(self.img_size)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_ids(batch: Dict[str, Any]) -> Optional[List[str]]:
        ids = batch.get('case_name', None)
        if ids is None:
            return None
        # default_collate turns a list[str] into a list[str] already.
        if isinstance(ids, (list, tuple)):
            return [str(x) for x in ids]
        if isinstance(ids, torch.Tensor):
            return [str(int(x)) for x in ids.view(-1).tolist()]
        return [str(ids)]

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.model.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # --- supervised loss on labeled data ---
        pred_l = self._forward(images_l)
        sup_loss = criterion(pred_l, labels)

        # --- "current" prediction on unlabeled (clean view) for the bank ---
        with torch.no_grad():
            pred_u_clean = self._forward(images_u)
            soft_u_clean = F.softmax(pred_u_clean, dim=1)

        # --- student prediction (possibly with strong aug) ---
        if self.use_strong_aug:
            x_student = self.strong_aug(images_u)
        else:
            x_student = images_u
        pred_u_student = self._forward(x_student)
        soft_u_student = F.softmax(pred_u_student, dim=1)

        # --- build per-sample EMA target ---
        ids = self._extract_ids(unlabeled_batch)
        if ids is None:
            if not self._warned_no_ids:
                warnings.warn(
                    "TemporalEnsembling: unlabeled batch has no 'case_name' "
                    "field; falling back to batch-local EMA target (no "
                    "cross-epoch accumulation).",
                    RuntimeWarning, stacklevel=2)
                self._warned_no_ids = True
            # Batch-local target: EMA over the most recent batch's clean soft.
            if self._batch_local_target is None \
                    or self._batch_local_target.shape != soft_u_clean.shape:
                target = soft_u_clean.detach()
            else:
                a = self.ensemble_alpha
                target = (a * self._batch_local_target
                          + (1.0 - a) * soft_u_clean.detach()).detach()
            self._batch_local_target = target.detach().to(soft_u_clean.device)
        else:
            # Per-sample target from the bank; update *after* loss is computed
            target = torch.stack([
                self._bank.get_target(sid, soft_u_clean[i])
                for i, sid in enumerate(ids)
            ], dim=0)

        cons_loss = F.mse_loss(soft_u_student, target)

        w = get_current_consistency_weight(epoch, self.consistency_weight,
                                           self.rampup_epochs)
        total_loss = sup_loss + w * cons_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- update bank with the clean softmax we just observed ---
        if ids is not None:
            for i, sid in enumerate(ids):
                self._bank.update(sid, soft_u_clean[i])

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": cons_loss.item(),
            "w": w,
            "bank_size": len(self._bank._z) if ids is not None else 0,
        }

    def get_eval_model(self) -> nn.Module:
        return self.model
