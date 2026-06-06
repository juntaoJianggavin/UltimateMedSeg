"""Tent: Fully Test-Time Adaptation by Entropy Minimization (ICLR 2021).

# Paper: https://arxiv.org/abs/2006.10726
# Reference: https://github.com/DequanWang/tent

Algorithm summary (from the paper):
    Tent adapts a pretrained model at test time by minimizing the Shannon
    entropy of its predictions on the unlabeled target stream.  Only the
    affine parameters (weight / bias) of the batch-norm layers are updated;
    every other parameter is frozen.  Crucially, BN runs in *train* mode at
    test time but with ``track_running_stats=False`` so the layer uses
    per-batch statistics rather than the source-domain running averages.
    The per-pixel loss is

        L_ent(x) = - sum_c softmax(f(x))_c * log_softmax(f(x))_c

    averaged over pixels (and the batch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("tent")
class TentLoss(nn.Module):
    """Tent: Fully Test-Time Adaptation by Entropy Minimization.

    Wang et al., ICLR 2021 (Spotlight).
    Reference implementation (not copied):
        https://github.com/DequanWang/tent

    To use this loss correctly you MUST first call
    ``TentLoss.configure_model(model)`` on the network so that:
        * all non-BN parameters are frozen,
        * BN affine parameters become trainable,
        * BN layers are switched to train-mode with
          ``track_running_stats=False`` (use per-batch statistics).

    Without that call, gradient flow through the full network defeats the
    purpose of Tent and the adapted model degrades.
    """

    def __init__(
        self,
        entropy_weight: float = 1.0,
        bn_only: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.entropy_weight = entropy_weight
        # bn_only is informational: it does not modify the network on its own
        # — the user must call configure_model() to actually freeze non-BN
        # weights. We keep the flag so it appears in checkpoints / configs
        # and to make the intent explicit.
        self.bn_only = bn_only

    # ------------------------------------------------------------------
    # Model configuration helper (paper-required setup)
    # ------------------------------------------------------------------
    @staticmethod
    def configure_model(model: nn.Module) -> nn.Module:
        """Configure ``model`` for Tent test-time adaptation.

        Implements the paper's required setup:

        1. Set all parameters ``requires_grad=False``.
        2. For each BN-style layer (BatchNorm1d/2d/3d, SyncBatchNorm):
              * switch to ``train()`` mode,
              * disable running-stats tracking
                (``track_running_stats=False``, ``running_mean=None``,
                ``running_var=None``),
              * re-enable gradients for ``weight`` and ``bias`` only.

        Returns the same model instance for convenience.
        """
        # Freeze everything first.
        for p in model.parameters():
            p.requires_grad_(False)

        # Then re-enable gradients on BN affines only.
        bn_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )
        for m in model.modules():
            if isinstance(m, bn_types):
                m.train()
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                if m.bias is not None:
                    m.bias.requires_grad_(True)
        return model

    @staticmethod
    def collect_params(model: nn.Module):
        """Return the BN affine params + their names, as in the official repo.

        Useful when constructing the optimizer:
            params, _ = TentLoss.collect_params(model)
            opt = torch.optim.Adam(params, lr=1e-3)
        """
        params, names = [], []
        bn_types = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )
        for nm, m in model.named_modules():
            if isinstance(m, bn_types):
                for pname, p in m.named_parameters(recurse=False):
                    if pname in ("weight", "bias"):
                        params.append(p)
                        names.append(f"{nm}.{pname}")
        return params, names

    # ------------------------------------------------------------------
    # Entropy loss
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_entropy(predictions: torch.Tensor) -> torch.Tensor:
        """Per-pixel softmax entropy averaged over pixels.

        Paper Eq.: H(y) = - sum_c softmax(x)_c * log softmax(x)_c
        """
        prob = F.softmax(predictions, dim=1)
        log_prob = F.log_softmax(predictions, dim=1)
        entropy = -(prob * log_prob).sum(dim=1)
        return entropy.mean()

    def forward(
        self,
        target_pred: torch.Tensor,
        labeled_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Entropy minimization on the target predictions.

        NOTE: For paper-correct Tent behaviour, ``TentLoss.configure_model``
        MUST have been called on the segmentation network before training
        starts. Otherwise this loss still runs but it gradient-updates the
        full network, which is not Tent.
        """
        entropy_loss = self._compute_entropy(target_pred)
        total_loss = self.entropy_weight * entropy_loss
        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss
