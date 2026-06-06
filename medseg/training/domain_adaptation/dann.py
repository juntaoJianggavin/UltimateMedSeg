"""DANN: Domain-Adversarial Training of Neural Networks (JMLR 2016).

# Paper: https://arxiv.org/abs/1505.07818
# Reference: https://github.com/fungtion/DANN

Algorithm summary (from the paper):
    DANN learns features that are simultaneously (a) discriminative for the
    source-supervised task and (b) invariant to the domain of origin. A
    small *domain classifier* is attached to the feature extractor through
    a Gradient Reversal Layer (GRL): during forward the GRL is the
    identity; during backward it multiplies the gradient by -lambda. The
    feature extractor therefore receives a gradient that *maximises* the
    domain classifier's loss, pushing the features to a domain-confused
    region, while the domain classifier itself is trained to minimise its
    loss. For segmentation we apply global average pooling on the
    predictions / features before the domain head.

        L_total = L_task(source) + lambda * L_domain(source ∪ target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


class _GradientReversalFn(torch.autograd.Function):
    """Identity in forward; gradient multiplied by -lambda in backward."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambda_)


@LOSS_REGISTRY.register("dann")
class DANNLoss(nn.Module):
    """Domain-Adversarial Neural Network adaptation loss.

    Ganin & Lempitsky / Ganin et al., JMLR 2016.
    Reference implementation (not copied):
        https://github.com/fungtion/DANN

    Args:
        lambda_: GRL coefficient. The paper schedules
            lambda_ = 2/(1+exp(-10p)) - 1 with p in [0,1] training progress;
            users can update at runtime via ``set_lambda(epoch, total)``.
        num_classes: number of segmentation classes (= channels of
            ``source_pred`` / ``target_pred``), used when no separate
            feature tensor is supplied.
        hidden_dim: width of the MLP domain head.
        domain_weight: scalar weight applied on top of ``lambda_``.
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        num_classes: int = 5,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 256,
        domain_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.lambda_ = float(lambda_)
        self.num_classes = num_classes
        # If features are not provided at runtime, the domain classifier
        # consumes the C-channel logits map after GAP -> C-vector.
        in_dim = feature_dim if feature_dim is not None else num_classes
        self.feature_dim = in_dim
        self.hidden_dim = hidden_dim
        self.domain_weight = domain_weight
        # fungtion/DANN uses a *single-logit* sigmoid head trained with BCE,
        # not a two-way softmax with cross-entropy. We follow that convention
        # here so the adversarial objective matches the official repo.
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Lambda scheduling (paper Eq. 14)
    # ------------------------------------------------------------------
    def set_lambda(self, epoch: int, total_epochs: int, gamma: float = 10.0):
        """Set lambda using the paper's schedule, 2/(1+e^{-gamma p}) - 1."""
        p = float(epoch) / max(1, total_epochs)
        self.lambda_ = float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))) - 1.0)

    def update_epoch(self, epoch):
        # Auto-called by the training loop. Use total=100 as a reasonable
        # default if the caller did not set ``total_epochs``; lambda then
        # ramps up over the first 100 epochs.
        if not hasattr(self, "_total_epochs"):
            self._total_epochs = 100
        self.set_lambda(epoch, self._total_epochs)

    # ------------------------------------------------------------------
    # Domain feature pooling
    # ------------------------------------------------------------------
    def _pool(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        # (B, C, H, W) -> (B, C) via global average pooling.
        if x.dim() == 4:
            return F.adaptive_avg_pool2d(x, 1).flatten(1)
        if x.dim() == 2:
            return x
        return x.flatten(1)

    def _resize_if_needed(self, vec: torch.Tensor) -> torch.Tensor:
        """If the pooled feature has a different width than the head expects,
        rebuild the head lazily so the loss still works."""
        if vec.shape[1] != self.feature_dim:
            self.feature_dim = vec.shape[1]
            device = vec.device
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
            ).to(device)
        return vec

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        source_pred: Optional[torch.Tensor] = None,
        target_pred: Optional[torch.Tensor] = None,
        source_features: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            source_pred / target_pred: segmentation logits, (B, C, H, W).
                Used as a fallback when explicit features are not provided.
            source_features / target_features: optional feature maps
                (B, F, H', W') from the backbone, GAP-ed before the head.
            labeled_loss: supervised loss on source.
        """
        device = (target_pred if target_pred is not None else source_pred).device
        total_loss = torch.tensor(0.0, device=device)

        # Pool source-side feature.
        src_feat = self._pool(
            source_features if source_features is not None else source_pred
        )
        tgt_feat = self._pool(
            target_features if target_features is not None else target_pred
        )

        if src_feat is not None and tgt_feat is not None:
            src_feat = self._resize_if_needed(src_feat)
            tgt_feat = self._resize_if_needed(tgt_feat)

            # GRL on both sides: feature extractor receives the *negated*
            # domain gradient, the domain head receives the normal one.
            src_dom = self.domain_classifier(grad_reverse(src_feat, self.lambda_))
            tgt_dom = self.domain_classifier(grad_reverse(tgt_feat, self.lambda_))

            # Sigmoid + BCE objective, following fungtion/DANN
            # (single-logit head, source=0, target=1).
            src_lbl = torch.zeros_like(src_dom)
            tgt_lbl = torch.ones_like(tgt_dom)
            dom_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(src_dom, src_lbl)
                + F.binary_cross_entropy_with_logits(tgt_dom, tgt_lbl)
            )
            total_loss = total_loss + self.domain_weight * dom_loss

        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss

        return total_loss
