"""AdvEnt: Adversarial Entropy Minimization (CVPR 2019).

# Paper: https://arxiv.org/abs/1811.12833
# Reference: https://github.com/valeoai/ADVENT

Algorithm summary (from the paper):
    AdvEnt formulates UDA for semantic segmentation as alignment of the
    weighted self-information maps  I_x = - p * log(p)  between source
    and target domains via an adversarial domain discriminator, plus a
    direct entropy-minimisation term on the target predictions.
    Eq.3 of the paper defines the per-pixel normalised entropy
        E_x = -1/log(C) * sum_c p_c log(p_c)
    and the discriminator is a fully-convolutional 4-layer
    LeakyReLU/Conv2d network operating on the entropy map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


def prob_2_entropy(prob: torch.Tensor) -> torch.Tensor:
    """Convert probability map to entropy map.
    Ref: ADVENT/advent/utils/loss.py prob_2_entropy()
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / torch.log2(torch.tensor(c, dtype=torch.float32, device=prob.device))


class GradientReversalFn(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        return -lambda_ * grads, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambda_)


@LOSS_REGISTRY.register("advent")
class AdvEntLoss(nn.Module):
    """Adversarial Entropy Minimization for domain adaptation.

    Vu et al., CVPR 2019.
    Official source: https://github.com/valeoai/ADVENT

    Key components from official code:
    - Entropy minimization on target domain
    - Adversarial training with domain discriminator
    - Input to discriminator: entropy map (not raw softmax)
    - Uses BCE loss for adversarial training (not MSE)

    loss = lambda_seg * CE_source
         + lambda_adv * BCE(D(entropy(target_pred)), source_label)
         + lambda_ent * entropy_loss(target_pred)
    """

    def __init__(
        self,
        entropy_weight: float = 0.001,
        adversarial_weight: float = 0.001,
        num_classes: int = 5,
        grl_lambda: float = 1.0,
        grl_gamma: float = 10.0,
        **kwargs
    ):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.adversarial_weight = adversarial_weight
        self.num_classes = num_classes
        # Initial GRL coefficient. Will be updated automatically by
        # ``update_epoch`` using the paper's schedule
        # (Eq. 9 of DANN, also used by AdvEnt for the adversarial GRL):
        #     lambda(p) = 2 / (1 + exp(-gamma * p)) - 1,  p = epoch / total_epochs
        self.grl_lambda = grl_lambda
        self.grl_gamma = grl_gamma
        # Default total epochs; ``update_epoch`` falls back to this when the
        # caller has not set _total_epochs explicitly.
        self._total_epochs = 100
        self.domain_classifier = self._build_classifier()

    # ------------------------------------------------------------------
    # GRL lambda scheduling (paper Eq. 9 / DANN schedule)
    # ------------------------------------------------------------------
    def set_lambda(self, epoch: int, total_epochs: int):
        """Update GRL lambda using ``2/(1+e^{-gamma p}) - 1``."""
        p = float(epoch) / max(1, total_epochs)
        self.grl_lambda = float(
            2.0 / (1.0 + torch.exp(torch.tensor(-self.grl_gamma * p))).item() - 1.0
        )

    def update_epoch(self, epoch: int):
        """Auto-called by the training loop each epoch.

        Uses ``self._total_epochs`` (default 100) so the GRL coefficient ramps
        up over training following the paper's schedule.
        """
        self.set_lambda(epoch, self._total_epochs)

    def _build_classifier(self):
        """Fully-convolutional 4-conv domain discriminator.

        Re-implemented from the paper description: a stack of
        stride-2 Conv2d/LeakyReLU(0.2) blocks with the standard
        ndf=64 channel-doubling schedule (64 -> 128 -> 256 -> 512)
        and a final 1-channel conv head producing per-patch
        domain logits.
        """
        ndf = 64
        return nn.Sequential(
            nn.Conv2d(self.num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=1),
        )

    def _compute_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute *normalised* prediction entropy (paper Eq. 3).

            E_x = -1/log(C) * sum_c p_c log p_c

        Dividing by log(C) keeps the per-pixel value in [0, 1] regardless of
        the number of classes, so the loss magnitude does not change when the
        problem dimensionality changes.
        """
        n, c, _, _ = predictions.shape
        prob = F.softmax(predictions, dim=1)
        log_prob = torch.log2(prob + 1e-30)
        # log2(C) for the normalisation so the result lies in [0, 1].
        norm = torch.log2(torch.tensor(c, dtype=prob.dtype, device=prob.device))
        entropy = -(prob * log_prob).sum(dim=1) / norm
        return entropy.mean()

    def forward(
        self,
        source_pred: Optional[torch.Tensor] = None,
        target_pred: Optional[torch.Tensor] = None,
        source_labels: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            source_pred: Source domain predictions (B, C, H, W)
            target_pred: Target domain predictions (B, C, H, W)
            source_labels: Source domain ground truth
            labeled_loss: Supervised loss on source
        """
        total_loss = torch.tensor(0.0, device=target_pred.device if target_pred is not None else next(self.parameters()).device)

        # Entropy minimization on target
        if target_pred is not None:
            entropy_loss = self._compute_entropy(target_pred)
            total_loss = total_loss + self.entropy_weight * entropy_loss

        # Adversarial loss with entropy maps as discriminator input
        if source_pred is not None and target_pred is not None:
            # Convert softmax probabilities to entropy maps
            # Ref: ADVENT train_UDA.py: d_main(prob_2_entropy(F.softmax(pred)))
            source_entropy = prob_2_entropy(F.softmax(source_pred, dim=1))
            target_entropy = prob_2_entropy(F.softmax(target_pred, dim=1))

            # Source entropy -> source label (0), target entropy -> target label (1)
            # Use gradient reversal on source side for adversarial game
            source_domain_pred = self.domain_classifier(
                gradient_reversal(source_entropy, self.grl_lambda)
            )
            target_domain_pred = self.domain_classifier(
                gradient_reversal(target_entropy, self.grl_lambda)
            )

            # BCE loss (official uses bce_loss, not MSE)
            # Ref: ADVENT/advent/utils/func.py bce_loss()
            source_domain_loss = F.binary_cross_entropy_with_logits(
                source_domain_pred,
                torch.zeros_like(source_domain_pred),
            )
            target_domain_loss = F.binary_cross_entropy_with_logits(
                target_domain_pred,
                torch.ones_like(target_domain_pred),
            )

            adversarial_loss = (source_domain_loss + target_domain_loss) / 2.0
            total_loss = total_loss + self.adversarial_weight * adversarial_loss

        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss

        return total_loss
