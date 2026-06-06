# Reference: https://github.com/yelantingfeng/pyScribble
# Paper: https://arxiv.org/abs/1604.05144
"""ScribbleSup — Scribble-Supervised CNNs for Semantic Segmentation.

Lin et al., CVPR 2016.
Paper: https://arxiv.org/abs/1604.05144
Reference repository: https://github.com/yelantingfeng/pyScribble
    (3rd-party implementation; the original Caffe code is not maintained.)

Faithful-in-spirit ``light`` variant suitable for in-tree training:

    L = CE(scribble pixels)
      + lambda_crf * GatedCRF(predictions, images)
      + lambda_ent * mean( H(softmax(predictions)) )

The original paper alternates training with a graph-cut step that
propagates scribble labels to the full image, then refits the CNN on the
propagated mask. Implementing a true GraphCut inside an autograd loop is
expensive and brings a non-differentiable dependency; instead we substitute
the differentiable **Gated CRF** loss (Obukhov et al., NeurIPS 2019), which
plays the same role of "labels should be smooth where colours are smooth"
and is already shipped in this repo. A predictor-entropy-minimisation term
sharpens predictions on unlabelled pixels, matching the EM behaviour of
the original alternating optimisation.

Strict no-fallback rule:
    If ``GatedCRFLoss`` cannot be imported (e.g. someone vendored a partial
    copy) we raise rather than silently dropping the CRF term — that would
    reduce ScribbleSup-light to plain sparse CE and quietly lie about the
    method being implemented.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY

try:
    from .gated_crf import GatedCRFLoss
except ImportError as exc:  # pragma: no cover - exercised at import time
    raise ImportError(
        "ScribbleSupLoss requires GatedCRFLoss from "
        "medseg.training.weakly_supervised.gated_crf but it could not be imported "
        "({}). Refusing to fall back to a no-CRF variant — that would "
        "silently change the algorithm.".format(exc)
    )


@LOSS_REGISTRY.register("scribble_sup")
class ScribbleSupLoss(nn.Module):
    """Scribble-supervised segmentation loss (light variant).

    Args:
        ignore_index: Pixel value in ``scribbles`` to treat as unlabelled
            (default -1, matches PyTorch's CE convention).
        crf_weight: Weight on the Gated-CRF surrogate for the paper's
            graph-cut propagation step (default 0.1).
        entropy_weight: Weight on the predictor-entropy minimisation term
            applied to all pixels (default 0.05).
        kernel_size: Gated-CRF kernel size (default 3, see GatedCRFLoss
            docstring for memory implications).
        dilation: Gated-CRF dilation (default 1).
        sigma_rgb: Colour-Gaussian sigma for the CRF (default 15.0).
        sigma_xy: Spatial-Gaussian sigma for the CRF (default 100.0).
    """

    def __init__(
        self,
        ignore_index: int = -1,
        crf_weight: float = 0.1,
        entropy_weight: float = 0.05,
        kernel_size: int = 3,
        dilation: int = 1,
        sigma_rgb: float = 15.0,
        sigma_xy: float = 100.0,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.crf_weight = crf_weight
        self.entropy_weight = entropy_weight
        # We intentionally build the CRF module ourselves rather than
        # accepting one from outside, so its hyperparameters are part of
        # this loss's config surface.
        self.crf = GatedCRFLoss(
            crf_weight=1.0,           # we apply crf_weight outside.
            kernel_size=kernel_size,
            dilation=dilation,
            sigma_rgb=sigma_rgb,
            sigma_xy=sigma_xy,
        )

    @staticmethod
    def _entropy(predictions: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(predictions, dim=1)
        return -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean()

    def forward(
        self,
        predictions: torch.Tensor,
        scribbles: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) semantic logits.
            scribbles: (B, H, W) integer labels; ``ignore_index`` for
                unlabelled pixels. May be omitted in the rare case of pure
                CRF/entropy-only regularisation.
            images: (B, 3, H, W) for the Gated-CRF colour kernel. Required
                whenever ``crf_weight > 0``; missing is a hard error.
            labeled_loss: Optional pre-computed supervised CE/dice term to
                add (mixed-supervision setting).
        """
        if predictions.dim() != 4:
            raise ValueError(
                f"predictions must be (B, C, H, W); got shape {tuple(predictions.shape)}"
            )

        total = predictions.new_zeros(())

        # ---- (1) Sparse CE on scribble pixels ---------------------------
        if scribbles is not None:
            if scribbles.dim() == 4 and scribbles.shape[1] == 1:
                scribbles = scribbles.squeeze(1)
            total = total + F.cross_entropy(
                predictions,
                scribbles.long(),
                ignore_index=self.ignore_index,
            )

        # ---- (2) Dense-CRF surrogate (Gated CRF) ------------------------
        if self.crf_weight > 0:
            if images is None:
                raise ValueError(
                    "ScribbleSupLoss with crf_weight>0 needs ``images`` "
                    "(B, 3, H, W) for the Gated-CRF colour kernel."
                )
            crf_term = self.crf(predictions, images)
            total = total + self.crf_weight * crf_term

        # ---- (3) Predictor-entropy minimisation -------------------------
        if self.entropy_weight > 0:
            total = total + self.entropy_weight * self._entropy(predictions)

        if labeled_loss is not None:
            total = total + labeled_loss
        return total
