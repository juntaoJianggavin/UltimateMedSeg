"""AdvCAM: Anti-Adversarially Manipulated Attributions — simplified training loss.

Lee et al., "Anti-Adversarially Manipulated Attributions for Weakly and
Semi-Supervised Semantic Segmentation", CVPR 2021.
Official: https://github.com/jbeomlee93/AdvCAM

IMPORTANT — scope clarification:
    The official AdvCAM ``obtain_CAM_masking.py`` implements the adversarial
    climbing procedure as an **offline inference-time** step:
        for it in range(adv_iter):
            regions = GradCAM(img_single)
            L_AD = sum(|regions - init_cam| * discriminative_mask)
            loss = -logit_loss - L_AD * AD_coeff
            loss.backward()
            img_single = adv_climb(img_single, stepsize, grad)

    This file provides a **training-time** classification loss that is
    NOT the official AdvCAM procedure.  It uses:
        cls_loss = multilabel_soft_margin_loss(GAP(predictions), labels)
        adv_loss = -mean(clamp(cam_accumulated - cam_orig.detach(), 0))
    which is a simplified surrogate encouraging expanded activations
    during training.  The true AdvCAM adversarial climbing on inputs
    must be run as a separate offline step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("advcam_loss")
class AdvCAMLoss(nn.Module):
    """AdvCAM training loss: multilabel_soft_margin_loss + optional adv climbing."""

    def __init__(self, adv_iter: int = 8, adv_alpha: float = 1.0,
                 adv_weight: float = 1.0, **kwargs):
        super().__init__()
        self.adv_iter = adv_iter
        self.adv_alpha = adv_alpha
        self.adv_weight = adv_weight

    def forward(self, predictions, image_labels, cam_accumulated=None,
                cam_orig=None, labeled_loss=None):
        if predictions.dim() == 4:
            predictions = predictions.mean(dim=[2, 3])
        cls_loss = F.multilabel_soft_margin_loss(predictions, image_labels.float())
        total_loss = cls_loss

        if cam_accumulated is not None and cam_orig is not None:
            expanded = torch.clamp(cam_accumulated - cam_orig.detach(), min=0.0)
            adv_loss = -expanded.mean()
            total_loss = total_loss + self.adv_weight * adv_loss

        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss
