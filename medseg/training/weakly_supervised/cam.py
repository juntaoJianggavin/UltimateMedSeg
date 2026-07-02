"""Class Activation Mapping based weak supervision loss.

Zhou et al., CVPR 2016 / Selvaraju et al., ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY

from .cam_generator import CAMGenerator


@LOSS_REGISTRY.register("cam_loss")
class CAMLoss(nn.Module):
    """Class Activation Mapping based weak supervision.

    Generates pseudo-labels from CAM and trains segmentation model.
    """

    def __init__(
        self,
        base_loss_fn=None,
        cam_threshold: float = 0.5,
        refine: bool = True,
        cam_weight: float = 1.0,
        seg_weight: float = 0.5,
        target_layer: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.cam_threshold = cam_threshold
        self.refine = refine
        self.cam_weight = cam_weight
        self.seg_weight = seg_weight
        self.target_layer = target_layer

        if base_loss_fn is None:
            from medseg.losses.compound_loss import CompoundLoss
            self.base_loss_fn = CompoundLoss()
        else:
            self.base_loss_fn = base_loss_fn

    def attach_generator(self, model) -> CAMGenerator:
        """Build a CAMGenerator bound to ``model`` using ``self.target_layer``."""
        return CAMGenerator(model, target_layer=self.target_layer)

    def forward(
        self,
        predictions: torch.Tensor,
        cams: torch.Tensor,
        image_labels: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total = predictions.new_zeros(())

        if target is not None:
            total = total + self.seg_weight * self.base_loss_fn(predictions, target)

        pseudo_labels = self._cam_to_pseudo_labels(cams, image_labels)
        total = total + self.cam_weight * self.base_loss_fn(predictions, pseudo_labels)

        return total

    def _cam_to_pseudo_labels(self, cams, image_labels):
        B, C, H, W = cams.shape
        pseudo_labels = torch.zeros(B, H, W, dtype=torch.long, device=cams.device)

        for b in range(B):
            present_classes = torch.where(image_labels[b] > 0)[0]
            # Filter to valid channel range in case image_labels has more classes
            # than model output channels (e.g. during smoke tests with patched data).
            present_classes = present_classes[present_classes < C]
            if len(present_classes) == 0:
                continue

            cam_values = cams[b, present_classes]
            max_cam, max_idx = cam_values.max(dim=0)
            foreground = max_cam > self.cam_threshold
            pseudo_labels[b, foreground] = present_classes[max_idx[foreground]]

        return pseudo_labels
