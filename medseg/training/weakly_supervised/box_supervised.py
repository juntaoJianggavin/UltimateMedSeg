"""BoxSup-style loss for bounding-box supervised segmentation.

Dai et al., ICCV 2015 (BoxSup).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("box_supervised")
class BoxSupervisedLoss(nn.Module):
    """Simplified BoxSup-style loss for bounding-box supervised segmentation.

    NOT a faithful re-implementation of BoxInst (Tian et al., CVPR 2021):
    that paper requires a projection loss along box rows/columns plus a
    pairwise color-affinity term, both of which we deliberately omit to
    keep the loss self-contained. What we run here is the "box mask =
    pseudo label" baseline of BoxSup (Dai et al., ICCV 2015):

        * inside-box pixels are treated as the present foreground class
          (using image-level labels or the argmax of the current prediction
          to disambiguate when several classes are present);
        * outside-box pixels are softly pushed toward background via the
          ``box_penalty * mean(P_fg)`` term.

    Refinement iterations (``refine_iterations``) are accepted for API
    parity with BoxSup but the current call site applies a single pass —
    the user can wrap multiple optimizer steps externally for the EM-style
    refinement described in the paper.
    """

    def __init__(
        self,
        base_loss_fn=None,
        box_penalty: float = 0.1,
        refine_iterations: int = 3,
        **kwargs
    ):
        super().__init__()
        self.box_penalty = box_penalty
        self.refine_iterations = refine_iterations

        if base_loss_fn is None:
            from medseg.losses.compound_loss import CompoundLoss
            self.base_loss_fn = CompoundLoss()
        else:
            self.base_loss_fn = base_loss_fn

    def forward(
        self,
        predictions: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        image_labels: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = predictions.shape

        if target is not None:
            return self.base_loss_fn(predictions, target)

        if boxes is not None:
            pseudo_labels = self._generate_pseudo_labels_from_boxes(
                predictions, boxes, image_labels
            )
            box_loss = self.base_loss_fn(predictions, pseudo_labels)
            outside_penalty = self._compute_outside_penalty(predictions, boxes)
            return box_loss + self.box_penalty * outside_penalty

        if image_labels is not None:
            return self._image_level_loss(predictions, image_labels)

        raise ValueError("Either boxes, image_labels, or target must be provided")

    def _parse_box(self, box):
        if isinstance(box, (list, tuple)):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        else:
            x1, y1, x2, y2 = box.int().tolist()
        return x1, y1, x2, y2

    def _generate_pseudo_labels_from_boxes(self, predictions, boxes, image_labels):
        B, C, H, W = predictions.shape
        pseudo_labels = torch.zeros(B, H, W, dtype=torch.long, device=predictions.device)

        for b in range(B):
            if image_labels is not None:
                present_classes = torch.where(image_labels[b] > 0)[0]
            else:
                present_classes = predictions[b].argmax(dim=0).unique()

            for box in boxes[b]:
                x1, y1, x2, y2 = self._parse_box(box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                if len(present_classes) == 1:
                    pseudo_labels[b, y1:y2, x1:x2] = present_classes[0]
                else:
                    box_pred = predictions[b, :, y1:y2, x1:x2]
                    pseudo_labels[b, y1:y2, x1:x2] = box_pred.mean(dim=[1, 2]).argmax()

        return pseudo_labels

    def _compute_outside_penalty(self, predictions, boxes):
        B, C, H, W = predictions.shape
        box_mask = torch.zeros(B, H, W, device=predictions.device)

        for b in range(B):
            for box in boxes[b]:
                x1, y1, x2, y2 = self._parse_box(box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                box_mask[b, y1:y2, x1:x2] = 1.0

        outside_mask = 1.0 - box_mask
        prob_outside = predictions[:, 1:, :, :].softmax(dim=1) * outside_mask.unsqueeze(1)
        return prob_outside.mean()

    def _image_level_loss(self, predictions, image_labels):
        global_pred = predictions.mean(dim=[2, 3])
        return F.binary_cross_entropy_with_logits(global_pred, image_labels.float())
