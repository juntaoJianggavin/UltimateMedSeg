"""Multi-Instance Learning for image-level supervised segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("mil_loss")
class MILLoss(nn.Module):
    """Multi-Instance Learning for image-level supervised segmentation.

    Treats image patches as instances and learns from image-level labels.
    """

    def __init__(self, patch_size: int = 32, aggregation: str = 'max', **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.aggregation = aggregation

    def forward(self, predictions: torch.Tensor, image_labels: torch.Tensor) -> torch.Tensor:
        B, C, H, W = predictions.shape
        patches = self._extract_patches(predictions)

        if patches.dim() == 5:
            patch_probs = patches.softmax(dim=2)
            aggregated = patch_probs.mean(dim=[0, 3, 4])
        else:
            aggregated = predictions.mean(dim=[2, 3])

        return F.binary_cross_entropy(aggregated, image_labels.float())

    def _extract_patches(self, predictions):
        B, C, H, W = predictions.shape
        p = self.patch_size
        patches = []
        for i in range(0, H, p):
            for j in range(0, W, p):
                patch = predictions[:, :, i:i+p, j:j+p]
                if patch.shape[2] == p and patch.shape[3] == p:
                    patches.append(patch)
        if len(patches) == 0:
            return predictions.unsqueeze(0)
        return torch.stack(patches, dim=0)
