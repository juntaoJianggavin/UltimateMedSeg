"""MCTformer: Multi-Class Token Transformer for weakly-supervised segmentation.

Xu et al., CVPR 2022.
Official: https://github.com/xulianuwa/MCTformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("mctformer_loss")
class MCTformerLoss(nn.Module):
    """MCTformer loss = cls_loss + loss_weight * reg_loss + patch_loss."""

    def __init__(self, loss_weight: float = 0.1, num_cct: int = 1, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_cct = num_cct

    def _regularizer_loss(self, class_token_embeddings, image_labels):
        L, B, C, D = class_token_embeddings.shape
        output_cls_embeddings = F.normalize(class_token_embeddings, dim=-1)
        scores = torch.matmul(output_cls_embeddings,
                              output_cls_embeddings.permute(0, 1, 3, 2))
        ground_truth = torch.arange(C, dtype=torch.long, device=scores.device)
        ground_truth = ground_truth.unsqueeze(0).unsqueeze(0).expand(L, B, C)
        reg_loss = F.cross_entropy(scores.permute(1, 2, 3, 0),
                                   ground_truth.permute(1, 2, 0), reduction='none')
        reg_loss = torch.mean(
            torch.mean(torch.sum(reg_loss * image_labels.float().unsqueeze(-1), dim=-2), dim=-1)
            / (torch.sum(image_labels.float(), dim=-1) + 1e-8))
        return reg_loss

    def forward(self, outputs, image_labels, patch_outputs=None,
                class_token_embeddings=None, labeled_loss=None):
        if outputs.dim() == 4:
            outputs = outputs.mean(dim=[2, 3])
        cls_loss = F.multilabel_soft_margin_loss(outputs, image_labels.float())
        total_loss = cls_loss

        if class_token_embeddings is not None:
            reg_loss = self._regularizer_loss(class_token_embeddings, image_labels)
            total_loss = total_loss + self.loss_weight * reg_loss

        if patch_outputs is not None:
            if patch_outputs.dim() == 4:
                patch_outputs = patch_outputs.mean(dim=[2, 3])
            patch_loss = F.multilabel_soft_margin_loss(patch_outputs, image_labels.float())
            total_loss = total_loss + patch_loss

        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss
