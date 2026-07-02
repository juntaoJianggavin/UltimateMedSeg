"""SEAM: Self-supervised Equivariant Attention Mechanism.

Wang et al., CVPR 2020 (Oral).
Official: https://github.com/YudeWang/SEAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("seam_loss")
class SEAMLoss(nn.Module):
    """SEAM training loss: loss_cls + loss_er + loss_ecr."""

    def __init__(self, scale_factor: float = 0.3, ecr_top_k_ratio: float = 0.2, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.ecr_top_k_ratio = ecr_top_k_ratio

    @staticmethod
    def _max_norm(cam):
        B, C, H, W = cam.shape
        cam_flat = cam.view(B, C, -1)
        cam_min = cam_flat.min(dim=2, keepdim=True)[0]
        cam_max = cam_flat.max(dim=2, keepdim=True)[0]
        cam_norm = (cam_flat - cam_min) / (cam_max - cam_min + 1e-8)
        return cam_norm.view(B, C, H, W)

    @staticmethod
    def _max_onehot(x):
        x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
        x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
        return x

    @staticmethod
    def _adaptive_min_pooling_loss(x):
        n, c, h, w = x.size()
        k = h * w // 4
        x = torch.max(x, dim=1)[0]
        y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
        y = F.relu(y, inplace=False)
        return torch.sum(y) / (k * n)

    def forward(self, cam1_raw, cam_rv1_raw, cam2_raw, cam_rv2_raw,
                image_labels, labeled_loss=None, **kwargs):
        N, C_total, H, W = cam1_raw.shape
        bg_score = torch.ones((N, 1), device=image_labels.device)
        label = torch.cat((bg_score, image_labels), dim=1).unsqueeze(2).unsqueeze(3)

        label1 = F.adaptive_avg_pool2d(cam1_raw, (1, 1))
        loss_rvmin1 = self._adaptive_min_pooling_loss((cam_rv1_raw * label)[:, 1:, :, :])
        cam1_norm = self._max_norm(cam1_raw) * label
        cam_rv1_norm = self._max_norm(cam_rv1_raw) * label

        label2 = F.adaptive_avg_pool2d(cam2_raw, (1, 1))
        loss_rvmin2 = self._adaptive_min_pooling_loss((cam_rv2_raw * label)[:, 1:, :, :])
        cam2_norm = self._max_norm(cam2_raw) * label
        cam_rv2_norm = self._max_norm(cam_rv2_raw) * label

        loss_cls1 = F.multilabel_soft_margin_loss(
            label1[:, 1:, :, :].squeeze(-1).squeeze(-1), image_labels)
        loss_cls2 = F.multilabel_soft_margin_loss(
            label2[:, 1:, :, :].squeeze(-1).squeeze(-1), image_labels)
        loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2

        # In the official SEAM (https://github.com/YudeWang/SEAM), cam1 is
        # from the original-scale image and cam2 from a downsampled image
        # (scale_factor ≈ 0.3).  F.interpolate is used to bring cam1 down
        # to cam2's resolution so that loss_er / loss_ecr compare features
        # across scales — the core equivariance signal.
        _, _, H2, W2 = cam2_norm.shape
        if (H, W) != (H2, W2):
            cam1_down = F.interpolate(cam1_norm, size=(H2, W2),
                                      mode='bilinear', align_corners=True)
            cam_rv1_down = F.interpolate(cam_rv1_norm, size=(H2, W2),
                                         mode='bilinear', align_corners=True)
        else:
            cam1_down = cam1_norm
            cam_rv1_down = cam_rv1_norm

        loss_er = torch.mean(torch.abs(cam1_down[:, 1:, :, :] - cam2_norm[:, 1:, :, :]))

        cam1_down[:, 0, :, :] = 1 - torch.max(cam1_down[:, 1:, :, :], dim=1)[0]
        cam2_norm[:, 0, :, :] = 1 - torch.max(cam2_norm[:, 1:, :, :], dim=1)[0]
        tensor_ecr1 = torch.abs(self._max_onehot(cam2_norm.detach().clone()) - cam_rv1_down)
        tensor_ecr2 = torch.abs(self._max_onehot(cam1_down.detach().clone()) - cam_rv2_norm)
        ns, cs, hs, ws = tensor_ecr1.shape
        k = int(cs * hs * ws * self.ecr_top_k_ratio)
        loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=k, dim=-1)[0])
        loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=k, dim=-1)[0])
        loss_ecr = loss_ecr1 + loss_ecr2

        total_loss = loss_cls + loss_er + loss_ecr
        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss
