"""CIRKD: Cross-Image Relational KD - Mini-batch pixel pairs (CVPR 2022).

Official source: https://github.com/winycg/CIRKD/blob/main/losses/cirkd_mini_batch.py

跨图像像素对的关系蒸馏，针对语义分割设计。
源码中B个样本两两计算pair-wise相似度并蒸馏。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("cirkd_minibatch")
class CIRKDMiniBatchLoss(nn.Module):
    """Cross-Image Relational KD - Mini-batch pixel pairs (CVPR 2022).

    Official source: winycg/CIRKD.
    """

    def __init__(self, temperature: float = 0.1, **kwargs):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _pair_wise_sim_map(fea_0, fea_1):
        """Compute pair-wise similarity map. fea_0/1: (C, H, W)"""
        C, H, W = fea_0.size()
        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)  # (HW, C)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)  # (HW, C)
        sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))  # (HW, HW)
        return sim_map_0_1

    def forward(self, feat_S: torch.Tensor, feat_T: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_S: (B, C, H, W) student features
            feat_T: (B, C, H, W) teacher features
        """
        B, C, H, W = feat_S.size()

        # Adaptive avgpool when feature map is large (strictly per source)
        if H >= 256 and W >= 256:
            patch_w = 4
            patch_h = 4
            maxpool = nn.AvgPool2d(
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w),
                padding=0,
                ceil_mode=True,
            )
            feat_S = maxpool(feat_S)
            feat_T = maxpool(feat_T)

        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        sim_dis = torch.tensor(0.0, device=feat_S.device)
        for i in range(B):
            for j in range(B):
                s_sim_map = self._pair_wise_sim_map(feat_S[i], feat_S[j])
                t_sim_map = self._pair_wise_sim_map(feat_T[i], feat_T[j])

                p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
                p_t = F.softmax(t_sim_map / self.temperature, dim=1)

                sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
                sim_dis = sim_dis + sim_dis_
        sim_dis = sim_dis / (B * B)
        return sim_dis
