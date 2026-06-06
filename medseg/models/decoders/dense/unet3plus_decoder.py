"""UNet 3+ Decoder — 全尺度密集连接解码器。
UNet 3+ Decoder — Full-scale dense-connection decoder.

Reference: https://github.com/ZJUGiveLab/UNet-Version
Paper: Huang et al., "UNet 3+: A Full-Scale Connected UNet for
       Medical Image Segmentation", ICASSP 2020.

核心思想 / Key idea:
    每个 decoder 层聚合**所有** encoder 层的特征（通过 pool 或 upsample
    对齐空间分辨率），而不是只用对应层的 skip connection。
    Each decoder level aggregates features from ALL encoder levels
    (via pool or upsample to align spatial size), not just the
    corresponding skip connection.

    这比 UNet++ 的密集连接更激进——UNet++ 只连接同层和相邻层，
    而 UNet 3+ 连接所有层。
    This is more aggressive than UNet++ dense connections — UNet++
    connects same-level and adjacent levels only, while UNet 3+
    connects ALL levels.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import DECODER_REGISTRY


class _SingleConv(nn.Module):
    """1x1 or 3x3 conv + BN + ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _FullScaleBlock(nn.Module):
    """全尺度聚合块：从所有 encoder 层 + 已解码层汇聚特征。
    Full-scale aggregation block: gather features from all encoder
    levels + already-decoded levels."""

    def __init__(self, all_channels: List[int], level: int,
                 decoded_channels: dict, cat_ch: int = 64):
        super().__init__()
        n = len(all_channels)
        self.level = level

        # 每个 encoder 层的 1x1 投影 / 1x1 projection for each encoder level
        self.enc_projs = nn.ModuleList([
            _SingleConv(c, cat_ch, 1) for c in all_channels
        ])

        # 已解码层的 1x1 投影 / 1x1 projection for already-decoded levels
        self.dec_levels = sorted(decoded_channels.keys())
        self.dec_projs = nn.ModuleList([
            _SingleConv(decoded_channels[dl], cat_ch, 1) for dl in self.dec_levels
        ])

        total = cat_ch * (n + len(self.dec_levels))
        self.fuse = _SingleConv(total, cat_ch * n, 3)
        self.out_channels = cat_ch * n

    def forward(self, enc_features, dec_features, target_size):
        parts = []

        for i, proj in enumerate(self.enc_projs):
            f = enc_features[i]
            if f.shape[2:] != target_size:
                if f.shape[2] > target_size[0]:
                    f = F.adaptive_max_pool2d(f, target_size)
                else:
                    f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            parts.append(proj(f))

        for idx, dl in enumerate(self.dec_levels):
            f = dec_features[dl]
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            parts.append(self.dec_projs[idx](f))

        return self.fuse(torch.cat(parts, dim=1))


@DECODER_REGISTRY.register("unet3plus")
class UNet3PlusDecoder(nn.Module):
    """UNet 3+ 全尺度密集连接解码器。
    UNet 3+ full-scale dense-connection decoder.

    与标准 UNet decoder 的区别 / Difference from standard UNet decoder:
        标准 UNet 每层只用一个 skip；UNet 3+ 每层聚合所有层的特征。
        Standard UNet uses one skip per level; UNet 3+ aggregates all levels.

    Args:
        encoder_channels: 各 encoder 层通道数（浅→深）/ Channel counts per encoder level.
        bottleneck_channels: 瓶颈层通道数 / Bottleneck channel count.
        skip_connection: 忽略（UNet3+ 有自己的全尺度连接）/ Ignored (UNet3+ has its own).
        cat_ch: 每个尺度投影到的统一通道数 / Unified channel count per scale.
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, cat_ch: int = 64, img_size: int = 224, **kwargs):
        super().__init__()

        # 所有层通道（encoder + bottleneck）
        all_channels = list(encoder_channels) + [bottleneck_channels]
        n_levels = len(encoder_channels)  # decoder 层数 = encoder 层数

        self.blocks = nn.ModuleList()
        decoded_so_far = {}

        # 从深到浅构建 decoder 层
        for i in range(n_levels - 1, -1, -1):
            block = _FullScaleBlock(all_channels, level=i,
                                   decoded_channels=dict(decoded_so_far),
                                   cat_ch=cat_ch)
            self.blocks.append(block)
            decoded_so_far[i] = block.out_channels

        self.blocks = nn.ModuleList(reversed(list(self.blocks)))
        self._out_channels = cat_ch * len(all_channels)

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        all_enc = list(skip_features) + [bottleneck_feat]
        dec_features = {}

        # 从深到浅逐层解码
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            target_h = all_enc[i].shape[2]
            target_w = all_enc[i].shape[3]
            out = block(all_enc, dec_features, (target_h, target_w))
            dec_features[i] = out

        # 返回最浅层的输出
        return dec_features[0]
