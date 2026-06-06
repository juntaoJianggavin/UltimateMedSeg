"""UNet++ (Nested UNet) Decoder.
Reference: Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"

UNet++ has its own dense nested skip connection mechanism.
External skip_connection parameter is IGNORED.
"""
# Source: https://github.com/MrGiovanni/UNetPlusPlus

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


@DECODER_REGISTRY.register("unetpp")
class UNetPPDecoder(nn.Module):
    """UNet++ decoder with dense nested skip connections.

    External skip_connection is IGNORED - UNet++ has its own dense skip mechanism.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, deep_supervision: bool = False, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision

        # encoder_channels: [c0, c1, c2, c3], bottleneck_channels: c4
        all_ch = list(encoder_channels) + [bottleneck_channels]
        n = len(all_ch)  # number of levels

        # Dense connections: X_{i,j} where i is depth, j is dense index
        self.dense_blocks = nn.ModuleDict()
        for j in range(1, n):
            for i in range(n - j):
                # Input: upsampled from X_{i+1, j-1} + all X_{i, 0..j-1}
                in_ch = all_ch[i + 1] + all_ch[i] * j
                out_ch = all_ch[i]
                self.dense_blocks[f"{i}_{j}"] = ConvBlock(in_ch, out_ch)

        self._out_channels = all_ch[0]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        all_features = list(skip_features) + [bottleneck_feat]
        n = len(all_features)

        # nodes[i][j] stores feature at depth i, dense index j
        nodes = [[None] * n for _ in range(n)]
        for i in range(n):
            nodes[i][0] = all_features[i]

        for j in range(1, n):
            for i in range(n - j):
                up = F.interpolate(nodes[i + 1][j - 1], size=nodes[i][0].shape[2:],
                                   mode='bilinear', align_corners=False)
                dense_inputs = [nodes[i][k] for k in range(j)]
                x = torch.cat([up] + dense_inputs, dim=1)
                nodes[i][j] = self.dense_blocks[f"{i}_{j}"](x)

        return nodes[0][n - 1]
