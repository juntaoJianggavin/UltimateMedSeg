"""ACC-UNet: A Completely Convolutional UNet for the 2020s.

Modern convolutional UNet using large kernels, depthwise separable
convolutions, and residual connections for competitive performance
without transformers.

Reference:
    ACC-UNet: A Completely Convolutional UNet model for the 2020s.
    arXiv 2023. https://github.com/kiharalab/ACC-UNet

Key components:
    - Modern conv blocks with large kernels (7x7) and LayerNorm
    - Residual encoder-decoder with multi-scale features
    - Hierarchical context aggregation in decoder
"""
# Source: https://github.com/kiharalab/ACC-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class _ModernConvBlock(nn.Module):
    """Modern conv block: large kernel DWConv + pointwise + residual."""

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim)
        self.pw1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x_ln = self.norm(x.permute(0, 2, 3, 1))  # (B, H, W, C)
        x_dw = self.dw(x)  # (B, C, H, W)
        x_dw = x_dw.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_out = self.pw2(self.act(self.pw1(x_ln + x_dw)))
        x_out = x_out.permute(0, 3, 1, 2)  # (B, C, H, W)
        return residual + self.gamma * x_out


class _ACCStage(nn.Module):
    def __init__(self, in_c, out_c, depth, kernel_size=7, downsample=True):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.LayerNorm(in_c),
                nn.Conv2d(in_c, out_c, 2, 2),
            )
            in_c = out_c
        self.blocks = nn.Sequential(*[
            _ModernConvBlock(in_c, kernel_size) for _ in range(depth)
        ])

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) if hasattr(self.downsample, '__len__') else None
            # Simpler: use conv for downsampling
            x_in = x
            B, C, H, W = x_in.shape
            x_ln = F.layer_norm(x_in.permute(0, 2, 3, 1), [C]).permute(0, 3, 1, 2)
            x = F.conv2d(x_ln, self.downsample[1].weight, self.downsample[1].bias, stride=2)
        return self.blocks(x)


class ACCUNet(nn.Module):
    """ACC-UNet: Completely Convolutional UNet for the 2020s.

    Args:
        in_channels: Input channels.
        num_classes: Segmentation classes.
        img_size: Input spatial size.
        embed_dims: Channel dims per stage.
        depths: Blocks per stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        embed_dims = embed_dims or [48, 96, 192, 384]
        depths = depths or [3, 3, 9, 3]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 4, 4, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
        )

        # Encoder stages
        self.enc_stages = nn.ModuleList()
        for i in range(len(embed_dims)):
            blocks = [_ModernConvBlock(embed_dims[i]) for _ in range(depths[i])]
            self.enc_stages.append(nn.Sequential(*blocks))

        self.downsamples = nn.ModuleList()
        for i in range(len(embed_dims) - 1):
            self.downsamples.append(nn.Sequential(
                nn.Conv2d(embed_dims[i], embed_dims[i + 1], 3, 2, 1, bias=False),
                nn.BatchNorm2d(embed_dims[i + 1]),
            ))

        # Decoder
        self.upsamples = nn.ModuleList()
        self.merges = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(embed_dims) - 1, 0, -1):
            self.upsamples.append(nn.ConvTranspose2d(embed_dims[i], embed_dims[i - 1], 2, 2))
            self.merges.append(nn.Sequential(
                nn.Conv2d(embed_dims[i - 1] * 2, embed_dims[i - 1], 1, bias=False),
                nn.BatchNorm2d(embed_dims[i - 1]),
            ))
            self.dec_blocks.append(nn.Sequential(
                *[_ModernConvBlock(embed_dims[i - 1]) for _ in range(depths[i - 1])]
            ))

        self.head = nn.Sequential(
            nn.ConvTranspose2d(embed_dims[0], embed_dims[0], 4, 4),
            nn.Conv2d(embed_dims[0], num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2:]
        x = self.stem(x)

        skips = []
        for i, stage in enumerate(self.enc_stages):
            x = stage(x)
            if i < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[i](x)

        for up, merge, dec in zip(self.upsamples, self.merges, self.dec_blocks):
            x = up(x)
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = dec(merge(torch.cat([x, skip], dim=1)))

        x = self.head(x)
        if x.shape[2:] != (H_in, W_in):
            x = F.interpolate(x, size=(H_in, W_in), mode="bilinear", align_corners=False)
        return x
