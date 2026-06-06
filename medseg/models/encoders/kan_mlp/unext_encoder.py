"""UNeXt encoder (ConvBlock stages + Tokenized MLP bottleneck).

Encoder portion of the UNeXt network (``medseg/networks/kan_mlp/unext.py``).
Four downsampling stages:

    enc1: ConvBlock(in, C)    -> ConvBlock(C, C)        -> /1, C
    pool + enc2: ConvBlock(C, 2C)  -> ConvBlock(2C, 2C) -> /2, 2C
    pool + enc3: ConvBlock(2C, 4C) -> ConvBlock(4C, 4C) -> /4, 4C
    pool + bottleneck_proj: ConvBlock(4C, 8C)
         + N x TokenizedMLP(8C)                          -> /8, 8C

The TokenizedMLP block at the bottleneck does spatial mixing through a
depthwise 1D conv on a flattened (B, N, C) token sequence, so it tolerates
arbitrary input resolutions.

Reference:
    Valanarasu & Patel, "UNeXt: MLP-based Rapid Medical Image Segmentation
    Network", MICCAI 2022.
"""
# Source: https://github.com/jeya-maria-jose/UNeXt-pytorch

import torch
import torch.nn as nn
from typing import List

from medseg.registry import ENCODER_REGISTRY


class _ConvBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _TokenizedMLP(nn.Module):
    """Tokenized MLP block: LN -> Linear -> shift DWConv1d -> GELU -> Linear (+ residual).

    Operates on tokens (B, N, C) where N = H*W. The depthwise 1D conv mixes
    along the token axis, so any spatial size is supported.
    """

    def __init__(self, dim: int, shift_size: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.dwconv = nn.Conv1d(
            dim * 2, dim * 2, shift_size,
            padding=shift_size // 2, groups=dim * 2,
        )
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # B, N, C
        x_res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 2, 1)  # B, 2C, N
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # B, N, 2C
        x = self.act(x)
        x = self.fc2(x)
        x = x + x_res
        return x.permute(0, 2, 1).reshape(B, C, H, W)


@ENCODER_REGISTRY.register("unext")
class UNeXtEncoder(nn.Module):
    """UNeXt encoder: 3 conv stages + tokenized-MLP bottleneck.

    Returns 4 multi-scale features (shallowest first, deepest last):
        [/1 base_ch, /2 base_ch*2, /4 base_ch*4, /8 base_ch*8]
    Default ``base_ch=32`` -> ``out_channels = [32, 64, 128, 256]``.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = False, base_ch: int = 32,
                 num_mlp_blocks: int = 3, **kwargs):
        super().__init__()
        # ``img_size`` and ``pretrained`` are accepted for API compatibility
        # but the encoder is fully convolutional / token-MLP so it adapts to
        # any input resolution and has no pretrained checkpoint.
        del img_size, pretrained

        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        self.out_channels: List[int] = chs

        # Optional 1x1 conv stem when caller passes non-RGB input
        if in_channels != 3:
            self.input_stem = nn.Conv2d(in_channels, 3, kernel_size=1)
            enc_in = 3
        else:
            self.input_stem = nn.Identity()
            enc_in = in_channels

        # Conv encoder stages
        self.enc1 = nn.Sequential(_ConvBlock(enc_in, chs[0]), _ConvBlock(chs[0], chs[0]))
        self.enc2 = nn.Sequential(_ConvBlock(chs[0], chs[1]), _ConvBlock(chs[1], chs[1]))
        self.enc3 = nn.Sequential(_ConvBlock(chs[1], chs[2]), _ConvBlock(chs[2], chs[2]))
        self.pool = nn.MaxPool2d(2)

        # Tokenized-MLP bottleneck (deepest stage)
        self.bottleneck_pool = nn.MaxPool2d(2)
        self.bottleneck_proj = _ConvBlock(chs[2], chs[3])
        self.mlp_blocks = nn.ModuleList(
            [_TokenizedMLP(chs[3]) for _ in range(num_mlp_blocks)]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_stem(x)

        e1 = self.enc1(x)                          # /1, C
        e2 = self.enc2(self.pool(e1))              # /2, 2C
        e3 = self.enc3(self.pool(e2))              # /4, 4C

        b = self.bottleneck_proj(self.bottleneck_pool(e3))  # /8, 8C
        for mlp in self.mlp_blocks:
            b = mlp(b)

        return [e1, e2, e3, b]  # deepest LAST
