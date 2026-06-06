"""MedNeXt encoder (2D).

Faithful port of the MedNeXt encoder blocks from MIC-DKFZ/MedNeXt.
Each MedNeXt block is a ConvNeXt-style residual unit:
    Depthwise Conv -> Norm -> 1x1 expansion -> GELU -> [GRN] -> 1x1 contraction
    (+ residual).

Architecture (4 downsampling stages, multi-scale output):
    stem (1x1 conv)                                -> /1, base_features
    stage1: depths[0] x MedNeXtBlock               -> /1
    down1 + stage2: depths[1] x MedNeXtBlock       -> /2, 2x ch
    down2 + stage3: depths[2] x MedNeXtBlock       -> /4, 4x ch
    down3 + stage4: depths[3] x MedNeXtBlock +
                    depths[4] x MedNeXtBlock       -> /8, 8x ch (bottleneck)

Reference:
    Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for
    Medical Image Segmentation", MICCAI 2023.
"""
# Source: https://github.com/MIC-DKFZ/MedNeXt

import torch
import torch.nn as nn
from typing import List, Sequence

from medseg.registry import ENCODER_REGISTRY


class _MedNeXtBlock(nn.Module):
    """MedNeXt block – faithful to MIC-DKFZ/MedNeXt ``MedNeXtBlock``.

    DWConv -> GroupNorm -> 1x1 expand -> GELU -> [GRN] -> 1x1 contract + residual.
    """

    def __init__(self, in_ch: int, out_ch: int, exp_r: int = 4,
                 kernel_size: int = 7, do_res: bool = True,
                 norm_type: str = 'group', n_groups=None,
                 grn: bool = False):
        super().__init__()
        self.do_res = do_res

        # Depthwise convolution (groups = in_ch per original)
        self.conv1 = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            padding=kernel_size // 2, groups=in_ch)

        # Normalization – GroupNorm with num_groups = in_ch (original default)
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=n_groups if n_groups else in_ch,
                num_channels=in_ch)
        else:
            self.norm = nn.LayerNorm(in_ch)

        # 1x1 expansion
        self.conv2 = nn.Conv2d(in_ch, exp_r * in_ch, 1)
        self.act = nn.GELU()

        # Optional Global Response Normalization (after GELU, before conv3)
        self.grn = grn
        if grn:
            exp_ch = exp_r * in_ch
            self.grn_gamma = nn.Parameter(torch.zeros(1, exp_ch, 1, 1))
            self.grn_beta = nn.Parameter(torch.zeros(1, exp_ch, 1, 1))

        # 1x1 contraction
        self.conv3 = nn.Conv2d(exp_r * in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, dummy_tensor=None) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class _MedNeXtDownBlock(nn.Module):
    """MedNeXt down block – faithful to MIC-DKFZ ``MedNeXtDownBlock``.

    Overrides the depthwise conv to stride=2 for 2x spatial downsampling.
    The main path has NO residual (per original). An optional stride-2
    1x1 residual projection can be added via ``do_res=True``.
    """

    def __init__(self, in_ch: int, out_ch: int, exp_r: int = 4,
                 kernel_size: int = 7, do_res: bool = False,
                 norm_type: str = 'group', grn: bool = False):
        super().__init__()
        # Inner block (no residual – matches original MedNeXtDownBlock)
        self.block = _MedNeXtBlock(
            in_ch, out_ch, exp_r, kernel_size,
            do_res=False, norm_type=norm_type, grn=grn)

        # Override conv1 with stride=2 for spatial downsampling
        self.block.conv1 = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            stride=2, padding=kernel_size // 2, groups=in_ch)

        # Optional stride-2 1x1 residual projection
        self.do_res = do_res
        if do_res:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block(x)
        if self.do_res:
            x1 = x1 + self.res_conv(x)
        return x1


def _make_stage(n_ch: int, num_blocks: int,
                exp_r: int, kernel_size: int, norm_type: str,
                grn: bool = False) -> nn.Sequential:
    """Build a stack of MedNeXt blocks (same-channel)."""
    return nn.Sequential(*[
        _MedNeXtBlock(n_ch, n_ch, exp_r, kernel_size,
                      do_res=True, norm_type=norm_type, grn=grn)
        for _ in range(num_blocks)
    ])


@ENCODER_REGISTRY.register("mednext")
class MedNeXtEncoder(nn.Module):
    """MedNeXt encoder with 4 multi-scale outputs.

    Faithful to MIC-DKFZ/MedNeXt encoder blocks with 1x1 stem,
    MedNeXtBlock (DWConv→Norm→expand→GELU→[GRN]→contract+residual),
    and MedNeXtDownBlock (stride-2 DWConv for downsampling).

    Args:
        in_channels: input image channels.
        img_size: input spatial size (unused, API consistency).
        base_features: channel width at scale /1.
        depths: tuple of 5 ints. First four are block counts for the four
            encoder stages. The fifth adds extra blocks at the deepest scale.
        exp_r: expansion ratio inside each MedNeXt block.
        kernel_size: depthwise convolution kernel size (default 7 per original).
        norm_type: 'group' (default) or 'layer'.
        grn: enable Global Response Normalization (default False).

    forward(x) returns a list of 4 feature tensors at scales /1, /2, /4, /8.
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 base_features: int = 32,
                 depths: Sequence[int] = (2, 2, 2, 2, 2),
                 exp_r: int = 4, kernel_size: int = 7,
                 norm_type: str = 'group', grn: bool = False,
                 **kwargs):
        super().__init__()

        depths = tuple(depths)
        if len(depths) < 4:
            raise ValueError(
                f"`depths` must have at least 4 entries, got {len(depths)}")
        if len(depths) == 4:
            depths = depths + (0,)

        n_ch = base_features
        channels = [n_ch, 2 * n_ch, 4 * n_ch, 8 * n_ch]
        self.out_channels: List[int] = channels

        # Stem: 1x1 conv (per original MedNeXt)
        self.stem = nn.Conv2d(in_channels, n_ch, 1)

        # Stage 1 at /1
        self.stage1 = _make_stage(n_ch, depths[0],
                                  exp_r, kernel_size, norm_type, grn)

        # Down + Stage 2 at /2
        self.down1 = _MedNeXtDownBlock(n_ch, 2 * n_ch, exp_r,
                                       kernel_size, norm_type=norm_type, grn=grn)
        self.stage2 = _make_stage(2 * n_ch, max(depths[1] - 1, 0),
                                  exp_r, kernel_size, norm_type, grn)

        # Down + Stage 3 at /4
        self.down2 = _MedNeXtDownBlock(2 * n_ch, 4 * n_ch, exp_r,
                                       kernel_size, norm_type=norm_type, grn=grn)
        self.stage3 = _make_stage(4 * n_ch, max(depths[2] - 1, 0),
                                  exp_r, kernel_size, norm_type, grn)

        # Down + Stage 4 at /8 (+ optional bottleneck refinement)
        self.down3 = _MedNeXtDownBlock(4 * n_ch, 8 * n_ch, exp_r,
                                       kernel_size, norm_type=norm_type, grn=grn)
        stage4_blocks = max(depths[3] - 1, 0) + max(depths[4], 0)
        self.stage4 = _make_stage(8 * n_ch, stage4_blocks,
                                  exp_r, kernel_size, norm_type, grn)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Handle single-channel input by replicating to in_channels
        if x.size(1) == 1 and self.stem.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        features: List[torch.Tensor] = []

        x = self.stem(x)
        x = self.stage1(x)
        features.append(x)        # /1

        x = self.down1(x)
        x = self.stage2(x)
        features.append(x)        # /2

        x = self.down2(x)
        x = self.stage3(x)
        features.append(x)        # /4

        x = self.down3(x)
        x = self.stage4(x)
        features.append(x)        # /8

        return features
