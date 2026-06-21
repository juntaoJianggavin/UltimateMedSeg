"""Dense skip connection (UNet++ style): concat + DoubleConv fusion.
    Dense 跳跃连接（UNet++ 风格）：拼接 + DoubleConv 融合。

Unlike plain ``concat`` (which outputs ``decoder_ch + skip_ch`` channels),
DenseSkip fuses the concatenated features through a DoubleConv block
(two 3×3 conv + BN + ReLU) and projects back to ``skip_ch`` channels,
mimicking the dense node fusion in UNet++ (Zhou et al., 2018).

Reference:
    Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image
    Segmentation", DLMIA 2018.  Each decoder node X^{i,j} receives the
    concatenation of all prior nodes at the same scale and applies a conv
    block to fuse them.  In this framework's per-pair interface (one
    decoder feature + one skip feature), the analogous operation is:
    concat → DoubleConv → project back to skip_ch.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


class _DoubleConv(nn.Module):
    """Two 3×3 conv + BN + ReLU (UNet building block).
        两个 3×3 conv + BN + ReLU（UNet 基础模块）。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


@SKIP_REGISTRY.register("dense")
class DenseSkip(nn.Module):
    """Dense skip (UNet++ style): concat + DoubleConv fusion.
        Dense 跳跃连接（UNet++ 风格）：拼接 + DoubleConv 融合。

    Concatenates decoder and skip features, then applies a DoubleConv
    block to fuse and project back to ``skip_ch`` channels.  Unlike
    plain ``concat`` (which outputs ``decoder_ch + skip_ch``), DenseSkip
    outputs ``skip_ch`` — the fusion happens inside the skip module.
    """

    def __init__(self, out_channels=None, **kwargs):
        super().__init__()
        # If None, output channels = skip_ch (project back to encoder size).
        # 若为 None, 输出通道 = skip_ch (投影回编码器通道数)。
        self._target_out = out_channels
        # Lazily-built submodules keyed by (decoder_ch, skip_ch).
        # 按 (decoder_ch, skip_ch) 懒构建子模块。
        self._convs = nn.ModuleDict()

    def get_out_channels(self, decoder_ch, skip_ch):
        if self._target_out is not None:
            return self._target_out
        return skip_ch

    def _key(self, dc, sc):
        return f"{dc}_{sc}"

    def _build(self, decoder_ch, skip_ch, device):
        key = self._key(decoder_ch, skip_ch)
        if key in self._convs:
            return
        out_ch = self._target_out if self._target_out is not None else skip_ch
        conv = _DoubleConv(decoder_ch + skip_ch, out_ch)
        self._convs[key] = conv.to(device)

    def forward(self, decoder_feat, skip_feat):
        # Spatial align skip to decoder if needed.
        # 空间对齐 skip 到 decoder (若需要)。
        if skip_feat.shape[-2:] != decoder_feat.shape[-2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        decoder_ch = decoder_feat.shape[1]
        skip_ch = skip_feat.shape[1]
        self._build(decoder_ch, skip_ch, decoder_feat.device)
        key = self._key(decoder_ch, skip_ch)

        x = torch.cat([decoder_feat, skip_feat], dim=1)
        return self._convs[key](x)
