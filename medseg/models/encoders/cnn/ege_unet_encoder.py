"""EGE-UNet Encoder: 6-stage encoder with Conv + GHPA (Group Hadamard Product Attention).
EGE-UNet 编码器：6 阶段编码器，前 3 阶段 Conv3x3+GN+GELU+MaxPool，后 3 阶段 GHPA。

Stages / 各阶段:
  1-3: Conv3x3 + GroupNorm + GELU + MaxPool2x2 (lightweight convolution)
       轻量卷积：Conv3x3 + GroupNorm + GELU + MaxPool2x2
  4-6: GHPA (Grouped multi-axis Hadamard Product Attention) + GN + GELU + MaxPool2x2
       分组多轴 Hadamard 乘积注意力 + GN + GELU + MaxPool2x2

out_channels: [8, 16, 24, 32, 48, 64] (default c_list)
"""
# Reference: https://github.com/JCruan519/EGE-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.utils.timm_compat import trunc_normal_

from medseg.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# LayerNorm (channels_first) / 层归一化（通道优先格式）
# ---------------------------------------------------------------------------

class _LayerNorm(nn.Module):
    """ConvNeXt-style LayerNorm supporting channels_first format.
    ConvNeXt 风格层归一化，支持通道优先格式。
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ---------------------------------------------------------------------------
# Depthwise Separable Conv / 深度可分离卷积
# ---------------------------------------------------------------------------

class DepthWiseConv2d(nn.Module):
    """Depthwise separable convolution: depthwise conv + GroupNorm + pointwise conv.
    深度可分离卷积：深度卷积 + GroupNorm + 逐点卷积。
    """
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


# ---------------------------------------------------------------------------
# Grouped Multi-Axis Hadamard Product Attention (GHPA)
# 分组多轴 Hadamard 乘积注意力
# ---------------------------------------------------------------------------

class GHPA(nn.Module):
    """Grouped multi-axis Hadamard Product Attention.
    分组多轴 Hadamard 乘积注意力。

    Splits channels into 4 groups, applies learnable params along
    xy/zx/zy axes + depthwise conv branch, then fuses via LDW.
    将通道分为 4 组，分别沿 xy/zx/zy 轴应用可学习参数 + 深度卷积分支，最后通过 LDW 融合。
    """
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        # xy branch / xy 分支
        self.params_xy = nn.Parameter(torch.ones(1, c_dim_in, x, y))
        self.conv_xy = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1))

        # zx branch / zx 分支
        self.params_zx = nn.Parameter(torch.ones(1, 1, c_dim_in, x))
        self.conv_zx = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1))

        # zy branch / zy 分支
        self.params_zy = nn.Parameter(torch.ones(1, 1, c_dim_in, y))
        self.conv_zy = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1))

        # depthwise branch / 深度卷积分支
        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1), nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in))

        # normalization + output projection / 归一化 + 输出投影
        self.norm1 = _LayerNorm(dim_in, data_format='channels_first')
        self.norm2 = _LayerNorm(dim_in, data_format='channels_first')
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1))

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        # xy branch / xy 分支
        x1 = x1 * self.conv_xy(F.interpolate(self.params_xy, size=x1.shape[2:4],
                                               mode='bilinear', align_corners=True))

        # zx branch / zx 分支
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2 * self.conv_zx(F.interpolate(self.params_zx, size=x2.shape[2:4],
                                               mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)

        # zy branch / zy 分支
        x3 = x3.permute(0, 2, 1, 3)
        x3 = x3 * self.conv_zy(F.interpolate(self.params_zy, size=x3.shape[2:4],
                                               mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)

        # depthwise branch / 深度卷积分支
        x4 = self.dw(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x


# ---------------------------------------------------------------------------
# EGE-UNet Encoder / EGE-UNet 编码器
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("ege_unet")
class EGEUNetEncoder(nn.Module):
    """EGE-UNet encoder: 6-stage with Conv + GHPA.
    EGE-UNet 编码器：6 阶段，前 3 阶段为轻量卷积，后 3 阶段为 GHPA。

    Stages / 各阶段:
      1: Conv3x3 -> GN -> GELU -> MaxPool2x2   (in_ch -> c0)
      2: Conv3x3 -> GN -> GELU -> MaxPool2x2   (c0 -> c1)
      3: Conv3x3 -> GN -> GELU -> MaxPool2x2   (c1 -> c2)
      4: GHPA    -> GN -> GELU -> MaxPool2x2   (c2 -> c3)
      5: GHPA    -> GN -> GELU -> MaxPool2x2   (c3 -> c4)
      6: GHPA    -> GELU (no pool, bottleneck)  (c4 -> c5)

    Returns first 5 stages as skip features + stage 6 as deepest feature.
    返回前 5 阶段作为跳跃连接特征，第 6 阶段作为最深特征。

    Args:
        in_channels: Input image channels (default 3). / 输入通道数。
        c_list: Channel list for 6 stages. / 6 个阶段的通道数列表。
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 c_list=(8, 16, 24, 32, 48, 64), **kwargs):
        super().__init__()
        c_list = list(c_list)

        # Encoder stages / 编码器各阶段
        # Stages 1-3: simple convolution / 第1-3阶段：简单卷积
        self.encoder1 = nn.Conv2d(in_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)

        # Stages 4-6: GHPA / 第4-6阶段：GHPA 注意力
        self.encoder4 = GHPA(c_list[2], c_list[3])
        self.encoder5 = GHPA(c_list[3], c_list[4])
        self.encoder6 = GHPA(c_list[4], c_list[5])

        # GroupNorm for stages 1-5 / 第1-5阶段的 GroupNorm
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        # Output channels (shallow -> deep) / 输出通道（由浅到深）
        self.out_channels: List[int] = c_list

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Weight initialization following EGE-UNet convention.
        按 EGE-UNet 惯例初始化权重。"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode input to 6 multi-scale feature maps.
        将输入编码为 6 个多尺度特征图。

        Returns: [t1(c0,H/2), t2(c1,H/4), t3(c2,H/8),
                  t4(c3,H/16), t5(c4,H/32), out6(c5,H/32)]
        """
        # Stage 1 / 阶段 1
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out

        # Stage 2 / 阶段 2
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out

        # Stage 3 / 阶段 3
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        # Stage 4 / 阶段 4
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out

        # Stage 5 / 阶段 5
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out

        # Stage 6: bottleneck (no pool) / 阶段 6：瓶颈层（无池化）
        out6 = F.gelu(self.encoder6(out))

        return [t1, t2, t3, t4, t5, out6]
