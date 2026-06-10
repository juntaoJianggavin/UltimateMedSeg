"""MALUNet Encoder: 6-stage encoder with Conv + EABlock + DilatedGatedAttention.
MALUNet 编码器：6 阶段编码器，前 3 阶段 Conv3x3+GN+GELU+MaxPool，后 3 阶段 EABlock+DGA。

Stages / 各阶段:
  1-3: Conv3x3 + GroupNorm + GELU + MaxPool2x2 (lightweight convolution)
       轻量卷积：Conv3x3 + GroupNorm + GELU + MaxPool2x2
  4-6: EABlock (External Attention) + DilatedGatedAttention + GN + GELU + MaxPool2x2
       外部注意力块 + 膨胀门控注意力 + GN + GELU + MaxPool2x2

out_channels: [8, 16, 24, 32, 48, 64] (default c_list)
"""
# Reference: https://github.com/JCruan519/MALUNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.utils.timm_compat import trunc_normal_

from medseg.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# DepthWise Conv / 深度可分离卷积
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
# Gated Attention Unit / 门控注意力单元
# ---------------------------------------------------------------------------

class GatedAttentionUnit(nn.Module):
    """Gated Attention Unit: dual-branch gating + output projection.
    门控注意力单元：双分支门控 + 输出投影。
    """
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.w1 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid())
        self.w2 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size + 2, padding=(kernel_size + 2) // 2),
            nn.GELU())
        self.wo = nn.Sequential(
            DepthWiseConv2d(in_c, out_c, kernel_size),
            nn.GELU())
        self.cw = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        return self.wo(self.w1(x) * self.w2(x)) + self.cw(x)


# ---------------------------------------------------------------------------
# Dilated Gated Attention / 膨胀门控注意力
# ---------------------------------------------------------------------------

class DilatedGatedAttention(nn.Module):
    """Multi-dilation grouped depthwise conv + Gated Attention Unit.
    多膨胀率分组深度卷积 + 门控注意力单元。
    """
    def __init__(self, in_c, out_c, k_size=3, dilated_ratio=(7, 5, 2, 1)):
        super().__init__()
        self.mdas = nn.ModuleList()
        for d in dilated_ratio:
            pad = (k_size + (k_size - 1) * (d - 1)) // 2
            self.mdas.append(nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                                        padding=pad, dilation=d, groups=in_c // 4))
        self.norm_layer = nn.GroupNorm(4, in_c)
        self.conv = nn.Conv2d(in_c, in_c, 1)
        self.gau = GatedAttentionUnit(in_c, out_c, 3)

    def forward(self, x):
        chunks = torch.chunk(x, 4, dim=1)
        outs = [self.mdas[i](chunks[i]) for i in range(4)]
        x = F.gelu(self.conv(self.norm_layer(torch.cat(outs, dim=1))))
        return self.gau(x)


# ---------------------------------------------------------------------------
# External Attention Block / 外部注意力块
# ---------------------------------------------------------------------------

class EABlock(nn.Module):
    """External Attention: low-rank linear attention.
    外部注意力：低秩线性注意力。
    """
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, 1)
        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)
        self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
        self.norm_layer = nn.GroupNorm(4, in_c)

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        attn = self.linear_0(x)
        attn = F.softmax(attn, dim=-1)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        x = self.linear_1(attn)
        x = x.view(b, c, h, w)
        x = self.norm_layer(self.conv2(x))
        return F.gelu(x + idn)


# ---------------------------------------------------------------------------
# MALUNet Encoder / MALUNet 编码器
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("malunet")
class MALUNetEncoder(nn.Module):
    """MALUNet encoder: 6-stage with Conv + EA + DGA.
    MALUNet 编码器：6 阶段，前 3 阶段为轻量卷积，后 3 阶段为注意力模块。

    Stages / 各阶段:
      1: Conv3x3 -> GN -> GELU -> MaxPool2x2   (in_ch -> c0)
      2: Conv3x3 -> GN -> GELU -> MaxPool2x2   (c0 -> c1)
      3: Conv3x3 -> GN -> GELU -> MaxPool2x2   (c1 -> c2)
      4: EA+DGA  -> GN -> GELU -> MaxPool2x2   (c2 -> c3)
      5: EA+DGA  -> GN -> GELU -> MaxPool2x2   (c3 -> c4)
      6: EA+DGA  -> GELU (no pool, bottleneck)  (c4 -> c5)

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

        # Stages 4-6: EA + DGA / 第4-6阶段：外部注意力 + 膨胀门控注意力
        self.encoder4 = nn.Sequential(EABlock(c_list[2]), DilatedGatedAttention(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(EABlock(c_list[3]), DilatedGatedAttention(c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(EABlock(c_list[4]), DilatedGatedAttention(c_list[4], c_list[5]))

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
        """Weight initialization following MALUNet convention.
        按 MALUNet 惯例初始化权重。"""
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
