"""EGE-UNet Decoder: 5-stage decoder with GHPA + Conv.
EGE-UNet 解码器：5 阶段解码器，前 3 阶段 GHPA，后 2 阶段 Conv3x3。

Stages / 各阶段:
  1-3: GHPA (Grouped multi-axis Hadamard Product Attention) + GN + GELU + Upsample
       分组多轴 Hadamard 乘积注意力 + GN + GELU + 上采样
  4-5: Conv3x3 + GN + GELU + Upsample
       Conv3x3 + GN + GELU + 上采样

Has internal skip connections (additive fusion with encoder features).
GroupAggregationBridge is a skip-level module, NOT part of the decoder core.
内部具有跳跃连接（与编码器特征相加融合）。
GroupAggregationBridge 是跳跃连接级模块，不属于解码器核心。

out_channels: c_list[0] = 8 (default)
"""
# Reference: https://github.com/JCruan519/EGE-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from timm.models.layers import trunc_normal_

from medseg.registry import DECODER_REGISTRY


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
# Group Aggregation Bridge (GAB) / 组聚合桥
# ---------------------------------------------------------------------------

class GroupAggregationBridge(nn.Module):
    """GAB: Fuses high-res and low-res features via grouped dilated convolution.
    组聚合桥：通过分组膨胀卷积融合高分辨率和低分辨率特征。
    """
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=(1, 2, 5, 7)):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.groups = nn.ModuleList()
        for d in d_list:
            pad = (k_size + (k_size - 1) * (d - 1)) // 2
            self.groups.append(nn.Sequential(
                _LayerNorm(group_size + 1, data_format='channels_first'),
                nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                          padding=pad, dilation=d, groups=group_size + 1)
            ))
        self.tail_conv = nn.Sequential(
            _LayerNorm(dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
        )

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)],
                           mode='bilinear', align_corners=True)
        xh_chunks = torch.chunk(xh, 4, dim=1)
        xl_chunks = torch.chunk(xl, 4, dim=1)
        outs = []
        for i, g in enumerate(self.groups):
            outs.append(g(torch.cat((xh_chunks[i], xl_chunks[i], mask), dim=1)))
        x = torch.cat(outs, dim=1)
        x = self.tail_conv(x)
        return x


# ---------------------------------------------------------------------------
# EGE-UNet Decoder / EGE-UNet 解码器
# ---------------------------------------------------------------------------

@DECODER_REGISTRY.register("ege_unet")
class EGEUNetDecoder(nn.Module):
    """EGE-UNet decoder: 5-stage with GHPA + Conv.
    EGE-UNet 解码器：5 阶段，前 3 阶段为 GHPA 注意力，后 2 阶段为卷积。

    Has internal skip connections (additive fusion) with optional GAB bridge.
    内部具有跳跃连接（加法融合），可选 GAB 桥。

    Architecture / 架构:
      dec1: GHPA(c5->c4)      + GAB5(out,t5) skip
      dec2: GHPA(c4->c3)      + upsample + GAB4(t5,t4) skip
      dec3: GHPA(c3->c2)      + upsample + GAB3(t4,t3) skip
      dec4: Conv3x3(c2->c1)   + upsample + GAB2(t3,t2) skip
      dec5: Conv3x3(c1->c0)   + upsample + GAB1(t2,t1) skip

    Args:
        encoder_channels: Encoder output channel list. / 编码器输出通道列表。
        bottleneck_channels: Bottleneck channels (same as encoder_channels[-1]).
                             瓶颈通道数（与 encoder_channels[-1] 相同）。
        c_list: Full channel list [c0..c5]. / 完整通道列表。
        bridge: Whether to use GroupAggregationBridge. / 是否使用 GAB 桥。
    """
    has_internal_skip = True
    required_skip_stages = 5
    requires_encoder = "ege_unet_enc"  # 5-stage encoder with specific spatial hierarchy

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 c_list=(8, 16, 24, 32, 48, 64),
                 bridge: bool = True, **kwargs):
        super().__init__()
        c_list = list(c_list)

        self.bridge = bridge

        # Project encoder features to c_list channels if mismatch
        # encoder_channels has 5 items (after adapter), map to c_list[0:5]
        self.projections = nn.ModuleList()
        self.bottleneck_proj = None
        if encoder_channels is not None and len(encoder_channels) == 5:
            for enc_ch, dec_ch in zip(encoder_channels, c_list[:5]):
                if enc_ch != dec_ch:
                    self.projections.append(nn.Conv2d(enc_ch, dec_ch, 1))
                else:
                    self.projections.append(nn.Identity())
            # Also project bottleneck to c_list[5] if needed
            if bottleneck_channels is not None and bottleneck_channels != c_list[5]:
                self.bottleneck_proj = nn.Conv2d(bottleneck_channels, c_list[5], 1)
        else:
            self.projections = None  # no projection needed

        # Bridge modules (GAB) / 桥接模块 (GAB)
        if bridge:
            self.GAB1 = GroupAggregationBridge(c_list[1], c_list[0])
            self.GAB2 = GroupAggregationBridge(c_list[2], c_list[1])
            self.GAB3 = GroupAggregationBridge(c_list[3], c_list[2])
            self.GAB4 = GroupAggregationBridge(c_list[4], c_list[3])
            self.GAB5 = GroupAggregationBridge(c_list[5], c_list[4])

        # GT deep supervision side convs (for GAB mask generation)
        # GT 深度监督侧输出卷积（用于 GAB mask 生成）
        self.gt_conv1 = nn.Conv2d(c_list[4], 1, 1)
        self.gt_conv2 = nn.Conv2d(c_list[3], 1, 1)
        self.gt_conv3 = nn.Conv2d(c_list[2], 1, 1)
        self.gt_conv4 = nn.Conv2d(c_list[1], 1, 1)
        self.gt_conv5 = nn.Conv2d(c_list[0], 1, 1)

        # Decoder stages / 解码器各阶段
        # Stages 1-3: GHPA / 第1-3阶段：GHPA 注意力
        self.decoder1 = GHPA(c_list[5], c_list[4])
        self.decoder2 = GHPA(c_list[4], c_list[3])
        self.decoder3 = GHPA(c_list[3], c_list[2])

        # Stages 4-5: Conv3x3 / 第4-5阶段：Conv3x3
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1)

        # GroupNorm for decoder / 解码器的 GroupNorm
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self._out_channels = c_list[0]

        self.apply(self._init_weights)

    @property
    def out_channels(self) -> int:
        """Output channel count of the decoder. / 解码器输出通道数。"""
        return self._out_channels

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

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """Decode bottleneck + skip features to output feature map.
        将瓶颈特征 + 跳跃连接特征解码为输出特征图。

        Args:
            bottleneck_feat: Deepest encoder feature (c5, H/32).
                             最深编码器特征。
            skip_features: [t1(c0,H/2), t2(c1,H/4), t3(c2,H/8), t4(c3,H/16), t5(c4,H/32)]
                           5 skip features from encoder (shallow to deep).
                           来自编码器的 5 个跳跃特征（由浅到深）。

        Returns: Decoded feature map (c0 channels). / 解码后的特征图。
        """
        # Apply projections if encoder channels don't match decoder channels
        if self.projections is not None:
            skip_features = [proj(feat) for proj, feat in zip(self.projections, skip_features)]
        if self.bottleneck_proj is not None:
            bottleneck_feat = self.bottleneck_proj(bottleneck_feat)

        # Unpack skip features / 解包跳跃特征
        t1, t2, t3, t4, t5 = skip_features[0], skip_features[1], skip_features[2], skip_features[3], skip_features[4]

        out = bottleneck_feat

        # Decoder stage 1: GHPA, skip with t5 via GAB / 解码阶段 1
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        gt_pre5 = self.gt_conv1(out5)
        if self.bridge:
            t5 = self.GAB5(out, t5, gt_pre5)
        out5 = out5 + t5

        # Decoder stage 2: GHPA + upsample, skip with t4 via GAB / 解码阶段 2
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        gt_pre4 = self.gt_conv2(out4)
        if self.bridge:
            t4 = self.GAB4(t5, t4, gt_pre4)
        out4 = out4 + t4

        # Decoder stage 3: GHPA + upsample, skip with t3 via GAB / 解码阶段 3
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        gt_pre3 = self.gt_conv3(out3)
        if self.bridge:
            t3 = self.GAB3(t4, t3, gt_pre3)
        out3 = out3 + t3

        # Decoder stage 4: Conv3x3 + upsample, skip with t2 via GAB / 解码阶段 4
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        gt_pre2 = self.gt_conv4(out2)
        if self.bridge:
            t2 = self.GAB2(t3, t2, gt_pre2)
        out2 = out2 + t2

        # Decoder stage 5: Conv3x3 + upsample, skip with t1 via GAB / 解码阶段 5
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        if self.bridge:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
        out1 = out1 + t1

        return out1
