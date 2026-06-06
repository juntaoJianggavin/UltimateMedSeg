"""MALUNet Decoder: 5-stage decoder with DGA + EABlock + Conv.
MALUNet 解码器：5 阶段解码器，前 3 阶段 DGA+EABlock，后 2 阶段 Conv3x3。

Stages / 各阶段:
  1-3: DilatedGatedAttention + EABlock + GN + GELU + Upsample
       膨胀门控注意力 + 外部注意力块 + GN + GELU + 上采样
  4-5: Conv3x3 + GN + GELU + Upsample
       Conv3x3 + GN + GELU + 上采样

Has internal skip connections (additive fusion with encoder features).
内部具有跳跃连接（与编码器特征相加融合）。

out_channels: c_list[0] = 8 (default)
"""
# Reference: https://github.com/JCruan519/MALUNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from timm.models.layers import trunc_normal_

from medseg.registry import DECODER_REGISTRY


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
# SC_Att_Bridge (Spatial-Channel Attention Bridge) / 空间-通道注意力桥
# ---------------------------------------------------------------------------

class _ChannelAttBridge(nn.Module):
    """Channel attention bridge for skip connection fusion.
    通道注意力桥，用于跳跃连接融合。
    """
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.atts = nn.ModuleList()
        for i in range(5):
            if split_att == 'fc':
                self.atts.append(nn.Linear(c_list_sum, c_list[i]))
            else:
                self.atts.append(nn.Conv1d(c_list_sum, c_list[i], 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):
        att = torch.cat([self.avgpool(f) for f in feats], dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        results = []
        for i, f in enumerate(feats):
            a = self.sigmoid(self.atts[i](att))
            if self.split_att == 'fc':
                a = a.transpose(-1, -2).unsqueeze(-1).expand_as(f)
            else:
                a = a.unsqueeze(-1).expand_as(f)
            results.append(a)
        return results


class _SpatialAttBridge(nn.Module):
    """Spatial attention bridge for skip connection fusion.
    空间注意力桥，用于跳跃连接融合。
    """
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3), nn.Sigmoid())

    def forward(self, feats):
        results = []
        for t in feats:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = self.shared_conv2d(torch.cat([avg_out, max_out], dim=1))
            results.append(att)
        return results


class SCABridge(nn.Module):
    """Spatial-Channel Attention Bridge for skip connection enhancement.
    空间-通道注意力桥，用于增强跳跃连接。
    """
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.catt = _ChannelAttBridge(c_list, split_att)
        self.satt = _SpatialAttBridge()

    def forward(self, feats):
        residuals = list(feats)
        satts = self.satt(feats)
        feats = [s * f for s, f in zip(satts, feats)]
        r2 = list(feats)
        feats = [f + r for f, r in zip(feats, residuals)]
        catts = self.catt(feats)
        feats = [c * f for c, f in zip(catts, feats)]
        return [f + r for f, r in zip(feats, r2)]


# ---------------------------------------------------------------------------
# MALUNet Decoder / MALUNet 解码器
# ---------------------------------------------------------------------------

@DECODER_REGISTRY.register("malunet")
class MALUNetDecoder(nn.Module):
    """MALUNet decoder: 5-stage with DGA + EABlock + Conv.
    MALUNet 解码器：5 阶段，前 3 阶段为注意力模块，后 2 阶段为卷积。

    Has internal skip connections (additive fusion).
    内部具有跳跃连接（加法融合）。

    Architecture / 架构:
      dec1: DGA(c5->c4) + EA(c4) + skip_t5
      dec2: DGA(c4->c3) + EA(c3) + upsample + skip_t4
      dec3: DGA(c3->c2) + EA(c2) + upsample + skip_t3
      dec4: Conv3x3(c2->c1) + upsample + skip_t2
      dec5: Conv3x3(c1->c0) + upsample + skip_t1

    Args:
        encoder_channels: Encoder output channel list. / 编码器输出通道列表。
        bottleneck_channels: Bottleneck channels (same as encoder_channels[-1]).
                             瓶颈通道数（与 encoder_channels[-1] 相同）。
        c_list: Full channel list [c0..c5]. / 完整通道列表。
        bridge: Whether to use SCABridge. / 是否使用 SCA 桥。
        split_att: SCABridge attention type. / SCA 桥注意力类型。
    """
    has_internal_skip = True
    required_skip_stages = 5
    requires_encoder = "malunet_enc"  # 5-stage encoder with specific spatial hierarchy

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 c_list=(8, 16, 24, 32, 48, 64),
                 bridge: bool = True, split_att: str = 'fc', **kwargs):
        super().__init__()
        c_list = list(c_list)

        self.bridge = bridge

        # Project encoder features to c_list channels if mismatch
        self.projections = nn.ModuleList()
        self.bottleneck_proj = None
        if encoder_channels is not None and len(encoder_channels) == 5:
            for enc_ch, dec_ch in zip(encoder_channels, c_list[:5]):
                if enc_ch != dec_ch:
                    self.projections.append(nn.Conv2d(enc_ch, dec_ch, 1))
                else:
                    self.projections.append(nn.Identity())
            if bottleneck_channels is not None and bottleneck_channels != c_list[5]:
                self.bottleneck_proj = nn.Conv2d(bottleneck_channels, c_list[5], 1)
        else:
            self.projections = None

        # Optional SCA bridge / 可选的 SCA 桥
        if bridge:
            self.scab = SCABridge(c_list, split_att)

        # Decoder stages / 解码器各阶段
        # Stages 1-3: DGA + EA (attention) / 第1-3阶段：DGA + EA（注意力）
        self.decoder1 = nn.Sequential(DilatedGatedAttention(c_list[5], c_list[4]), EABlock(c_list[4]))
        self.decoder2 = nn.Sequential(DilatedGatedAttention(c_list[4], c_list[3]), EABlock(c_list[3]))
        self.decoder3 = nn.Sequential(DilatedGatedAttention(c_list[3], c_list[2]), EABlock(c_list[2]))

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

        # Apply SCA bridge if enabled / 如果启用则应用 SCA 桥
        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab([t1, t2, t3, t4, t5])

        out = bottleneck_feat

        # Decoder stage 1: DGA+EA, skip with t5 / 解码阶段 1
        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = out5 + t5

        # Decoder stage 2: DGA+EA + upsample, skip with t4 / 解码阶段 2
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out4 = out4 + t4

        # Decoder stage 3: DGA+EA + upsample, skip with t3 / 解码阶段 3
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out3 = out3 + t3

        # Decoder stage 4: Conv3x3 + upsample, skip with t2 / 解码阶段 4
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out2 = out2 + t2

        # Decoder stage 5: Conv3x3 + upsample, skip with t1 / 解码阶段 5
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out1 = out1 + t1

        return out1
