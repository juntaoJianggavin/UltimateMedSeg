"""MALUNet: Multi-Axis Large-kernel UNet for Medical Image Segmentation.

Faithful reimplementation from:
  https://github.com/JCruan519/MALUNet  (2023)

Key innovations:
  - DilatedGatedAttention (DGA): Multi-dilation grouped depthwise conv + Gated Attention Unit.
  - External Attention block (EA): Low-rank linear attention.
  - SC_Att_Bridge: Channel + Spatial attention bridge for skip connections.
"""
# Source: https://github.com/JCruan519/MALUNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.utils.timm_compat import trunc_normal_


# ---------------------------------------------------------------------------
# DepthWise Conv
# ---------------------------------------------------------------------------

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


# ---------------------------------------------------------------------------
# Gated Attention Unit
# ---------------------------------------------------------------------------

class GatedAttentionUnit(nn.Module):
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
# Dilated Gated Attention
# ---------------------------------------------------------------------------

class DilatedGatedAttention(nn.Module):
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
# External Attention Block
# ---------------------------------------------------------------------------

class EABlock(nn.Module):
    """External Attention: low-rank linear attention."""
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
# SC_Att_Bridge (Spatial-Channel Attention Bridge)
# ---------------------------------------------------------------------------

class _ChannelAttBridge(nn.Module):
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
# MALUNet
# ---------------------------------------------------------------------------

class MALUNet(nn.Module):
    """MALUNet: Multi-Axis Large-kernel UNet."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 c_list=(8, 16, 24, 32, 48, 64), split_att='fc', bridge=True, deep_supervision=False, **kwargs):
        super().__init__()
        c_list = list(c_list)
        self.bridge = bridge
        self.deep_supervision = deep_supervision

        # Encoder
        self.encoder1 = nn.Conv2d(in_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        self.encoder4 = nn.Sequential(EABlock(c_list[2]), DilatedGatedAttention(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(EABlock(c_list[3]), DilatedGatedAttention(c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(EABlock(c_list[4]), DilatedGatedAttention(c_list[4], c_list[5]))

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        if bridge:
            self.scab = SCABridge(c_list, split_att)

        # Decoder
        self.decoder1 = nn.Sequential(DilatedGatedAttention(c_list[5], c_list[4]), EABlock(c_list[4]))
        self.decoder2 = nn.Sequential(DilatedGatedAttention(c_list[4], c_list[3]), EABlock(c_list[3]))
        self.decoder3 = nn.Sequential(DilatedGatedAttention(c_list[3], c_list[2]), EABlock(c_list[2]))
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1)

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        # Deep supervision side output heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(c_list[4], num_classes, 1),
                nn.Conv2d(c_list[3], num_classes, 1),
                nn.Conv2d(c_list[2], num_classes, 1),
                nn.Conv2d(c_list[1], num_classes, 1),
            ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out

        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab([t1, t2, t3, t4, t5])

        out = F.gelu(self.encoder6(out))

        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = out5 + t5
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out4 = out4 + t4
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out3 = out3 + t3
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out2 = out2 + t2
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out1 = out1 + t1

        out0 = F.interpolate(self.final(out1), scale_factor=2,
                             mode='bilinear', align_corners=True)

        if self.training and self.deep_supervision:
            return self._ds_forward(out0, [out5, out4, out3, out2], out0.shape[2:])

        return out0

    def _ds_forward(self, out0, intermediates, input_size):
        """Deep supervision: return [main_output, aux1, aux2, ...]."""
        aux = []
        for feat, head in zip(intermediates, self.ds_heads):
            a = head(feat)
            if a.shape[2:] != input_size:
                a = F.interpolate(a, size=input_size, mode='bilinear', align_corners=True)
            aux.append(a)
        return [out0] + aux
