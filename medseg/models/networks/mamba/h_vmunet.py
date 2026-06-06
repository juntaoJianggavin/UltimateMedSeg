"""H-vmunet: High-order Vision Mamba UNet for Medical Image Segmentation.

Faithful reimplementation from:
  https://github.com/wurenkai/H-vmunet  (Neurocomputing 2025, 135+ stars)

Key innovations:
  - H_SS2D: High-order Selective Scan 2D that splits channels into multiple orders
    and applies pointwise conv multiplication + SS2D at each order level.
  - SC_Att_Bridge: Spatial-Channel Attention Bridge for skip connections.
  - Local_SS2D: Local frequency + depthwise conv + SS2D branch.

Reuses SS2D from the project's vmunet_encoder to avoid code duplication.
"""
# Source: https://github.com/wurenkai/H-vmunet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

from medseg.models.encoders.vmunet_encoder import SS2D


# ---------------------------------------------------------------------------
# LayerNorm (channels_first / channels_last)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm supporting both channels_last (B,H,W,C) and channels_first (B,C,H,W)."""
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
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# ---------------------------------------------------------------------------
# Local_SS2D: local depthwise conv + SS2D branch
# ---------------------------------------------------------------------------

class Local_SS2D(nn.Module):
    """Local branch: depthwise conv + SS2D on split channels."""
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1,
                            bias=False, groups=dim // 2)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.SS2D = SS2D(d_model=dim // 2, dropout=0, d_state=16)

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)
        B, C, a, b = x2.shape
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.SS2D(x2)
        x2 = x2.permute(0, 3, 1, 2)
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


# ---------------------------------------------------------------------------
# H_SS2D: High-order Selective Scan 2D
# ---------------------------------------------------------------------------

class H_SS2D(nn.Module):
    """High-order SS2D: splits channels into multiple orders,
    applies pointwise conv multiplication + SS2D at each order level.

    Args:
        dim: Input channel dimension.
        order: Number of hierarchical orders (default 5).
        gflayer: Optional global filter layer (default Local_SS2D).
        s: Scale factor for depthwise conv branch.
        d_state: SS2D state dimension.
    """
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0, d_state=16):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims),
                                    kernel_size=7, padding=3, bias=True,
                                    groups=sum(self.dims))
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        # SS2D at each order level
        self.ss2d_in = SS2D(d_model=self.dims[0], dropout=0, d_state=d_state)
        self.ss2d_layers = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.ss2d_layers.append(SS2D(d_model=self.dims[i], dropout=0, d_state=d_state))

        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)

        # First order: multiply + SS2D
        x = pwa * dw_list[0]
        x = x.permute(0, 2, 3, 1)
        x = self.ss2d_in(x)
        x = x.permute(0, 3, 1, 2)

        # Higher orders
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
            x = x.permute(0, 2, 3, 1)
            x = self.ss2d_layers[i](x)
            x = x.permute(0, 3, 1, 2)

        x = self.proj_out(x)
        return x


# ---------------------------------------------------------------------------
# H_VSS Block
# ---------------------------------------------------------------------------

class HVSSBlock(nn.Module):
    """H-VSS Block: LayerNorm→H_SS2D + LayerScale → LayerNorm→FFN(4x) + LayerScale."""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,
                 h_ss2d_cls=None):
        super().__init__()
        if h_ss2d_cls is None:
            h_ss2d_cls = partial(H_SS2D, order=2, s=1 / 3, gflayer=Local_SS2D)

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.h_ss2d = h_ss2d_cls(dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0 else None)
        self.gamma2 = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0 else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        gamma1 = self.gamma1.view(C, 1, 1) if self.gamma1 is not None else 1
        x = x + self.drop_path(gamma1 * self.h_ss2d(self.norm1(x)))

        inp = x
        x = x.permute(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
        x = inp + self.drop_path(x)
        return x


# ---------------------------------------------------------------------------
# SC_Att_Bridge: Spatial-Channel Attention Bridge
# ---------------------------------------------------------------------------

class Channel_Att_Bridge(nn.Module):
    """Channel attention bridge across encoder features."""
    def __init__(self, c_list):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.atts = nn.ModuleList([
            nn.Linear(c_list_sum, c_list[i]) for i in range(5)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, *features):
        att = torch.cat([self.avgpool(f) for f in features], dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        results = []
        for i, att_layer in enumerate(self.atts):
            a = self.sigmoid(att_layer(att))
            a = a.transpose(-1, -2).unsqueeze(-1).expand_as(features[i])
            results.append(a)
        return results


class Spatial_Att_Bridge(nn.Module):
    """Spatial attention bridge with shared conv."""
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
            nn.Sigmoid())

    def forward(self, *features):
        results = []
        for t in features:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = self.shared_conv2d(torch.cat([avg_out, max_out], dim=1))
            results.append(att)
        return results


class SC_Att_Bridge(nn.Module):
    """Spatial-Channel Attention Bridge: combines channel + spatial attention."""
    def __init__(self, c_list):
        super().__init__()
        self.catt = Channel_Att_Bridge(c_list)
        self.satt = Spatial_Att_Bridge()

    def forward(self, *features):
        r = list(features)
        satts = self.satt(*features)
        features = [s * f for s, f in zip(satts, features)]
        r_ = list(features)
        features = [f + ri for f, ri in zip(features, r)]
        catts = self.catt(*features)
        features = [c * f for c, f in zip(catts, features)]
        return [f + ri for f, ri in zip(features, r_)]


# ---------------------------------------------------------------------------
# HVMUNet: main model
# ---------------------------------------------------------------------------

class HVMUNet(nn.Module):
    """H-vmunet: High-order Vision Mamba UNet.

    Architecture:
      6-stage conv encoder with H_SS2D blocks (stages 3-6) + MaxPool2d downsample
      → SC_Att_Bridge for skip connections
      → 5-stage decoder with H_SS2D blocks + bilinear upsample

    Args:
        in_channels: Input channels.
        num_classes: Number of output classes.
        img_size: Input image size.
        c_list: Channel list for 6 encoder stages.
        depths: Number of H_SS2D blocks per stage (4 stages with blocks).
        drop_path_rate: Stochastic depth rate.
        bridge: Whether to use SC_Att_Bridge.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 c_list=None, depths=None, drop_path_rate=0.0,
                 bridge=True, deep_supervision=False, **kwargs):
        super().__init__()
        if c_list is None:
            c_list = [8, 16, 32, 64, 128, 256]
        if depths is None:
            depths = [2, 2, 2, 2]

        self.bridge = bridge
        self.deep_supervision = deep_supervision

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # H_SS2D configs for 4 block stages (order 2,3,4,5)
        h_ss2d_configs = [
            partial(H_SS2D, order=2, s=1 / 3, gflayer=Local_SS2D),
            partial(H_SS2D, order=3, s=1 / 3, gflayer=Local_SS2D),
            partial(H_SS2D, order=4, s=1 / 3, gflayer=Local_SS2D),
            partial(H_SS2D, order=5, s=1 / 3, gflayer=Local_SS2D),
        ]

        # Encoder
        self.encoder1 = nn.Conv2d(in_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)

        self.encoder3 = nn.Sequential(
            *[HVSSBlock(c_list[1], drop_path=dp_rates[j],
                        h_ss2d_cls=h_ss2d_configs[0]) for j in range(depths[0])],
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1))

        self.encoder4 = nn.Sequential(
            *[HVSSBlock(c_list[2], drop_path=dp_rates[sum(depths[:1]) + j],
                        h_ss2d_cls=h_ss2d_configs[1]) for j in range(depths[1])],
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1))

        self.encoder5 = nn.Sequential(
            *[HVSSBlock(c_list[3], drop_path=dp_rates[sum(depths[:2]) + j],
                        h_ss2d_cls=h_ss2d_configs[2]) for j in range(depths[2])],
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1))

        self.encoder6 = nn.Sequential(
            *[HVSSBlock(c_list[4], drop_path=dp_rates[sum(depths[:3]) + j],
                        h_ss2d_cls=h_ss2d_configs[3]) for j in range(depths[3])],
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1))

        # SC_Att_Bridge
        if bridge:
            self.scab = SC_Att_Bridge(c_list)

        # Decoder
        self.decoder1 = nn.Sequential(
            *[HVSSBlock(c_list[5], drop_path=dp_rates[sum(depths[:3])],
                        h_ss2d_cls=h_ss2d_configs[3]) for _ in range(depths[3])],
            nn.Conv2d(c_list[5], c_list[4], 3, stride=1, padding=1))

        self.decoder2 = nn.Sequential(
            *[HVSSBlock(c_list[4], drop_path=dp_rates[sum(depths[:2]) + j],
                        h_ss2d_cls=h_ss2d_configs[2]) for j in range(depths[2])],
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1))

        self.decoder3 = nn.Sequential(
            *[HVSSBlock(c_list[3], drop_path=dp_rates[sum(depths[:1]) + j],
                        h_ss2d_cls=h_ss2d_configs[1]) for j in range(depths[1])],
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1))

        self.decoder4 = nn.Sequential(
            *[HVSSBlock(c_list[2], drop_path=dp_rates[j],
                        h_ss2d_cls=h_ss2d_configs[0]) for j in range(depths[0])],
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1))

        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1)

        # GroupNorm
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
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
        t1 = out  # (B, c0, H/2, W/2)

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # (B, c1, H/4, W/4)

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # (B, c2, H/8, W/8)

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # (B, c3, H/16, W/16)

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # (B, c4, H/32, W/32)

        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)

        out = F.gelu(self.encoder6(out))  # (B, c5, H/32, W/32)

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
            input_size = out0.shape[2:]
            aux = []
            for feat, head in zip([out5, out4, out3, out2], self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear', align_corners=True)
                aux.append(a)
            return [out0] + aux

        return out0
