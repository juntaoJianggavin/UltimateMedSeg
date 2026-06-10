"""UltraLight-VM-UNet: Ultra-Lightweight Vision Mamba UNet.

Faithful reimplementation from:
  https://github.com/wurenkai/UltraLight-VM-UNet  (Patterns/Cell 2024)

Key innovations:
  - PVMLayer: Parallel Vision Mamba layer - splits channels into 4 groups,
    applies Mamba SSM to each group independently.
  - SC_Att_Bridge: Spatial-Channel Attention Bridge for skip connections.
  - Ultra-lightweight (~0.05M params with default c_list).
"""
# Source: https://github.com/wurenkai/UltraLight-VM-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.utils.timm_compat import trunc_normal_

from .umamba import MambaSSM


# ---------------------------------------------------------------------------
# PVMLayer: Parallel Vision Mamba
# ---------------------------------------------------------------------------

class PVMLayer(nn.Module):
    """Parallel Vision Mamba: splits channels into 4 groups, applies Mamba to each."""
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = MambaSSM(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        return x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)


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
# UltraLight-VM-UNet
# ---------------------------------------------------------------------------

class UltraLightVMUNet(nn.Module):
    """UltraLight-VM-UNet: Ultra-Lightweight Vision Mamba UNet."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 c_list=(8, 16, 24, 32, 48, 64), split_att='fc', bridge=True, deep_supervision=False, **kwargs):
        super().__init__()
        c_list = list(c_list)
        self.bridge = bridge
        self.deep_supervision = deep_supervision

        # Encoder (first 3 stages: conv, last 3: PVMLayer)
        self.encoder1 = nn.Conv2d(in_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        self.encoder4 = PVMLayer(c_list[2], c_list[3])
        self.encoder5 = PVMLayer(c_list[3], c_list[4])
        self.encoder6 = PVMLayer(c_list[4], c_list[5])

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        if bridge:
            self.scab = SCABridge(c_list, split_att)

        # Decoder
        self.decoder1 = PVMLayer(c_list[5], c_list[4])
        self.decoder2 = PVMLayer(c_list[4], c_list[3])
        self.decoder3 = PVMLayer(c_list[3], c_list[2])
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
            input_size = out0.shape[2:]
            aux = []
            for feat, head in zip([out5, out4, out3, out2], self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear', align_corners=True)
                aux.append(a)
            return [out0] + aux

        return out0
