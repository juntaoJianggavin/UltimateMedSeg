"""UltraLBM-UNet: Ultra-Lightweight Bidirectional Mamba UNet.

Faithful reimplementation from:
  https://github.com/LinLinLin-X/UltraLBM-UNet  (2024)

Key innovations:
  - LMBP: Local Mamba Block with parallel DW separable conv branches.
  - GLMBP: Global-Local Mamba Block with bidirectional Mamba scan + DW conv.
  - Learnable skip scale parameter.
"""
# Source: https://github.com/wurenkai/UltraLight-VM-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from medseg.models.networks.mamba.umamba import MambaSSM


# ---------------------------------------------------------------------------
# Depthwise Separable Conv with BN + Act
# ---------------------------------------------------------------------------

class DWSepConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.act_dw = nn.GELU()
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.act_pw = nn.GELU()

    def forward(self, x):
        x = self.act_dw(self.bn_dw(self.dw(x)))
        return self.act_pw(self.bn_pw(self.pw(x)))


# ---------------------------------------------------------------------------
# LMBP: Local Mamba Block (Parallel DW conv branches, no Mamba)
# ---------------------------------------------------------------------------

class LMBP(nn.Module):
    """Local Mamba Block Parallel: 3 DW conv branches + 1 identity."""
    def __init__(self, input_dim, output_dim, sep_conv_kernel=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        padding = (sep_conv_kernel - 1) // 2
        c_quarter = input_dim // 4
        self.sep_conv1 = DWSepConvBNAct(c_quarter, c_quarter, sep_conv_kernel, 1, padding)
        self.sep_conv2 = DWSepConvBNAct(c_quarter, c_quarter, sep_conv_kernel, 1, padding)
        self.sep_conv3 = DWSepConvBNAct(c_quarter, c_quarter, sep_conv_kernel, 1, padding)
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        c_quarter = C // 4

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        # Branch 1-3: DW sep conv
        def _conv_branch(xi, conv):
            xi_img = xi.transpose(-1, -2).reshape(B, c_quarter, *img_dims)
            xi_img = conv(xi_img)
            return xi_img.reshape(B, c_quarter, n_tokens).transpose(-1, -2)

        x_proc1 = _conv_branch(x1, self.sep_conv1) + self.skip_scale * x1
        x_proc2 = _conv_branch(x2, self.sep_conv2) + self.skip_scale * x2
        x_proc3 = _conv_branch(x3, self.sep_conv3) + self.skip_scale * x3
        x_proc4 = x4 + self.skip_scale * x4  # identity

        x_mamba = torch.cat([x_proc1, x_proc2, x_proc3, x_proc4], dim=2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        return x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)


# ---------------------------------------------------------------------------
# GLMBP: Global-Local Mamba Block (bidirectional Mamba + DW conv)
# ---------------------------------------------------------------------------

class GLMBP(nn.Module):
    """Global-Local Mamba Block: bidirectional Mamba for 2 branches + DW conv for 1 + identity."""
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2,
                 sep_conv_kernel=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = MambaSSM(
            d_model=input_dim // 4, d_state=d_state, d_conv=d_conv, expand=expand)
        padding = (sep_conv_kernel - 1) // 2
        self.sep_conv = DWSepConvBNAct(input_dim // 4, input_dim // 4,
                                        sep_conv_kernel, 1, padding)
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        c_quarter = C // 4

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        # Branch 1-2: bidirectional Mamba
        x1_fwd = self.mamba(x1)
        x1_bwd = torch.flip(self.mamba(torch.flip(x1, dims=[1])), dims=[1])
        x_proc1 = x1_fwd + x1_bwd + self.skip_scale * x1

        x2_fwd = self.mamba(x2)
        x2_bwd = torch.flip(self.mamba(torch.flip(x2, dims=[1])), dims=[1])
        x_proc2 = x2_fwd + x2_bwd + self.skip_scale * x2

        # Branch 3: DW sep conv
        x3_img = x3.transpose(-1, -2).reshape(B, c_quarter, *img_dims)
        x3_img = self.sep_conv(x3_img)
        x3_flat = x3_img.reshape(B, c_quarter, n_tokens).transpose(-1, -2)
        x_proc3 = x3_flat + self.skip_scale * x3

        # Branch 4: identity
        x_proc4 = x4 + self.skip_scale * x4

        x_mamba = torch.cat([x_proc1, x_proc2, x_proc3, x_proc4], dim=2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        return x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)


# ---------------------------------------------------------------------------
# UltraLBM-UNet
# ---------------------------------------------------------------------------

class UltraLBMUNet(nn.Module):
    """UltraLBM-UNet: Ultra-Lightweight Bidirectional Mamba UNet."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 c_list=(8, 16, 24, 32, 48, 64), channel_multiplier=1.0,
                 learnable_skip=True, deep_supervision=False, **kwargs):
        super().__init__()
        if channel_multiplier != 1.0:
            c_list = [max(4, (int(c * channel_multiplier) // 4) * 4) for c in c_list]
        c_list = list(c_list)

        # Encoder
        self.encoder1 = nn.Conv2d(in_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        self.encoder4 = LMBP(c_list[2], c_list[3], sep_conv_kernel=3)
        self.encoder5 = GLMBP(c_list[3], c_list[4], sep_conv_kernel=5)
        self.encoder6 = GLMBP(c_list[4], c_list[5], sep_conv_kernel=7)

        self.ebn1 = nn.GroupNorm(min(4, c_list[0]), c_list[0])
        self.ebn2 = nn.GroupNorm(min(4, c_list[1]), c_list[1])
        self.ebn3 = nn.GroupNorm(min(4, c_list[2]), c_list[2])
        self.ebn4 = nn.GroupNorm(min(4, c_list[3]), c_list[3])
        self.ebn5 = nn.GroupNorm(min(4, c_list[4]), c_list[4])

        # Decoder
        self.decoder1 = GLMBP(c_list[5], c_list[4], sep_conv_kernel=7)
        self.decoder2 = GLMBP(c_list[4], c_list[3], sep_conv_kernel=5)
        self.decoder3 = LMBP(c_list[3], c_list[2], sep_conv_kernel=3)
        self.decoder4 = nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1)

        self.dbn1 = nn.GroupNorm(min(4, c_list[4]), c_list[4])
        self.dbn2 = nn.GroupNorm(min(4, c_list[3]), c_list[3])
        self.dbn3 = nn.GroupNorm(min(4, c_list[2]), c_list[2])
        self.dbn4 = nn.GroupNorm(min(4, c_list[1]), c_list[1])
        self.dbn5 = nn.GroupNorm(min(4, c_list[0]), c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.deep_supervision = deep_supervision
        self.learnable_skip = learnable_skip
        if learnable_skip:
            self.k = nn.Parameter(torch.ones(1))

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
        k = self.k if self.learnable_skip else 1.0

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

        out = F.gelu(self.encoder6(out))

        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = out5 + k * t5
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out4 = out4 + k * t4
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out3 = out3 + k * t3
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out2 = out2 + k * t2
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        out1 = out1 + k * t1

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
