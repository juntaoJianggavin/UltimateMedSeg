"""EGE-UNet: Efficient Group Enhanced UNet for Skin Lesion Segmentation.

Faithful reimplementation from:
  https://github.com/JCruan519/EGE-UNet  (MICCAI 2023 Workshop)

Key innovations:
  - Group multi-axis Hadamard Product Attention (GHPA): splits channels into 4 groups,
    applies learnable params along xy/zx/zy axes + depthwise conv.
  - Group Aggregation Bridge (GAB): multi-dilation grouped conv skip connection fusion.
  - Deep Supervision via side outputs at each decoder level.
"""
# Source: https://github.com/JCruan519/EGE-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.utils.timm_compat import trunc_normal_


# ---------------------------------------------------------------------------
# LayerNorm (channels_first)
# ---------------------------------------------------------------------------

class _LayerNorm(nn.Module):
    """ConvNeXt-style LayerNorm supporting channels_first."""
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
# Depthwise Separable Conv
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
# Group Aggregation Bridge (GAB)
# ---------------------------------------------------------------------------

class GroupAggregationBridge(nn.Module):
    """GAB: Fuses high-res and low-res features via grouped dilated convolution."""
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
# Grouped Multi-Axis Hadamard Product Attention (GHPA)
# ---------------------------------------------------------------------------

class GHPA(nn.Module):
    """Grouped multi-axis Hadamard Product Attention."""
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.ones(1, c_dim_in, x, y))
        self.conv_xy = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.ones(1, 1, c_dim_in, x))
        self.conv_zx = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.ones(1, 1, c_dim_in, y))
        self.conv_zy = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
            nn.GELU(),
            nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1), nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in))

        self.norm1 = _LayerNorm(dim_in, data_format='channels_first')
        self.norm2 = _LayerNorm(dim_in, data_format='channels_first')
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1))

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # xy
        x1 = x1 * self.conv_xy(F.interpolate(self.params_xy, size=x1.shape[2:4],
                                               mode='bilinear', align_corners=True))
        # zx
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2 * self.conv_zx(F.interpolate(self.params_zx, size=x2.shape[2:4],
                                               mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # zy
        x3 = x3.permute(0, 2, 1, 3)
        x3 = x3 * self.conv_zy(F.interpolate(self.params_zy, size=x3.shape[2:4],
                                               mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        # dw
        x4 = self.dw(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x


# ---------------------------------------------------------------------------
# EGE-UNet
# ---------------------------------------------------------------------------

class EGEUNet(nn.Module):
    """EGE-UNet: Efficient Group Enhanced UNet."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 c_list=(8, 16, 24, 32, 48, 64), bridge=True, deep_supervision=False, **kwargs):
        super().__init__()
        c_list = list(c_list)
        self.bridge = bridge
        self.deep_supervision = deep_supervision

        # Encoder
        self.encoder1 = nn.Conv2d(in_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        self.encoder4 = GHPA(c_list[2], c_list[3])
        self.encoder5 = GHPA(c_list[3], c_list[4])
        self.encoder6 = GHPA(c_list[4], c_list[5])

        # Encoder batch norms
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        # Bridge
        if bridge:
            self.GAB1 = GroupAggregationBridge(c_list[1], c_list[0])
            self.GAB2 = GroupAggregationBridge(c_list[2], c_list[1])
            self.GAB3 = GroupAggregationBridge(c_list[3], c_list[2])
            self.GAB4 = GroupAggregationBridge(c_list[4], c_list[3])
            self.GAB5 = GroupAggregationBridge(c_list[5], c_list[4])

        # GT deep supervision side convs
        self.gt_conv1 = nn.Conv2d(c_list[4], 1, 1)
        self.gt_conv2 = nn.Conv2d(c_list[3], 1, 1)
        self.gt_conv3 = nn.Conv2d(c_list[2], 1, 1)
        self.gt_conv4 = nn.Conv2d(c_list[1], 1, 1)
        self.gt_conv5 = nn.Conv2d(c_list[0], 1, 1)

        # Decoder
        self.decoder1 = GHPA(c_list[5], c_list[4])
        self.decoder2 = GHPA(c_list[4], c_list[3])
        self.decoder3 = GHPA(c_list[3], c_list[2])
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
        t1 = out  # c0, H/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # c1, H/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # c2, H/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # c3, H/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # c4, H/32

        out = F.gelu(self.encoder6(out))  # c5, H/32

        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        gt_pre5 = self.gt_conv1(out5)
        if self.bridge:
            t5 = self.GAB5(out, t5, gt_pre5)
        out5 = out5 + t5

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        gt_pre4 = self.gt_conv2(out4)
        if self.bridge:
            t4 = self.GAB4(t5, t4, gt_pre4)
        out4 = out4 + t4

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        gt_pre3 = self.gt_conv3(out3)
        if self.bridge:
            t3 = self.GAB3(t4, t3, gt_pre3)
        out3 = out3 + t3

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        gt_pre2 = self.gt_conv4(out2)
        if self.bridge:
            t2 = self.GAB2(t3, t2, gt_pre2)
        out2 = out2 + t2

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),
                                     scale_factor=2, mode='bilinear', align_corners=True))
        if self.bridge:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
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
