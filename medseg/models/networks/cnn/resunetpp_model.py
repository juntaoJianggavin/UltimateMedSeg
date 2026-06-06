"""ResUNet++: An Advanced Architecture for Medical Image Segmentation.

Reference:
    Jha et al., "ResUNet++: An Advanced Architecture for Medical Image
    Segmentation", IEEE ISM 2019.
    https://github.com/DebeshJha/ResUNetPlusPlus
    https://github.com/rishikksh20/ResUnet/blob/master/core/res_unet_plus.py

Architecture: Residual encoder + SE attention + ASPP bottleneck + attention
gated decoder with skip connections.
"""
# Source: https://github.com/DebeshJha/ResUNetPlusPlus

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualConv(nn.Module):
    """Residual conv block faithful to original ResUNet++."""

    def __init__(self, in_ch, out_ch, stride=1, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class _SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block faithful to original."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling faithful to original."""

    def __init__(self, in_ch, out_ch, rate=None):
        super().__init__()
        if rate is None:
            rate = [6, 12, 18]
        self.blocks = nn.ModuleList()
        for r in rate:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=r, dilation=r),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch),
            ))
        self.output = nn.Conv2d(len(rate) * out_ch, out_ch, 1)

    def forward(self, x):
        outs = [block(x) for block in self.blocks]
        out = torch.cat(outs, dim=1)
        return self.output(out)


class _AttentionBlock(nn.Module):
    """Attention block faithful to original ResUNet++."""

    def __init__(self, input_encoder, input_decoder, output_dim):
        super().__init__()
        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )
        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )
        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class ResUNetPP(nn.Module):
    """ResUNet++ with SE blocks, ASPP bottleneck, and attention-gated decoder.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: spatial size (unused, kept for API consistency).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224):
        super().__init__()
        filters = [32, 64, 128, 256, 512]

        # Encoder
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], 3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], 3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], 3, padding=1),
        )

        self.se1 = _SqueezeExcite(filters[0])
        self.res_conv1 = _ResidualConv(filters[0], filters[1], 2, 1)

        self.se2 = _SqueezeExcite(filters[1])
        self.res_conv2 = _ResidualConv(filters[1], filters[2], 2, 1)

        self.se3 = _SqueezeExcite(filters[2])
        self.res_conv3 = _ResidualConv(filters[2], filters[3], 2, 1)

        # ASPP bottleneck
        self.aspp_bridge = _ASPP(filters[3], filters[4])

        # Decoder with attention gates
        self.attn1 = _AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear',
                                      align_corners=True)
        self.up_res_conv1 = _ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = _AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear',
                                      align_corners=True)
        self.up_res_conv2 = _ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = _AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear',
                                      align_corners=True)
        self.up_res_conv3 = _ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        # ASPP output
        self.aspp_out = _ASPP(filters[1], filters[0])

        # Final output
        self.out_conv = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        inp_size = x.shape[2:]

        # Encoder
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.se1(x1)
        x2 = self.res_conv1(x2)

        x3 = self.se2(x2)
        x3 = self.res_conv2(x3)

        x4 = self.se3(x3)
        x4 = self.res_conv3(x4)

        # ASPP bridge
        x5 = self.aspp_bridge(x4)

        # Decoder
        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_res_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_res_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_res_conv3(x8)

        # ASPP output + final conv
        x9 = self.aspp_out(x8)
        out = self.out_conv(x9)

        # Upsample if needed
        if out.shape[-2:] != torch.Size(inp_size):
            out = F.interpolate(out, size=inp_size, mode='bilinear',
                                align_corners=True)
        return out
