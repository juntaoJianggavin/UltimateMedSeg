"""VM-UNet-V2: Rethinking Vision Mamba UNet for Medical Image Segmentation.

Faithful port from https://github.com/nobodyplayer1/VM-UNetV2.

Architecture:
    Encoder:  VSSM backbone (PatchEmbed2D + 4x VSSLayer w/ SS2D selective scan)
    Refinement: CBAM (ChannelAttention + SpatialAttention) per scale
    Projection: TransLayer (1x1 Conv+BN+ReLU) -> mid_channel
    Aggregation: SDI (Spatial-Dimensional Integration) per scale
    Decoder:  Progressive ConvTranspose2d upsampling + skip addition
    Head:     Deep supervision with sigmoid (binary) or raw logits (multi-class)

Reference:
    VM-UNET-V2: Rethinking Vision Mamba UNet for Medical Image
    Segmentation. arXiv 2024.
"""
# Source: https://github.com/nobodyplayer1/VM-UNetV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# CBAM modules (faithful to original vmunet_v2.py)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """CBAM Channel Attention."""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


# ---------------------------------------------------------------------------
# Helper modules (faithful to original vmunet_v2.py)
# ---------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    """Conv + BN + ReLU (used as TransLayer)."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SDI(nn.Module):
    """Spatial-Dimensional Integration module.

    Takes a list of 4 multi-scale features and an anchor tensor.
    Each feature is resized to the anchor's spatial size, passed through
    a 3x3 conv, then all are element-wise multiplied together.
    """

    def __init__(self, channel):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
            for _ in range(4)
        ])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                  mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


# ---------------------------------------------------------------------------
# VMUNetV2 – full model
# ---------------------------------------------------------------------------

class VMUNetV2(nn.Module):
    """VM-UNet-V2: Vision Mamba UNet V2.

    Faithful to https://github.com/nobodyplayer1/VM-UNetV2

    Args:
        in_channels: Number of input channels (default 3).
        num_classes: Number of segmentation classes (default 2).
        img_size: Input spatial size (default 224, must be divisible by 32).
        embed_dim: Base embedding dimension for VSSM encoder (default 64).
        depths: Block counts per encoder stage (default [2, 2, 6, 2]).
        depths_decoder: Block counts for decoder stages (used by VSSM init).
        mid_channel: Intermediate channel after TransLayer projection.
            Defaults to embed_dim // 2.
        drop_path_rate: Stochastic depth rate (default 0.2).
        deep_supervision: Enable deep supervision output (default True).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=64, depths=None, depths_decoder=None,
                 mid_channel=None, drop_path_rate=0.2,
                 deep_supervision=True, **kwargs):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        depths_decoder = depths_decoder or [2, 6, 2, 2]
        mid_channel = mid_channel or embed_dim // 2
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        # ---- Encoder: VSSM backbone ----
        from medseg.models.encoders.mamba.vm_unet_v2_encoder import VMUNetV2Encoder
        self.encoder = VMUNetV2Encoder(
            in_channels=in_channels, img_size=img_size,
            embed_dim=embed_dim, depths=depths,
            drop_path_rate=drop_path_rate,
        )

        # ---- CBAM attention per scale ----
        self.ca_1 = ChannelAttention(dims[0])
        self.sa_1 = SpatialAttention()
        self.ca_2 = ChannelAttention(dims[1])
        self.sa_2 = SpatialAttention()
        self.ca_3 = ChannelAttention(dims[2])
        self.sa_3 = SpatialAttention()
        self.ca_4 = ChannelAttention(dims[3])
        self.sa_4 = SpatialAttention()

        # ---- TransLayer: project each scale to mid_channel ----
        self.Translayer_1 = BasicConv2d(dims[0], mid_channel, 1)
        self.Translayer_2 = BasicConv2d(dims[1], mid_channel, 1)
        self.Translayer_3 = BasicConv2d(dims[2], mid_channel, 1)
        self.Translayer_4 = BasicConv2d(dims[3], mid_channel, 1)

        # ---- SDI: cross-scale integration ----
        self.sdi_1 = SDI(mid_channel)
        self.sdi_2 = SDI(mid_channel)
        self.sdi_3 = SDI(mid_channel)
        self.sdi_4 = SDI(mid_channel)

        # ---- Segmentation heads (one per scale) ----
        self.seg_outs = nn.ModuleList([
            nn.Conv2d(mid_channel, num_classes, 1, 1) for _ in range(4)
        ])

        # ---- Decoder deconvolution layers ----
        # ConvTranspose2d with output_padding=1 for exact 2x upsampling
        self.deconv2 = nn.ConvTranspose2d(mid_channel, mid_channel,
                                          kernel_size=4, stride=2,
                                          padding=1, output_padding=1,
                                          bias=False)
        self.deconv3 = nn.ConvTranspose2d(mid_channel, mid_channel,
                                          kernel_size=4, stride=2,
                                          padding=1, output_padding=1,
                                          bias=False)
        self.deconv4 = nn.ConvTranspose2d(mid_channel, mid_channel,
                                          kernel_size=4, stride=2,
                                          padding=1, output_padding=1,
                                          bias=False)
        # Deep supervision merge: upsample second-finet seg_out by 2x
        self.deconv6 = nn.ConvTranspose2d(num_classes, num_classes, 3,
                                          stride=2, padding=1,
                                          output_padding=1)

    def forward(self, x):
        H_in, W_in = x.shape[2:]

        # Handle single-channel input
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # ---- Encoder ----
        feats = self.encoder(x)  # 4 tensors: f1(H/4), f2(H/8), f3(H/16), f4(H/32)
        f1, f2, f3, f4 = feats  # already in (B, C, H, W) format

        # ---- CBAM refinement + TransLayer projection ----
        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)   # -> mid_channel

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)   # -> mid_channel

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)   # -> mid_channel

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)   # -> mid_channel

        # ---- SDI aggregation ----
        f41 = self.sdi_4([f1, f2, f3, f4], f4)   # at f4 resolution
        f31 = self.sdi_3([f1, f2, f3, f4], f3)   # at f3 resolution
        f21 = self.sdi_2([f1, f2, f3, f4], f2)   # at f2 resolution
        f11 = self.sdi_1([f1, f2, f3, f4], f1)   # at f1 resolution

        # ---- Decoder: progressive upsampling + skip addition ----
        seg_outs = []
        seg_outs.append(self.seg_outs[0](f41))

        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))

        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))

        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))

        # ---- Upsample all seg outputs to patch-embed resolution (4x) ----
        for i in range(len(seg_outs)):
            seg_outs[i] = F.interpolate(seg_outs[i], scale_factor=4,
                                        mode='bilinear', align_corners=False)

        # ---- Final output ----
        if self.deep_supervision:
            # Merge two finest levels: best + upsampled 2nd-best
            out_best = seg_outs[-1]
            out_second = self.deconv6(seg_outs[-2])
            result = out_best + out_second
            if self.num_classes == 1:
                result = torch.sigmoid(result)
            # Crop/pad to original input size
            if result.shape[2:] != (H_in, W_in):
                result = F.interpolate(result, size=(H_in, W_in),
                                       mode='bilinear', align_corners=False)
            return result
        else:
            result = seg_outs[-1]
            if self.num_classes == 1:
                result = torch.sigmoid(result)
            if result.shape[2:] != (H_in, W_in):
                result = F.interpolate(result, size=(H_in, W_in),
                                       mode='bilinear', align_corners=False)
            return result
