"""CASCADE: Cascaded Attention Decoder for Medical Image Segmentation.

Reference:
    Md Mostafijur Rahman, Radu Marculescu.
    "Medical Image Segmentation via Cascaded Attention Decoding."
    WACV 2023.
    Upstream code: https://github.com/SLDGroup/CASCADE

Architecture overview:
    - Encoder: PVTv2-B2 (Pyramid Vision Transformer v2) producing 4 multi-scale
      features at strides {4, 8, 16, 32} with channels {64, 128, 320, 512}.
    - Cascaded Attention Decoder (CAD): three decoding stages that progressively
      go /16 -> /8 -> /4 (then bilinear up to input). Each stage applies a
      Convolutional Attention Module (CAM = channel attention + spatial
      attention) and an Aggregation Attention Module (AAM) that merges the
      upsampled deeper feature with the attended skip via concat + 3x3 conv.
    - Final 1x1 head + bilinear upsample to the input resolution.

Self-contained: only torch and timm (for the PVT backbone) are required.
"""
# Source: https://github.com/SLDGroup/CASCADE

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _ConvBNReLU(nn.Module):
    """Conv + BN + ReLU."""

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, relu=True):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _ChannelAttention(nn.Module):
    """SE-style channel attention (avg-pool + max-pool branches)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = self.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn


class _SpatialAttention(nn.Module):
    """Spatial attention from concat([avg, max]) -> 7x7 conv -> sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class _CAM(nn.Module):
    """Convolutional Attention Module: channel attention then spatial attention.

    Applied to a skip feature to suppress noise before fusion with the deeper
    (upsampled) decoder stream. Followed by a 3x3 refinement conv.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction=reduction)
        self.sa = _SpatialAttention(kernel_size=7)
        self.refine = _ConvBNReLU(channels, channels, kernel_size=3)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return self.refine(x)


class _AAM(nn.Module):
    """Aggregation Attention Module.

    Aggregates an upsampled deeper feature ``x_up`` (in_up channels) with the
    attended skip ``x_skip`` (in_skip channels). Following CASCADE, the deeper
    branch is gated by attention computed from the skip stream so that the
    decoder selectively passes context from the previous stage. Output is the
    concat -> 3x3 conv at ``out_channels``.
    """

    def __init__(self, in_up, in_skip, out_channels):
        super().__init__()
        # Project both branches to a common channel dim for gating.
        self.proj_up = _ConvBNReLU(in_up, out_channels, kernel_size=1)
        self.proj_skip = _ConvBNReLU(in_skip, out_channels, kernel_size=1)
        # Gate produced from the skip stream modulates the upsampled branch.
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = _ConvBNReLU(2 * out_channels, out_channels, kernel_size=3)

    def forward(self, x_up, x_skip):
        # Align spatial size: x_up should already be at x_skip's resolution,
        # but be defensive in case of off-by-one rounding.
        if x_up.shape[-2:] != x_skip.shape[-2:]:
            x_up = F.interpolate(
                x_up, size=x_skip.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        u = self.proj_up(x_up)
        s = self.proj_skip(x_skip)
        g = self.gate(s)
        u = u * g
        return self.fuse(torch.cat([u, s], dim=1))


class _DecoderStage(nn.Module):
    """One stage of CASCADE's Cascaded Attention Decoder.

    Steps:
      1. Bilinear-upsample the deeper feature to the skip's resolution.
      2. Apply CAM to the skip feature.
      3. Apply AAM to fuse (upsampled deeper, attended skip).
    """

    def __init__(self, in_deep, in_skip, out_channels):
        super().__init__()
        self.cam = _CAM(in_skip)
        self.aam = _AAM(in_deep, in_skip, out_channels)

    def forward(self, x_deep, x_skip):
        x_up = F.interpolate(
            x_deep, size=x_skip.shape[-2:],
            mode='bilinear', align_corners=False,
        )
        x_skip = self.cam(x_skip)
        return self.aam(x_up, x_skip)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class CASCADE(nn.Module):
    """CASCADE end-to-end segmentation network (WACV 2023).

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation classes.
        img_size: nominal input spatial size (not enforced; forward is fully
            convolutional and accepts arbitrary H=W).
        decoder_channels: channel dims for the three decoder stages, from
            deepest (/16) to shallowest (/4).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 decoder_channels=(320, 128, 64), **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # --- Backbone -------------------------------------------------------
        def _create_backbone(pretrained):
            return timm.create_model(
                'pvt_v2_b2',
                features_only=True,
                pretrained=pretrained,
                in_chans=in_channels,
            )

        self.backbone = load_with_ssl_fallback(_create_backbone, pretrained=True)

        enc_channels = self.backbone.feature_info.channels()
        if len(enc_channels) != 4:
            raise RuntimeError(
                f'Expected 4 encoder features, got {len(enc_channels)}: {enc_channels}'
            )
        c1, c2, c3, c4 = enc_channels  # strides 4, 8, 16, 32

        d3, d2, d1 = decoder_channels  # channels at strides 16, 8, 4

        # --- Cascaded Attention Decoder -------------------------------------
        # Stage 1: /32 (c4) -> /16 (c3) using skip c3
        self.stage1 = _DecoderStage(in_deep=c4, in_skip=c3, out_channels=d3)
        # Stage 2: /16 (d3) -> /8 (c2) using skip c2
        self.stage2 = _DecoderStage(in_deep=d3, in_skip=c2, out_channels=d2)
        # Stage 3: /8 (d2) -> /4 (c1) using skip c1
        self.stage3 = _DecoderStage(in_deep=d2, in_skip=c1, out_channels=d1)

        # --- Heads ----------------------------------------------------------
        self.head = nn.Conv2d(d1, num_classes, kernel_size=1)
        # Auxiliary heads at deeper scales (kept for completeness; CASCADE
        # supports deep supervision during training).
        self.aux_head3 = nn.Conv2d(d3, num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv2d(d2, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2:]
        f1, f2, f3, f4 = self.backbone(x)  # strides 4, 8, 16, 32

        d3 = self.stage1(f4, f3)   # /16
        d2 = self.stage2(d3, f2)   # /8
        d1 = self.stage3(d2, f1)   # /4

        logits = self.head(d1)
        logits = F.interpolate(
            logits, size=(H, W), mode='bilinear', align_corners=False,
        )
        return logits
