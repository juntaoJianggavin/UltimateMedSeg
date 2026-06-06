"""Lightweight HoVerNet – nuclei segmentation & classification.

Inspired by:
    Graham et al., "HoVer-Net: Simultaneous Segmentation and Classification
    of Nuclei in Multi-Tissue Histology Images", Medical Image Analysis, 2019.
    https://github.com/vqdang/hover_net

Architecture highlights
-----------------------
* Multi-branch encoder-decoder (nuclei pixel + horizontal/vertical maps + type)
* Residual convolutional encoder with progressive downsampling
* Skip connections at each resolution level

Adapted for the project's standard interface:
    HoverNetLite(in_channels, num_classes, img_size, pretrained, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ──────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Two 3×3 convs + BN + ReLU with a residual connection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        s = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + s)


class _EncBlock(nn.Module):
    """Encoder block: ConvBlock + MaxPool, returns pre-pool skip + pooled."""

    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = _ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2) if pool else None

    def forward(self, x):
        pre = self.conv(x)       # pre-pool (skip for decoder)
        nxt = self.pool(pre) if self.pool is not None else pre
        return pre, nxt


class _DecBlock(nn.Module):
    """Decoder block: upsample + concat skip + ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential spatial mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Main model ───────────────────────────────────────────────────────────────

class HoverNetLite(nn.Module):
    """Lightweight HoVerNet for nuclei segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 3).
    num_classes : int
        Number of output segmentation classes (nuclei types).
    img_size : int
        Input image size (used only for interface compatibility).
    pretrained : bool
        Not used (kept for interface compatibility).
    base_filters : int
        Number of filters in the first encoder block. Doubled at each level.
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 pretrained=True, base_filters=32, **kwargs):
        super().__init__()
        f = base_filters

        # Encoder (5 levels)
        self.enc1 = _EncBlock(in_channels, f, pool=True)       # -> f, H/2
        self.enc2 = _EncBlock(f, f * 2, pool=True)             # -> 2f, H/4
        self.enc3 = _EncBlock(f * 2, f * 4, pool=True)         # -> 4f, H/8
        self.enc4 = _EncBlock(f * 4, f * 8, pool=True)         # -> 8f, H/16

        # Bottleneck
        self.bottleneck = _ConvBlock(f * 8, f * 16)            # -> 16f, H/16

        # Nuclei pixel (NP) decoder – primary segmentation branch
        self.np_dec4 = _DecBlock(f * 16, f * 8, f * 8)
        self.np_dec3 = _DecBlock(f * 8, f * 4, f * 4)
        self.np_dec2 = _DecBlock(f * 4, f * 2, f * 2)
        self.np_dec1 = _DecBlock(f * 2, f, f)
        self.np_head = nn.Conv2d(f, num_classes, 1)

        # Horizontal-Vertical (HV) decoder – instance boundary branch
        self.hv_dec4 = _DecBlock(f * 16, f * 8, f * 8)
        self.hv_dec3 = _DecBlock(f * 8, f * 4, f * 4)
        self.hv_dec2 = _DecBlock(f * 4, f * 2, f * 2)
        self.hv_dec1 = _DecBlock(f * 2, f, f)
        self.hv_head = nn.Conv2d(f, 2, 1)  # horizontal + vertical maps

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        s1, x = self.enc1(x)    # s1: f@H/2
        s2, x = self.enc2(x)    # s2: 2f@H/4
        s3, x = self.enc3(x)    # s3: 4f@H/8
        s4, x = self.enc4(x)    # s4: 8f@H/16

        # Bottleneck
        x = self.bottleneck(x)  # 16f@H/16

        # NP branch
        np = self.np_dec4(x, s4)
        np = self.np_dec3(np, s3)
        np = self.np_dec2(np, s2)
        np = self.np_dec1(np, s1)
        np_out = self.np_head(np)

        # HV branch
        hv = self.hv_dec4(x, s4)
        hv = self.hv_dec3(hv, s3)
        hv = self.hv_dec2(hv, s2)
        hv = self.hv_dec1(hv, s1)
        # hv_out = self.hv_head(hv)  # HV maps not used at inference

        # Decoder output is already at original resolution
        return np_out
