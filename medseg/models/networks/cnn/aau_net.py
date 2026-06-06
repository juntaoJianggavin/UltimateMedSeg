"""AAU-Net: Adaptive Attention U-Net for Breast Ultrasound Image Segmentation.

Reference:
    Chen et al., "AAU-Net: An Adaptive Attention U-Net for Breast Lesions
    Segmentation in Ultrasound Images", IEEE TMI 2022.
    https://github.com/CGPxy/AAU-net

Architecture summary:
    - Standard 5-level encoder/decoder UNet with double Conv-BN-ReLU blocks
      (channels [64, 128, 256, 512, 1024]).
    - Each skip connection is processed by a Hybrid Adaptive Attention
      Module (HAAM / AAB) before being concatenated with the decoder
      feature.  The AAB module combines:
        * a channel attention branch built from a dilated 3x3 conv and a
          5x5 conv, fused via Squeeze-and-Excitation style weighting that
          adaptively re-balances the two receptive fields ("adaptive
          dilation rate");
        * a spatial attention branch that combines the channel-attention
          output with a fresh spatial encoding via a 1x1 conv + sigmoid.
    - Final 1x1 conv yields ``num_classes`` channels and the output is
      bilinearly resized to the input H/W if necessary.

The file is self contained (only ``torch`` required).
"""
# Source: https://github.com/CGPxy/AAU-net

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _load_with_ssl_fallback(load_fn, *args, **kwargs):
    """Wrap any pretrained-loading call with an SSL/offline fallback.

    Kept for API parity with the rest of the medseg networks package even
    though AAU-Net itself has no pretrained weights to download.
    """
    import ssl, warnings
    try:
        return load_fn(*args, **kwargs)
    except Exception:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            warnings.warn(
                "Pretrained download failed (%s); using random init." % e2
            )
            return load_fn(*args, **{**kwargs, "pretrained": False})
        finally:
            ssl._create_default_https_context = prev


class _ConvBNReLU(nn.Module):
    """Conv -> BN -> ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 padding=None, dilation=1):
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DoubleConv(nn.Module):
    """Standard double Conv-BN-ReLU block used in vanilla UNet."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            _ConvBNReLU(in_ch, out_ch),
            _ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# AAB / HAAM module
# ---------------------------------------------------------------------------
class _ChannelBlock(nn.Module):
    """Channel-attention branch of the AAB module.

    Two parallel sub-branches with different receptive fields:
      * 3x3 conv with an adaptive (per-level) dilation rate;
      * 5x5 conv (standard receptive field).
    A SE-style gating computed on the concatenation adaptively weights the
    two branches and produces an output with ``out_ch`` channels.
    """

    def __init__(self, in_ch, out_ch, dilation=3):
        super().__init__()
        # dilated 3x3 branch
        self.branch_dil = _ConvBNReLU(
            in_ch, out_ch, kernel_size=3, dilation=dilation
        )
        # 5x5 branch
        self.branch_5x5 = _ConvBNReLU(in_ch, out_ch, kernel_size=5)

        # SE-style channel gating on the concat of the two branches
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_ch * 2, out_ch, bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch, bias=True),
            nn.Sigmoid(),
        )

        # 1x1 fusion conv after re-weighting
        self.fuse = _ConvBNReLU(out_ch * 2, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        b1 = self.branch_dil(x)
        b2 = self.branch_5x5(x)

        cat = torch.cat([b1, b2], dim=1)
        v = self.gap(cat).flatten(1)              # (B, 2C)
        if v.size(0) == 1:
            # BatchNorm1d cannot operate on a single sample in train mode;
            # only apply BN over the Linear layers when batch > 1.
            # We therefore inline the FC computation manually:
            w = torch.sigmoid(
                self.fc[3](self.fc[2](self.fc[0](v)))
            )
        else:
            w = self.fc(v)                        # (B, C) in [0, 1]

        a = w.view(-1, w.size(1), 1, 1)
        a_inv = 1.0 - a

        y1 = b1 * a
        y2 = b2 * a_inv

        out = self.fuse(torch.cat([y1, y2], dim=1))
        return out


class _SpatialBlock(nn.Module):
    """Spatial-attention branch of the AAB module."""

    def __init__(self, in_ch, out_ch, fuse_kernel=3):
        super().__init__()
        # spatial pathway computed directly from the raw input
        self.spatial_a = _ConvBNReLU(in_ch, out_ch, kernel_size=3)
        self.spatial_b = _ConvBNReLU(out_ch, out_ch, kernel_size=1, padding=0)

        # 1x1 conv to compress to a single attention channel
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        pad = fuse_kernel // 2
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=fuse_kernel,
                      padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x, channel_data):
        s = self.spatial_a(x)
        s = self.spatial_b(s)             # spatial feature, (B, C, H, W)

        merged = F.relu(channel_data + s, inplace=True)
        att = self.gate(merged)           # (B, 1, H, W)
        att_inv = 1.0 - att

        y = channel_data * att
        y1 = s * att_inv

        return self.fuse(torch.cat([y, y1], dim=1))


class _AAB(nn.Module):
    """Adaptive Attention Block (a.k.a. HAAM in the original repo).

    Combines a channel-attention branch with an adaptive dilation rate and
    a spatial-attention branch.  Output has ``out_ch`` channels.
    """

    def __init__(self, in_ch, out_ch, dilation=3, fuse_kernel=3):
        super().__init__()
        self.channel = _ChannelBlock(in_ch, out_ch, dilation=dilation)
        self.spatial = _SpatialBlock(in_ch, out_ch, fuse_kernel=fuse_kernel)

    def forward(self, x):
        c = self.channel(x)
        return self.spatial(x, c)


# ---------------------------------------------------------------------------
# Encoder / decoder pieces
# ---------------------------------------------------------------------------
class _Down(nn.Module):
    """MaxPool 2x then double-conv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):
    """Up-conv 2x, concatenate skip, then double-conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad to match skip spatial dims if needed (odd sizes after pooling).
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(
                x,
                [diff_w // 2, diff_w - diff_w // 2,
                 diff_h // 2, diff_h - diff_h // 2],
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------
class AAUNet(nn.Module):
    """AAU-Net.

    5-level UNet with an Adaptive Attention Block (HAAM) on every skip
    connection.

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation output classes.
        img_size: spatial size hint (kept for API consistency; the model
            is fully convolutional and works at arbitrary H/W).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 img_size: int = 224, **kwargs):
        super().__init__()
        chs = [64, 128, 256, 512, 1024]
        # Per-level dilation rates for the AAB channel branch.  Deeper
        # features already have a large receptive field, so we shrink the
        # dilation rate as we go down -- this realises the "adaptive
        # dilation rate" of the AAB module.
        dilations = [3, 3, 2, 2, 1]

        # Encoder ------------------------------------------------------
        self.inc = _DoubleConv(in_channels, chs[0])
        self.down1 = _Down(chs[0], chs[1])
        self.down2 = _Down(chs[1], chs[2])
        self.down3 = _Down(chs[2], chs[3])
        self.down4 = _Down(chs[3], chs[4])

        # Skip-connection adaptive attention blocks -------------------
        # Channel count is preserved by each AAB.
        self.aab1 = _AAB(chs[0], chs[0], dilation=dilations[0])
        self.aab2 = _AAB(chs[1], chs[1], dilation=dilations[1])
        self.aab3 = _AAB(chs[2], chs[2], dilation=dilations[2])
        self.aab4 = _AAB(chs[3], chs[3], dilation=dilations[3])

        # Decoder ------------------------------------------------------
        self.up1 = _Up(chs[4], chs[3], chs[3])
        self.up2 = _Up(chs[3], chs[2], chs[2])
        self.up3 = _Up(chs[2], chs[1], chs[1])
        self.up4 = _Up(chs[1], chs[0], chs[0])

        # Output head --------------------------------------------------
        self.out_conv = nn.Conv2d(chs[0], num_classes, kernel_size=1)

        # Standard Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h_in, w_in = x.shape[-2], x.shape[-1]

        # Pad up to multiple of 16 to guarantee the 4 pooling stages
        # roundtrip without losing spatial information.
        pad_h = (16 - h_in % 16) % 16
        pad_w = (16 - w_in % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h])

        # Encoder
        e1 = self.inc(x)            # (B,  64, H,    W)
        e2 = self.down1(e1)         # (B, 128, H/2,  W/2)
        e3 = self.down2(e2)         # (B, 256, H/4,  W/4)
        e4 = self.down3(e3)         # (B, 512, H/8,  W/8)
        b = self.down4(e4)          # (B, 1024, H/16, W/16)

        # Skip-connection adaptive attention
        s1 = self.aab1(e1)
        s2 = self.aab2(e2)
        s3 = self.aab3(e3)
        s4 = self.aab4(e4)

        # Decoder
        d4 = self.up1(b,  s4)
        d3 = self.up2(d4, s3)
        d2 = self.up3(d3, s2)
        d1 = self.up4(d2, s1)

        logits = self.out_conv(d1)

        # Crop back to original input size
        if pad_h or pad_w:
            logits = logits[..., :h_in, :w_in]

        # Safety net: ensure spatial dims match the original input.
        if logits.shape[-2] != h_in or logits.shape[-1] != w_in:
            logits = F.interpolate(
                logits, size=(h_in, w_in),
                mode="bilinear", align_corners=False,
            )
        return logits
