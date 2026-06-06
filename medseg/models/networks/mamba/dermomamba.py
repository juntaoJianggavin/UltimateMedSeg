"""DermoMamba: Cross-Scale Mamba for Skin Lesion Segmentation.

Reference:
    Hoang et al., "DermoMamba: A cross-scale Mamba-based model with Guide
    Fusion Loss for skin lesion segmentation in dermoscopy images",
    Pattern Analysis and Applications, 2025.
    https://github.com/hnkhai25/DermoMamba

Architecture:
    * 5-stage UNet encoder-decoder.
    * Encoder: ResMambaBlock (Cross-Scale Mamba Block + residual) then
      Conv + BN + ReLU + MaxPool.
    * Cross-Scale Mamba Block: splits channels into 4 groups, applies
      axial depthwise convolutions with different dilation rates (1,2,3)
      followed by SS2D (VSS), then concatenates.
    * Skip connections: CBAM attention on each skip.
    * Bottleneck: PCA (channel attention) + Sweep_Mamba (3-directional SS2D).
    * Decoder: Upsample + concat + Conv.

Constructor:
    DermoMamba(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/hnkhai25/DermoMamba

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.vmunet_encoder import SS2D


# ── CBAM (Channel + Spatial attention) ────────────────────────────────────────

class _ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * a


class _SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ch = _ChannelGate(channels, reduction)
        self.sp = _SpatialGate()

    def forward(self, x):
        return self.sp(self.ch(x))


# ── Axial Spatial Depthwise Conv ──────────────────────────────────────────────

class _AxialSpatialDW(nn.Module):
    def __init__(self, dim, kernel=7, dilation=1):
        super().__init__()
        self.mixer_h = nn.Conv2d(dim, dim, (kernel, 1), padding="same",
                                 groups=dim, dilation=dilation)
        self.mixer_w = nn.Conv2d(dim, dim, (1, kernel), padding="same",
                                 groups=dim, dilation=dilation)
        self.conv = nn.Conv2d(dim, dim, 3, padding="same",
                              groups=dim, dilation=dilation)

    def forward(self, x):
        return self.conv(self.mixer_h(self.mixer_w(x))) + x


# ── Cross-Scale Mamba Block ──────────────────────────────────────────────────

class _CrossScaleMambaBlock(nn.Module):
    """Splits channels into 4 groups, applies axial DW + SS2D with
    different dilation rates on first 3 groups, passes 4th through."""

    def __init__(self, dim):
        super().__init__()
        q = max(dim // 4, 4)
        self.dw1 = _AxialSpatialDW(q, 7, dilation=1)
        self.dw2 = _AxialSpatialDW(q, 7, dilation=2)
        self.dw3 = _AxialSpatialDW(q, 7, dilation=3)
        self.vss = SS2D(d_model=q, dropout=0, d_state=16)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        chunks = torch.chunk(x, 4, dim=1)
        parts = []
        for i, (xi, dw) in enumerate(zip(chunks, [self.dw1, self.dw2, self.dw3])):
            xi = dw(xi)
            xi = self.vss(xi.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            parts.append(xi)
        parts.append(chunks[3])
        x = torch.cat(parts, dim=1)
        return self.act(self.bn(x))


# ── ResMambaBlock ────────────────────────────────────────────────────────────

class _ResMambaBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_c, affine=True)
        self.act = nn.LeakyReLU(0.01)
        self.block = _CrossScaleMambaBlock(in_c)
        self.conv = nn.Conv2d(in_c, in_c, 3, padding="same")
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = self.block(x)
        out = self.act(self.norm(self.conv(out))) + x * self.scale
        return out


# ── Encoder Block ────────────────────────────────────────────────────────────

class _EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.resmamba = _ResMambaBlock(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 3, padding="same")
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.resmamba(x)
        skip = self.act(self.bn(self.pw(x)))
        x = self.down(skip)
        return x, skip


# ── PCA (Channel Attention) ──────────────────────────────────────────────────

class _PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 9, groups=dim, padding="same")
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        c = x.mean(dim=[2, 3])                     # (B, C)
        x = self.dw(x)
        c_ = x.mean(dim=[2, 3])                    # (B, C)
        raise_ch = self.softmax(c_ - c)
        att = torch.sigmoid(c_ * (1 + raise_ch))
        return x * att.unsqueeze(-1).unsqueeze(-1)


# ── Sweep_Mamba (Bottleneck) ─────────────────────────────────────────────────

class _SweepMamba(nn.Module):
    """3-directional SS2D bottleneck with channel reduction."""

    def __init__(self, dim, ratio=8):
        super().__init__()
        red = max(dim // ratio, 8)
        self.ln = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, red, 1)
        self.mamba1 = SS2D(d_model=red, dropout=0, d_state=16)
        self.mamba2 = SS2D(d_model=red, dropout=0, d_state=16)
        self.mamba3 = SS2D(d_model=red, dropout=0, d_state=16)
        self.act = nn.SiLU()
        self.relu = nn.ReLU()
        self.proj_out = nn.Linear(red, dim, 1)
        self.scale = nn.Parameter(torch.ones(1))
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        # x: (B, H, W, C)
        skip = x
        x = self.proj_in(self.ln(x))               # (B, H, W, red)
        x1 = self.mamba1(x)
        # Transposed scans for directional diversity
        x2 = self.mamba2(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x3 = self.mamba3(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        w = self.act(x)
        out = w * x1 + w * x2 + w * x3
        out = self.proj_out(out) + skip * self.scale
        return out


# ── Decoder Block ────────────────────────────────────────────────────────────

class _DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.pw = nn.Conv2d(in_c * 2, in_c, 1)
        self.pw2 = nn.Conv2d(in_c, out_c, 3, padding="same")
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.pw(x)
        return self.bn(self.act(self.pw2(x)))


# ── DermoMamba ───────────────────────────────────────────────────────────────

class DermoMamba(nn.Module):
    """Cross-scale Mamba with CBAM skip + PCA/SweepMamba bottleneck.

    Channel progression: 16 -> 32 -> 64 -> 128 -> 256 -> 512.
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 base_channels=16, **kwargs):
        super().__init__()
        c = base_channels  # 16

        self.pw_in = nn.Conv2d(in_channels, c, 1)

        # Encoder
        self.e1 = _EncoderBlock(c, c * 2)
        self.e2 = _EncoderBlock(c * 2, c * 4)
        self.e3 = _EncoderBlock(c * 4, c * 8)
        self.e4 = _EncoderBlock(c * 8, c * 16)
        self.e5 = _EncoderBlock(c * 16, c * 32)

        # Skip connections (CBAM)
        self.s1 = CBAM(c * 2)
        self.s2 = CBAM(c * 4)
        self.s3 = CBAM(c * 8)
        self.s4 = CBAM(c * 16)
        self.s5 = CBAM(c * 32)

        # Bottleneck
        self.pca = _PCA(c * 32)
        self.sweep = _SweepMamba(c * 32)

        # Decoder
        self.d5 = _DecoderBlock(c * 32, c * 16)
        self.d4 = _DecoderBlock(c * 16, c * 8)
        self.d3 = _DecoderBlock(c * 8, c * 4)
        self.d2 = _DecoderBlock(c * 4, c * 2)
        self.d1 = _DecoderBlock(c * 2, c)

        self.conv_out = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        # Pad to multiple of 32
        pH = (32 - H % 32) % 32
        pW = (32 - W % 32) % 32
        if pH > 0 or pW > 0:
            x = F.pad(x, [0, pW, 0, pH], mode="reflect")

        x = self.pw_in(x)

        # Encoder
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        # Skip attention
        skip1 = self.s1(skip1)
        skip2 = self.s2(skip2)
        skip3 = self.s3(skip3)
        skip4 = self.s4(skip4)
        skip5 = self.s5(skip5)

        # Bottleneck
        x = self.pca(x)
        # Sweep_Mamba operates in (B,H,W,C) space
        x = self.sweep(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Decoder
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)

        out = self.conv_out(x)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
