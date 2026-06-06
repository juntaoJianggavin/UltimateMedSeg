"""UU-Mamba: Uncertainty-aware U-Mamba for Cardiac/Ultrasound Segmentation.

Reference:
    Tsai et al., "Uncertainty-aware U-Mamba for Cardiac Image Segmentation",
    IEEE MIPR 2024.
    https://github.com/tiffany9056/UU-Mamba

Architecture:
    * U-Mamba backbone: hybrid CNN + SSM encoder-decoder.
    * 4-stage encoder with Mamba blocks at bottleneck.
    * Uncertainty-aware output: predicts mean + variance per pixel.
    * Standard UNet decoder with skip concatenation.

Constructor:
    UUMamba(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/tiffany9056/UU-Mamba

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.vmunet_encoder import SS2D


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(8, out_ch), nn.GELU())
    def forward(self, x):
        return self.block(x)


class _MambaBlock2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ss2d = SS2D(d_model=dim, d_state=16, d_conv=3, expand=2)
    def forward(self, x):
        B, C, H, W = x.shape
        x_hw = x.permute(0, 2, 3, 1)
        x_m = self.ss2d(self.norm(x_hw))
        return x_m.permute(0, 3, 1, 2) + x


class UUMamba(nn.Module):
    """UU-Mamba for cardiac/ultrasound segmentation."""
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 base_channels=32, **kwargs):
        super().__init__()
        C = base_channels
        self.num_classes = num_classes

        # Encoder
        self.enc1 = _DoubleConv(in_channels, C)
        self.enc2 = _DoubleConv(C, C * 2)
        self.enc3 = _DoubleConv(C * 2, C * 4)
        self.enc4 = _DoubleConv(C * 4, C * 8)
        self.pool = nn.MaxPool2d(2)

        # Mamba bottleneck
        self.mamba1 = _MambaBlock2D(C * 8)
        self.mamba2 = _MambaBlock2D(C * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(C * 8, C * 4, 2, stride=2)
        self.dec4 = _DoubleConv(C * 8, C * 4)
        self.up3 = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec3 = _DoubleConv(C * 4, C * 2)
        self.up2 = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
        self.dec2 = _DoubleConv(C * 2, C)

        # Output: mean + variance (uncertainty-aware)
        self.head_mean = nn.Conv2d(C, num_classes, 1)
        self.head_var = nn.Sequential(
            nn.Conv2d(C, C, 1), nn.Softplus(),
            nn.Conv2d(C, num_classes, 1), nn.Softplus())

    def forward(self, x):
        H, W = x.shape[2:]
        pH = ((H + 15) // 16) * 16
        pW = ((W + 15) // 16) * 16
        if pH != H or pW != W:
            x = F.pad(x, [0, pW - W, 0, pH - H], mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.mamba2(self.mamba1(e4))

        d4 = self.dec4(torch.cat([self.up4(b)[:, :, :e3.shape[2], :e3.shape[3]], e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4)[:, :, :e2.shape[2], :e2.shape[3]], e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3)[:, :, :e1.shape[2], :e1.shape[3]], e1], dim=1))

        mean = self.head_mean(d2)
        if self.training:
            var = self.head_var(d2)
            mean = F.interpolate(mean, size=(H, W), mode='bilinear', align_corners=False)
            var = F.interpolate(var, size=(H, W), mode='bilinear', align_corners=False)
            return mean, var
        mean = F.interpolate(mean, size=(H, W), mode='bilinear', align_corners=False)
        return mean
