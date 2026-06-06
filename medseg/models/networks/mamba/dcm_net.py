"""DCM-Net: Dual-encoder CNN-Mamba for Medical Image Segmentation.

Reference:
    "DCM-Net: Dual-encoder CNN-Mamba network with cross-branch fusion
    for robust medical image segmentation", BMC Medical Imaging 2025.
    https://github.com/mecusbans/DCM-Net

Architecture:
    * Dual-branch encoder: CNN branch + Mamba branch.
    * Cross-Branch Fusion Module (CBFM): fuses CNN local + Mamba global.
    * 4-stage UNet decoder with skip connections.
    * Designed for breast ultrasound lesion segmentation.

Constructor:
    DCMNet(in_channels=3, num_classes=2, img_size=224, **kwargs)
"""
# Source: https://github.com/mecusbans/DCM-Net

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.vmunet_encoder import SS2D


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
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
        x_n = self.norm(x_hw)
        x_m = self.ss2d(x_n)
        return x_m.permute(0, 3, 1, 2) + x


class _CrossBranchFusion(nn.Module):
    """Cross-Branch Fusion Module: fuses CNN and Mamba features."""
    def __init__(self, dim):
        super().__init__()
        self.attn_cnn = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.ReLU(True), nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid())
        self.attn_mamba = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.ReLU(True), nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True))

    def forward(self, cnn_feat, mamba_feat):
        a_cnn = self.attn_cnn(cnn_feat)
        a_mamba = self.attn_mamba(mamba_feat)
        fused = self.fuse(torch.cat([cnn_feat * a_cnn, mamba_feat * a_mamba], dim=1))
        return fused


class DCMNet(nn.Module):
    """DCM-Net for breast ultrasound segmentation."""
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 base_channels=32, **kwargs):
        super().__init__()
        C = base_channels
        self.num_classes = num_classes

        # CNN encoder branch
        self.cnn_enc1 = _DoubleConv(in_channels, C)
        self.cnn_enc2 = _DoubleConv(C, C * 2)
        self.cnn_enc3 = _DoubleConv(C * 2, C * 4)
        self.cnn_enc4 = _DoubleConv(C * 4, C * 8)
        self.pool = nn.MaxPool2d(2)

        # Mamba encoder branch
        self.mamba_enc1 = _MambaBlock2D(C)
        self.mamba_enc2 = _MambaBlock2D(C * 2)
        self.mamba_enc3 = _MambaBlock2D(C * 4)
        self.mamba_enc4 = _MambaBlock2D(C * 8)

        # Cross-branch fusion
        self.cbf1 = _CrossBranchFusion(C)
        self.cbf2 = _CrossBranchFusion(C * 2)
        self.cbf3 = _CrossBranchFusion(C * 4)
        self.cbf4 = _CrossBranchFusion(C * 8)

        # Bottleneck
        self.bottleneck = _MambaBlock2D(C * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(C * 8, C * 4, 2, stride=2)
        self.dec4 = _DoubleConv(C * 8, C * 4)
        self.up3 = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec3 = _DoubleConv(C * 4, C * 2)
        self.up2 = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
        self.dec2 = _DoubleConv(C * 2, C)

        self.head = nn.Conv2d(C, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        pH = ((H + 15) // 16) * 16
        pW = ((W + 15) // 16) * 16
        if pH != H or pW != W:
            x = F.pad(x, [0, pW - W, 0, pH - H], mode='reflect')

        # Encoder with dual branches
        c1 = self.cnn_enc1(x)
        m1 = self.mamba_enc1(c1)
        f1 = self.cbf1(c1, m1)

        c2 = self.cnn_enc2(self.pool(f1))
        m2 = self.mamba_enc2(c2)
        f2 = self.cbf2(c2, m2)

        c3 = self.cnn_enc3(self.pool(f2))
        m3 = self.mamba_enc3(c3)
        f3 = self.cbf3(c3, m3)

        c4 = self.cnn_enc4(self.pool(f3))
        m4 = self.mamba_enc4(c4)
        f4 = self.cbf4(c4, m4)

        # Bottleneck
        b = self.bottleneck(f4)
        b = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b)[:, :, :f3.shape[2], :f3.shape[3]], f3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4)[:, :, :f2.shape[2], :f2.shape[3]], f2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3)[:, :, :f1.shape[2], :f1.shape[3]], f1], dim=1))

        out = self.head(d2)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
