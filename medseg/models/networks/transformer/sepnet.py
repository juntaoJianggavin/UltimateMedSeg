"""SEPNet: Semantic Enhanced Perceptual Network for Polyp Segmentation.

Reference:
    Wang et al., "Polyp Segmentation via Semantic Enhanced Perceptual
    Network", IEEE TCSVT 2024.
    https://github.com/wangtong627/SEPNet

Architecture:
    * PVT-v2-B2 backbone (timm, ImageNet pretrained).
    * MAP module: Receptive-Field Block (RFB) for multi-scale feature reduction.
    * CRC module: Dynamic Focus and Mining for progressive feature refinement.
    * Multi-output: 4 logit maps at different scales, fused during inference.

Constructor:
    SEPNet(in_channels=3, num_classes=2, img_size=352, **kwargs)
"""
# Source: https://github.com/wangtong627/SEPNet

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.encoders.transformer.pvtv2_encoder import PVTv2Encoder


class _BasicRFB(nn.Module):
    """Receptive Field Block for multi-scale feature enhancement."""
    def __init__(self, in_ch, out_ch, reduction=4, se_reduction=16):
        super().__init__()
        mid = max(in_ch // reduction, 8)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, 1, 3, dilation=3, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, 1, 5, dilation=5, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, 1, 7, dilation=7, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True))
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 4, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        # SE attention
        se_mid = max(out_ch // se_reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, se_mid, 1), nn.ReLU(True),
            nn.Conv2d(se_mid, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = self.fuse(torch.cat([b0, b1, b2, b3], dim=1))
        return out * self.se(out)


class _DynamicFocusMining(nn.Module):
    """Dynamic Focus and Mining module for progressive refinement."""
    def __init__(self, channels=128, reduction=4):
        super().__init__()
        mid = channels // max(reduction, 1)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 2, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(True))
        self.focus = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(True))
        self.out = nn.Conv2d(mid, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_low, x_high):
        x_high_up = F.interpolate(x_high, size=x_low.shape[2:],
                                  mode='bilinear', align_corners=False)
        merged = self.reduce(torch.cat([x_low, x_high_up], dim=1))
        focus_map = self.sigmoid(self.focus(merged))
        refined = merged * focus_map + merged
        return self.out(refined)


class SEPNet(nn.Module):
    """SEPNet for polyp segmentation (PVT-v2-B2 backbone)."""
    def __init__(self, in_channels=3, num_classes=2, img_size=352,
                 mid_channels=128, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        C = mid_channels

        # PVT-v2-B2 encoder (timm, ImageNet pretrained)
        # Output channels: [64, 128, 320, 512]
        self.encoder = PVTv2Encoder(in_channels=in_channels, img_size=img_size,
                                     pretrained=True)
        enc_channels = self.encoder.out_channels  # [64, 128, 320, 512]

        # MAP (RFB) modules
        self.rfb1 = _BasicRFB(enc_channels[0], C)
        self.rfb2 = _BasicRFB(enc_channels[1], C)
        self.rfb3 = _BasicRFB(enc_channels[2], C)
        self.rfb4 = _BasicRFB(enc_channels[3], C)

        # CRC (Dynamic Focus and Mining) modules
        self.crc3 = _DynamicFocusMining(C, reduction=4)
        self.crc2 = _DynamicFocusMining(C, reduction=4)
        self.crc1 = _DynamicFocusMining(C, reduction=4)

        # Heads
        self.head1 = nn.Sequential(nn.Conv2d(C, C, 3, 1, 1), nn.ReLU(True),
                                   nn.Conv2d(C, num_classes, 1))
        self.head2 = nn.Sequential(nn.Conv2d(C, num_classes, 1))
        self.head3 = nn.Sequential(nn.Conv2d(C, num_classes, 1))
        self.head4 = nn.Sequential(nn.Conv2d(C, num_classes, 1))

    def forward(self, x):
        H, W = x.shape[2:]
        feats = self.encoder(x)
        e1, e2, e3, e4 = feats

        cr1 = self.rfb1(e1)
        cr2 = self.rfb2(e2)
        cr3 = self.rfb3(e3)
        cr4 = self.rfb4(e4)

        # Progressive refinement
        f3 = self.crc3(cr3, cr4)
        f2 = self.crc2(cr2, f3)
        f1 = self.crc1(cr1, f2)

        # Multi-scale outputs
        out1 = self.head1(f1)
        out1 = F.interpolate(out1, size=(H, W), mode='bilinear', align_corners=False)

        if self.training:
            out2 = F.interpolate(self.head2(f2), size=(H, W),
                                 mode='bilinear', align_corners=False)
            out3 = F.interpolate(self.head3(f3), size=(H, W),
                                 mode='bilinear', align_corners=False)
            out4 = F.interpolate(self.head4(cr4), size=(H, W),
                                 mode='bilinear', align_corners=False)
            return out1, out2, out3, out4
        return out1
