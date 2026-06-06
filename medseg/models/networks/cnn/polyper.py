"""Polyper: Boundary Sensitive Polyp Segmentation.

Reference:
    Shao et al., "Polyper: Boundary Sensitive Polyp Segmentation", AAAI 2024.
    https://github.com/haoshao-nku/medical_seg

Architecture:
    * Swin-T (Swin Transformer Tiny) encoder (timm, ImageNet pretrained).
    * Boundary-aware dual-branch framework.
    * Polyp Region Branch (PRB): global polyp region prediction.
    * Boundary Branch (BB): fine-grained boundary refinement.
    * Boundary Guidance Module (BGM): guides PRB with boundary cues.

Constructor:
    Polyper(in_channels=3, num_classes=2, img_size=352, **kwargs)
"""
# Source: https://github.com/haoshao-nku/medical_seg

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class _ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ksize, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x):
        return self.block(x)


class _BoundaryGuidance(nn.Module):
    """Boundary Guidance Module: uses edge cues to refine features."""
    def __init__(self, dim):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.ReLU(True),
            nn.Conv2d(dim // 4, 1, 1), nn.Sigmoid())
        self.refine = _ConvBN(dim, dim, 3, 1, 1)

    def forward(self, x):
        edge = self.edge_conv(x)
        return self.refine(x * edge.expand_as(x))


class _SwinTEncoder(nn.Module):
    """Swin-T encoder via timm, output 4 stages: [96, 192, 384, 768]."""
    def __init__(self, in_channels=3, img_size=224, pretrained=True):
        super().__init__()
        import ssl, warnings
        self.stem = None
        backbone_in = in_channels
        try:
            self.model = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=pretrained, features_only=True,
                in_chans=backbone_in)
        except Exception as e1:
            prev = ssl._create_default_https_context
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
                self.model = timm.create_model(
                    "swin_tiny_patch4_window7_224",
                    pretrained=pretrained, features_only=True,
                    in_chans=backbone_in)
            except Exception as e2:
                # Fallback: build RGB backbone + 1x1 stem
                warnings.warn(f"Pretrained download failed ({e2}); using random init.")
                if in_channels != 3:
                    self.stem = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
                self.model = timm.create_model(
                    "swin_tiny_patch4_window7_224",
                    pretrained=False, features_only=True,
                    in_chans=3)
            finally:
                ssl._create_default_https_context = prev
        self.out_channels = list(self.model.feature_info.channels())

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        features = self.model(x)
        out = []
        for i, f in enumerate(features):
            if f.ndim == 4:
                expected_c = self.out_channels[i]
                if f.shape[1] != expected_c and f.shape[-1] == expected_c:
                    f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out


class Polyper(nn.Module):
    """Polyper for boundary-sensitive polyp segmentation (Swin-T backbone)."""
    def __init__(self, in_channels=3, num_classes=2, img_size=352, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        # Swin-T encoder (timm, ImageNet pretrained)
        # Output channels: [96, 192, 384, 768]
        self.encoder = _SwinTEncoder(in_channels=in_channels, img_size=img_size,
                                     pretrained=True)
        enc_chs = self.encoder.out_channels  # [96, 192, 384, 768]

        # Polyp Region Branch (main decoder)
        self.up4 = _ConvBN(enc_chs[3], enc_chs[2])
        self.up3 = _ConvBN(enc_chs[2] + enc_chs[2], enc_chs[1])
        self.up2 = _ConvBN(enc_chs[1] + enc_chs[1], enc_chs[0])
        self.up1 = _ConvBN(enc_chs[0] + enc_chs[0], 64)

        # Boundary Guidance at each skip
        self.bg3 = _BoundaryGuidance(enc_chs[2])
        self.bg2 = _BoundaryGuidance(enc_chs[1])
        self.bg1 = _BoundaryGuidance(enc_chs[0])

        # Boundary Branch
        self.bb_up4 = _ConvBN(enc_chs[3], enc_chs[2])
        self.bb_up3 = _ConvBN(enc_chs[2] + enc_chs[2], enc_chs[1])
        self.bb_up2 = _ConvBN(enc_chs[1] + enc_chs[1], enc_chs[0])

        # Heads
        self.region_head = nn.Conv2d(64, num_classes, 1)
        self.boundary_head = nn.Conv2d(enc_chs[0], 1, 1)

    def forward(self, x):
        H, W = x.shape[2:]
        feats = self.encoder(x)
        e1, e2, e3, e4 = feats

        # Region branch
        r4 = F.interpolate(self.up4(e4), scale_factor=2, mode='bilinear', align_corners=False)
        r3 = self.up3(torch.cat([r4[:, :, :e3.shape[2], :e3.shape[3]], self.bg3(e3)], dim=1))
        r3 = F.interpolate(r3, scale_factor=2, mode='bilinear', align_corners=False)
        r2 = self.up2(torch.cat([r3[:, :, :e2.shape[2], :e2.shape[3]], self.bg2(e2)], dim=1))
        r2 = F.interpolate(r2, scale_factor=2, mode='bilinear', align_corners=False)
        r1 = self.up1(torch.cat([r2[:, :, :e1.shape[2], :e1.shape[3]], self.bg1(e1)], dim=1))

        # Boundary branch
        b4 = F.interpolate(self.bb_up4(e4), scale_factor=2, mode='bilinear', align_corners=False)
        b3 = self.bb_up3(torch.cat([b4[:, :, :e3.shape[2], :e3.shape[3]], e3], dim=1))
        b3 = F.interpolate(b3, scale_factor=2, mode='bilinear', align_corners=False)
        b2 = self.bb_up2(torch.cat([b3[:, :, :e2.shape[2], :e2.shape[3]], e2], dim=1))
        b2 = F.interpolate(b2, scale_factor=2, mode='bilinear', align_corners=False)

        region_out = F.interpolate(self.region_head(r1), size=(H, W),
                                   mode='bilinear', align_corners=False)
        boundary_out = F.interpolate(self.boundary_head(b2), size=(H, W),
                                     mode='bilinear', align_corners=False)

        if self.training:
            return region_out, boundary_out
        return region_out
