"""TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation.

Self-contained port of github.com/Rayicer/TransFuse (MICCAI 2021),
TransFuse-S variant.

Architecture:
    - CNN branch: ResNet34 producing features at /4 (64ch), /8 (128ch),
      /16 (256ch), /32 (512ch, unused as in original).
    - ViT branch: DeiT-Small/16 (embed_dim=384, depth=12, heads=6) with
      pos_embed interpolated for any img_size, producing tokens at /16.
    - BiFusion: at each of three scales (/4, /8, /16) fuse CNN feature and
      ViT feature via channel attention (on CNN), spatial attention
      (on ViT) and a bilinear-product bridge, then concatenate.
    - Decoder: 3-stage upsample + concat of fused features, final 1x1 head
      followed by bilinear upsample to the input resolution.
"""
# Source: https://github.com/Rayicer/TransFuse

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvmodels

import timm

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


# ---------------------------------------------------------------------------
# DeiT-Small wrapper (timm)
# ---------------------------------------------------------------------------
class _DeiTSmall(nn.Module):
    def __init__(self, img_size=224, in_chans=3, pretrained=True):
        super().__init__()
        self.embed_dim = 384
        self.patch_size = 16
        self.img_size = img_size

        self.vit = load_with_ssl_fallback(
            timm.create_model,
            'deit_small_patch16_224',
            pretrained=bool(pretrained),
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,
        )

    def forward(self, x):
        feat = self.vit.forward_features(x)  # (B, 1+N, C)
        # Strip prefix tokens (cls, and dist if present) by keeping the trailing N
        B, T, C = feat.shape
        h = x.shape[-2] // self.patch_size
        w = x.shape[-1] // self.patch_size
        n = h * w
        if T > n:
            feat = feat[:, T - n:, :]
        return feat, h, w


# ---------------------------------------------------------------------------
# BiFusion module
# ---------------------------------------------------------------------------
class _BiFusion(nn.Module):
    """Fuse a CNN feature and a ViT feature at the same spatial scale.

    Combines:
      - channel attention applied to the CNN feature,
      - spatial attention applied to the ViT feature,
      - a bilinear-product bridge between projected CNN/ViT features.

    The three branches are concatenated and passed through a residual
    bottleneck producing ``ch_out`` channels.
    """

    def __init__(self, ch_vit, ch_cnn, ch_out, r=4):
        super().__init__()
        # Channel attention on ViT branch (faithful to official: ch_2 = ViT)
        red_ca = max(ch_vit // r, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(ch_vit, red_ca, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_ca, ch_vit, 1, bias=False),
            nn.Sigmoid(),
        )
        # Spatial attention on CNN branch (faithful to official: ch_1 = CNN)
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.W_v = nn.Sequential(
            nn.Conv2d(ch_vit, ch_out, 1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.W_c = nn.Sequential(
            nn.Conv2d(ch_cnn, ch_out, 1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.W = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        in_concat = ch_vit + ch_cnn + ch_out
        self.residual = nn.Sequential(
            nn.Conv2d(in_concat, ch_out, 1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, vit_feat, cnn_feat):
        if vit_feat.shape[-2:] != cnn_feat.shape[-2:]:
            vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[-2:],
                                     mode='bilinear', align_corners=False)
        # Channel attention on ViT branch (faithful to official: ch_2 = ViT)
        ca_w = self.ca(self.gap(vit_feat))
        vit_att = vit_feat * ca_w
        # Spatial attention on CNN branch (faithful to official: ch_1 = CNN)
        sa_in = torch.cat([
            cnn_feat.mean(dim=1, keepdim=True),
            cnn_feat.max(dim=1, keepdim=True)[0],
        ], dim=1)
        sa_w = self.sa(sa_in)
        cnn_att = cnn_feat * sa_w
        Wv = self.W_v(vit_feat)
        Wc = self.W_c(cnn_feat)
        bp = self.W(Wv * Wc)
        out = torch.cat([vit_att, cnn_att, bp], dim=1)
        return self.residual(out)


class _UpConv(nn.Module):
    """Bilinear 2x upsample + 3x3 conv + BN + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=False)
        return self.conv(x)


# ---------------------------------------------------------------------------
# TransFuse (TransFuse-S)
# ---------------------------------------------------------------------------
class TransFuse(nn.Module):
    """TransFuse-S: parallel ResNet34 + DeiT-Small with BiFusion decoder."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 pretrained=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # Input adapter so both branches see 3 channels.
        if in_channels != 3:
            self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_adapter = nn.Identity()

        # CNN branch: ResNet34
        resnet = load_with_ssl_fallback(
            tvmodels.resnet34, pretrained=pretrained)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # /4,  64
        self.layer2 = resnet.layer2  # /8,  128
        self.layer3 = resnet.layer3  # /16, 256
        self.layer4 = resnet.layer4  # /32, 512 (kept for completeness; unused)

        # ViT branch: DeiT-Small/16
        self.vit = _DeiTSmall(img_size=img_size, in_chans=3,
                              pretrained=pretrained)
        vit_dim = self.vit.embed_dim  # 384

        # ViT feature projection + upsamples to /8 and /4
        self.vit_proj_16 = nn.Sequential(
            nn.Conv2d(vit_dim, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up_v_8 = _UpConv(256, 128)
        self.up_v_4 = _UpConv(128, 64)

        # BiFusion modules
        self.bifusion_16 = _BiFusion(ch_vit=256, ch_cnn=256, ch_out=256)
        self.bifusion_8 = _BiFusion(ch_vit=128, ch_cnn=128, ch_out=128)
        self.bifusion_4 = _BiFusion(ch_vit=64, ch_cnn=64, ch_out=64)

        # Decoder
        self.dec_16_to_8 = _UpConv(256, 128)
        self.dec_fuse_8 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec_8_to_4 = _UpConv(128, 64)
        self.dec_fuse_4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        x_in = self.input_adapter(x)

        # CNN branch
        x0 = self.stem(x_in)
        c1 = self.layer1(x0)   # /4,  64
        c2 = self.layer2(c1)   # /8,  128
        c3 = self.layer3(c2)   # /16, 256
        # self.layer4(c3) intentionally unused (TransFuse-S decoder is /4-/16)

        # ViT branch -> /16 spatial
        vit_tokens, hv, wv = self.vit(x_in)
        B = vit_tokens.shape[0]
        v16 = vit_tokens.transpose(1, 2).reshape(
            B, self.vit.embed_dim, hv, wv)
        # Align to CNN /16 grid if rounding differs
        if v16.shape[-2:] != c3.shape[-2:]:
            v16 = F.interpolate(v16, size=c3.shape[-2:], mode='bilinear',
                                align_corners=False)
        v16 = self.vit_proj_16(v16)                    # 256 @ /16
        v8 = self.up_v_8(v16)                          # 128 @ /8
        if v8.shape[-2:] != c2.shape[-2:]:
            v8 = F.interpolate(v8, size=c2.shape[-2:], mode='bilinear',
                               align_corners=False)
        v4 = self.up_v_4(v8)                           # 64  @ /4
        if v4.shape[-2:] != c1.shape[-2:]:
            v4 = F.interpolate(v4, size=c1.shape[-2:], mode='bilinear',
                               align_corners=False)

        # BiFusion at three scales
        f16 = self.bifusion_16(v16, c3)
        f8 = self.bifusion_8(v8, c2)
        f4 = self.bifusion_4(v4, c1)

        # Decoder
        d8 = self.dec_16_to_8(f16)
        if d8.shape[-2:] != f8.shape[-2:]:
            d8 = F.interpolate(d8, size=f8.shape[-2:], mode='bilinear',
                               align_corners=False)
        d8 = self.dec_fuse_8(torch.cat([d8, f8], dim=1))
        d4 = self.dec_8_to_4(d8)
        if d4.shape[-2:] != f4.shape[-2:]:
            d4 = F.interpolate(d4, size=f4.shape[-2:], mode='bilinear',
                               align_corners=False)
        d4 = self.dec_fuse_4(torch.cat([d4, f4], dim=1))

        out = self.head(d4)
        out = F.interpolate(out, size=(H, W), mode='bilinear',
                            align_corners=False)
        return out
