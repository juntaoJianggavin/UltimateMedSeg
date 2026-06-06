"""CTNet: Contrastive Transformer Network for Polyp Segmentation.

Reference:
    Xiao et al., "CTNet: Contrastive Transformer Network for Polyp
    Segmentation", IEEE TCYB 2024.
    https://github.com/Fhujinwu/CTNet

Architecture:
    * MiT-B3 encoder (Mix Transformer, layers=[3,4,18,3]).
    * SMI Module (SMIM): multi-scale interaction for deep features.
    * CI Module (CIM): cross-level interaction for shallow+deep fusion.
    * 1x1 channel reduction at each encoder stage.
    * ProjectionHead for contrastive learning.
    * Single output prediction head.

Constructor:
    CTNet(in_channels=3, num_classes=2, img_size=352, **kwargs)
"""
# Source: https://github.com/Fhujinwu/CTNet

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.networks.transformer.missformer_model import _MiT


class _BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride, padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _SMIModule(nn.Module):
    """Scale-aware Multi-scale Interaction Module."""
    def __init__(self, out_channel=32):
        super().__init__()
        self.conv1 = _BasicConv2d(out_channel, out_channel, 3, 1, 1, dilation=1)
        self.conv2 = _BasicConv2d(out_channel, out_channel, 3, 1, 2, dilation=2)
        self.conv3 = _BasicConv2d(out_channel, out_channel, 3, 1, 3, dilation=3)
        self.fuse = _BasicConv2d(out_channel * 3, out_channel, 1, 1, 0)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        return self.fuse(torch.cat([c1, c2, c3], dim=1)) + x


class _CIModule(nn.Module):
    """Cross-level Interaction Module for multi-scale fusion."""
    def __init__(self, out_channel=32):
        super().__init__()
        self.up = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
        self.reduce = _BasicConv2d(out_channel * 2, out_channel, 1, 1, 0)
        self.refine = _BasicConv2d(out_channel, out_channel, 3, 1, 1)

    def forward(self, x_deep, x_mid, x_shallow):
        # Upsample deep and fuse with mid
        x_deep_up = F.interpolate(x_deep, size=x_mid.shape[2:],
                                  mode='bilinear', align_corners=False)
        fused = self.reduce(torch.cat([x_deep_up, x_mid], dim=1))
        # Further fuse with shallow
        fused_up = F.interpolate(fused, size=x_shallow.shape[2:],
                                 mode='bilinear', align_corners=False)
        out = self.reduce(torch.cat([fused_up, x_shallow], dim=1))
        return self.refine(out)


class _ProjectionHead(nn.Module):
    """Projection head for contrastive learning (upstream Contrastive_head.py)."""
    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # x: (B, C, H, W) -> global avg pool -> project
        x = x.mean(dim=[2, 3])  # (B, C)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


class CTNet(nn.Module):
    """CTNet for polyp segmentation (MiT-B3 backbone)."""
    def __init__(self, in_channels=3, num_classes=2, img_size=352,
                 channel=32, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        # MiT-B3 encoder: dims=[64,128,320,512], layers=[3,4,18,3]
        self.encoder = _MiT(img_size=img_size, in_ch=in_channels,
                            dims=(64, 128, 320, 512),
                            layers=(3, 4, 18, 3))
        enc_chs = [64, 128, 320, 512]

        # Channel reduction
        self.trans1 = _BasicConv2d(enc_chs[0], channel, 1, 1, 0)
        self.trans2 = _BasicConv2d(enc_chs[1], channel, 1, 1, 0)
        self.trans3 = _BasicConv2d(enc_chs[2], channel, 1, 1, 0)
        self.trans4 = _BasicConv2d(enc_chs[3], channel, 1, 1, 0)

        # Projection head for contrastive learning
        self.contrasthead = _ProjectionHead(dim=channel)

        # SMI and CI modules
        self.smim = _SMIModule(channel)
        self.cim = _CIModule(channel)

        # Upsample + prediction
        self.uper = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, 4, 2, 1),
            _BasicConv2d(channel, channel, 3, 1, 1))
        self.pred = nn.Sequential(
            _BasicConv2d(channel, channel, 3, 1, 1),
            nn.Conv2d(channel, num_classes, 1))

    def forward(self, x):
        H, W = x.shape[2:]
        feats = self.encoder(x)
        c1, c2, c3, c4 = feats

        x1 = self.trans1(c1)
        x2 = self.trans2(c2)
        x3 = self.trans3(c3)
        x4 = self.trans4(c4)

        # Contrastive embedding from deepest feature
        emb = self.contrasthead(x4)

        # SMI on deep features
        x4 = self.smim(x4)
        x3 = self.smim(x3)
        x2 = self.smim(x2)

        # Cross-level interaction
        fused = self.cim(x4, x3, x2)

        # Fuse with shallow
        x1 = x1 + self.uper(fused)
        pred = self.pred(x1)
        pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)

        if self.training:
            return pred, emb
        return pred
