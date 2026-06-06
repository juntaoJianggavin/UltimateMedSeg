"""UNet 3+: Full-Scale Skip Connections and Deep Supervision.

Reference: Huang et al., "UNet 3+: A Full-Scale Connected UNet for Medical Image
Segmentation", ICASSP 2020.

Key idea: Each decoder node aggregates FULL-SCALE features from ALL encoder levels
AND all higher-resolution decoder levels, using a unified channel count for each
scale's contribution. This captures both fine-grained and coarse-grained semantics.
"""
# Source: https://github.com/ZJUGiveLab/UNet-Version

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU x2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SingleConv(nn.Module):
    """Single Conv-BN-ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FullScaleBlock(nn.Module):
    """Full-scale skip connection block for UNet 3+.

    Aggregates features from ALL encoder levels and decoded higher-resolution levels
    by adapting each to a unified channel count `cat_ch`, then concatenating.

    For decoder level i (0=shallowest), the inputs are:
      - Encoder levels 0..i: max-pooled to target spatial size, each projected to cat_ch
      - Encoder levels i+1..4 (deeper): upsampled to target spatial size, each projected to cat_ch
      - Already-decoded levels for i+1..3 (if available): upsampled similarly
    """
    def __init__(self, enc_channels, dec_channels_so_far, level, cat_ch=64):
        """
        Args:
            enc_channels: List of 5 encoder channel counts [e0, e1, e2, e3, e4].
            dec_channels_so_far: Dict mapping decoder level -> channel count for
                already-computed decoder outputs at deeper (lower-res) levels.
            level: Current decoder level index (0=shallowest, 3=deepest decoder).
            cat_ch: Unified channel count per contributing scale.
        """
        super().__init__()
        n_enc = len(enc_channels)  # 5

        self.level = level
        self.n_enc = n_enc
        self.cat_ch = cat_ch

        # --- From encoder levels ---
        self.enc_convs = nn.ModuleList()
        for i in range(n_enc):
            self.enc_convs.append(SingleConv(enc_channels[i], cat_ch))

        # --- From already-decoded deeper levels ---
        self.dec_convs = nn.ModuleList()
        self.dec_levels = sorted(dec_channels_so_far.keys())  # deeper decoder levels
        for dl in self.dec_levels:
            self.dec_convs.append(SingleConv(dec_channels_so_far[dl], cat_ch))

        # Total concatenated channels
        total_cat = cat_ch * (n_enc + len(self.dec_levels))
        self.fuse = SingleConv(total_cat, cat_ch * n_enc)

    def forward(self, enc_features, dec_features, target_size):
        """
        Args:
            enc_features: List of 5 encoder feature maps [e0, e1, ..., e4].
            dec_features: Dict mapping decoder level -> feature for already-decoded levels.
            target_size: (H, W) spatial size for this decoder level.
        """
        parts = []

        # From encoder levels
        for i in range(self.n_enc):
            feat = enc_features[i]
            if feat.shape[2:] != target_size:
                if feat.shape[2] > target_size[0]:
                    feat = F.adaptive_max_pool2d(feat, target_size)
                else:
                    feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            parts.append(self.enc_convs[i](feat))

        # From already-decoded deeper levels
        for idx, dl in enumerate(self.dec_levels):
            feat = dec_features[dl]
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            parts.append(self.dec_convs[idx](feat))

        cat = torch.cat(parts, dim=1)
        return self.fuse(cat)


class UNet3Plus(nn.Module):
    """UNet 3+: Full-Scale Connected UNet.

    5-level architecture where each decoder node aggregates features from ALL
    encoder levels and all previously-decoded decoder levels via full-scale
    skip connections. Supports optional deep supervision.

    Args:
        in_channels: Number of input channels (default: 3).
        num_classes: Number of output classes (default: 2).
        img_size: Input image size (default: 224).
        base_ch: Base channel count (default: 64).
        cat_ch: Channel count per contributing scale in full-scale fusion (default: 64).
        deep_supervision: If True, output averaged predictions from all decoder levels.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 base_ch=64, cat_ch=64, deep_supervision=False, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        n = 5

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc0 = ConvBlock(in_channels, chs[0])
        self.enc1 = ConvBlock(chs[0], chs[1])
        self.enc2 = ConvBlock(chs[1], chs[2])
        self.enc3 = ConvBlock(chs[2], chs[3])
        self.enc4 = ConvBlock(chs[3], chs[4])  # bottleneck

        # Decoder: from deepest (level 3) to shallowest (level 0)
        # level 3: target spatial = enc3 spatial
        # level 2: target spatial = enc2 spatial
        # level 1: target spatial = enc1 spatial
        # level 0: target spatial = enc0 spatial

        fused_ch = cat_ch * n  # output channels per decoder level

        # Level 3 decoder (deepest decoder, no previous decoder outputs)
        self.fsb3 = FullScaleBlock(chs, {}, level=3, cat_ch=cat_ch)
        # Level 2 decoder (has decoder level 3)
        self.fsb2 = FullScaleBlock(chs, {3: fused_ch}, level=2, cat_ch=cat_ch)
        # Level 1 decoder (has decoder levels 3, 2)
        self.fsb1 = FullScaleBlock(chs, {3: fused_ch, 2: fused_ch}, level=1, cat_ch=cat_ch)
        # Level 0 decoder (has decoder levels 3, 2, 1)
        self.fsb0 = FullScaleBlock(chs, {3: fused_ch, 2: fused_ch, 1: fused_ch}, level=0, cat_ch=cat_ch)

        # Segmentation heads
        if deep_supervision:
            self.heads = nn.ModuleList([
                nn.Conv2d(fused_ch, num_classes, 1) for _ in range(4)
            ])
        else:
            self.head = nn.Conv2d(fused_ch, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        enc_features = [e0, e1, e2, e3, e4]

        # Decoder (deepest to shallowest)
        d3 = self.fsb3(enc_features, {}, target_size=e3.shape[2:])
        d2 = self.fsb2(enc_features, {3: d3}, target_size=e2.shape[2:])
        d1 = self.fsb1(enc_features, {3: d3, 2: d2}, target_size=e1.shape[2:])
        d0 = self.fsb0(enc_features, {3: d3, 2: d2, 1: d1}, target_size=e0.shape[2:])

        if self.deep_supervision:
            outputs = []
            for i, d in enumerate([d0, d1, d2, d3]):
                out_i = self.heads[i](d)
                if out_i.shape[2:] != input_size:
                    out_i = F.interpolate(out_i, size=input_size, mode='bilinear', align_corners=False)
                outputs.append(out_i)
            if self.training:
                return outputs  # [main, aux1, aux2, aux3]
            return outputs[0]
        else:
            out = self.head(d0)
            if out.shape[2:] != input_size:
                out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out
