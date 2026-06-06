"""MultiResUNet – self-contained port from nibtehaz/MultiResUNet.

MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical
image segmentation (Ibtehaz & Rahman, Neural Networks 2020).

Architecture: Replaces standard convolution blocks with MultiResBlocks
(multi-resolution 3x3/5x5/7x7 convolutions) and uses ResPath for
skip connections to reduce the semantic gap.
"""
# Source: https://github.com/nibtehaz/MultiResUNet

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _ConvBlock(nn.Module):
    """Single Conv-BN-ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _MultiResBlock(nn.Module):
    """Multi-resolution block: parallel 3x3, 5x5, 7x7 convolutions.

    Approximated with cascaded 3x3 convolutions for efficiency.
    Also includes a 1x1 shortcut + residual connection.
    Final 1x1 conv ensures exact out_ch output regardless of rounding.
    """

    def __init__(self, in_ch, out_ch, alpha=1.67):
        super().__init__()
        W = out_ch * alpha
        self.W3x3 = int(W * 0.167)
        self.W5x5 = int(W * 0.333)
        self.W7x7 = int(W * 0.5)
        self.total_ch = self.W3x3 + self.W5x5 + self.W7x7

        # Cascaded 3x3 convolutions
        self.conv3x3 = _ConvBlock(in_ch, self.W3x3, 3, 1, 1)
        self.conv5x5 = _ConvBlock(self.W3x3, self.W5x5, 3, 1, 1)
        self.conv7x7 = _ConvBlock(self.W5x5, self.W7x7, 3, 1, 1)

        # Shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, self.total_ch, 1, bias=False),
            nn.BatchNorm2d(self.total_ch),
        )
        self.bn1 = nn.BatchNorm2d(self.total_ch)
        self.bn2 = nn.BatchNorm2d(self.total_ch)
        self.relu = nn.ReLU(inplace=True)
        # Project back to exact out_ch
        self.reduce = nn.Sequential(
            nn.Conv2d(self.total_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s = self.shortcut(x)
        a = self.conv3x3(x)
        b = self.conv5x5(a)
        c = self.conv7x7(b)
        out = torch.cat([a, b, c], dim=1)
        out = self.bn1(out)
        out = self.relu(out + s)
        out = self.bn2(out)
        out = self.relu(out)
        return self.reduce(out)


class _ResPath(nn.Module):
    """Residual path for skip connections.

    Stacks `depth` residual conv blocks (stride=1) to reduce the semantic
    gap between encoder and decoder features while keeping spatial dims.
    """

    def __init__(self, in_ch, out_ch, depth):
        super().__init__()
        self.shortcuts = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(depth):
            c_in = in_ch if i == 0 else out_ch
            self.shortcuts.append(nn.Conv2d(c_in, out_ch, 1, bias=False))
            self.convs.append(nn.Conv2d(c_in, out_ch, 3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for shortcut, conv, bn in zip(self.shortcuts, self.convs, self.bns):
            s = shortcut(x)
            x = self.relu(bn(conv(x)) + s)
        return x


# ---------------------------------------------------------------------------
# MultiResUNet
# ---------------------------------------------------------------------------
class MultiResUNet(nn.Module):
    """MultiResUNet with 4 encoder levels.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224, unused).
        alpha: MultiResBlock width scaling factor (default 1.67).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 alpha=1.67, **kwargs):
        super().__init__()
        self.alpha = alpha
        # Encoder (MultiResBlocks) — output ch = base (via reduce conv)
        self.enc1 = _MultiResBlock(in_channels, 32, alpha)
        self.enc2 = _MultiResBlock(32, 64, alpha)
        self.enc3 = _MultiResBlock(64, 128, alpha)
        self.enc4 = _MultiResBlock(128, 256, alpha)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _MultiResBlock(256, 512, alpha)

        # ResPath skip connections (output ch = base)
        self.respath1 = _ResPath(32, 32, depth=4)
        self.respath2 = _ResPath(64, 64, depth=3)
        self.respath3 = _ResPath(128, 128, depth=2)
        self.respath4 = _ResPath(256, 256, depth=1)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _MultiResBlock(256 + 256, 256, alpha)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _MultiResBlock(128 + 128, 128, alpha)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _MultiResBlock(64 + 64, 64, alpha)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _MultiResBlock(32 + 32, 32, alpha)

        # Output
        self.out_conv = nn.Conv2d(32, num_classes, 1)

    @staticmethod
    def _out_ch(base, alpha):
        """Compute MultiResBlock output channels."""
        W = int(base * alpha)
        return int(W * 0.167) + int(W * 0.333) + int(W * 0.5)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with ResPath skip connections
        d4 = self.up4(b)
        e4 = self.respath4(e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.respath3(e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.respath2(e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.respath1(e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear',
                                align_corners=False)
        return out
