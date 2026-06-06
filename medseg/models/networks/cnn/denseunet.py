"""DenseUNet: DenseNet-based UNet for medical image segmentation.

Reference:
    Li et al., "UNet++: A Nested U-Net Architecture for Medical Image
    Segmentation" + DenseNet backbone inspired by:
    https://github.com/zh320/medical-segmentation-pytorch (DenseUNet)
    https://github.com/nibtehaz/MultiResUNet (DenseNet encoder idea)

Uses DenseNet-style dense blocks in the encoder with transition layers
for downsampling, and a standard UNet decoder with skip connections.
"""
# Source: https://github.com/zh320/medical-segmentation-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DenseLayer(nn.Module):
    """Single dense layer: BN → ReLU → 1×1 conv → BN → ReLU → 3×3 conv.

    Bottleneck design with 1×1 conv reducing channels before 3×3 conv.
    """

    def __init__(self, in_ch, growth_rate, bottleneck=4):
        super().__init__()
        mid_ch = bottleneck * growth_rate
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)


class _DenseBlock(nn.Module):
    """Stack of dense layers. Output channels = in_ch + num_layers * growth_rate."""

    def __init__(self, in_ch, num_layers, growth_rate=32, bottleneck=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_ch + i * growth_rate, growth_rate, bottleneck))
        self.block = nn.Sequential(*layers)
        self.out_channels = in_ch + num_layers * growth_rate

    def forward(self, x):
        return self.block(x)


class _TransitionDown(nn.Module):
    """Transition layer for downsampling: BN → ReLU → 1×1 conv → AvgPool2."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class _TransitionUp(nn.Module):
    """Upsample via transposed conv, concat skip, then 1×1 conv to reduce."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = nn.Sequential(
            _ConvBNReLU(in_ch + skip_ch, out_ch),
            _ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DenseUNet(nn.Module):
    """DenseUNet: DenseNet encoder + UNet decoder.

    Encoder uses 4 dense blocks with transition-down layers.
    Decoder uses transposed-conv upsampling with skip connections.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: spatial size (unused, kept for API consistency).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224):
        super().__init__()
        growth_rate = 32
        init_features = 64

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, init_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
        )

        # Dense encoder
        self.dense1 = _DenseBlock(init_features, num_layers=4, growth_rate=growth_rate)
        # out: 64 + 4*32 = 192
        self.trans1 = _TransitionDown(192, 96)

        self.dense2 = _DenseBlock(96, num_layers=4, growth_rate=growth_rate)
        # out: 96 + 4*32 = 224
        self.trans2 = _TransitionDown(224, 112)

        self.dense3 = _DenseBlock(112, num_layers=4, growth_rate=growth_rate)
        # out: 112 + 4*32 = 240
        self.trans3 = _TransitionDown(240, 120)

        self.dense4 = _DenseBlock(120, num_layers=4, growth_rate=growth_rate)
        # out: 120 + 4*32 = 248
        self.trans4 = _TransitionDown(248, 124)

        # Bottleneck dense block
        self.dense5 = _DenseBlock(124, num_layers=4, growth_rate=growth_rate)
        # out: 124 + 4*32 = 252

        # Decoder (skip from dense block outputs)
        self.up4 = _TransitionUp(252, 248, 128)   # /16→/8, concat 248
        self.up3 = _TransitionUp(128, 240, 128)   # /8→/4,  concat 240
        self.up2 = _TransitionUp(128, 224, 64)    # /4→/2,  concat 224
        self.up1 = _TransitionUp(64, 192, 64)     # /2→/1,  concat 192

        # Final upsample to stem resolution + output
        self.final_up = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.final_conv = nn.Sequential(
            _ConvBNReLU(64 + init_features, 64),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        # Stem
        s0 = self.stem(x)  # /1, 64 ch

        # Encoder — save dense block outputs as skips
        d1 = self.dense1(s0)       # /1, 192 ch
        t1 = self.trans1(d1)       # /2,  96 ch
        d2 = self.dense2(t1)       # /2, 224 ch
        t2 = self.trans2(d2)       # /4, 112 ch
        d3 = self.dense3(t2)       # /4, 240 ch
        t3 = self.trans3(d3)       # /8, 120 ch
        d4 = self.dense4(t3)       # /8, 248 ch
        t4 = self.trans4(d4)       # /16, 124 ch

        # Bottleneck
        b = self.dense5(t4)        # /16, 252 ch

        # Decoder — skip from dense block outputs
        u = self.up4(b, d4)        # /8, 252+248→128
        u = self.up3(u, d3)        # /4, 128+240→128
        u = self.up2(u, d2)        # /2, 128+224→64
        u = self.up1(u, d1)        # /1,  64+192→64

        # Final upsample to input resolution
        u = self.final_up(u)
        # pad to match s0
        dh = s0.size(2) - u.size(2)
        dw = s0.size(3) - u.size(3)
        u = F.pad(u, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        u = torch.cat([u, s0], dim=1)

        return self.final_conv(u)
