"""DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation.

Faithful port from github.com/DebeshJha/Doubleunet_pytorch.

Key components (matching official source):
    - Encoder1: VGG19 pretrained backbone
    - ASPP: Atrous Spatial Pyramid Pooling (avgpool(2,2) + dilations 6,12,18)
    - Decoder1: upsampling + conv_block with squeeze-excitation
    - Encoder2: custom CNN (no VGG), takes image * sigmoid(pred1)
    - Decoder2: fuses skip connections from BOTH encoder1 and encoder2
    - Squeeze-excitation blocks in conv_block

Reference:
    Jha et al., DoubleU-Net: A Deep Convolutional Neural Network for
    Medical Image Segmentation. IEEE CBMS 2020.
"""
# Source: https://github.com/DebeshJha/Doubleunet_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


class Conv2D(nn.Module):
    """Official Conv2D block with optional SE-style activation."""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1,
                 bias=False, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding,
                      dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class squeeze_excitation_block(nn.Module):
    """Squeeze-and-Excitation block (official implementation)."""
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling - faithful to official source.

    Uses AdaptiveAvgPool2d((2,2)) and dilations (6, 12, 18).
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )
        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)
        self.c5 = Conv2D(out_c * 5, out_c, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear",
                           align_corners=True)
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        y = self.c5(xc)
        return y


class conv_block(nn.Module):
    """Conv block with squeeze-excitation (official implementation)."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.a1 = squeeze_excitation_block(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.a1(x)
        return x


class encoder1(nn.Module):
    """VGG19 pretrained encoder (official implementation)."""
    def __init__(self):
        super().__init__()
        network = load_with_ssl_fallback(vgg19, pretrained=True)
        self.x1 = network.features[:4]
        self.x2 = network.features[4:9]
        self.x3 = network.features[9:18]
        self.x4 = network.features[18:27]
        self.x5 = network.features[27:36]

    def forward(self, x):
        x0 = x
        x1 = self.x1(x0)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        return x5, [x4, x3, x2, x1]


class decoder1(nn.Module):
    """Decoder with bilinear upsampling + conv_block + SE (official)."""
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=True)
        self.c1 = conv_block(64 + 512, 256)
        self.c2 = conv_block(512, 128)
        self.c3 = conv_block(256, 64)
        self.c4 = conv_block(128, 32)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip

        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s3], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, s4], axis=1)
        x = self.c4(x)

        return x


class encoder2(nn.Module):
    """Custom CNN encoder for second UNet (official implementation).

    Takes image * sigmoid(pred1) as input (NOT VGG).
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d((2, 2))
        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x0 = x
        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        return p4, [x4, x3, x2, x1]


class decoder2(nn.Module):
    """Decoder that fuses skip connections from BOTH encoders (official)."""
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=True)
        self.c1 = conv_block(832, 256)
        self.c2 = conv_block(640, 128)
        self.c3 = conv_block(320, 64)
        self.c4 = conv_block(160, 32)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x


class build_doubleunet(nn.Module):
    """Official DoubleU-Net architecture.

    Returns (y1, y2) tuple from both UNet outputs.
    """
    def __init__(self):
        super().__init__()
        self.e1 = encoder1()
        self.a1 = ASPP(512, 64)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.e2 = encoder2()
        self.a2 = ASPP(256, 64)
        self.d2 = decoder2()
        self.y2 = nn.Conv2d(32, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x0 = x
        x, skip1 = self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)

        input_x = x0 * self.sigmoid(y1)
        x, skip2 = self.e2(input_x)
        x = self.a2(x)
        x = self.d2(x, skip1, skip2)
        y2 = self.y2(x)

        return y1, y2


class DoubleUNet(nn.Module):
    """DoubleU-Net wrapper with standard interface.

    Faithful to official source: VGG19 encoder, SE blocks, ASPP(avgpool(2,2)),
    dual-skip decoder2, image*sigmoid(pred1) for second UNet.

    Args:
        in_channels: Input channels (default 3; projected to 3 if different).
        num_classes: Output segmentation classes.
        img_size: Input spatial size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Input projection if in_channels != 3
        self.input_proj = (
            nn.Conv2d(in_channels, 3, 1, bias=False)
            if in_channels != 3 else nn.Identity()
        )

        self.model = build_doubleunet()

        # y1 MUST stay at 1 channel (used for sigmoid gating with input image).
        # Only replace y2 for final num_classes output.
        if num_classes != 1:
            self.model.y2 = nn.Conv2d(32, num_classes, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        _, y2 = self.model(x_proj)
        # Interpolate to input size if needed
        if y2.shape[2:] != x.shape[2:]:
            y2 = F.interpolate(y2, size=x.shape[2:], mode="bilinear",
                               align_corners=True)
        return y2
