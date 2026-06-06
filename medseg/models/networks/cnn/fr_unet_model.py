"""FR-UNet: Full-Resolution Network for vessel segmentation.

Reference:
    Liu et al., "Full-Resolution Network and Dual-Threshold Iteration for
    Retinal Vessel and Coronary Angiograph Segmentation", JBHI 2022.
    https://github.com/lseventeen/FR-UNet

Key innovations:
  - Multi-resolution interaction blocks with up/down sampling
  - Feature fusion: 1×1 + 3×3 + dilated 3×3 convolutions
  - Deep supervision with 5 output heads averaged
"""
# Source: https://github.com/lseventeen/FR-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Conv(nn.Module):
    """Double conv block with LeakyReLU, faithful to original FR-UNet."""

    def __init__(self, in_c, out_c, dp=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _FeatureFuse(nn.Module):
    """Multi-scale feature fusion: 1×1 + 3×3 + dilated 3×3."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv11 = nn.Conv2d(in_c, out_c, 1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(in_c, out_c, 3, padding=2, bias=False,
                                    dilation=2)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv33_di(x)
        return self.norm(x1 + x2 + x3)


class _Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x):
        return self.up(x)


class _Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, 2, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class _Block(nn.Module):
    """FR-UNet block with optional up/down sampling paths."""

    def __init__(self, in_c, out_c, dp=0, is_up=False, is_down=False,
                 fuse=False):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c

        if fuse:
            self.fuse = _FeatureFuse(in_c, out_c)
        else:
            self.fuse = nn.Conv2d(in_c, out_c, 1, 1)

        self.is_up = is_up
        self.is_down = is_down
        self.conv = _Conv(out_c, out_c, dp=dp)

        if self.is_up:
            self.up = _Up(out_c, out_c // 2)
        if self.is_down:
            self.down = _Down(out_c, out_c * 2)

    def forward(self, x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)

        if not self.is_up and not self.is_down:
            return x
        elif self.is_up and not self.is_down:
            return x, self.up(x)
        elif not self.is_up and self.is_down:
            return x, self.down(x)
        else:
            return x, self.up(x), self.down(x)


class FRUNet(nn.Module):
    """FR-UNet: Full-Resolution Network.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: spatial size (unused, kept for API consistency).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224):
        super().__init__()
        feature_scale = 2
        dropout = 0.2
        fuse = True

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        # [32, 64, 128, 256, 512]

        # Level 1 blocks
        self.block1_3 = _Block(in_channels, filters[0], dp=dropout,
                                is_up=False, is_down=True, fuse=fuse)
        self.block1_2 = _Block(filters[0], filters[0], dp=dropout,
                                is_up=False, is_down=True, fuse=fuse)
        self.block1_1 = _Block(filters[0] * 2, filters[0], dp=dropout,
                                is_up=False, is_down=True, fuse=fuse)
        self.block10 = _Block(filters[0] * 2, filters[0], dp=dropout,
                               is_up=False, is_down=True, fuse=fuse)
        self.block11 = _Block(filters[0] * 2, filters[0], dp=dropout,
                               is_up=False, is_down=True, fuse=fuse)
        self.block12 = _Block(filters[0] * 2, filters[0], dp=dropout,
                               is_up=False, is_down=False, fuse=fuse)
        self.block13 = _Block(filters[0] * 2, filters[0], dp=dropout,
                               is_up=False, is_down=False, fuse=fuse)

        # Level 2 blocks
        self.block2_2 = _Block(filters[1], filters[1], dp=dropout,
                                is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = _Block(filters[1] * 2, filters[1], dp=dropout,
                                is_up=True, is_down=True, fuse=fuse)
        self.block20 = _Block(filters[1] * 3, filters[1], dp=dropout,
                               is_up=True, is_down=True, fuse=fuse)
        self.block21 = _Block(filters[1] * 3, filters[1], dp=dropout,
                               is_up=True, is_down=False, fuse=fuse)
        self.block22 = _Block(filters[1] * 3, filters[1], dp=dropout,
                               is_up=True, is_down=False, fuse=fuse)

        # Level 3 blocks
        self.block3_1 = _Block(filters[2], filters[2], dp=dropout,
                                is_up=True, is_down=True, fuse=fuse)
        self.block30 = _Block(filters[2] * 2, filters[2], dp=dropout,
                               is_up=True, is_down=False, fuse=fuse)
        self.block31 = _Block(filters[2] * 3, filters[2], dp=dropout,
                               is_up=True, is_down=False, fuse=fuse)

        # Level 4 block
        self.block40 = _Block(filters[3], filters[3], dp=dropout,
                               is_up=True, is_down=False, fuse=fuse)

        # Deep supervision heads
        self.final1 = nn.Conv2d(filters[0], num_classes, 1)
        self.final2 = nn.Conv2d(filters[0], num_classes, 1)
        self.final3 = nn.Conv2d(filters[0], num_classes, 1)
        self.final4 = nn.Conv2d(filters[0], num_classes, 1)
        self.final5 = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        inp_size = x.shape[2:]

        # Forward pass faithful to original FR-UNet
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)

        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)

        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(
            torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)

        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(
            torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))

        _, x_up40 = self.block40(x_down3_1)

        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(
            torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(
            torch.cat([x_down20, x30, x_up40], dim=1))

        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))

        # Deep supervision: average 5 heads
        output = (self.final1(x1_1) + self.final2(x10) +
                  self.final3(x11) + self.final4(x12) +
                  self.final5(x13)) / 5.0

        if output.shape[-2:] != torch.Size(inp_size):
            output = F.interpolate(output, size=inp_size, mode='bilinear',
                                   align_corners=True)
        return output
