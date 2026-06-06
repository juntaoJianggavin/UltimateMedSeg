"""R2U-Net - self-contained port from LeeJunHyun/Image_Segmentation.

R2U-Net: Recurrent Residual Convolutional Neural Network based on U-Net
(Alom et al., 2018).

Architecture:
  - Five-level UNet encoder/decoder (4 downsamples via MaxPool2d(2),
    4 upsamples via ConvTranspose2d(2, stride=2)).
  - Each "double conv" block is replaced by an RRCNN block: a 1x1 conv
    expands channels, then two stacked RCNN blocks (Recurrent CNN with
    t iterations) are applied with a residual addition.
  - Base feature width = 64; channels double per level (64, 128, 256,
    512, 1024).
  - Final 1x1 conv maps to ``num_classes``.
"""
# Source: https://github.com/LeeJunHyun/Image_Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Recurrent / Recurrent-Residual blocks
# ---------------------------------------------------------------------------
class _RecurrentBlock(nn.Module):
    """Recurrent convolution: Conv3x3 + BN + ReLU iterated ``t`` times,
    adding the original input back to the accumulating activation at each
    iteration (RCNN cell from Alom et al., 2018).
    """

    def __init__(self, ch_out, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = None
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class _RRCNNBlock(nn.Module):
    """Recurrent Residual CNN block.

    1x1 conv first projects the incoming features to ``ch_out`` channels,
    then two stacked recurrent conv blocks are applied. A residual skip
    adds the projected features back to the recurrent output.
    """

    def __init__(self, ch_in, ch_out, t=2):
        super().__init__()
        self.RCNN = nn.Sequential(
            _RecurrentBlock(ch_out, t=t),
            _RecurrentBlock(ch_out, t=t),
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1,
                                  padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


# ---------------------------------------------------------------------------
# R2U-Net
# ---------------------------------------------------------------------------
class R2UNet(nn.Module):
    """Recurrent Residual U-Net (R2U-Net).

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size:    Expected input spatial resolution (default 224, unused).
        t:           Number of recurrent iterations in each RCNN cell
                     (default 2).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, t=2,
                 **kwargs):
        super().__init__()
        base = 64
        ch = [base, base * 2, base * 4, base * 8, base * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)  # unused (kept for parity)

        # Encoder
        self.RRCNN1 = _RRCNNBlock(in_channels, ch[0], t=t)
        self.RRCNN2 = _RRCNNBlock(ch[0], ch[1], t=t)
        self.RRCNN3 = _RRCNNBlock(ch[1], ch[2], t=t)
        self.RRCNN4 = _RRCNNBlock(ch[2], ch[3], t=t)
        self.RRCNN5 = _RRCNNBlock(ch[3], ch[4], t=t)

        # Decoder (transposed conv upsample + RRCNN over concatenated skip)
        self.Up5 = nn.ConvTranspose2d(ch[4], ch[3], kernel_size=2, stride=2)
        self.Up_RRCNN5 = _RRCNNBlock(ch[4], ch[3], t=t)

        self.Up4 = nn.ConvTranspose2d(ch[3], ch[2], kernel_size=2, stride=2)
        self.Up_RRCNN4 = _RRCNNBlock(ch[3], ch[2], t=t)

        self.Up3 = nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2)
        self.Up_RRCNN3 = _RRCNNBlock(ch[2], ch[1], t=t)

        self.Up2 = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2)
        self.Up_RRCNN2 = _RRCNNBlock(ch[1], ch[0], t=t)

        # Output 1x1
        self.Conv_1x1 = nn.Conv2d(ch[0], num_classes, kernel_size=1, stride=1,
                                  padding=0)

    @staticmethod
    def _match(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode='bilinear',
                               align_corners=False)
        return up

    def forward(self, x):
        in_size = x.shape[-2:]

        # Encoder
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # Decoder
        d5 = self.Up5(x5)
        d5 = self._match(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = self._match(d4, x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = self._match(d3, x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = self._match(d2, x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv_1x1(d2)
        if out.shape[-2:] != in_size:
            out = F.interpolate(out, size=in_size, mode='bilinear',
                                align_corners=False)
        return out
