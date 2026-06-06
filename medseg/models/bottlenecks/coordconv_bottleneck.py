"""CoordConv bottleneck — NeurIPS 2018.

Official source: https://github.com/mkocabas/CoordConv-pytorch

Reference:
    Liu et al., "An Intriguing Failing of Convolutional Neural Networks and
    the CoordConv Solution", NeurIPS 2018.

Appends (x, y) coordinate channels before convolution so the network can
learn spatially-aware transformations at the bottleneck.
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


class AddCoords(nn.Module):
    """Append normalised (x, y) coordinate maps to the input tensor.

    Faithful port of the ``AddCoords`` class from the official PyTorch
    implementation. Coordinates are in [-1, 1] range.
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim, device=input_tensor.device,
                                  dtype=torch.float32).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim, device=input_tensor.device,
                                  dtype=torch.float32).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel / (x_dim - 1)
        yy_channel = yy_channel / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor),
        ], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
                + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2)
            )
            ret = torch.cat([ret, rr], dim=1)

        return ret


@BOTTLENECK_REGISTRY.register("coordconv")
class CoordConvBottleneck(nn.Module):
    """CoordConv bottleneck with residual connection.

    The coordinate channels (+2 for x, y) are prepended before a 3×3 conv
    so the convolution can condition on absolute spatial position.

    Args:
        in_channels: Number of input/output channels.
        with_r: If True, also append radial coordinate r = sqrt(x² + y²).
    """

    def __init__(self, in_channels, with_r=False, **kwargs):
        super().__init__()
        extra = 3 if with_r else 2
        self.add_coords = AddCoords(with_r=with_r)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + extra, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.conv(self.add_coords(x)) + x
