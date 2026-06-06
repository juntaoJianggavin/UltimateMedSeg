"""Standard UNet Decoder (Ronneberger et al., MICCAI 2015).

Reference: https://github.com/milesial/Pytorch-UNet
Paper: https://arxiv.org/abs/1505.04597

The classic UNet decoder uses transposed convolution for upsampling,
followed by concatenation with the corresponding encoder skip feature,
then two 3x3 conv-BN-ReLU blocks.
"""

import torch
import torch.nn as nn
from typing import List
from medseg.registry import DECODER_REGISTRY


class _DoubleConv(nn.Module):
    """Two consecutive 3x3 conv-BN-ReLU blocks (UNet building block)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


@DECODER_REGISTRY.register("unet")
class UNetDecoder(nn.Module):
    """Standard UNet decoder with transposed convolution upsampling.

    Architecture per stage:
        1. ConvTranspose2d (2x upsample, halve channels)
        2. Concatenate with encoder skip feature
        3. DoubleConv (conv-BN-ReLU × 2) to fuse

    Args:
        encoder_channels: list of channel dims from encoder stages
                          (shallow → deep, e.g. [64, 128, 256, 512]).
        bottleneck_channels: channel dim of the bottleneck output.
        skip_connection: optional skip connection module (if None, uses cat).
        decoder_channels: optional list specifying output channels per
                          decoder stage. If None, mirrors encoder_channels
                          in reverse.
    """

    has_internal_skip = False

    def __init__(
        self,
        encoder_channels: List[int],
        bottleneck_channels: int,
        skip_connection=None,
        decoder_channels: List[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.skip_connection = skip_connection

        # Decoder stages go from deep to shallow
        skip_channels = list(reversed(encoder_channels))

        if decoder_channels is not None:
            out_channels_list = decoder_channels
        else:
            out_channels_list = skip_channels

        self.up_convs = nn.ModuleList()
        self.double_convs = nn.ModuleList()

        in_ch = bottleneck_channels
        for i, skip_ch in enumerate(skip_channels):
            out_ch = out_channels_list[i]

            # Transposed convolution for 2x upsample
            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            )

            # After upconv, channels = in_ch // 2
            # After concat with skip, channels = in_ch // 2 + skip_ch
            if skip_connection is not None:
                merged_ch = skip_connection.get_out_channels(in_ch // 2, skip_ch)
            else:
                merged_ch = in_ch // 2 + skip_ch

            self.double_convs.append(_DoubleConv(merged_ch, out_ch))
            in_ch = out_ch

        self._out_channels = out_channels_list[-1] if out_channels_list else bottleneck_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))
        x = bottleneck_feat

        for i, (up, dconv) in enumerate(zip(self.up_convs, self.double_convs)):
            x = up(x)
            skip = skips[i]

            # Handle size mismatch (pad if needed, as in original UNet)
            if x.shape[2:] != skip.shape[2:]:
                diff_h = skip.shape[2] - x.shape[2]
                diff_w = skip.shape[3] - x.shape[3]
                x = nn.functional.pad(x, [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                ])

            if self.skip_connection is not None:
                x = self.skip_connection(x, skip)
            else:
                x = torch.cat([x, skip], dim=1)

            x = dconv(x)

        return x
