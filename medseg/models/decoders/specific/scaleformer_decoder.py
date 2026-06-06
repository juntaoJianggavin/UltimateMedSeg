"""ScaleFormer decoder module.

Extracted from networks/transformer/scaleformer_model.py for modular reuse.
Faithful to the original ScaleFormer decoder: 4-stage UNet decoder with
bilinear 2x upsample + concat skip + 2x Conv-BN-ReLU + final decoder block.
"""
# Source: https://github.com/ZJUGiveLab/ScaleFormer

import torch
import torch.nn as nn
from typing import List
from medseg.registry import DECODER_REGISTRY


class _Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super().__init__(conv, bn, relu)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _Conv2dReLU(in_channels, out_channels, 3, padding=1)
        self.conv2 = _Conv2dReLU(out_channels, out_channels, 3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class _SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                         padding=kernel_size // 2)
        super().__init__(conv)


@DECODER_REGISTRY.register("scaleformer")
class ScaleFormerDecoder(nn.Module):
    """ScaleFormer 4-stage UNet decoder + segmentation head.

    Standard interface: ``forward(bottleneck_feat, skip_features)``
    where skip_features = [sk1, sk2, sk3, sk4] (shallow→deep).

    The parallel encoder produces 5 features (including bottleneck).
    Decoder: 4 concat-decode stages + 1 final decode + seg head.
    """

    has_internal_skip = True
    required_skip_stages = 4
    requires_encoder = "resnet"  # Designed for ResNet-50 parallel encoder

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 enc_ch=None,
                 **kwargs):
        super().__init__()
        # Derive enc_ch from encoder_channels if not explicitly provided
        if enc_ch is None:
            if encoder_channels is not None and len(encoder_channels) == 4:
                # encoder_channels = [sk1, sk2, sk3, sk4], bottleneck is last
                enc_ch = [bottleneck_channels] + list(reversed(encoder_channels))
            else:
                enc_ch = [1024, 512, 256, 128, 64]
        self.decoder1 = _DecoderBlock(enc_ch[0] + enc_ch[0], enc_ch[1])
        self.decoder2 = _DecoderBlock(enc_ch[1] + enc_ch[1], enc_ch[2])
        self.decoder3 = _DecoderBlock(enc_ch[2] + enc_ch[2], enc_ch[3])
        self.decoder4 = _DecoderBlock(enc_ch[3] + enc_ch[3], enc_ch[4])
        self.decoder_final = _DecoderBlock(64, 64)
        self.segmentation_head = _SegmentationHead(64, 2)
        self._out_channels = 2

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # skip_features: [sk1(64), sk2(128), sk3(256), sk4(512)] shallow→deep
        # bottleneck_feat = sk5 (1024)
        x = self.decoder1(bottleneck_feat, skip_features[-1])
        x = self.decoder2(x, skip_features[-2])
        x = self.decoder3(x, skip_features[-3])
        x = self.decoder4(x, skip_features[-4])
        x = self.decoder_final(x, None)
        return self.segmentation_head(x)
