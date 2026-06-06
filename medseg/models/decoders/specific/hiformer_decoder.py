"""HiFormer decoder module.

Extracted from networks/transformer/hiformer_model.py for modular reuse.
Faithful to the original: ConvUpsample layers (Conv-GroupNorm-ReLU-Upsample
cascade) + conv_pred + seg_head prediction path.
"""
# Source: https://github.com/amirhossein-kz/HiFormer

import torch
import torch.nn as nn
from typing import List
from medseg.registry import DECODER_REGISTRY


class _ConvUpsample(nn.Module):
    """Multi-layer Conv-GroupNorm-ReLU with optional bilinear upsampling."""

    def __init__(self, in_chans=384, out_chans=(128,), upsample=True):
        super().__init__()
        self.out_chans = list(out_chans)
        layers = []
        c_in = in_chans
        for i, c_out in enumerate(self.out_chans):
            if i > 0:
                c_in = self.out_chans[i - 1]
            layers.append(nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False))
            layers.append(nn.GroupNorm(32, c_out))
            layers.append(nn.ReLU(inplace=False))
            if upsample:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear',
                                          align_corners=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@DECODER_REGISTRY.register("hiformer")
class HiFormerDecoder(nn.Module):
    """HiFormer decoder: ConvUpsample + conv_pred + seg_head.

    Standard interface: ``forward(bottleneck_feat, skip_features)``
    where bottleneck_feat is the deepest embedding and skip_features
    contains the shallower embedding.

    Architecture:
        conv_up_l(shallow_ch->128, no upsample) + conv_up_s(deep_ch->128->128, 2x upsample)
        -> element-wise add -> conv_pred (1x1+ReLU+4x upsample) -> seg_head (3x3)
    """

    has_internal_skip = True
    required_skip_stages = 1
    requires_encoder = "hiformer_enc"  # Designed for HiFormer's dual-branch encoder

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 num_classes=2,
                 **kwargs):
        super().__init__()
        # Derive channel sizes from encoder_channels (default to original HiFormer values)
        shallow_ch = encoder_channels[0] if encoder_channels else 96
        deep_ch = bottleneck_channels if bottleneck_channels else 384
        self.conv_up_s = _ConvUpsample(deep_ch, [128, 128], upsample=True)
        self.conv_up_l = _ConvUpsample(shallow_ch, [128], upsample=False)
        self.conv_pred = nn.Sequential(
            nn.Conv2d(128, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear',
                        align_corners=False))
        self.seg_head = nn.Conv2d(16, num_classes, 3, 1, 1)
        self._out_channels = num_classes

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # bottleneck_feat = deep embedding (384ch, patch_size=16)
        # skip_features[0] = shallow embedding (96ch, patch_size=4)
        shallow = skip_features[0] if skip_features else bottleneck_feat
        deep = bottleneck_feat
        r0 = self.conv_up_l(shallow)
        r1 = self.conv_up_s(deep)
        combined = r0 + r1
        combined = self.conv_pred(combined)
        return self.seg_head(combined)
