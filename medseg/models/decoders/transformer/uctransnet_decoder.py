"""UCTransNet decoder module.

Extracted from networks/transformer/uctransnet_model.py for modular reuse.
Faithful to the original: CCA (Channel-wise Cross Attention) skip +
UpBlockAttention (bilinear 2x up + CCA-gated skip + concat + N convs).
"""
# Source: https://github.com/McGregorWwww/UCTransNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class _ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        act_cls = getattr(nn, activation)
        self.activation = act_cls(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


def _make_nConv(in_ch, out_ch, nb_Conv, activation='ReLU'):
    layers = [_ConvBatchNorm(in_ch, out_ch, activation)]
    for _ in range(nb_Conv - 1):
        layers.append(_ConvBatchNorm(out_ch, out_ch, activation))
    return nn.Sequential(*layers)


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class _CCA(nn.Module):
    """Channel-wise Cross Attention for skip connection gating."""

    def __init__(self, F_g, F_x):
        super().__init__()
        self.F_x = F_x
        self.mlp_x = nn.Sequential(_Flatten(), nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(_Flatten(), nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        ch_att_x = self.mlp_x(avg_x)
        avg_g = F.avg_pool2d(g, (g.size(2), g.size(3)))
        ch_att_g = self.mlp_g(avg_g)
        scale = torch.sigmoid((ch_att_x + ch_att_g) / 2.0)
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)
        return self.relu(x * scale)


class _UpBlockAttention(nn.Module):
    """Single decoder step: bilinear up -> CCA-gated skip -> concat -> N convs."""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU',
                 up_channels=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        # up_channels: number of channels after upsample (before concat with skip)
        # If not provided, assume in_channels is evenly split (original behavior)
        if up_channels is not None:
            self.coatt = _CCA(F_g=up_channels, F_x=out_channels)
        else:
            self.coatt = _CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv,
                                  activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x)


@DECODER_REGISTRY.register("uctransnet")
class UCTransNetDecoder(nn.Module):
    """UCTransNet 4-stage decoder with CCA skip connections.

    Standard interface: ``forward(bottleneck_feat, skip_features)``
    where skip_features = [x1, x2, x3, x4] (shallow→deep).

    Architecture: 4 UpBlockAttention blocks with CCA-gated skip concat.
    """

    has_internal_skip = True
    required_skip_stages = 4
    requires_encoder = "uctransnet_enc"  # Requires UCTransNet's ChannelTransformer encoder

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 base_channel=64,
                 **kwargs):
        super().__init__()
        bc = base_channel
        # Derive channel sizes from encoder_channels if available
        # UCTransNet original: bottleneck=1024, skips=[64, 128, 256, 512]
        if encoder_channels is not None and len(encoder_channels) == 4:
            # Use actual encoder channels for CCA input sizes
            enc_chs = list(encoder_channels)
            # up4: input=bottleneck(up=enc_chs[-1]), skip=enc_chs[-1] -> concat -> enc_chs[-1]
            self.up4 = _UpBlockAttention(bottleneck_channels + enc_chs[-1], enc_chs[-1], 2,
                                          up_channels=bottleneck_channels)
            self.up3 = _UpBlockAttention(enc_chs[-1] + enc_chs[-2], enc_chs[-2], 2,
                                          up_channels=enc_chs[-1])
            self.up2 = _UpBlockAttention(enc_chs[-2] + enc_chs[-3], enc_chs[-3], 2,
                                          up_channels=enc_chs[-2])
            self.up1 = _UpBlockAttention(enc_chs[-3] + enc_chs[-4], enc_chs[-4], 2,
                                          up_channels=enc_chs[-3])
            self._out_channels = enc_chs[-4]
        else:
            self.up4 = _UpBlockAttention(bc * 16, bc * 4, 2)
            self.up3 = _UpBlockAttention(bc * 8, bc * 2, 2)
            self.up2 = _UpBlockAttention(bc * 4, bc, 2)
            self.up1 = _UpBlockAttention(bc * 2, bc, 2)
            self._out_channels = bc

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # skip_features: [x1, x2, x3, x4] shallow→deep
        x1, x2, x3, x4 = (skip_features[0], skip_features[1],
                           skip_features[2], skip_features[3])
        x = self.up4(bottleneck_feat, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return x
