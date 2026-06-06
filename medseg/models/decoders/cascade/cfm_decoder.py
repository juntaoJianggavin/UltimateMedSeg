"""CFM Decoder - Cascaded Fusion Module (Polyp-PVT style).

Reference: Dong et al. "Polyp-PVT: Polyp Segmentation with Pyramid Vision
Transformers" (2021). The original CFM fuses multi-scale encoder features by
first projecting each one to a common low-dim channel via an RFB block, then
cascading from the deepest to the shallowest stage with elementwise
multiplication, a 3x3 conv, and bilinear up-sampling at every step.

This decoder owns its skip topology (``has_internal_skip = True``) because the
RFB + cascade-multiply path subsumes any external skip fusion; the framework's
``skip_connection`` argument is therefore accepted for API symmetry but
ignored.
"""
# Source: https://github.com/DengPingFan/Polyp-PVT

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class _BasicConv2d(nn.Module):
    """Conv -> BN -> ReLU helper used throughout the RFB block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size, stride: int = 1,
                 padding=0, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _RFB(nn.Module):
    """Receptive-Field Block (Polyp-PVT variant).

    Four parallel branches with asymmetric + dilated convolutions widen the
    receptive field, then their concatenation is fused with a residual 1x1
    projection of the input.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.branch0 = _BasicConv2d(in_ch, out_ch, kernel_size=1)
        self.branch1 = nn.Sequential(
            _BasicConv2d(in_ch, out_ch, kernel_size=1),
            _BasicConv2d(out_ch, out_ch, kernel_size=(1, 3), padding=(0, 1)),
            _BasicConv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0)),
            _BasicConv2d(out_ch, out_ch, kernel_size=3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            _BasicConv2d(in_ch, out_ch, kernel_size=1),
            _BasicConv2d(out_ch, out_ch, kernel_size=(1, 5), padding=(0, 2)),
            _BasicConv2d(out_ch, out_ch, kernel_size=(5, 1), padding=(2, 0)),
            _BasicConv2d(out_ch, out_ch, kernel_size=3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            _BasicConv2d(in_ch, out_ch, kernel_size=1),
            _BasicConv2d(out_ch, out_ch, kernel_size=(1, 7), padding=(0, 3)),
            _BasicConv2d(out_ch, out_ch, kernel_size=(7, 1), padding=(3, 0)),
            _BasicConv2d(out_ch, out_ch, kernel_size=3, padding=7, dilation=7),
        )
        self.conv_cat = _BasicConv2d(4 * out_ch, out_ch, kernel_size=3, padding=1)
        self.conv_res = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_res = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        cat = self.conv_cat(torch.cat([x0, x1, x2, x3], dim=1))
        res = self.bn_res(self.conv_res(x))
        return self.relu(cat + res)


@DECODER_REGISTRY.register("cfm")
class CFMDecoder(nn.Module):
    """Cascaded Fusion Module decoder.

    Args:
        encoder_channels: Skip-connection channels (shallow -> deep), typically
            ``encoder.out_channels[:-1]`` as wired by ``model_builder``.
        bottleneck_channels: Channels of the deepest (bottleneck) feature.
        skip_connection: Ignored (CFM fuses skips internally via multiply).
        img_size: Kept for API symmetry; not used internally.
        common_channels: Shared channel width every input is projected to via
            RFB. Defaults to 32 per the Polyp-PVT spec.
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection: nn.Module = None, img_size: int = 224,
                 common_channels: int = 32, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.common_channels = common_channels
        # ``skip_connection`` is accepted for API symmetry; CFM ignores it.
        self.skip_connection = None

        # One RFB per skip stage (shallow -> deep) and one for the bottleneck.
        self.skip_rfbs = nn.ModuleList(
            [_RFB(c, common_channels) for c in encoder_channels]
        )
        self.bottleneck_rfb = _RFB(bottleneck_channels, common_channels)

        # One 3x3 fusion conv per skip (applied after the cascade multiply).
        self.cascade_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(common_channels, common_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(common_channels),
                nn.ReLU(inplace=True),
            )
            for _ in encoder_channels
        ])

        # Output sits at the shallowest skip's spatial size with ``common_channels``.
        self._out_channels = common_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # ``skip_features`` is ordered shallow -> deep.
        n_skips = len(skip_features)
        if n_skips == 0:
            # Degenerate case: no skips, just project the bottleneck.
            return self.bottleneck_rfb(bottleneck_feat)
        if n_skips > len(self.skip_rfbs):
            raise ValueError(
                f"CFMDecoder built for at most {len(self.skip_rfbs)} skips "
                f"but received {n_skips}."
            )

        # Project each input to ``common_channels`` via its matching RFB.
        # When ``n_skips < len(self.skip_rfbs)`` we use the shallowest N RFBs,
        # matching the convention that skip_features[i] has channels
        # encoder_channels[i].
        skip_feats = [self.skip_rfbs[i](skip_features[i]) for i in range(n_skips)]
        x = self.bottleneck_rfb(bottleneck_feat)

        # Cascade deepest -> shallowest: upsample, elementwise multiply with
        # the matching skip, then 3x3 conv.
        for i in range(n_skips - 1, -1, -1):
            skip = skip_feats[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            x = x * skip
            x = self.cascade_convs[i](x)
        return x
