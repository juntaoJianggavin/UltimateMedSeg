"""EMCAD (Efficient Multi-scale Convolutional Attention Decoding) Decoder.
Faithfully ported from: https://github.com/SLDGroup/EMCAD/blob/main/lib/decoders.py

EMCAD has its own internal skip connection mechanism (LGAG + additive aggregation),
so external skip_connection module is IGNORED.
"""
# Source: https://github.com/SLDGroup/EMCAD

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class MSDC(nn.Module):
    """Multi-scale depth-wise convolution."""
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, ks, stride, ks // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
            )
            for ks in kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MSCB(nn.Module):
    """Multi-scale convolution block."""
    def __init__(self, in_channels, out_channels, stride=1, kernel_sizes=[1, 3, 5],
                 expansion_factor=2, dw_parallel=True, add=True):
        super().__init__()
        self.stride = stride
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip = (stride == 1)

        ex_channels = int(in_channels * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ex_channels),
            nn.ReLU6(inplace=True),
        )
        self.msdc = MSDC(ex_channels, kernel_sizes, stride, dw_parallel=dw_parallel)

        combined_channels = ex_channels if add else ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if self.use_skip and in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self._combined_channels = combined_channels
        self._out_channels = out_channels
        self._in_channels = in_channels

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = sum(msdc_outs)
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, _gcd(self._combined_channels, self._out_channels))
        out = self.pconv2(dout)
        if self.use_skip:
            if self._in_channels != self._out_channels:
                x = self.conv1x1(x)
            return x + out
        return out


def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5],
              expansion_factor=2, dw_parallel=True, add=True):
    convs = [MSCB(in_channels, out_channels, stride, kernel_sizes, expansion_factor, dw_parallel, add)]
    for _ in range(1, n):
        convs.append(MSCB(out_channels, out_channels, 1, kernel_sizes, expansion_factor, dw_parallel, add))
    return nn.Sequential(*convs)


class EUCB(nn.Module):
    """Efficient up-convolution block."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, kernel_size // 2,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pwc = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


class LGAG(nn.Module):
    """Large-kernel grouped attention gate."""
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, 1, kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, 1, kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class CAB(nn.Module):
    """Channel attention block."""
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        if in_channels < ratio:
            ratio = in_channels
        reduced = in_channels // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SAB(nn.Module):
    """Spatial attention block."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


@DECODER_REGISTRY.register("emcad")
class EMCADDecoder(nn.Module):
    """EMCAD decoder - faithful port from original repo.

    Has its own internal skip mechanism (LGAG + additive aggregation + CAB + SAB).
    External skip_connection parameter is IGNORED.
    """
    has_internal_skip = True
    """

    Args:
        encoder_channels: List of encoder stage output channels (shallow to deep).
        bottleneck_channels: Deepest feature channels.
        kernel_sizes: MSDC kernel sizes.
        expansion_factor: MSCB expansion factor.
        dw_parallel: Whether MSDC runs in parallel.
        add: Whether to add (True) or concat (False) MSDC outputs.
        lgag_ks: LGAG kernel size.
    """

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, kernel_sizes=None, expansion_factor=6,
                 dw_parallel=True, add=True, lgag_ks=3, **kwargs):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]

        # channels: [deepest, ..., shallowest] matching original EMCAD convention
        # encoder_channels is [shallow, ..., deep-1], bottleneck_channels is deepest
        # Reverse encoder_channels so skips go from deep to shallow
        skip_chs = list(reversed(encoder_channels))  # [deep-1, ..., shallow]
        channels = [bottleneck_channels] + skip_chs

        eucb_ks = 3

        # Stage 4 (deepest): MSCB + MSCAM
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1,
                                kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                                dw_parallel=dw_parallel, add=add)
        self.cab4 = CAB(channels[0])

        # Build decoder stages for each skip connection
        self.eucbs = nn.ModuleList()
        self.lgags = nn.ModuleList()
        self.mscbs = nn.ModuleList()
        self.cabs = nn.ModuleList()

        for i in range(len(skip_chs)):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.eucbs.append(EUCB(in_ch, out_ch, kernel_size=eucb_ks))
            self.lgags.append(LGAG(F_g=out_ch, F_l=out_ch, F_int=out_ch // 2,
                                    kernel_size=lgag_ks, groups=out_ch // 2))
            self.mscbs.append(MSCBLayer(out_ch, out_ch, n=1, stride=1,
                                         kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                                         dw_parallel=dw_parallel, add=add))
            self.cabs.append(CAB(out_ch))

        self.sab = SAB()
        self._out_channels = skip_chs[-1] if skip_chs else bottleneck_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))  # deep to shallow

        # Stage 4: MSCAM + MSCB on bottleneck
        d = self.cab4(bottleneck_feat) * bottleneck_feat
        d = self.sab(d) * d
        d = self.mscb4(d)

        # Progressive decoding with LGAG gating
        for i in range(len(self.eucbs)):
            # EUCB (upsample)
            d = self.eucbs[i](d)
            # Match spatial size to skip feature (needed for ViT encoders
            # whose pyramid ratios differ from the standard 2x stride)
            skip_i = skips[i]
            if d.shape[2:] != skip_i.shape[2:]:
                d = F.interpolate(d, size=skip_i.shape[2:],
                                  mode='bilinear', align_corners=False)
            # LGAG (attention gate on skip feature)
            x_skip = self.lgags[i](g=d, x=skip_i)
            # Additive aggregation
            d = d + x_skip
            # MSCAM + MSCB
            d = self.cabs[i](d) * d
            d = self.sab(d) * d
            d = self.mscbs[i](d)

        return d
