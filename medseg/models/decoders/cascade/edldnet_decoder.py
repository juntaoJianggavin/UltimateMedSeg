"""EDLDNet Decoder – Efficient Dual-Line Decoder with Multi-Scale Convolutional Attention.

Faithfully ported from: https://github.com/riadhassan/EDLDNet
Paper: An Efficient Dual-Line Decoder Network with Multi-Scale Convolutional
       Attention for Multi-organ Segmentation
       (Biomedical Signal Processing and Control, 2025, Hassan et al.)

The decoder pipeline per stage:
    bottleneck -> MSCAM4 -> UCB3 + AG3(skip) -> add -> MSCAM3
             -> UCB2 + AG2(skip) -> add -> MSCAM2
             -> UCB1 + AG1(skip) -> add -> MSCAM1

MSCAM = CAB (Channel Attention Block) -> SAB (Spatial Attention Block)
       -> MSCB (Multi-Scale Convolution Block)

The paper uses dual-line decoders (noise-free + noisy) during training.
This implementation provides a single decoder line; the noisy training
variant can be handled externally by the training loop.

External skip_connection parameter is IGNORED (``has_internal_skip = True``).
"""
# Source: https://github.com/riadhassan/EDLDNet

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
    """Channel shuffle (from ShuffleNet)."""
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


# ── MSDC (Multi-Scale Depth-wise Convolution) ────────────────────────────────

class MSDC(nn.Module):
    """Multi-scale depth-wise convolution."""

    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True,
                 activation='relu6'):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, ks, stride, ks // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True) if activation == 'relu6'
                else nn.ReLU(inplace=True),
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


# ── MSCB (Multi-Scale Convolution Block) ─────────────────────────────────────

class MSCB(nn.Module):
    """Multi-scale convolution block."""

    def __init__(self, in_channels, out_channels, stride=1,
                 kernel_sizes=(1, 3, 5), expansion_factor=2,
                 dw_parallel=True, add=True, activation='relu6'):
        super().__init__()
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip = (stride == 1)
        ex_channels = int(in_channels * expansion_factor)

        relu_act = nn.ReLU6(inplace=True) if activation == 'relu6' \
            else nn.ReLU(inplace=True)

        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ex_channels),
            relu_act,
        )
        self.msdc = MSDC(ex_channels, kernel_sizes, stride,
                         dw_parallel=dw_parallel, activation=activation)

        combined = ex_channels if add else ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(combined, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if self.use_skip and in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        self._combined = combined
        self._out_channels = out_channels
        self._in_channels = in_channels

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add:
            dout = sum(msdc_outs)
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, _gcd(self._combined, self._out_channels))
        out = self.pconv2(dout)
        if self.use_skip:
            if self._in_channels != self._out_channels:
                x = self.conv1x1(x)
            return x + out
        return out


def _mscb_layer(in_ch, out_ch, n=1, stride=1, kernel_sizes=(1, 3, 5),
                expansion_factor=2, dw_parallel=True, add=True,
                activation='relu6'):
    layers = [MSCB(in_ch, out_ch, stride, kernel_sizes, expansion_factor,
                   dw_parallel, add, activation)]
    for _ in range(1, n):
        layers.append(MSCB(out_ch, out_ch, 1, kernel_sizes, expansion_factor,
                           dw_parallel, add, activation))
    return nn.Sequential(*layers)


# ── UCB (Up-Convolution Block) ───────────────────────────────────────────────

class UCB(nn.Module):
    """Up-convolution block: upsample + DW conv + BN + ReLU + 1x1 conv."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        relu_act = nn.ReLU6(inplace=True) if activation == 'relu6' \
            else nn.ReLU(inplace=True)
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, kernel_size, 1,
                      kernel_size // 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            relu_act,
        )
        self.pwc = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


# ── AG (Attention Gate) ─────────────────────────────────────────────────────

class AG(nn.Module):
    """Attention gate: refines skip features using gating signal.

    AG(x_s, x_u) = sigmoid(BN(Conv1x1(relu(BN(Conv3x3(x_u)) + BN(Conv3x3(x_s)))))) * x_s
    """

    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1,
                 activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, 1, kernel_size // 2,
                      groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, 1, kernel_size // 2,
                      groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        relu_act = nn.ReLU6(inplace=True) if activation == 'relu6' \
            else nn.ReLU(inplace=True)
        self.act = relu_act

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ── CAB (Channel Attention Block) ────────────────────────────────────────────

class CAB(nn.Module):
    """CBAM-style channel attention: avg+max pool -> shared MLP -> sigmoid."""

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


# ── SAB (Spatial Attention Block) ────────────────────────────────────────────

class SAB(nn.Module):
    """CBAM-style spatial attention: avg+max along channel -> conv -> sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


# ── EDLDNet Decoder ─────────────────────────────────────────────────────────

@DECODER_REGISTRY.register("edldnet")
class EDLDNetDecoder(nn.Module):
    """EDLDNet decoder with MSCAM + AG + UCB.

    Pipeline:
        bottleneck -> MSCAM4 -> UCB3 -> AG3(skip[0]) -> add -> MSCAM3
                 -> UCB2 -> AG2(skip[1]) -> add -> MSCAM2
                 -> UCB1 -> AG1(skip[2]) -> add -> MSCAM1

    MSCAM = CAB -> SAB -> MSCB (Multi-Scale Convolution Block).

    Args:
        encoder_channels: List of encoder stage output channels (shallow to deep).
        bottleneck_channels: Channels of the bottleneck feature.
        kernel_sizes: MSDC kernel sizes.
        expansion_factor: MSCB expansion factor.
        dw_parallel: Whether MSDC runs in parallel.
        add: Whether to add (True) or concat (False) MSDC outputs.
        ag_ks: Attention Gate kernel size.
        activation: Activation function ('relu' or 'relu6').
        skip_connection: IGNORED (internal skip used).
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int],
                 bottleneck_channels: int,
                 skip_connection=None,
                 kernel_sizes=None, expansion_factor=6,
                 dw_parallel=True, add=True,
                 ag_ks=3, activation='relu6',
                 **kwargs):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 5]

        # channels: [bottleneck, skip_deep, ..., skip_shallow]
        skip_chs = list(reversed(encoder_channels))
        channels = [bottleneck_channels] + skip_chs  # e.g. [512, 320, 128, 64]

        ucb_ks = 3

        # Stage 4 (deepest): MSCAM on bottleneck
        self.mscb4 = _mscb_layer(channels[0], channels[0], n=1, stride=1,
                                  kernel_sizes=kernel_sizes,
                                  expansion_factor=expansion_factor,
                                  dw_parallel=dw_parallel, add=add,
                                  activation=activation)
        self.cab4 = CAB(channels[0])

        # Build decoder stages
        self.ucbs = nn.ModuleList()
        self.ags = nn.ModuleList()
        self.mscbs = nn.ModuleList()
        self.cabs = nn.ModuleList()

        for i in range(len(skip_chs)):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            self.ucbs.append(UCB(in_ch, out_ch, kernel_size=ucb_ks,
                                 activation=activation))
            self.ags.append(AG(F_g=out_ch, F_l=out_ch, F_int=out_ch // 2,
                               kernel_size=ag_ks, groups=out_ch // 2,
                               activation=activation))
            self.mscbs.append(_mscb_layer(out_ch, out_ch, n=1, stride=1,
                                           kernel_sizes=kernel_sizes,
                                           expansion_factor=expansion_factor,
                                           dw_parallel=dw_parallel, add=add,
                                           activation=activation))
            self.cabs.append(CAB(out_ch))

        self.sab = SAB()
        self._out_channels = skip_chs[-1] if skip_chs else bottleneck_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        skips = list(reversed(skip_features))  # deep to shallow

        # Stage 4: MSCAM on bottleneck
        d = self.cab4(bottleneck_feat) * bottleneck_feat
        d = self.sab(d) * d
        d = self.mscb4(d)

        # Progressive decoding with AG gating
        for i in range(len(self.ucbs)):
            # UCB (upsample)
            d = self.ucbs[i](d)
            # Match spatial size to skip feature
            skip_i = skips[i]
            if d.shape[2:] != skip_i.shape[2:]:
                d = F.interpolate(d, size=skip_i.shape[2:],
                                  mode='bilinear', align_corners=False)
            # AG (attention gate on skip feature)
            x_skip = self.ags[i](g=d, x=skip_i)
            # Additive aggregation
            d = d + x_skip
            # MSCAM (CAB -> SAB -> MSCB)
            d = self.cabs[i](d) * d
            d = self.sab(d) * d
            d = self.mscbs[i](d)

        return d
