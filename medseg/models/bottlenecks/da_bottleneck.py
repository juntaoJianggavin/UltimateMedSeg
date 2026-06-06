"""Dual Attention (DA) bottleneck — faithful port from
https://github.com/junfu1115/DANet (Fu et al., CVPR 2019,
"Dual Attention Network for Scene Segmentation").

Files referenced (master branch):
  - encoding/nn/da_att.py   -> PAM_Module, CAM_Module
  - encoding/models/sseg/danet.py -> DANetHead

Notes:
  * ``PAM_Module`` / ``CAM_Module`` keep the official names and tensor
    shapes so that pretrained weights (if available) can be mapped 1-to-1.
  * ``CAM_Module`` includes the official ``energy_new = max(energy) - energy``
    step that some unofficial reproductions omit.
  * ``DABottleneck`` mirrors ``DANetHead`` (conv5a / conv5c projections to
    ``in_channels // 4`` -> PAM / CAM -> conv51 / conv52 -> element-wise sum)
    and finally projects back to ``in_channels`` to fit the project's
    bottleneck interface (the original head's classifier convs 6/7/8 are
    omitted because they would conflict with the modular decoder).
"""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Parameter, Softmax

from medseg.registry import BOTTLENECK_REGISTRY


# ====================================================================
#  PAM_Module — 1:1 from encoding/nn/da_att.py
# ====================================================================
class PAM_Module(Module):
    """Position attention module (CVPR 2019, junfu1115/DANet)."""

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        return self.gamma * out + x


# ====================================================================
#  CAM_Module — 1:1 from encoding/nn/da_att.py (incl. ``max - energy``)
# ====================================================================
class CAM_Module(Module):
    """Channel attention module (CVPR 2019, junfu1115/DANet)."""

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        # Official trick: subtract per-row max for numerical stability so that
        # the channel similarity matrix focuses on the *relative* differences.
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        return self.gamma * out + x


# ====================================================================
#  DABottleneck — mirrors DANetHead (conv5a/5c -> PAM/CAM -> conv51/52 -> sum)
# ====================================================================
@BOTTLENECK_REGISTRY.register("dual_attention")
class DABottleneck(nn.Module):
    """Dual Attention bottleneck (DANet head, junfu1115/DANet, CVPR 2019).

    Pipeline reproduced from ``DANetHead`` in
    ``encoding/models/sseg/danet.py``::

        x ─ conv5a ─ PAM ─ conv51 ─┐
                                   ├─(+)─ proj-back-to-C ─ out
        x ─ conv5c ─ CAM ─ conv52 ─┘

    The original head's classifier convs (conv6/7/8) are dropped because
    classification is performed by the decoder in this codebase; we instead
    add a 1×1 projection from ``inter_channels`` back to ``in_channels`` so
    the module can drop in as a generic bottleneck.
    """

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        inter_channels = max(in_channels // 4, 1)

        self.conv5a = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5c = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)

        self.conv51 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.conv52 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        # Project the fused dual-attention feature back to in_channels so that
        # this module fulfils the project's BOTTLENECK_REGISTRY contract
        # (input and output channel counts are equal).
        self.proj_out = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        # PAM branch
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        # CAM branch
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        # Fusion (element-wise sum, exactly as in DANetHead)
        feat_sum = sa_conv + sc_conv

        return self.proj_out(feat_sum)
