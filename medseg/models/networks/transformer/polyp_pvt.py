"""Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformer.

Reference:
    Bo Dong, Wenhai Wang, Deng-Ping Fan, Jinpeng Li, Huazhu Fu, Ling Shao.
    "Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers."
    Machine Intelligence Research (MIR), 2023.
    Upstream code: https://github.com/DengPingFan/Polyp-PVT

Architecture overview:
    - Encoder: PVTv2-B2 (Pyramid Vision Transformer v2) producing 4 multi-scale
      features at strides {4, 8, 16, 32} with channels {64, 128, 320, 512}.
    - Channel reduction via Receptive Field Block (RFB) on the top-3 stages.
    - Cascaded Fusion Module (CFM): aggregates high-level features 4-3-2 via
      element-wise multiplication + 3x3 conv + bilinear upsample.
    - Similarity Aggregation Module (SAM): refines the CFM prediction using the
      low-level (stride-4) feature with channel & spatial attention.
    - 1x1 head + bilinear upsample to the input resolution.

Self-contained: only torch and timm (for the PVT backbone) are required.
"""
# Source: https://github.com/DengPingFan/Polyp-PVT

import os

# Limit huggingface_hub retry/timeout budgets so a network outage does not
# stall model construction for minutes. Must be set before importing timm.
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '3')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '5')

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _BasicConv2d(nn.Module):
    """Conv + BN + ReLU, supporting tuple kernel/padding for non-square kernels."""

    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _RFB(nn.Module):
    """Receptive Field Block: reduces channels and aggregates multi-scale context.

    Used in Polyp-PVT to bring all encoder stages to a common channel dim while
    enlarging their receptive fields via dilated 3x3 convolutions.
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.branch0 = _BasicConv2d(in_c, out_c, kernel_size=1)
        self.branch1 = nn.Sequential(
            _BasicConv2d(in_c, out_c, kernel_size=1),
            _BasicConv2d(out_c, out_c, kernel_size=(1, 3), padding=(0, 1)),
            _BasicConv2d(out_c, out_c, kernel_size=(3, 1), padding=(1, 0)),
            _BasicConv2d(out_c, out_c, kernel_size=3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            _BasicConv2d(in_c, out_c, kernel_size=1),
            _BasicConv2d(out_c, out_c, kernel_size=(1, 5), padding=(0, 2)),
            _BasicConv2d(out_c, out_c, kernel_size=(5, 1), padding=(2, 0)),
            _BasicConv2d(out_c, out_c, kernel_size=3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            _BasicConv2d(in_c, out_c, kernel_size=1),
            _BasicConv2d(out_c, out_c, kernel_size=(1, 7), padding=(0, 3)),
            _BasicConv2d(out_c, out_c, kernel_size=(7, 1), padding=(3, 0)),
            _BasicConv2d(out_c, out_c, kernel_size=3, padding=7, dilation=7),
        )
        self.conv_cat = _BasicConv2d(4 * out_c, out_c, kernel_size=3, padding=1)
        self.conv_res = _BasicConv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        cat = self.conv_cat(torch.cat([x0, x1, x2, x3], dim=1))
        return self.relu(cat + self.conv_res(x))


class _CFM(nn.Module):
    """Cascaded Fusion Module.

    Fuses three top-level features (x1=stride32, x2=stride16, x3=stride8) via
    cascaded element-wise multiplication, 3x3 conv refinement, and bilinear
    upsampling. Returns a feature map at the resolution of x3 (stride 8).
    """

    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)
        self.conv_upsample1 = _BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv_upsample2 = _BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv_upsample3 = _BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv_upsample4 = _BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv_upsample5 = _BasicConv2d(2 * channel, 2 * channel, kernel_size=3, padding=1)
        self.conv_concat2 = _BasicConv2d(2 * channel, 2 * channel, kernel_size=3, padding=1)
        self.conv_concat3 = _BasicConv2d(3 * channel, 3 * channel, kernel_size=3, padding=1)
        self.conv4 = _BasicConv2d(3 * channel, channel, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3):
        # x1: highest level (smallest spatial)
        # x2: middle level
        # x3: lowest level (largest spatial)
        # Faithful to official: upsample chain then conv
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat([x2_1, self.conv_upsample4(self.upsample(x1_1))], dim=1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat([x3_1, self.conv_upsample5(self.upsample(x2_2))], dim=1)
        x3_2 = self.conv_concat3(x3_2)

        return self.conv4(x3_2)


class _GCN(nn.Module):
    """Graph Convolutional Network module (faithful to official Polyp-PVT)."""

    def __init__(self, num_state, num_node, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class _SAM(nn.Module):
    """Similarity Aggregation Module (faithful to official Polyp-PVT).

    Uses GCN-based graph reasoning with edge-guided priors to refine the CFM
    output using the low-level (stride-4) feature.
    """

    def __init__(self, channel, num_in=32, plane_mid=16, mids=4):
        super().__init__()
        self.normalize = False
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = _GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.interpolate(edge, size=(x.size()[-2], x.size()[-1]),
                             mode='bilinear', align_corners=True)

        n, c, h, w = x.size()
        edge = F.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(
            x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(
            x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + self.conv_extend(x_state)

        return out


class _ChannelAttention(nn.Module):
    """Channel attention (faithful to official Polyp-PVT)."""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class _SpatialAttention(nn.Module):
    """Spatial attention (faithful to official Polyp-PVT)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class PolypPVT(nn.Module):
    """Polyp-PVT segmentation network.

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation classes.
        img_size: nominal input spatial size (not enforced; forward is fully
            convolutional and accepts arbitrary H=W).
        channel: shared channel dimension for the decoder (RFB output).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 channel=32, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # --- Backbone -------------------------------------------------------
        self.backbone = load_with_ssl_fallback(
            timm.create_model,
            'pvt_v2_b2',
            features_only=True,
            pretrained=True,
            in_chans=in_channels,
        )

        enc_channels = self.backbone.feature_info.channels()
        # Expect 4 stages from pvt_v2_b2: [64, 128, 320, 512]
        if len(enc_channels) != 4:
            raise RuntimeError(
                f'Expected 4 encoder features, got {len(enc_channels)}: {enc_channels}'
            )
        c1, c2, c3, c4 = enc_channels

        # --- Decoder --------------------------------------------------------
        # CIM: channel + spatial attention on low-level feature
        self.ca = _ChannelAttention(c1)
        self.sa = _SpatialAttention()

        self.rfb2 = _RFB(c2, channel)
        self.rfb3 = _RFB(c3, channel)
        self.rfb4 = _RFB(c4, channel)
        self.cfm = _CFM(channel)

        # SAM: needs CIM feature reduced to `channel` dim + downsampled
        self.translayer2_0 = _BasicConv2d(c1, channel, kernel_size=1)
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear',
                                  align_corners=True)
        self.sam = _SAM(channel, num_in=channel)

        # --- Heads (faithful: separate CFM and SAM prediction heads) --------
        self.out_cfm = nn.Conv2d(channel, num_classes, kernel_size=1)
        self.out_sam = nn.Conv2d(channel, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2:]
        f1, f2, f3, f4 = self.backbone(x)

        # CIM: channel + spatial attention on low-level feature (faithful)
        f1_ca = self.ca(f1) * f1
        cim_feature = self.sa(f1_ca) * f1_ca

        # CFM: fuse high-level features (faithful)
        r2 = self.rfb2(f2)
        r3 = self.rfb3(f3)
        r4 = self.rfb4(f4)
        cfm_feature = self.cfm(r4, r3, r2)       # stride 8

        # SAM: refine CFM output with CIM feature (faithful)
        T2 = self.translayer2_0(cim_feature)
        T2 = self.down05(T2)                      # stride 8
        sam_feature = self.sam(cfm_feature, T2)

        # Two prediction heads (faithful to official)
        pred_cfm = self.out_cfm(cfm_feature)
        pred_sam = self.out_sam(sam_feature)
        pred_cfm = F.interpolate(
            pred_cfm, size=(H, W), mode='bilinear', align_corners=False)
        pred_sam = F.interpolate(
            pred_sam, size=(H, W), mode='bilinear', align_corners=False)
        # Return SAM prediction (typically more refined)
        return pred_sam
