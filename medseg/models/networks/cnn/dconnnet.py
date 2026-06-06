"""DconnNet: Directional Connectivity-based Network for Medical Image Segmentation.

Faithful self-contained reimplementation from:
  https://github.com/Zyun-Y/DconnNet  (IEEE TMI 2024)

Key ideas:
  - ResNet34 encoder (torchvision, with SSL fallback for pretrained weights).
  - Sub-path Directional Excitation (SDE) module on the deepest feature.
  - Space-aware decoder with global-context modulation (SpaceBlock).
  - Lightweight multi-scale decoder (LWdecoder) aggregating pyramid features.
  - Two parallel heads sharing the encoder/decoder pyramid:
      (a) Connectivity head (8-direction map, num_classes * 8 channels).
      (b) Segmentation head (num_classes channels) -- returned tensor.

For our framework forward(x) returns only the segmentation tensor of shape
(B, num_classes, H, W) matching the input H/W.
"""
# Source: https://github.com/Zyun-Y/DconnNet

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


# ---------------------------------------------------------------------------
# ResNet34 encoder (torchvision wrapper)
# ---------------------------------------------------------------------------

class _ResNet34Encoder(nn.Module):
    """Torchvision ResNet34 surfaced as a 5-stage feature extractor.

    Returns features at strides 2, 4, 8, 16, 32 with channel counts
    (64, 64, 128, 256, 512) given a 3-channel input. For other in_channels,
    the stem conv is rebuilt and pretrained weights skipped on the stem.
    """

    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        from torchvision import models
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        except Exception:
            weights = None
        if weights is not None:
            resnet = load_with_ssl_fallback(models.resnet34, weights=weights)
        else:
            resnet = load_with_ssl_fallback(models.resnet34, pretrained=pretrained)

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                     padding=3, bias=False)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)          # 1/2,   64
        x = self.maxpool(c1)
        c2 = self.layer1(x)        # 1/4,   64
        c3 = self.layer2(c2)       # 1/8,  128
        c4 = self.layer3(c3)       # 1/16, 256
        c5 = self.layer4(c4)       # 1/32, 512
        return c1, c2, c3, c4, c5


# ---------------------------------------------------------------------------
# Position / Channel attention (DANet, used in SDE)
# ---------------------------------------------------------------------------

class _PAM(nn.Module):
    """Position attention module (SAGAN/DANet style)."""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, max(in_dim // 8, 1), 1)
        self.key_conv = nn.Conv2d(in_dim, max(in_dim // 8, 1), 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, h * w)
        attn = self.softmax(torch.bmm(q, k))
        v = self.value_conv(x).view(b, -1, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


class _CAM(nn.Module):
    """Channel attention module (DANet style)."""

    def __init__(self, in_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        q = x.view(b, c, -1)
        k = x.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(q, k)
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attn = self.softmax(energy)
        v = x.view(b, c, -1)
        out = torch.bmm(attn, v).view(b, c, h, w)
        return self.gamma * out + x


# ---------------------------------------------------------------------------
# DANet head used inside the SDE module
# ---------------------------------------------------------------------------

class _DANetHead(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv5a = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels), nn.ReLU())
        self.conv5c = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels), nn.ReLU())
        self.sa = _PAM(inter_channels)
        self.sc = _CAM(inter_channels)
        self.conv51 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels), nn.ReLU())
        self.conv52 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels), nn.ReLU())
        self.conv8 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, inter_channels, 1))

    def forward(self, x, enc_feat):
        sa = self.conv51(self.sa(self.conv5a(x)))
        sc = self.conv52(self.sc(self.conv5c(x)))
        feat_sum = (sa + sc) * torch.sigmoid(enc_feat)
        return self.conv8(feat_sum)


# ---------------------------------------------------------------------------
# Sub-path Directional Excitation (SDE) module
# ---------------------------------------------------------------------------

class _SDEModule(nn.Module):
    """Splits the deepest feature into 8 directional groups, each refined by a
    DANet head conditioned on the directional prior encoded from the GAP.
    """

    def __init__(self, in_channels, out_channels, num_class):
        super().__init__()
        self.inter_channels = in_channels // 8
        self.heads = nn.ModuleList([
            _DANetHead(self.inter_channels, self.inter_channels) for _ in range(8)
        ])
        self.final_conv = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(in_channels, out_channels, 1))

        if num_class < 32:
            self.reencoder = nn.Sequential(
                nn.Conv2d(num_class, num_class * 8, 1),
                nn.ReLU(True),
                nn.Conv2d(num_class * 8, in_channels, 1))
        else:
            self.reencoder = nn.Sequential(
                nn.Conv2d(num_class, in_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 1))

    def forward(self, x, d_prior):
        enc_feat = self.reencoder(d_prior)
        ic = self.inter_channels
        feats = []
        for i, head in enumerate(self.heads):
            feats.append(head(
                x[:, i * ic:(i + 1) * ic],
                enc_feat[:, i * ic:(i + 1) * ic]))
        feat = torch.cat(feats, dim=1)
        return self.final_conv(feat) + x


# ---------------------------------------------------------------------------
# Space-aware block: scene-context gated feature refinement
# ---------------------------------------------------------------------------

class _SpaceBlock(nn.Module):
    def __init__(self, in_channels, channel_in, out_channels):
        super().__init__()
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1),
        )
        self.content_encoders = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.feature_reencoders = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features):
        content_feats = self.content_encoders(features)
        scene_feat = self.scene_encoder(scene_feature)
        relations = self.normalizer((scene_feat * content_feats).sum(dim=1, keepdim=True))
        p_feats = self.feature_reencoders(features)
        return relations * p_feats


# ---------------------------------------------------------------------------
# Lightweight multi-scale decoder (LWdecoder) aggregating pyramid features
# ---------------------------------------------------------------------------

class _ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad,
                 dilation=1, groups=1, has_bn=True, has_relu=True, has_bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.bn = nn.BatchNorm2d(out_planes) if has_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if has_relu else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _FeatureBlock(nn.Module):
    def __init__(self, in_planes, out_planes, scale=2, relu=True, last=False):
        super().__init__()
        self.conv_3x3 = _ConvBnRelu(in_planes, in_planes, 3, 1, 1)
        self.conv_1x1 = _ConvBnRelu(in_planes, out_planes, 1, 1, 0)
        self.scale = scale
        self.last = last
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        if not self.last:
            x = self.conv_3x3(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x


class _LWDecoder(nn.Module):
    def __init__(self, in_channels, out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4):
        super().__init__()
        self.blocks = nn.ModuleList()
        for dec_level, in_feat_os in enumerate(in_feat_output_strides):
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))
            num_layers = num_upsample if num_upsample != 0 else 1
            layers = []
            for idx in range(num_layers):
                cin = in_channels[dec_level] if idx == 0 else out_channels
                layers.append(nn.Sequential(
                    nn.Conv2d(cin, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                ))
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, feat_list):
        outs = []
        for block, feat in zip(self.blocks, feat_list):
            outs.append(block(feat))
        # Align all to the same spatial size (smallest output of the stack).
        target = outs[0].shape[-2:]
        for i in range(1, len(outs)):
            if outs[i].shape[-2:] != target:
                outs[i] = F.interpolate(outs[i], size=target,
                                        mode='bilinear', align_corners=True)
        return sum(outs) / float(len(outs))


# ---------------------------------------------------------------------------
# DconnNet
# ---------------------------------------------------------------------------

class DconnNet(nn.Module):
    """Directional Connectivity Network for medical image segmentation.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of segmentation classes (default 2).
        img_size:    Reference square input size; used only as a hint for
                     pre-padding to the encoder's stride requirement.
    """

    _STRIDE = 32  # ResNet34 requires input H/W divisible by 32.

    def __init__(self, in_channels=3, num_classes=2, img_size=224, pretrained=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        out_planes = num_classes * 8  # 8 direction maps per class

        # --- Encoder ---
        self.backbone = _ResNet34Encoder(in_channels=in_channels, pretrained=pretrained)

        # --- SDE module on the deepest feature (c5: 512 channels) ---
        self.sde_module = _SDEModule(512, 512, out_planes)

        # --- Decoder feature blocks (upsample-conv) ---
        self.fb5 = _FeatureBlock(512, 256, relu=False, last=True)   # 1/32 -> 1/16
        self.fb4 = _FeatureBlock(256, 128, relu=False)              # 1/16 -> 1/8
        self.fb3 = _FeatureBlock(128, 64, relu=False)               # 1/8  -> 1/4
        self.fb2 = _FeatureBlock(64, 64)                            # 1/4  -> 1/2

        # --- Space-aware refinement blocks ---
        self.sb1 = _SpaceBlock(512, 512, 512)
        self.sb2 = _SpaceBlock(512, 256, 256)
        self.sb3 = _SpaceBlock(256, 128, 128)
        self.sb4 = _SpaceBlock(128, 64, 64)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        # --- Lightweight multi-scale decoder ---
        self.final_decoder = _LWDecoder(
            in_channels=[64, 64, 128, 256], out_channels=32,
            in_feat_output_strides=(4, 8, 16, 32), out_feat_output_stride=4)

        # --- Connectivity head: produces 8-direction connectivity map ---
        self.cls_pred_conv_2 = nn.Conv2d(32, out_planes, 1)
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_mapping = nn.Sequential(
            nn.Conv2d(512, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )
        self.direc_reencode = nn.Conv2d(out_planes, out_planes, 1)

        # --- Segmentation head: aggregates 8-direction map -> num_classes ---
        # A simple 1x1 conv that learns to combine the 8 directional channels
        # of each class into a single per-class score map.
        self.seg_head = nn.Conv2d(out_planes, num_classes, 1)

    # ------------------------------------------------------------------
    # Padding / cropping to encoder stride
    # ------------------------------------------------------------------

    def _pad_to_multiple(self, x):
        _, _, h, w = x.shape
        ph = (self._STRIDE - h % self._STRIDE) % self._STRIDE
        pw = (self._STRIDE - w % self._STRIDE) % self._STRIDE
        if ph == 0 and pw == 0:
            return x, (0, 0, 0, 0)
        # Pad right and bottom.
        x = F.pad(x, (0, pw, 0, ph))
        return x, (0, pw, 0, ph)

    def forward(self, x):
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        x, _ = self._pad_to_multiple(x)

        c1, c2, c3, c4, c5 = self.backbone(x)

        # Directional Prior from deepest feature.
        directional_c5 = self.channel_mapping(c5)
        mapped_c5 = F.interpolate(directional_c5, scale_factor=32,
                                  mode='bilinear', align_corners=True)
        mapped_c5 = self.direc_reencode(mapped_c5)
        d_prior = self.gap(mapped_c5)

        # SDE on c5.
        c5 = self.sde_module(c5, d_prior)

        # Space-aware decoder path.
        c6 = self.gap(c5)
        r5 = self.sb1(c6, c5)                              # 512, 1/32
        d4 = self.relu(self.fb5(r5) + c4)                  # 256, 1/16
        r4 = self.sb2(self.gap(r5), d4)
        d3 = self.relu(self.fb4(r4) + c3)                  # 128, 1/8
        r3 = self.sb3(self.gap(r4), d3)
        d2 = self.relu(self.fb3(r3) + c2)                  # 64,  1/4
        r2 = self.sb4(self.gap(r3), d2)
        d1 = self.fb2(r2) + c1                             # 64,  1/2

        # Multi-scale aggregation (LWdecoder consumes 4 levels).
        feat_list = [d1, d2, d3, d4]
        final_feat = self.final_decoder(feat_list)

        # Connectivity output: 8 * num_classes channels.
        conn = self.cls_pred_conv_2(final_feat)
        conn = self.upsample2x(conn)

        # Segmentation output: aggregate 8 directions per class.
        seg = self.seg_head(conn)

        # Match input size exactly.
        if seg.shape[-2] != orig_h or seg.shape[-1] != orig_w:
            seg = F.interpolate(seg, size=(orig_h, orig_w),
                                mode='bilinear', align_corners=True)
        return seg
