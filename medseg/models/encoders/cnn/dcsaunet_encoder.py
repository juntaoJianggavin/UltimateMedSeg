"""DCSAU-Net Encoder: faithful port from https://github.com/xq141839/DCSAU-Net

Reference: Xu et al., "DCSAU-Net: A Deeper and More Compact Split-Attention U-Net
           for Medical Image Segmentation"
Files: DCSAU_Net.py, encoder.py, resnet.py, splat.py
All class/attribute names match the original for pretrained weight loading.
"""
# Source: https://github.com/xq141839/DCSAU-Net

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.nn.modules.utils import _pair

from medseg.registry import ENCODER_REGISTRY


# ============= rSoftMax (from splat.py) =============
class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


# ============= SplAtConv2d (from splat.py) =============
class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d"""
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob

        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                              groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
            self.bn2 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)

        self.rsoftmax = rSoftMax(radix, groups)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation,
                               groups=groups * radix, bias=bias, **kwargs)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)
        batch, rchannel = x.shape[:2]

        x1, x2 = torch.split(x, rchannel // self.radix, dim=1)
        x2 = x2 + x1
        x2 = self.conv2(x2)
        if self.use_bn:
            x2 = self.bn2(x2)
        x2 = self.relu(x2)

        splited = (x1, x2)
        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        attens = torch.split(atten, rchannel // self.radix, dim=1)
        out = sum([att * split for (att, split) in zip(attens, splited)])
        return out.contiguous()


# ============= Bottleneck (from resnet.py) =============
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=2, cardinality=1,
                 bottleneck_width=64, avd=False, avd_first=False, dilation=1,
                 is_first=False, rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, norm_layer=norm_layer)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# ============= ResNet with CSA (from resnet.py) =============
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet_CSA(nn.Module):
    """ResNet backbone with Split-Attention (8 layers: 4 encoder + 4 decoder)."""

    def __init__(self, block, layers, radix=2, groups=1, bottleneck_width=64,
                 deep_stem=True, stem_width=32, avg_down=True,
                 avd=True, avd_first=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ResNet_CSA, self).__init__()
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avd = avd
        self.avd_first = avd_first
        self.radix = radix

        # Deep stem
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Encoder layers
        self.layer1 = self._make_layer(block, 16, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2, norm_layer=norm_layer)

        # Decoder layers
        self.layer5 = self._make_layer(block, 64, layers[3], stride=1, norm_layer=norm_layer)
        self.layer6 = self._make_layer(block, 32, layers[2], stride=1, norm_layer=norm_layer)
        self.layer7 = self._make_layer(block, 16, layers[1], stride=1, norm_layer=norm_layer)
        self.layer8 = self._make_layer(block, 16, layers[0], stride=1, norm_layer=norm_layer)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avd and stride > 1:
                down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
                down_layers.append(conv1x1(self.inplanes, planes * block.expansion))
            else:
                down_layers.append(conv1x1(self.inplanes, planes * block.expansion, stride))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            radix=self.radix, cardinality=self.cardinality,
                            bottleneck_width=self.bottleneck_width,
                            avd=self.avd, avd_first=self.avd_first,
                            norm_layer=norm_layer, is_first=is_first))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)


# ============= PFC (from DCSAU_Net.py) =============
class PFC(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))

    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x


# ============= Up (from DCSAU_Net.py) =============
class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


# ============= DCSAU-Net Encoder =============
@ENCODER_REGISTRY.register("dcsaunet")
class DCSAUNetEncoder(nn.Module):
    """DCSAU-Net Encoder: PFC + CSA ResNet backbone.
    Faithful to https://github.com/xq141839/DCSAU-Net
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained_path: str = None,
        **kwargs,
    ):
        super().__init__()
        csa_block = ResNet_CSA(Bottleneck, [2, 2, 2, 2],
                               radix=2, groups=1, bottleneck_width=64,
                               deep_stem=True, stem_width=32, avg_down=True,
                               avd=True, avd_first=False)

        self.pfc = PFC(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.down1 = csa_block.layer1
        self.down2 = csa_block.layer2
        self.down3 = csa_block.layer3
        self.down4 = csa_block.layer4

        # out channels: PFC=64, after CSA layers: 64, 128, 256, 512
        self._out_channels = [64, 64, 128, 256, 512]

        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)

    @property
    def out_channels(self):
        return self._out_channels

    def _load_pretrained(self, path):
        state = torch.load(path, map_location='cpu')
        msg = self.load_state_dict(state, strict=False)
        print(f"DCSAU-Net encoder loaded: {msg}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.pfc(x)
        x2 = self.maxpool(x1)
        x3 = self.down1(x2)
        x4 = self.maxpool(x3)
        x5 = self.down2(x4)
        x6 = self.maxpool(x5)
        x7 = self.down3(x6)
        x8 = self.maxpool(x7)
        x9 = self.down4(x8)

        return [x1, x3, x5, x7, x9]
