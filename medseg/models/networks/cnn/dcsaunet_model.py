"""DCSAU-Net – self-contained port from github.com/xq141839/DCSAU-Net.

Combines splat.py, resnet.py, encoder.py, and DCSAU_Net.py into one file.

Standard interface:
    model = DCSAUNet(in_channels=3, num_classes=2, img_size=224)
    out = model(x)  # -> (B, num_classes, H, W)
"""
# Source: https://github.com/xq141839/DCSAU-Net

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair


# ── Split-Attention (from splat.py) ──────────────────────────────────────────
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


class SplAtConv2d(Module):
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4, rectify=False,
                 rectify_avg=False, norm_layer=None, dropblock_prob=0.0,
                 **kwargs):
        super().__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = Conv2d(in_channels, channels * radix, kernel_size,
                           stride, padding, dilation,
                           groups=groups * radix, bias=bias)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
            self.bn2 = norm_layer(channels)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1,
                          groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)
        self.conv2 = Conv2d(channels, channels, kernel_size, stride,
                            padding, dilation, groups=groups * radix,
                            bias=bias)

    def forward(self, x):
        x = self.relu(self.bn0(self.conv(x)))
        batch, rchannel = x.shape[:2]
        x1, x2 = torch.split(x, rchannel // self.radix, dim=1)
        x2 = self.relu(self.bn2(self.conv2(x2 + x1)))
        splited = (x1, x2)
        gap = F.adaptive_avg_pool2d(sum(splited), 1)
        gap = self.relu(self.bn1(self.fc1(gap)))
        atten = self.rsoftmax(self.fc2(gap)).view(batch, -1, 1, 1)
        attens = torch.split(atten, rchannel // self.radix, dim=1)
        out = sum([att * split for (att, split) in zip(attens, splited)])
        return out.contiguous()


# ── ResNet with Split-Attention Bottleneck (from resnet.py) ──────────────────
class _DCSABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 norm_layer=None, custom=0):
        super().__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        if custom != 0:
            inplanes = custom
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1
        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation, groups=cardinality,
                bias=False, radix=radix, norm_layer=norm_layer)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3,
                                   stride=stride, padding=dilation,
                                   dilation=dilation, groups=cardinality,
                                   bias=False)
            self.bn2 = norm_layer(group_width)
        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        out = self.conv2(out)
        if self.radix == 0:
            out = self.relu(self.bn2(out))
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _DCSAResNet(nn.Module):
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, avd_first=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2
        self.avg_down = avg_down
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        ConvFea = [32, 64, 128, 256, 512]
        self.layer1 = self._make_layer(block, ConvFea[0], layers[0],
                                       norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, ConvFea[1], layers[1],
                                       stride=1, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, ConvFea[2], layers[2],
                                       stride=1, dilation=1,
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, ConvFea[2], layers[3],
                                       stride=1, dilation=1,
                                       norm_layer=norm_layer)
        self.layer5 = self._make_layer(block, ConvFea[1], layers[0],
                                       stride=1, dilation=1,
                                       norm_layer=norm_layer, inchannel=1024)
        self.layer6 = self._make_layer(block, ConvFea[0], layers[1],
                                       stride=1, dilation=1,
                                       norm_layer=norm_layer, inchannel=512)
        self.layer7 = self._make_layer(block, ConvFea[0] // 2, layers[2],
                                       stride=1, norm_layer=norm_layer,
                                       inchannel=256)
        self.layer8 = self._make_layer(block, ConvFea[0] // 2, layers[3],
                                       norm_layer=norm_layer, is_first=False,
                                       inchannel=128)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    norm_layer=None, is_first=True, inchannel=0):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion
                or inchannel != 0):
            if inchannel != 0:
                self.inplanes = inchannel
            down_layers = []
            if self.avg_down:
                down_layers.append(nn.AvgPool2d(kernel_size=stride,
                                                stride=stride,
                                                ceil_mode=True,
                                                count_include_pad=False)
                                   if dilation == 1
                                   else nn.AvgPool2d(kernel_size=1, stride=1,
                                                     ceil_mode=True,
                                                     count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes,
                                             planes * block.expansion,
                                             kernel_size=1, stride=1,
                                             bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes,
                                             planes * block.expansion,
                                             kernel_size=1, stride=stride,
                                             bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)
        layers = []
        d_val = 1 if dilation in (1, 2) else 2
        layers.append(block(self.inplanes, planes, stride,
                            downsample=downsample, radix=self.radix,
                            cardinality=self.cardinality,
                            bottleneck_width=self.bottleneck_width,
                            avd=self.avd, avd_first=self.avd_first,
                            dilation=d_val, is_first=is_first,
                            norm_layer=norm_layer, custom=inchannel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)


def _build_csa():
    return _DCSAResNet(
        _DCSABottleneck, [2, 2, 2, 2],
        radix=2, groups=1, bottleneck_width=64,
        deep_stem=True, stem_width=32, avg_down=True,
        avd=True, avd_first=False)


# ── DCSAU-Net Model ─────────────────────────────────────────────────────────
class _Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x2, x1], dim=1)


class _PFC(nn.Module):
    def __init__(self, channels, kernel_size=7, in_channels=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, groups=channels,
                      padding=kernel_size // 2),
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
        return self.pointwise(x)


class _DCSAUModel(nn.Module):
    def __init__(self, img_channels=3, n_classes=1):
        super().__init__()
        csa = _build_csa()
        self.pfc = _PFC(64, in_channels=img_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1,
                                  stride=1, padding=0)
        self.up_conv1 = _Up()
        self.up_conv2 = _Up()
        self.up_conv3 = _Up()
        self.up_conv4 = _Up()
        self.down1 = csa.layer1
        self.down2 = csa.layer2
        self.down3 = csa.layer3
        self.down4 = csa.layer4
        self.up1 = csa.layer5
        self.up2 = csa.layer6
        self.up3 = csa.layer7
        self.up4 = csa.layer8

    def forward(self, x):
        x1 = self.pfc(x)
        x2 = self.maxpool(x1)
        x3 = self.down1(x2)
        x4 = self.maxpool(x3)
        x5 = self.down2(x4)
        x6 = self.maxpool(x5)
        x7 = self.down3(x6)
        x8 = self.maxpool(x7)
        x9 = self.down4(x8)
        x10 = self.up_conv1(x9, x7)
        x11 = self.up1(x10)
        x12 = self.up_conv2(x11, x5)
        x13 = self.up2(x12)
        x14 = self.up_conv3(x13, x3)
        x15 = self.up3(x14)
        x16 = self.up_conv4(x15, x1)
        x17 = self.up4(x16)
        return self.out_conv(x17)


class DCSAUNet(nn.Module):
    """DCSAU-Net wrapper with standard interface.

    Args:
        in_channels (int): Number of input channels (default: 3).
        num_classes (int): Number of output classes (default: 2).
        img_size (int): Input image size (default: 224, unused by model).
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        self.model = _DCSAUModel(img_channels=in_channels, n_classes=num_classes)

    def forward(self, x):
        return self.model(x)
