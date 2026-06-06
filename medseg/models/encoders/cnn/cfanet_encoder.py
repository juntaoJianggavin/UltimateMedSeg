"""CFA-Net Encoder.

Faithful 1:1 port of the official implementation in
    https://github.com/taozh2017/CFANet
        lib/model.py
        lib/res2net_v1b_base.py

Paper:
    Cross-level Feature Aggregation Network for Polyp Segmentation,
    Zhou et al., Pattern Recognition 2023.

The official CFA-Net is built on a Res2Net-50 v1b backbone (``Res2Net_Ours``
returning 5 stages ``x0..x4`` at strides ``[4, 4, 8, 16, 32]``) followed by a
boundary-aware dual-branch decoder composed of ``GateFusion``, ``CFF``,
``BAM``, ``global_module`` and ``ChannelAttention`` modules.

For integration with the project's generic 4-stage encoder/decoder pipeline,
only the backbone path (``Res2Net_Ours``) is invoked at ``forward`` time,
returning the 4 deep stages ``[x1, x2, x3, x4]`` with channels
``[256, 512, 1024, 2048]`` at strides ``[4, 8, 16, 32]``.

The full ``CFANet`` segmentation head (boundary + dual-branch decoder) and
all of its building blocks are ported verbatim into this file so the
implementation stays bit-for-bit faithful to the original source even
though the project decoder may not use them.
"""
# Source: https://github.com/taozh2017/CFANet

import math
from typing import List

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from medseg.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# Res2Net v1b backbone (1:1 from lib/res2net_v1b_base.py)
# ---------------------------------------------------------------------------


model_urls = {
    'res2net50_v1b_26w_4s':
        'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s':
        'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 baseWidth=26, scale=4, stype='normal'):
        super().__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride,
                                   padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Res2Net(nn.Module):
    """Vanilla Res2Net v1b classifier (kept for state-dict compatibility)."""

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000,
                 in_channels: int = 3):
        self.inplanes = 64
        super().__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.avgpool(x4)
        x6 = x5.view(x5.size(0), -1)
        x7 = self.fc(x6)
        return x7


class Res2Net_Ours(nn.Module):
    """Res2Net v1b backbone returning the 5 intermediate stages used by CFA-Net.

    Output strides for ``in_size=H``: ``x0=H/4`` (after stem+maxpool, 64ch),
    ``x1=H/4`` (256ch), ``x2=H/8`` (512ch), ``x3=H/16`` (1024ch), ``x4=H/32``
    (2048ch).
    """

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000,
                 in_channels: int = 3):
        self.inplanes = 64
        super().__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


def res2net50_v1b(pretrained=False, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'],
                                                  map_location='cpu'))
    return model


def res2net50_v1b_Ours(pretrained=False, **kwargs):
    model = Res2Net_Ours(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'],
                                                  map_location='cpu'))
    return model


def Res2Net_model(ind: int = 50, pretrained: bool = False, in_channels: int = 3):
    """Build the Res2Net_Ours backbone and optionally seed its weights from
    the corresponding ImageNet-pretrained ``Res2Net`` classifier.
    """
    if ind == 50:
        model_base = res2net50_v1b(pretrained=pretrained, in_channels=in_channels)
        model = res2net50_v1b_Ours(in_channels=in_channels)
    else:
        raise NotImplementedError(f"Res2Net_model only ports ind=50 (got {ind}).")

    if pretrained:
        pretrained_dict = model_base.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


# ---------------------------------------------------------------------------
# CFA-Net building blocks (1:1 from lib/model.py)
# ---------------------------------------------------------------------------


class global_module(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        out_channels = int(channels // r)
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        xg = self.global_att(x)
        return self.sig(xg)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class GateFusion(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.gate_1 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        cat_fea = torch.cat([x1, x2], dim=1)
        att_vec_1 = self.gate_1(cat_fea)
        att_vec_2 = self.gate_2(cat_fea)
        att_vec_cat = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)
        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        return x1 * att_soft_1 + x2 * att_soft_2


class BAM(nn.Module):
    """Boundary Aware Module."""

    def __init__(self, channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.global_att = global_module(channel)
        self.conv_layer = BasicConv2d(channel * 2, channel, 3, padding=1)

    def forward(self, x, x_boun_atten):
        out1 = self.conv_layer(torch.cat((x, x_boun_atten), dim=1))
        out2 = self.global_att(out1)
        out3 = out1.mul(out2)
        return x + out3


class CFF(nn.Module):
    """Cross Feature Fusion."""

    def __init__(self, in_channel1, in_channel2, out_channel):
        super().__init__()
        act_fn = nn.ReLU(inplace=True)
        self.layer0 = BasicConv2d(in_channel1, out_channel // 2, 1)
        self.layer1 = BasicConv2d(in_channel2, out_channel // 2, 1)

        self.layer3_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer3_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer5_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer5_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channel // 2), act_fn)
        self.layer_out = nn.Sequential(
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel), act_fn)

    def forward(self, x0, x1):
        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)
        x_3_1 = self.layer3_1(torch.cat((x0_1, x1_1), dim=1))
        x_5_1 = self.layer5_1(torch.cat((x1_1, x0_1), dim=1))
        x_3_2 = self.layer3_2(torch.cat((x_3_1, x_5_1), dim=1))
        x_5_2 = self.layer5_2(torch.cat((x_5_1, x_3_1), dim=1))
        return self.layer_out(x0_1 + x1_1 + torch.mul(x_3_2, x_5_2))


# ---------------------------------------------------------------------------
# Full CFA-Net network (preserved verbatim for reference / standalone use)
# ---------------------------------------------------------------------------


class CFANet(nn.Module):
    """Full CFA-Net segmentation network from the official lib/model.py.

    This is the boundary-aware dual-branch network used in the paper. The
    project's encoder/decoder pipeline does not call into this class but it
    is kept here so the file remains a faithful 1:1 port of the source.
    """

    def __init__(self, channel: int = 64, in_channels: int = 3,
                 pretrained_backbone: bool = False):
        super().__init__()
        act_fn = nn.ReLU(inplace=True)

        self.resnet = Res2Net_model(50, pretrained=pretrained_backbone,
                                    in_channels=in_channels)
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.layer0 = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer1 = nn.Sequential(
            nn.Conv2d(256, channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channel), act_fn)

        self.low_fusion = GateFusion(channel)
        self.high_fusion1 = CFF(256, 512, channel)
        self.high_fusion2 = CFF(1024, 2048, channel)

        self.layer_edge0 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_edge1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_edge2 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_edge3 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

        self.layer_cat_ori1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig01 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_cat11 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig11 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_cat21 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig21 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_cat31 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_hig31 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

        self.layer_cat_ori2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig02 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_cat12 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig12 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_cat22 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig22 = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_cat32 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_hig32 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

        self.layer_fil = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1))

        self.atten_edge_0 = ChannelAttention(channel)
        self.atten_edge_1 = ChannelAttention(channel)
        self.atten_edge_2 = ChannelAttention(channel)
        self.atten_edge_ori = ChannelAttention(channel)

        self.cat_01 = BAM(channel); self.cat_11 = BAM(channel)
        self.cat_21 = BAM(channel); self.cat_31 = BAM(channel)
        self.cat_02 = BAM(channel); self.cat_12 = BAM(channel)
        self.cat_22 = BAM(channel); self.cat_32 = BAM(channel)

        self.downSample = nn.MaxPool2d(2, stride=2)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, xx):
        x0, x1, x2, x3, x4 = self.resnet(xx)

        x0_1 = self.layer0(x0)
        x1_1 = self.layer1(x1)
        low_x = self.low_fusion(x0_1, x1_1)

        edge_out0 = self.layer_edge0(self.up_2(low_x))
        edge_out1 = self.layer_edge1(self.up_2(edge_out0))
        edge_out2 = self.layer_edge2(self.up_2(edge_out1))
        edge_out3 = self.layer_edge3(edge_out2)

        etten_edge_ori = self.atten_edge_ori(low_x)
        etten_edge_0 = self.atten_edge_0(edge_out0)
        etten_edge_1 = self.atten_edge_1(edge_out1)
        etten_edge_2 = self.atten_edge_2(edge_out2)

        high_x01 = self.high_fusion1(self.downSample(x1), x2)
        high_x02 = self.high_fusion2(self.up_2(x3), self.up_4(x4))

        cat_out_01 = self.cat_01(high_x01, low_x.mul(etten_edge_ori))
        hig_out01 = self.layer_hig01(self.up_2(cat_out_01))
        cat_out11 = self.cat_11(hig_out01, edge_out0.mul(etten_edge_0))
        hig_out11 = self.layer_hig11(self.up_2(cat_out11))
        cat_out21 = self.cat_21(hig_out11, edge_out1.mul(etten_edge_1))
        hig_out21 = self.layer_hig21(self.up_2(cat_out21))
        cat_out31 = self.cat_31(hig_out21, edge_out2.mul(etten_edge_2))
        sal_out1 = self.layer_hig31(cat_out31)

        cat_out_02 = self.cat_02(high_x02, low_x.mul(etten_edge_ori))
        hig_out02 = self.layer_hig02(self.up_2(cat_out_02))
        cat_out12 = self.cat_12(hig_out02, edge_out0.mul(etten_edge_0))
        hig_out12 = self.layer_hig12(self.up_2(cat_out12))
        cat_out22 = self.cat_22(hig_out12, edge_out1.mul(etten_edge_1))
        hig_out22 = self.layer_hig22(self.up_2(cat_out22))
        cat_out32 = self.cat_32(hig_out22, edge_out2.mul(etten_edge_2))
        sal_out2 = self.layer_hig32(cat_out32)

        sal_out3 = self.layer_fil(cat_out31 + cat_out32)

        return edge_out3, sal_out1, sal_out2, sal_out3


# ---------------------------------------------------------------------------
# Encoder wrapper for project's 4-stage decoder pipeline
# ---------------------------------------------------------------------------


@ENCODER_REGISTRY.register("cfanet")
class CFANetEncoder(nn.Module):
    """CFA-Net encoder (Res2Net-50 v1b backbone) returning 4 deep stages.

    The official network couples a Res2Net-50 v1b backbone with a
    boundary-aware dual-branch decoder. The decoder is project-specific and
    is replaced here by the generic decoder selected in the YAML config, so
    only the backbone path is exposed:

        ``forward(x) -> [x1, x2, x3, x4]``

    with channels ``[256, 512, 1024, 2048]`` at strides ``[4, 8, 16, 32]``
    for an input of ``H x W`` divisible by 32.
    """

    def __init__(self, pretrained: bool = False, in_channels: int = 3,
                 img_size: int = 224, pretrained_path: str | None = None, **kwargs):
        super().__init__()
        self.resnet = Res2Net_model(50, pretrained=False, in_channels=in_channels)
        self.out_channels = [256, 512, 1024, 2048]

        if pretrained:
            try:
                self.resnet = Res2Net_model(50, pretrained=True, in_channels=in_channels)
            except Exception as exc:  # network unavailable / cache miss
                print(f"[cfanet] ImageNet pretrained Res2Net50_v1b unavailable ({exc!r}); "
                      "using random init.")

        if pretrained_path:
            state = torch.load(pretrained_path, map_location='cpu')
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            self.load_state_dict(state, strict=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # ``Res2Net_Ours.conv1`` is fixed to in_channels=3 internally only in
        # the official lib; here we have re-parameterised it through
        # ``Res2Net_model(in_channels=...)`` so we just forward as-is.
        _x0, x1, x2, x3, x4 = self.resnet(x)
        return [x1, x2, x3, x4]


# Alias matching the paper's class name for clarity.
CFANetEncoderBackbone = CFANetEncoder
