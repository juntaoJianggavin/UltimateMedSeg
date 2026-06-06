"""CFANet – self-contained port from github.com/taozh2017/CFANet.

Cross-level Feature Aggregation Network for Polyp Segmentation
(Pattern Recognition 2023).

Architecture: Res2Net-50 encoder + cross-level feature aggregation decoder
              with BAM modules and gate fusion.
"""
# Source: https://github.com/taozh2017/CFANet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Res2Net building blocks
# ---------------------------------------------------------------------------
class _Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 baseWidth=26, scale=4, stype='normal'):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = 1 if scale == 1 else scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs, bns = [], []
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
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if (i == 0 or self.stype == 'stage') else sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _Res2Net(nn.Module):
    """Res2Net variant that returns 5 feature maps (stem + 4 stages)."""

    def __init__(self, block, layers, baseWidth=26, scale=4, in_channels=3):
        self.inplanes = 64
        super().__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
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
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.baseWidth,
                        scale=self.scale)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


def _Res2Net_model(depth, in_channels=3):
    cfgs = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}
    return _Res2Net(_Bottle2neck, cfgs[depth], in_channels=in_channels)


# ---------------------------------------------------------------------------
# Attention / Fusion modules
# ---------------------------------------------------------------------------
class _GlobalModule(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        out_ch = channels // r
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_ch, 1), nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, channels, 1), nn.BatchNorm2d(channels))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.global_att(x))


class _BasicConv2d(nn.Module):
    def __init__(self, in_p, out_p, kernel_size, stride=1, padding=0,
                 dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_p, out_p, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class _ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu1(self.fc1(self.max_pool(x)))))


class _GateFusion(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.gate_1 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        cat_fea = torch.cat([x1, x2], dim=1)
        a1 = self.gate_1(cat_fea)
        a2 = self.gate_2(cat_fea)
        att = self.softmax(torch.cat([a1, a2], dim=1))
        return x1 * att[:, 0:1] + x2 * att[:, 1:2]


class _BAM(nn.Module):
    """Boundary Attention Module."""

    def __init__(self, channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.global_att = _GlobalModule(channel)
        self.conv_layer = _BasicConv2d(channel * 2, channel, 3, padding=1)

    def forward(self, x, x_boun_atten):
        out1 = self.conv_layer(torch.cat((x, x_boun_atten), dim=1))
        out2 = self.global_att(out1)
        return x + out1.mul(out2)


class _CFF(nn.Module):
    """Cross-level Feature Fusion."""

    def __init__(self, in_ch1, in_ch2, out_channel):
        super().__init__()
        act_fn = nn.ReLU(inplace=True)
        oc = out_channel // 2
        self.layer0 = _BasicConv2d(in_ch1, oc, 1)
        self.layer1 = _BasicConv2d(in_ch2, oc, 1)
        self.layer3_1 = nn.Sequential(nn.Conv2d(out_channel, oc, 3, 1, 1),
                                      nn.BatchNorm2d(oc), act_fn)
        self.layer3_2 = nn.Sequential(nn.Conv2d(out_channel, oc, 3, 1, 1),
                                      nn.BatchNorm2d(oc), act_fn)
        self.layer5_1 = nn.Sequential(nn.Conv2d(out_channel, oc, 5, 1, 2),
                                      nn.BatchNorm2d(oc), act_fn)
        self.layer5_2 = nn.Sequential(nn.Conv2d(out_channel, oc, 5, 1, 2),
                                      nn.BatchNorm2d(oc), act_fn)
        self.layer_out = nn.Sequential(nn.Conv2d(oc, out_channel, 3, 1, 1),
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
# CFANet
# ---------------------------------------------------------------------------
class CFANet(nn.Module):
    """Cross-level Feature Aggregation Network.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        channel = 64
        act_fn = nn.ReLU(inplace=True)
        self.resnet = _Res2Net_model(50, in_channels)
        self.downSample = nn.MaxPool2d(2, stride=2)
        # Feature adaptation
        self.layer0 = nn.Sequential(
            nn.Conv2d(64, channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer1 = nn.Sequential(
            nn.Conv2d(256, channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.low_fusion = _GateFusion(channel)
        self.high_fusion1 = _CFF(512, 1024, channel)
        self.high_fusion2 = _CFF(1024, 2048, channel)
        # Edge branch
        self.layer_edge0 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_edge1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_edge2 = nn.Sequential(
            nn.Conv2d(channel, 64, 3, 1, 1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_edge3 = nn.Sequential(nn.Conv2d(64, 1, 1))
        # Main decode branch 1
        self.layer_cat_ori1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig01 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_cat11 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig11 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_cat21 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig21 = nn.Sequential(
            nn.Conv2d(channel, 64, 3, 1, 1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_cat31 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_hig31 = nn.Sequential(nn.Conv2d(64, num_classes, 1))
        # Main decode branch 2
        self.layer_cat_ori2 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig02 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_cat12 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig12 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_cat22 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer_hig22 = nn.Sequential(
            nn.Conv2d(channel, 64, 3, 1, 1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_cat32 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64), act_fn)
        self.layer_hig32 = nn.Sequential(nn.Conv2d(64, num_classes, 1))
        self.layer_fil = nn.Sequential(nn.Conv2d(64, num_classes, 1))
        # Attention modules
        self.atten_edge_0 = _ChannelAttention(channel)
        self.atten_edge_1 = _ChannelAttention(channel)
        self.atten_edge_2 = _ChannelAttention(channel)
        self.atten_edge_ori = _ChannelAttention(channel)
        # BAM modules
        self.cat_01 = _BAM(channel)
        self.cat_11 = _BAM(channel)
        self.cat_21 = _BAM(channel)
        self.cat_31 = _BAM(channel)
        self.cat_02 = _BAM(channel)
        self.cat_12 = _BAM(channel)
        self.cat_22 = _BAM(channel)
        self.cat_32 = _BAM(channel)
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear',
                                align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear',
                                align_corners=True)

    def forward(self, xx):
        x0, x1, x2, x3, x4 = self.resnet(xx)
        # Feature adaptation
        x0_a = self.layer0(x0)
        x1_a = self.layer1(x1)
        low_fused = self.low_fusion(x0_a, x1_a)
        high1 = self.high_fusion1(x2, F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=True))
        high2 = self.high_fusion2(x3, F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=True))
        # Edge branch
        edge_atten = self.atten_edge_ori(low_fused)
        edge = self.cat_01(low_fused, low_fused * edge_atten)
        edge = self.layer_cat_ori1(torch.cat((edge, low_fused), dim=1))
        edge = self.layer_hig01(edge)
        edge_atten1 = self.atten_edge_0(high1)
        # Upsample high1 to match edge's spatial size
        high1_up = F.interpolate(high1, size=edge.shape[2:], mode='bilinear', align_corners=True)
        high1_atten_up = F.interpolate(edge_atten1, size=edge.shape[2:], mode='bilinear', align_corners=True)
        edge = self.cat_11(edge, high1_up * high1_atten_up)
        edge = self.layer_cat11(torch.cat((edge, high1_up), dim=1))
        edge = self.layer_hig11(self.up_2(edge))
        edge_atten2 = self.atten_edge_1(high2)
        # Upsample high2 to match edge's spatial size
        high2_up = F.interpolate(high2, size=edge.shape[2:], mode='bilinear', align_corners=True)
        high2_atten_up = F.interpolate(edge_atten2, size=edge.shape[2:], mode='bilinear', align_corners=True)
        edge = self.cat_21(edge, high2_up * high2_atten_up)
        edge = self.layer_cat21(torch.cat((edge, high2_up), dim=1))
        edge = self.layer_hig21(self.up_2(edge))
        # Upsample x0_a to match edge's spatial size
        x0_a_up_edge = F.interpolate(x0_a, size=edge.shape[2:], mode='bilinear', align_corners=True)
        edge = self.layer_cat31(torch.cat((edge, x0_a_up_edge), dim=1))
        edge_out = self.layer_hig31(self.up_2(edge))
        # Main branch
        main_atten = self.atten_edge_2(high2)
        main = self.cat_02(high2, high2 * main_atten)
        main = self.layer_cat_ori2(torch.cat((main, high2), dim=1))
        main = self.layer_hig02(main)
        # Upsample main to match high1's spatial size
        main_up = F.interpolate(main, size=high1.shape[2:], mode='bilinear', align_corners=True)
        main = self.cat_12(main_up, high1)
        main = self.layer_cat12(torch.cat((main, high1), dim=1))
        main = self.layer_hig12(self.up_2(main))
        # Upsample low_fused to match main's spatial size
        low_fused_up = F.interpolate(low_fused, size=main.shape[2:], mode='bilinear', align_corners=True)
        main = self.cat_22(main, low_fused_up)
        main = self.layer_cat22(torch.cat((main, low_fused_up), dim=1))
        main = self.layer_hig22(self.up_2(main))
        # Upsample x0_a to match main's spatial size
        x0_a_up_main = F.interpolate(x0_a, size=main.shape[2:], mode='bilinear', align_corners=True)
        main = self.layer_cat32(torch.cat((main, x0_a_up_main), dim=1))
        main_out = self.layer_hig32(self.up_2(main))
        # Final fusion
        final = self.layer_fil(
            self.up_2(self.cat_31(main, edge)))
        # Ensure output matches input spatial size
        if final.shape[-2:] != xx.shape[-2:]:
            final = F.interpolate(final, size=xx.shape[-2:],
                                  mode='bilinear', align_corners=True)
        if edge_out.shape[-2:] != xx.shape[-2:]:
            edge_out = F.interpolate(edge_out, size=xx.shape[-2:],
                                     mode='bilinear', align_corners=True)
        if main_out.shape[-2:] != xx.shape[-2:]:
            main_out = F.interpolate(main_out, size=xx.shape[-2:],
                                     mode='bilinear', align_corners=True)
        return final + edge_out + main_out
