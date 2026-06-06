"""CFANet decoder – extracted from networks/cnn/cfanet_model.py.

Cross-level Feature Aggregation decoder (Pattern Recognition 2023).
Dual-branch (edge + main) decoding with BAM, CFF, GateFusion, ChannelAttention.

Decoder contract
----------------
forward(bottleneck_feat, skip_features) -> Tensor
    bottleneck_feat : x4 (deepest Res2Net feature, [B,2048,H/32,W/32])
    skip_features   : [x0, x1, x2, x3] from Res2Net encoder
                      x0: [B,64,H/4,W/4],  x1: [B,256,H/8,W/8]
                      x2: [B,512,H/16,W/16], x3: [B,1024,H/32,W/32]
"""
# Source: https://github.com/taozh2017/CFANet

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import DECODER_REGISTRY


# ---------------------------------------------------------------------------
# Building blocks (inlined for self-containment)
# ---------------------------------------------------------------------------
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
# Public decoder wrapper
# ---------------------------------------------------------------------------
@DECODER_REGISTRY.register("cfanet")
class CFANetDecoder(nn.Module):
    """CFANet dual-branch decoder (edge + main with BAM & CFF).

    has_internal_skip = True  (consumes all encoder features internally)
    out_channels = num_classes (produces final logits)
    """

    has_internal_skip = True
    required_skip_stages = 5
    requires_encoder = "res2net"  # Spatial hierarchy assumes Res2Net stride pattern

    def __init__(self, encoder_channels=None, bottleneck_channels=None,
                 skip_connection=None, num_classes=2, channel=64, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.channel = channel
        act_fn = nn.ReLU(inplace=True)

        # Derive channel sizes from encoder_channels (with adapter-provided defaults)
        # Default to Res2Net-style: [64, 256, 512, 1024, 2048]
        if encoder_channels is not None and len(encoder_channels) >= 5:
            c = list(encoder_channels)
        else:
            c = [64, 256, 512, 1024, 2048]

        # Feature adaptation layers
        self.layer0 = nn.Sequential(
            nn.Conv2d(c[0], channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.layer1 = nn.Sequential(
            nn.Conv2d(c[1], channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(channel), act_fn)
        self.downSample = nn.MaxPool2d(2, stride=2)
        self.low_fusion = _GateFusion(channel)
        self.high_fusion1 = _CFF(c[2], c[3], channel)
        self.high_fusion2 = _CFF(c[3], c[4], channel)

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
        self._out_channels = num_classes

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat, skip_features):
        """
        Args:
            bottleneck_feat: x4 [B,2048,H/32,W/32] (unused directly,
                             included in skip_features as last element).
            skip_features: [x0, x1, x2, x3, x4] from Res2Net encoder.
        """
        x0, x1, x2, x3, x4 = skip_features

        # Feature adaptation
        x0_a = self.layer0(x0)
        x1_a = self.layer1(x1)
        low_fused = self.low_fusion(x0_a, x1_a)
        high1 = self.high_fusion1(
            x2, F.interpolate(x3, size=x2.shape[2:],
                              mode='bilinear', align_corners=True))
        high2 = self.high_fusion2(
            x3, F.interpolate(x4, size=x3.shape[2:],
                              mode='bilinear', align_corners=True))

        # Edge branch
        edge_atten = self.atten_edge_ori(low_fused)
        edge = self.cat_01(low_fused, low_fused * edge_atten)
        edge = self.layer_cat_ori1(torch.cat((edge, low_fused), dim=1))
        edge = self.layer_hig01(edge)
        edge_atten1 = self.atten_edge_0(high1)
        high1_up = F.interpolate(high1, size=edge.shape[2:],
                                 mode='bilinear', align_corners=True)
        high1_atten_up = F.interpolate(edge_atten1, size=edge.shape[2:],
                                       mode='bilinear', align_corners=True)
        edge = self.cat_11(edge, high1_up * high1_atten_up)
        edge = self.layer_cat11(torch.cat((edge, high1_up), dim=1))
        edge = self.layer_hig11(self.up_2(edge))
        edge_atten2 = self.atten_edge_1(high2)
        high2_up = F.interpolate(high2, size=edge.shape[2:],
                                 mode='bilinear', align_corners=True)
        high2_atten_up = F.interpolate(edge_atten2, size=edge.shape[2:],
                                       mode='bilinear', align_corners=True)
        edge = self.cat_21(edge, high2_up * high2_atten_up)
        edge = self.layer_cat21(torch.cat((edge, high2_up), dim=1))
        edge = self.layer_hig21(self.up_2(edge))
        x0_a_up_edge = F.interpolate(x0_a, size=edge.shape[2:],
                                     mode='bilinear', align_corners=True)
        edge = self.layer_cat31(torch.cat((edge, x0_a_up_edge), dim=1))
        edge_out = self.layer_hig31(self.up_2(edge))

        # Main branch
        main_atten = self.atten_edge_2(high2)
        main = self.cat_02(high2, high2 * main_atten)
        main = self.layer_cat_ori2(torch.cat((main, high2), dim=1))
        main = self.layer_hig02(main)
        main_up = F.interpolate(main, size=high1.shape[2:],
                                mode='bilinear', align_corners=True)
        main = self.cat_12(main_up, high1)
        main = self.layer_cat12(torch.cat((main, high1), dim=1))
        main = self.layer_hig12(self.up_2(main))
        low_fused_up = F.interpolate(low_fused, size=main.shape[2:],
                                     mode='bilinear', align_corners=True)
        main = self.cat_22(main, low_fused_up)
        main = self.layer_cat22(torch.cat((main, low_fused_up), dim=1))
        main = self.layer_hig22(self.up_2(main))
        x0_a_up_main = F.interpolate(x0_a, size=main.shape[2:],
                                     mode='bilinear', align_corners=True)
        main = self.layer_cat32(torch.cat((main, x0_a_up_main), dim=1))
        main_out = self.layer_hig32(self.up_2(main))

        # Final fusion
        final = self.layer_fil(self.up_2(self.cat_31(main, edge)))
        return final + edge_out + main_out
