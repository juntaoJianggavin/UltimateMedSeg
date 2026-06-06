"""ScaleFormer – self-contained port from github.com/ZJUGiveLab/ScaleFormer.

ScaleFormer: Revisiting the Transformer-based Backbones from a Scale-wise
Perspective for Medical Image Segmentation (IJCAI 2022).

Architecture: ParallelEncoder (CNN + TransEncoder + SpatialAwareTrans fusion)
              + UNet-style decoder with skip connections.
"""
# Source: https://github.com/ZJUGiveLab/ScaleFormer

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------
class _Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super().__init__(conv, bn, relu)


class _ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1,
                 activation=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(cin, cout, 3, 1, padding=1),
            _ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class _DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=None):
        super().__init__()
        if groups is None:
            groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=groups, bias=True)

    def forward(self, x):
        return self.depthwise(x)


# ---------------------------------------------------------------------------
# CNN Encoder (U-Net style, 5 stages)
# ---------------------------------------------------------------------------
class _UEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.res1 = _DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.res2 = _DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.res3 = _DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = _DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.res5 = _DoubleConv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x); features.append(x)
        x = self.pool1(x)
        x = self.res2(x); features.append(x)
        x = self.pool2(x)
        x = self.res3(x); features.append(x)
        x = self.pool3(x)
        x = self.res4(x); features.append(x)
        x = self.pool4(x)
        x = self.res5(x); features.append(x)
        x = self.pool5(x)
        features.append(x)
        return features


# ---------------------------------------------------------------------------
# Dual-axis attention (lightweight MHSA)
# ---------------------------------------------------------------------------
class _DualAxis(nn.Module):
    def __init__(self, input_size, channels, d_h, d_v, d_w, heads, dropout):
        super().__init__()
        self.dwconv_qh = _DWConv(channels, channels)
        self.dwconv_kh = _DWConv(channels, channels)
        self.pool_qh = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_kh = nn.AdaptiveAvgPool2d((None, 1))
        self.fc_qh = nn.Linear(channels, heads * d_h)
        self.fc_kh = nn.Linear(channels, heads * d_h)
        self.dwconv_v = _DWConv(channels, channels)
        self.fc_v = nn.Linear(channels, heads * d_v)
        self.dwconv_qw = _DWConv(channels, channels)
        self.dwconv_kw = _DWConv(channels, channels)
        self.pool_qw = nn.AdaptiveAvgPool2d((1, None))
        self.pool_kw = nn.AdaptiveAvgPool2d((1, None))
        self.fc_qw = nn.Linear(channels, heads * d_w)
        self.fc_kw = nn.Linear(channels, heads * d_w)
        self.fc_o = nn.Linear(heads * d_v, channels)
        self.channels = channels
        self.d_h, self.d_v, self.d_w = d_h, d_v, d_w
        self.heads = heads
        self.scaled_factor_h = d_h ** -0.5
        self.scaled_factor_w = d_w ** -0.5
        self.Bh = nn.Parameter(torch.Tensor(1, heads, input_size, input_size),
                               requires_grad=True)
        self.Bw = nn.Parameter(torch.Tensor(1, heads, input_size, input_size),
                               requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        qh = self.pool_qh(self.dwconv_qh(x)).squeeze(-1).permute(0, 2, 1)
        qh = self.fc_qh(qh).view(b, h, self.heads, self.d_h).permute(0, 2, 1, 3).contiguous()
        kh = self.pool_kh(self.dwconv_kh(x)).squeeze(-1).permute(0, 2, 1)
        kh = self.fc_kh(kh).view(b, h, self.heads, self.d_h).permute(0, 2, 1, 3).contiguous()
        attn_h = torch.einsum('... i d, ... j d -> ... i j', qh, kh) * self.scaled_factor_h
        attn_h = attn_h + self.Bh
        attn_h = torch.softmax(attn_h, dim=-1)

        v = self.dwconv_v(x)
        vb, vc, vh, vw = v.shape
        v = v.view(vb, vc, vh * vw).permute(0, 2, 1).contiguous()
        v = self.fc_v(v).view(vb, vh, vw, self.heads, self.d_v)
        v = v.permute(0, 3, 1, 2, 4).contiguous()
        v = v.view(vb, self.heads, vh, vw * self.d_v).contiguous()

        qw = self.pool_qw(self.dwconv_qw(x)).squeeze(-2).permute(0, 2, 1)
        qw = self.fc_qw(qw).view(b, w, self.heads, self.d_w).permute(0, 2, 1, 3).contiguous()
        kw = self.pool_kw(self.dwconv_kw(x)).squeeze(-2).permute(0, 2, 1)
        kw = self.fc_kw(kw).view(b, w, self.heads, self.d_w).permute(0, 2, 1, 3).contiguous()
        attn_w = torch.einsum('... i d, ... j d -> ... i j', qw, kw) * self.scaled_factor_w
        attn_w = attn_w + self.Bw
        attn_w = torch.softmax(attn_w, dim=-1)

        result = torch.matmul(attn_h, v)
        result = result.view(b, self.heads, h, w, self.d_v).permute(0, 1, 2, 4, 3).contiguous()
        result = result.view(b, self.heads, h * self.d_v, w).contiguous()
        result = torch.matmul(result, attn_w)
        result = result.view(b, self.heads, h, self.d_v, w).permute(0, 2, 4, 1, 3).contiguous()
        result = result.view(b, h * w, self.heads * self.d_v).contiguous()
        result = self.fc_o(result).view(b, self.channels, h, w)
        return result


# ---------------------------------------------------------------------------
# FFN with multi-LayerNorm + DWConv
# ---------------------------------------------------------------------------
class _FFNMultiLN(nn.Module):
    def __init__(self, in_channels, img_size, R=4):
        super().__init__()
        exp_ch = in_channels * R
        self.h = self.w = img_size
        self.fc1 = nn.Linear(in_channels, exp_ch)
        self.dwconv = _DWConv(exp_ch, exp_ch)
        self.ln1 = nn.LayerNorm(exp_ch, eps=1e-6)
        self.ln2 = nn.LayerNorm(exp_ch, eps=1e-6)
        self.ln3 = nn.LayerNorm(exp_ch, eps=1e-6)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(exp_ch, in_channels)

    def forward(self, x):
        x = self.fc1(x)
        b, n, c = x.shape
        h = x
        x = x.view(b, self.h, self.w, c).permute(0, 3, 1, 2)
        x = self.dwconv(x).view(b, c, self.h * self.w).permute(0, 2, 1)
        x = self.ln1(x + h)
        x = self.ln2(x + h)
        x = self.ln3(x + h)
        x = self.act(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Intra-scale transformer block (Dual-axis + IRFFN)
# ---------------------------------------------------------------------------
class _IntraTransBlock(nn.Module):
    def __init__(self, img_size, d_h, d_v, d_w, num_heads, R=4, in_channels=46):
        super().__init__()
        self.SlayerNorm = nn.LayerNorm(in_channels, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(in_channels, eps=1e-6)
        self.lmhsa = _DualAxis(img_size, in_channels, d_h, d_v, d_w, num_heads, 0.0)
        self.irffn = _FFNMultiLN(in_channels, img_size, R)

    def forward(self, x):
        x_pre = x
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x = self.SlayerNorm(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = self.lmhsa(x)
        x = x_pre + x
        x_pre = x
        x = x.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x = self.ElayerNorm(x)
        x = self.irffn(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x_pre + x


# ---------------------------------------------------------------------------
# TransEncoder (4-stage transformer encoder)
# ---------------------------------------------------------------------------
class _TransEncoder(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
        base = img_size // 4  # 56 for 224
        self.block_layer = [2, 2, 2, 1]
        self.size = [base, base // 2, base // 4, base // 8]
        self.channels = [256, 512, 1024, 1024]
        stages = []
        for idx in range(4):
            s = []
            for _ in range(self.block_layer[idx]):
                s.append(_IntraTransBlock(
                    img_size=self.size[idx], in_channels=self.channels[idx],
                    d_h=self.channels[idx] // 8, d_v=self.channels[idx] // 8,
                    d_w=self.channels[idx] // 8, num_heads=8))
            stages.append(nn.Sequential(*s))
        self.stages = nn.ModuleList(stages)
        self.downlayers = nn.ModuleList([
            _ConvBNReLU(256, 512, 2, 2, padding=0),
            _ConvBNReLU(512, 1024, 2, 2, padding=0),
            _ConvBNReLU(1024, 2048, 2, 2, padding=0),
        ])
        self.squeelayers = nn.ModuleList([
            nn.Conv2d(512 * 2, 512, 1, 1),
            nn.Conv2d(1024 * 2, 1024, 1, 1),
        ])
        self.squeeze_final = nn.Conv2d(1024 * 3, 1024, 1, 1)

    def forward(self, features):
        _, _, f0, f1, f2, f3 = features
        t0 = self.stages[0](f0)
        t0d = self.downlayers[0](t0)
        f1_in = self.squeelayers[0](torch.cat((f1, t0d), dim=1))
        t1 = self.stages[1](f1_in)
        t1d = self.downlayers[1](t1)
        f2_in = self.squeelayers[1](torch.cat((f2, t1d), dim=1))
        t2 = self.stages[2](f2_in)
        t2d = self.downlayers[2](t2)
        f3_in = self.squeeze_final(torch.cat((f3, t2d), dim=1))
        t3 = self.stages[3](f3_in)
        return [t0, t1, t2, t3]


# ---------------------------------------------------------------------------
# Multi-scale attention & inter-scale transformer
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        return self.dropout(self.fc2(x))


class _MultiScaleAtten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head) ** 0.5

    def forward(self, x):
        B, nb, _, _, C = x.shape
        qkv = self.qkv_linear(x).reshape(B, nb, nb, -1, 3, self.num_head,
                                          C // self.num_head)
        qkv = qkv.permute(4, 0, 1, 2, 5, 3, 6).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        atten = self.softmax(q @ k.transpose(-1, -2).contiguous())
        out = (atten @ v).transpose(-2, -3).contiguous().reshape(B, nb, nb, -1, C)
        return self.proj(out)


class _InterTransBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = _MultiScaleAtten(dim)
        self.FFN = _MLP(dim)

    def forward(self, x):
        h = x
        x = h + self.Attention(self.SlayerNorm_1(x))
        return x + self.FFN(self.SlayerNorm_2(x))


class _SpatialAwareTrans(nn.Module):
    def __init__(self, dim=256, num=1):
        super().__init__()
        self.ini_win_size = 2
        self.channels = [256, 512, 1024, 1024]
        self.dim = dim
        self.depth = 4
        self.fc_module = nn.ModuleList(
            [nn.Linear(ch, dim) for ch in self.channels])
        self.fc_rever_module = nn.ModuleList(
            [nn.Linear(dim, ch) for ch in self.channels])
        self.group_attention = nn.Sequential(
            *[_InterTransBlock(dim) for _ in range(num)])
        self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]

    def forward(self, x):
        x = [self.fc_module[i](item.permute(0, 2, 3, 1))
             for i, item in enumerate(x)]
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            ws = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, H // ws, ws, W // ws, ws, C)
            item = item.permute(0, 1, 3, 2, 4, 5).contiguous()
            item = item.reshape(B, H // ws, W // ws, ws * ws, C).contiguous()
            x[j] = item
        x = torch.cat(tuple(x), dim=-2)
        for layer in self.group_attention:
            x = layer(x)
        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)
        for j, item in enumerate(x):
            B, nb, _, N, C = item.shape
            ws = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, nb, nb, ws, ws, C)
            item = item.permute(0, 1, 3, 2, 4, 5).contiguous()
            item = item.reshape(B, nb * ws, nb * ws, C)
            item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
            x[j] = item
        return x


# ---------------------------------------------------------------------------
# Parallel encoder (CNN + TransEncoder + fusion)
# ---------------------------------------------------------------------------
class _ParallEncoder(nn.Module):
    def __init__(self, in_channels=3, img_size=224):
        super().__init__()
        self.Encoder1 = _UEncoder(in_channels)
        self.Encoder2 = _TransEncoder(img_size)
        self.inter_trans = _SpatialAwareTrans(dim=256)
        self.num_module = 4
        channel_list = [128, 256, 512, 1024]
        fusion_list = [256, 512, 1024, 1024]
        self.squeelayers = nn.ModuleList([
            nn.Conv2d(fl * 2, fl, 1, 1) for fl in fusion_list])

    def forward(self, x):
        features = self.Encoder1(x)
        feature_trans = self.Encoder2(features)
        feature_trans = self.inter_trans(feature_trans)
        skips = list(features[:2])
        for i in range(self.num_module):
            skip = self.squeelayers[i](
                torch.cat((feature_trans[i], features[i + 2]), dim=1))
            skips.append(skip)
        return skips


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _Conv2dReLU(in_channels, out_channels, 3, padding=1)
        self.conv2 = _Conv2dReLU(out_channels, out_channels, 3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class _SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                         padding=kernel_size // 2)
        super().__init__(conv)


# ---------------------------------------------------------------------------
# ScaleFormer
# ---------------------------------------------------------------------------
class ScaleFormer(nn.Module):
    """ScaleFormer: ParallelEncoder + 4-stage UNet decoder.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        enc_ch = [1024, 512, 256, 128, 64]
        self.p_encoder = _ParallEncoder(in_channels, img_size)
        self.decoder1 = _DecoderBlock(enc_ch[0] + enc_ch[0], enc_ch[1])
        self.decoder2 = _DecoderBlock(enc_ch[1] + enc_ch[1], enc_ch[2])
        self.decoder3 = _DecoderBlock(enc_ch[2] + enc_ch[2], enc_ch[3])
        self.decoder4 = _DecoderBlock(enc_ch[3] + enc_ch[3], enc_ch[4])
        self.decoder_final = _DecoderBlock(64, 64)
        self.segmentation_head = _SegmentationHead(64, num_classes)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        skips = self.p_encoder(x)
        x = self.decoder1(skips[-1], skips[-2])
        x = self.decoder2(x, skips[-3])
        x = self.decoder3(x, skips[-4])
        x = self.decoder4(x, skips[-5])
        x = self.decoder_final(x, None)
        return self.segmentation_head(x)
