"""UCTransNet – self-contained port from github.com/McGregorWwww/UCTransNet.

UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-wise
Perspective with Transformer (AAAI 2022).

Architecture: UNet encoder + ChannelTransformer skip connections + UNet decoder.
"""
# Source: https://github.com/McGregorWwww/UCTransNet

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair


# ---------------------------------------------------------------------------
# Config helper – replaces the external Config object
# ---------------------------------------------------------------------------
class _UCTConfig:
    """Minimal config for UCTransNet's CTrans module."""

    def __init__(self, img_size=224, base_channel=64, patch_sizes=None):
        self.base_channel = base_channel
        self.patch_sizes = patch_sizes or [28, 14, 7, 4]
        self.KV_size = base_channel + base_channel * 2 + base_channel * 4 + base_channel * 8
        self.KV_size_S = base_channel + base_channel * 2 + base_channel * 4 + base_channel * 8
        self.expand_ratio = 4
        self.transformer = {
            "num_heads": 4,
            "num_layers": 4,
            "embeddings_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "dropout_rate": 0.1,
        }


# ---------------------------------------------------------------------------
# CTrans: Channel Transformer module
# ---------------------------------------------------------------------------
class _ChannelEmbeddings(nn.Module):
    def __init__(self, patchsize, img_size, in_channels, dropout_rate):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        return self.dropout(embeddings)


class _Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None
        B, n_patch, hidden = x.size()
        h = w = int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)
        return self.activation(self.norm(self.conv(x)))


class _AttentionOrg(nn.Module):
    def __init__(self, config, vis, channel_num):
        super().__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]
        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()
        for _ in range(self.num_attention_heads):
            self.query1.append(copy.deepcopy(nn.Linear(channel_num[0], channel_num[0], bias=False)))
            self.query2.append(copy.deepcopy(nn.Linear(channel_num[1], channel_num[1], bias=False)))
            self.query3.append(copy.deepcopy(nn.Linear(channel_num[2], channel_num[2], bias=False)))
            self.query4.append(copy.deepcopy(nn.Linear(channel_num[3], channel_num[3], bias=False)))
            self.key.append(copy.deepcopy(nn.Linear(self.KV_size, self.KV_size, bias=False)))
            self.value.append(copy.deepcopy(nn.Linear(self.KV_size, self.KV_size, bias=False)))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        multi_head_Q1 = torch.stack([q(emb1) for q in self.query1], dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack([q(emb2) for q in self.query2], dim=1) if emb2 is not None else None
        multi_head_Q3 = torch.stack([q(emb3) for q in self.query3], dim=1) if emb3 is not None else None
        multi_head_Q4 = torch.stack([q(emb4) for q in self.query4], dim=1) if emb4 is not None else None
        multi_head_K = torch.stack([k(emb_all) for k in self.key], dim=1)
        multi_head_V = torch.stack([v(emb_all) for v in self.value], dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2) if emb4 is not None else None

        attn_scores1 = torch.matmul(multi_head_Q1, multi_head_K) / math.sqrt(self.KV_size) if emb1 is not None else None
        attn_scores2 = torch.matmul(multi_head_Q2, multi_head_K) / math.sqrt(self.KV_size) if emb2 is not None else None
        attn_scores3 = torch.matmul(multi_head_Q3, multi_head_K) / math.sqrt(self.KV_size) if emb3 is not None else None
        attn_scores4 = torch.matmul(multi_head_Q4, multi_head_K) / math.sqrt(self.KV_size) if emb4 is not None else None

        attn_probs1 = self.softmax(self.psi(attn_scores1)) if emb1 is not None else None
        attn_probs2 = self.softmax(self.psi(attn_scores2)) if emb2 is not None else None
        attn_probs3 = self.softmax(self.psi(attn_scores3)) if emb3 is not None else None
        attn_probs4 = self.softmax(self.psi(attn_scores4)) if emb4 is not None else None

        if self.vis:
            weights = [
                attn_probs1.mean(1) if emb1 is not None else None,
                attn_probs2.mean(1) if emb2 is not None else None,
                attn_probs3.mean(1) if emb3 is not None else None,
                attn_probs4.mean(1) if emb4 is not None else None,
            ]
        else:
            weights = None

        attn_probs1 = self.attn_dropout(attn_probs1) if emb1 is not None else None
        attn_probs2 = self.attn_dropout(attn_probs2) if emb2 is not None else None
        attn_probs3 = self.attn_dropout(attn_probs3) if emb3 is not None else None
        attn_probs4 = self.attn_dropout(attn_probs4) if emb4 is not None else None

        multi_head_V = multi_head_V.transpose(-1, -2)
        ctx1 = torch.matmul(attn_probs1, multi_head_V).permute(0, 3, 2, 1).contiguous().mean(dim=3) if emb1 is not None else None
        ctx2 = torch.matmul(attn_probs2, multi_head_V).permute(0, 3, 2, 1).contiguous().mean(dim=3) if emb2 is not None else None
        ctx3 = torch.matmul(attn_probs3, multi_head_V).permute(0, 3, 2, 1).contiguous().mean(dim=3) if emb3 is not None else None
        ctx4 = torch.matmul(attn_probs4, multi_head_V).permute(0, 3, 2, 1).contiguous().mean(dim=3) if emb4 is not None else None

        O1 = self.proj_dropout(self.out1(ctx1)) if emb1 is not None else None
        O2 = self.proj_dropout(self.out2(ctx2)) if emb2 is not None else None
        O3 = self.proj_dropout(self.out3(ctx3)) if emb3 is not None else None
        O4 = self.proj_dropout(self.out4(ctx4)) if emb4 is not None else None
        return O1, O2, O3, O4, weights


class _Mlp(nn.Module):
    def __init__(self, config, channel_num):
        super().__init__()
        expand_ratio = config.expand_ratio
        self.ffn1 = nn.Sequential(
            nn.Linear(channel_num[0], channel_num[0] * expand_ratio),
            nn.GELU(),
            Dropout(config.transformer["dropout_rate"]),
            nn.Linear(channel_num[0] * expand_ratio, channel_num[0]),
            Dropout(config.transformer["dropout_rate"]),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(channel_num[1], channel_num[1] * expand_ratio),
            nn.GELU(),
            Dropout(config.transformer["dropout_rate"]),
            nn.Linear(channel_num[1] * expand_ratio, channel_num[1]),
            Dropout(config.transformer["dropout_rate"]),
        )
        self.ffn3 = nn.Sequential(
            nn.Linear(channel_num[2], channel_num[2] * expand_ratio),
            nn.GELU(),
            Dropout(config.transformer["dropout_rate"]),
            nn.Linear(channel_num[2] * expand_ratio, channel_num[2]),
            Dropout(config.transformer["dropout_rate"]),
        )
        self.ffn4 = nn.Sequential(
            nn.Linear(channel_num[3], channel_num[3] * expand_ratio),
            nn.GELU(),
            Dropout(config.transformer["dropout_rate"]),
            nn.Linear(channel_num[3] * expand_ratio, channel_num[3]),
            Dropout(config.transformer["dropout_rate"]),
        )

    def forward(self, x1, x2, x3, x4):
        x1 = self.ffn1(x1) if x1 is not None else None
        x2 = self.ffn2(x2) if x2 is not None else None
        x3 = self.ffn3(x3) if x3 is not None else None
        x4 = self.ffn4(x4) if x4 is not None else None
        return x1, x2, x3, x4


class _Block(nn.Module):
    def __init__(self, config, vis, channel_num):
        super().__init__()
        self.attn = _AttentionOrg(config, vis, channel_num)
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.attn_norm = LayerNorm(config.KV_size, eps=1e-6)
        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.ffn = _Mlp(config, channel_num)

    def forward(self, emb1, emb2, emb3, emb4):
        # Build emb_all by concatenating non-None embeddings
        embcat = []
        for emb in [emb1, emb2, emb3, emb4]:
            if emb is not None:
                embcat.append(emb)
        emb_all = torch.cat(embcat, dim=2)

        org1, org2, org3, org4 = emb1, emb2, emb3, emb4
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)
        cx1, cx2, cx3, cx4, weights = self.attn(cx1, cx2, cx3, cx4, emb_all)
        emb1 = org1 + cx1 if emb1 is not None and cx1 is not None else emb1
        emb2 = org2 + cx2 if emb2 is not None and cx2 is not None else emb2
        emb3 = org3 + cx3 if emb3 is not None and cx3 is not None else emb3
        emb4 = org4 + cx4 if emb4 is not None and cx4 is not None else emb4

        org1, org2, org3, org4 = emb1, emb2, emb3, emb4
        x1 = self.ffn_norm1(emb1) if emb1 is not None else None
        x2 = self.ffn_norm2(emb2) if emb2 is not None else None
        x3 = self.ffn_norm3(emb3) if emb3 is not None else None
        x4 = self.ffn_norm4(emb4) if emb4 is not None else None
        x1, x2, x3, x4 = self.ffn(x1, x2, x3, x4)
        emb1 = emb1 + x1 if emb1 is not None and x1 is not None else emb1
        emb2 = emb2 + x2 if emb2 is not None and x2 is not None else emb2
        emb3 = emb3 + x3 if emb3 is not None and x3 is not None else emb3
        emb4 = emb4 + x4 if emb4 is not None and x4 is not None else emb4
        return emb1, emb2, emb3, emb4, weights


class _ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num, patchSize):
        super().__init__()
        self.vis = vis
        self.embeddings = nn.ModuleList([
            _ChannelEmbeddings(patchSize[i], img_size // (2 ** i),
                               channel_num[i],
                               config.transformer["embeddings_dropout_rate"])
            for i in range(len(channel_num))
        ])
        self.encoder_layers = nn.ModuleList([
            _Block(config, vis, channel_num)
            for _ in range(config.transformer["num_layers"])
        ])
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.reconstruct = nn.ModuleList([
            _Reconstruct(channel_num[i], channel_num[i],
                         kernel_size=1, scale_factor=patchSize[i])
            for i in range(len(channel_num))
        ])

    def forward(self, en1, en2, en3, en4):
        emb1 = self.embeddings[0](en1) if en1 is not None else None
        emb2 = self.embeddings[1](en2) if en2 is not None else None
        emb3 = self.embeddings[2](en3) if en3 is not None else None
        emb4 = self.embeddings[3](en4) if en4 is not None else None
        for layer in self.encoder_layers:
            emb1, emb2, emb3, emb4, attn_weights = layer(emb1, emb2, emb3, emb4)
        # Apply encoder norms
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        # Reconstruct with target spatial sizes
        enc1 = self.reconstruct[0](emb1) if emb1 is not None else en1
        enc2 = self.reconstruct[1](emb2) if emb2 is not None else en2
        enc3 = self.reconstruct[2](emb3) if emb3 is not None else en3
        enc4 = self.reconstruct[3](emb4) if emb4 is not None else en4
        if enc1 is not None and en1 is not None:
            enc1 = enc1 + en1
        if enc2 is not None and en2 is not None:
            enc2 = enc2 + en2
        if enc3 is not None and en3 is not None:
            enc3 = enc3 + en3
        if enc4 is not None and en4 is not None:
            enc4 = enc4 + en4
        return enc1, enc2, enc3, enc4, attn_weights


# ---------------------------------------------------------------------------
# UNet encoder / decoder building blocks
# ---------------------------------------------------------------------------
def _get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    return nn.ReLU()


class _ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = _get_activation(activation)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


def _make_nConv(in_ch, out_ch, nb_Conv, activation='ReLU'):
    layers = [_ConvBatchNorm(in_ch, out_ch, activation)]
    for _ in range(nb_Conv - 1):
        layers.append(_ConvBatchNorm(out_ch, out_ch, activation))
    return nn.Sequential(*layers)


class _DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        return self.nConvs(self.maxpool(x))


class _Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class _CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(_Flatten(), nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(_Flatten(), nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        ch_att_x = self.mlp_x(avg_x)
        avg_g = F.avg_pool2d(g, (g.size(2), g.size(3)))
        ch_att_g = self.mlp_g(avg_g)
        scale = torch.sigmoid((ch_att_x + ch_att_g) / 2.0)
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)
        return self.relu(x * scale)


class _UpBlockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = _CCA(F_g=in_channels // 2, F_x=in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x)


# ---------------------------------------------------------------------------
# UCTransNet
# ---------------------------------------------------------------------------
class UCTransNetEnc(nn.Module):
    """UCTransNet: UNet with Channel Transformer skip connections.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        base_channel = 64
        self.n_channels = in_channels
        self.n_classes = num_classes
        # Config for CTrans
        # Feature sizes: x1=img_size, x2=img_size/2, x3=img_size/4, x4=img_size/8
        # Patch sizes must divide feature sizes evenly and produce square number of patches
        # Using patch sizes: img/16, img/32, img/64, img/128 (rounded to int)
        patch_sizes = [max(1, img_size // 16), max(1, img_size // 32),
                       max(1, img_size // 64), max(1, img_size // 128)]
        config = _UCTConfig(img_size, base_channel, patch_sizes)
        # Encoder
        self.inc = _ConvBatchNorm(in_channels, base_channel)
        self.down1 = _DownBlock(base_channel, base_channel * 2, 2)
        self.down2 = _DownBlock(base_channel * 2, base_channel * 4, 2)
        self.down3 = _DownBlock(base_channel * 4, base_channel * 8, 2)
        self.down4 = _DownBlock(base_channel * 8, base_channel * 8, 2)
        # CTrans
        self.mtc = _ChannelTransformer(
            config, vis=False, img_size=img_size,
            channel_num=[base_channel, base_channel * 2,
                         base_channel * 4, base_channel * 8],
            patchSize=patch_sizes)
        # Decoder
        self.up4 = _UpBlockAttention(base_channel * 16, base_channel * 4, 2)
        self.up3 = _UpBlockAttention(base_channel * 8, base_channel * 2, 2)
        self.up2 = _UpBlockAttention(base_channel * 4, base_channel, 2)
        self.up1 = _UpBlockAttention(base_channel * 2, base_channel, 2)
        self.outc = nn.Conv2d(base_channel, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1, x2, x3, x4, _ = self.mtc(x1, x2, x3, x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.outc(x)
