"""TransUNet – self-contained port from github.com/Beckschen/TransUNet.

Combines vit_seg_configs.py, vit_seg_modeling_resnet_skip.py, and
vit_seg_modeling.py into a single file with no external medseg imports.

Standard interface:
    model = TransUNet(in_channels=3, num_classes=2, img_size=224)
    out = model(x)  # -> (B, num_classes, H, W)
"""
# Source: https://github.com/Beckschen/TransUNet

from __future__ import absolute_import, division, print_function

import copy
import math
from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, CrossEntropyLoss, Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules.utils import _pair


# ── lightweight config (replaces ml_collections.ConfigDict) ──────────────────
class _Cfg(dict):
    """Dict that also supports attribute access."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return v

    def __setattr__(self, k, v):
        self[k] = v

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default


def _get_r50_b16_config():
    c = _Cfg()
    c.patches = _Cfg({"grid": (16, 16)})
    c.hidden_size = 768
    c.transformer = _Cfg(mlp_dim=3072, num_heads=12, num_layers=12,
                         attention_dropout_rate=0.0, dropout_rate=0.1)
    c.resnet = _Cfg(num_layers=(3, 4, 9), width_factor=1)
    c.classifier = "seg"
    c.decoder_channels = (256, 128, 64, 16)
    c.skip_channels = [512, 256, 64, 16]
    c.n_classes = 2
    c.n_skip = 3
    c.activation = "softmax"
    return c


# ── ResNetV2 with skip connections (from vit_seg_modeling_resnet_skip.py) ───
def _np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def _conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1,
                     bias=bias, groups=groups)


def _conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0,
                     bias=bias)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = _conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = _conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = _conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.downsample = _conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.root = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(3, width, kernel_size=7, stride=2, bias=False,
                               padding=3)),
            ("gn", nn.GroupNorm(32, width, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))
        self.body = nn.Sequential(OrderedDict([
            ("block1", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width, cout=width * 4,
                                            cmid=width))] +
                [(f"unit{i}", PreActBottleneck(cin=width * 4, cout=width * 4,
                                               cmid=width))
                 for i in range(2, block_units[0] + 1)]))),
            ("block2", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 4, cout=width * 8,
                                            cmid=width * 2, stride=2))] +
                [(f"unit{i}", PreActBottleneck(cin=width * 8, cout=width * 8,
                                               cmid=width * 2))
                 for i in range(2, block_units[1] + 1)]))),
            ("block3", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 8, cout=width * 16,
                                            cmid=width * 4, stride=2))] +
                [(f"unit{i}", PreActBottleneck(cin=width * 16, cout=width * 16,
                                               cmid=width * 4))
                 for i in range(2, block_units[2] + 1)]))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                feat = torch.zeros((b, x.size()[1], right_size, right_size),
                                   device=x.device)
                feat[:, :, : x.size()[2], : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# ── ViT building blocks (from vit_seg_modeling.py) ──────────────────────────
def _swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": _swish}


class Attention(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads,
                                      self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.attention_head_size)
        probs = self.softmax(scores)
        weights = probs if self.vis else None
        probs = self.attn_dropout(probs)
        ctx = torch.matmul(probs, v)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(*ctx.size()[:-2], self.all_head_size)
        out = self.proj_dropout(self.out(ctx))
        return out, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act_fn(self.fc1(x)))))


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0],
                          img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = ((img_size[0] // patch_size_real[0]) *
                         (img_size[1] // patch_size_real[1]))
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = ((img_size[0] // patch_size[0]) *
                         (img_size[1] // patch_size[1]))
            self.hybrid = False
        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        features = None
        if self.hybrid:
            x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        return self.dropout(embeddings), features


class Block(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x, weights = self.attn(self.attention_norm(x))
        x = x + h
        h = x
        x = self.ffn(self.ffn_norm(x))
        return x + h, weights


class Encoder(nn.Module):
    def __init__(self, config, vis=False):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([
            copy.deepcopy(Block(config, vis))
            for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        return self.encoder_norm(hidden_states), attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis=False):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        emb, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(emb)
        return encoded, attn_weights, features


# ── Decoder (CUP) ────────────────────────────────────────────────────────────
class Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, stride=1,
                 use_batchnorm=True):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                         padding=padding, bias=not use_batchnorm)
        bn = nn.BatchNorm2d(out_ch)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_ch + skip_ch, out_ch, 3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_ch, out_ch, 3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, padding=kernel_size // 2)
        up = (nn.UpsamplingBilinear2d(scale_factor=upsampling)
              if upsampling > 1 else nn.Identity())
        super().__init__(conv2d, up)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, 3,
                                    padding=1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        if config.n_skip != 0:
            skip_channels = list(config.skip_channels)
            for i in range(4 - config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]
        blocks = [DecoderBlock(ic, oc, sc)
                  for ic, oc, sc in zip(in_channels, out_channels,
                                        skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = (features[i] if (features is not None and
                                    i < self.config.n_skip) else None)
            x = decoder_block(x, skip=skip)
        return x


# ── Top-level model ──────────────────────────────────────────────────────────
class _VisionTransformer(nn.Module):
    """Original TransUNet VisionTransformer."""
    def __init__(self, config, img_size=224, num_classes=2, vis=False):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config["decoder_channels"][-1],
            out_channels=config["n_classes"], kernel_size=3)
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        return self.segmentation_head(x)


class TransUNet(nn.Module):
    """TransUNet wrapper with standard interface.

    Args:
        in_channels (int): Number of input channels (default: 3).
        num_classes (int): Number of output classes (default: 2).
        img_size (int): Input image size (default: 224).
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        config = _get_r50_b16_config()
        config.n_classes = num_classes
        # Compute grid dynamically based on img_size
        grid = img_size // 16  # 14 for 224, 16 for 256
        config.patches["grid"] = (grid, grid)
        # Allow kwargs to override config
        if "hidden_size" in kwargs:
            config.hidden_size = kwargs["hidden_size"]
        if "num_layers" in kwargs:
            config.transformer["num_layers"] = kwargs["num_layers"]
        if "num_heads" in kwargs:
            config.transformer["num_heads"] = kwargs["num_heads"]
        if "mlp_dim" in kwargs:
            config.transformer["mlp_dim"] = kwargs["mlp_dim"]
        if "resnet_num_layers" in kwargs:
            config.resnet["num_layers"] = tuple(kwargs["resnet_num_layers"])
        if "grid_size" in kwargs:
            config.patches["grid"] = tuple(kwargs["grid_size"])
        if "decoder_channels" in kwargs:
            config.decoder_channels = tuple(kwargs["decoder_channels"])
        if "skip_channels" in kwargs:
            config.skip_channels = list(kwargs["skip_channels"])
        if "n_skip" in kwargs:
            config.n_skip = kwargs["n_skip"]
        self.model = _VisionTransformer(config, img_size=img_size,
                                        num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
