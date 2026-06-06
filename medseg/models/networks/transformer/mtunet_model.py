"""MTUNet – self-contained port from github.com/Dootmaan/MT-UNet.

Mixed Transformer UNet for Medical Image Segmentation (ICASSP 2022).

Architecture: CNN stem + mixed transformer encoder/decoder (CSAttention +
MEAttention + Gaussian-biased axial attention) + UNet decoder stem.
"""
# Source: https://github.com/Dootmaan/MT-UNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Default configs (from original repo)
# ---------------------------------------------------------------------------
_CFGS = {
    "win_size": 4,
    "head": 8,
    "encoder": [256, 512],
    "bottleneck": 1024,
    "decoder": [1024, 512],
}


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------
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
        x = self.bn(self.conv(x))
        return self.relu(x) if self.activation else x


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(cin, cout, 3, 1, padding=1),
            _ConvBNReLU(cout, cout, 3, 1, padding=1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.bn(self.conv1(x))
        return self.relu(h + x)


# ---------------------------------------------------------------------------
# CNN U-Encoder / U-Decoder (3-stage)
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

    def forward(self, x):
        features = []
        x = self.res1(x); features.append(x)
        x = self.pool1(x)
        x = self.res2(x); features.append(x)
        x = self.pool2(x)
        x = self.res3(x); features.append(x)
        x = self.pool3(x)
        return x, features


class _UDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res1 = _DoubleConv(512, 256)
        self.trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res2 = _DoubleConv(256, 128)
        self.trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res3 = _DoubleConv(128, 64)

    def forward(self, x, feature):
        x = self.trans1(x)
        x = torch.cat((feature[2], x), dim=1)
        x = self.res1(x)
        x = self.trans2(x)
        x = torch.cat((feature[1], x), dim=1)
        x = self.res2(x)
        x = self.trans3(x)
        x = torch.cat((feature[0], x), dim=1)
        return self.res3(x)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------
class _MEAttention(nn.Module):
    """Memory-Efficient (linear) attention."""

    def __init__(self, dim, configs):
        super().__init__()
        self.coef = 4
        self.num_heads = configs["head"] * self.coef
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)
        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = self.linear_0(x).softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.proj(x)


class _Attention(nn.Module):
    """Multi-head self-attention with optional axial mode."""

    def __init__(self, dim, configs, axial=False):
        super().__init__()
        self.axial = axial
        self.dim = dim
        self.num_head = configs["head"]
        self.attention_head_size = dim // configs["head"]
        self.all_head_size = self.num_head * self.attention_head_size
        self.query_layer = nn.Linear(dim, self.all_head_size)
        self.key_layer = nn.Linear(dim, self.all_head_size)
        self.value_layer = nn.Linear(dim, self.all_head_size)
        self.out = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        return x.view(*new_shape)

    def forward(self, x):
        if self.axial:
            b, h, w, c = x.shape
            q = self.query_layer(x)
            k = self.key_layer(x)
            v = self.value_layer(x)
            # Row attention
            q_x = q.view(b * h, w, -1)
            k_x = k.view(b * h, w, -1).transpose(-1, -2)
            attn_x = torch.matmul(q_x, k_x).view(b, -1, w, w)
            # Col attention
            q_y = q.permute(0, 2, 1, 3).contiguous().view(b * w, h, -1)
            k_y = k.permute(0, 2, 1, 3).contiguous().view(b * w, h, -1).transpose(-1, -2)
            attn_y = torch.matmul(q_y, k_y).view(b, -1, h, h)
            return attn_x, attn_y, v
        else:
            q = self.transpose_for_scores(self.query_layer(x)).permute(
                0, 1, 2, 4, 3, 5).contiguous()
            k = self.transpose_for_scores(self.key_layer(x)).permute(
                0, 1, 2, 4, 3, 5).contiguous()
            v = self.transpose_for_scores(self.value_layer(x)).permute(
                0, 1, 2, 4, 3, 5).contiguous()
            scores = torch.matmul(q, k.transpose(-1, -2))
            scores = scores / math.sqrt(self.attention_head_size)
            probs = self.softmax(scores)
            ctx = torch.matmul(probs, v)
            ctx = ctx.permute(0, 1, 2, 4, 3, 5).contiguous()
            new_shape = ctx.size()[:-2] + (self.all_head_size,)
            ctx = ctx.view(*new_shape)
            return self.out(ctx)


class _WinAttention(nn.Module):
    def __init__(self, configs, dim):
        super().__init__()
        self.window_size = configs["win_size"]
        self.attention = _Attention(dim, configs)

    def forward(self, x):
        b, n, c = x.shape
        h = w = int(np.sqrt(n))
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
        ws = self.window_size
        if h % ws != 0:
            rs = h + ws - h % ws
            new_x = torch.zeros((b, c, rs, rs), device=x.device, dtype=x.dtype)
            new_x[:, :, :h, :w] = x
            new_x[:, :, h:, w:] = x[:, :, h - rs:, w - rs:]
            x = new_x
            b, c, h, w = x.shape
        x = x.view(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b, h // ws, w // ws, ws * ws, c)
        return self.attention(x)


class _DlightConv(nn.Module):
    def __init__(self, dim, configs):
        super().__init__()
        self.linear = nn.Linear(dim, configs["win_size"] * configs["win_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = x
        avg_x = torch.mean(x, dim=-2)
        x_prob = self.softmax(self.linear(avg_x))
        x = torch.mul(h, x_prob.unsqueeze(-1))
        return torch.sum(x, dim=-2)


class _GaussianTrans(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(-torch.abs(torch.randn(1)))
        self.shift = nn.Parameter(torch.abs(torch.randn(1)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, atten_x, atten_y, value = x
        device = x.device
        new_value = torch.zeros_like(value)
        for r in range(x.shape[1]):
            for c in range(x.shape[2]):
                ax = atten_x[:, r, c, :]
                ay = atten_y[:, c, r, :]
                dx = torch.tensor(
                    [(hc - c) ** 2 for hc in range(x.shape[2])],
                    device=device, dtype=x.dtype)
                dy = torch.tensor(
                    [(wr - r) ** 2 for wr in range(x.shape[1])],
                    device=device, dtype=x.dtype)
                dx = -(self.shift * dx + self.bias)
                dy = -(self.shift * dy + self.bias)
                ax = self.softmax(dx + ax)
                ay = self.softmax(dy + ay)
                new_value[:, r, c, :] = torch.sum(
                    ax.unsqueeze(-1) * value[:, r, :, :] +
                    ay.unsqueeze(-1) * value[:, :, c, :], dim=-2)
        return new_value


class _CSAttention(nn.Module):
    """Combined local (window) + global (axial + Gaussian) attention."""

    def __init__(self, dim, configs):
        super().__init__()
        self.win_atten = _WinAttention(configs, dim)
        self.dlightconv = _DlightConv(dim, configs)
        self.global_atten = _Attention(dim, configs, axial=True)
        self.gaussiantrans = _GaussianTrans()
        self.up = nn.UpsamplingBilinear2d(scale_factor=configs["win_size"])
        self.queeze = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, x):
        b, n, c = x.shape
        origin_h = origin_w = int(np.sqrt(n))
        x = self.win_atten(x)
        b, p, p, win, c = x.shape
        h = x.view(b, p, p, int(np.sqrt(win)), int(np.sqrt(win)), c)
        h = h.permute(0, 1, 3, 2, 4, 5).contiguous()
        h = h.view(b, p * int(np.sqrt(win)), p * int(np.sqrt(win)), c)
        h = h.permute(0, 3, 1, 2).contiguous()
        x = self.dlightconv(x)
        atten_x, atten_y, mixed_value = self.global_atten(x)
        x = self.gaussiantrans((x, atten_x, atten_y, mixed_value))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.up(x)
        # Crop to original spatial dims before concat (handles padding)
        x = x[:, :, :origin_h, :origin_w].contiguous()
        h = h[:, :, :origin_h, :origin_w].contiguous()
        x = self.queeze(torch.cat((x, h), dim=1)).permute(0, 2, 3, 1).contiguous()
        return x.view(b, -1, c)


class _EAmodule(nn.Module):
    """Encoder/Attention module (CSAttention + MEAttention)."""

    def __init__(self, dim, configs):
        super().__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.CSAttention = _CSAttention(dim, configs)
        self.EAttention = _MEAttention(dim, configs)

    def forward(self, x):
        h = x
        x = h + self.CSAttention(self.SlayerNorm(x))
        h = x
        return h + self.EAttention(self.ElayerNorm(x))


# ---------------------------------------------------------------------------
# Stem / Encoder / Decoder blocks
# ---------------------------------------------------------------------------
class _Stem(nn.Module):
    # UEncoder applies three 2x maxpools → stride 8.
    _STEM_STRIDE = 8

    def __init__(self, in_channels=3, img_size=224, embed_dim=256):
        super().__init__()
        self.model = _UEncoder(in_channels)
        self.trans_dim = _ConvBNReLU(embed_dim, embed_dim, 1, 1, 0)
        # Derive token-grid size from img_size at init.
        self.grid_size = max(img_size // self._STEM_STRIDE, 1)
        n_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim
        self.position_embedding = nn.Parameter(
            torch.zeros((1, n_patches, embed_dim)))

    def _resize_pos_embed(self, h, w):
        # Interpolate the learned pos embedding over a 2D grid to match
        # runtime spatial dims (h, w) of the token map.
        if h == self.grid_size and w == self.grid_size:
            return self.position_embedding
        pe = self.position_embedding  # (1, N, C)
        pe = pe.transpose(1, 2).reshape(
            1, self.embed_dim, self.grid_size, self.grid_size)
        pe = F.interpolate(pe, size=(h, w), mode="bilinear",
                           align_corners=False)
        pe = pe.flatten(2).transpose(1, 2)  # (1, h*w, C)
        return pe

    def forward(self, x):
        x, features = self.model(x)
        x = self.trans_dim(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(-2, -1)
        pos = self._resize_pos_embed(H, W)
        x = x + pos
        return x, features


class _EncoderBlock(nn.Module):
    def __init__(self, dim, configs):
        super().__init__()
        self.block = nn.ModuleList([
            _EAmodule(dim, configs),
            _EAmodule(dim, configs),
            _ConvBNReLU(dim, dim * 2, 2, stride=2, padding=0)])

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)
        B, N, C = x.shape
        h = w = int(np.sqrt(N))
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)
        skip = x
        x = self.block[2](x)
        return x, skip


class _DecoderBlock(nn.Module):
    def __init__(self, dim, flag, configs):
        super().__init__()
        self.flag = flag
        if not flag:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),
                nn.Conv2d(dim, dim // 2, 1, 1),
                _EAmodule(dim // 2, configs),
                _EAmodule(dim // 2, configs)])
        else:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),
                _EAmodule(dim, configs),
                _EAmodule(dim, configs)])

    def forward(self, x, skip):
        if not self.flag:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = self.block[1](x)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[2](x)
            return self.block[3](x)
        else:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[1](x)
            return self.block[2](x)


# ---------------------------------------------------------------------------
# MTUNet
# ---------------------------------------------------------------------------
class MTUNet(nn.Module):
    """Mixed Transformer UNet.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        configs = dict(_CFGS)
        self.img_size = img_size
        self.stem = _Stem(in_channels, img_size=img_size,
                          embed_dim=configs["encoder"][0])
        self.encoder = nn.ModuleList([
            _EncoderBlock(dim, configs) for dim in configs["encoder"]])
        self.bottleneck = nn.Sequential(
            _EAmodule(configs["bottleneck"], configs),
            _EAmodule(configs["bottleneck"], configs))
        self.decoder = nn.ModuleList()
        for i in range(len(configs["decoder"]) - 1):
            self.decoder.append(_DecoderBlock(configs["decoder"][i], False, configs))
        self.decoder.append(_DecoderBlock(configs["decoder"][-1], True, configs))
        self.decoder_stem = _UDecoder()
        self.segmentation_head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)
        skips = []
        for enc in self.encoder:
            x, skip = enc(x)
            skips.append(skip)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        x = self.bottleneck(x)
        B, N, C = x.shape
        x = x.view(B, int(np.sqrt(N)), -1, C).permute(0, 3, 1, 2)
        for i, dec in enumerate(self.decoder):
            x = dec(x, skips[len(self.decoder) - i - 1])
            B, N, C = x.shape
            x = x.view(B, int(np.sqrt(N)), int(np.sqrt(N)), C).permute(0, 3, 1, 2)
        x = self.decoder_stem(x, features)
        return self.segmentation_head(x)
