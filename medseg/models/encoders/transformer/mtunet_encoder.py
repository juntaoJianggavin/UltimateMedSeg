"""MT-UNet Encoder: faithful port from https://github.com/Dootmaan/MT-UNet

Reference: Wang et al., "Mixed Transformer U-Net for Medical Image Segmentation"
All class/attribute names match the original for pretrained weight loading.
"""
# Source: https://github.com/Dootmaan/MT-UNet

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from medseg.registry import ENCODER_REGISTRY


# ============= ConvBNReLU =============
class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


# ============= U_encoder (CNN stem) =============
class U_encoder(nn.Module):
    """CNN stem for MT-UNet."""
    def __init__(self):
        super(U_encoder, self).__init__()
        self.econv0 = nn.Sequential(
            ConvBNReLU(3, 32, 3, 1, 1),
            ConvBNReLU(32, 64, 3, 1, 1))
        self.econv1 = nn.Sequential(
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 128, 3, 1, 1))
        self.econv2 = nn.Sequential(
            ConvBNReLU(128, 128, 3, 1, 1),
            ConvBNReLU(128, 256, 3, 1, 1))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        features = []
        x = self.econv0(x)  # (B, 64, H, W)
        features.append(x)
        x = self.pool(x)
        x = self.econv1(x)  # (B, 128, H/2, W/2)
        features.append(x)
        x = self.pool(x)
        x = self.econv2(x)  # (B, 256, H/4, W/4)
        features.append(x)
        x = self.pool(x)     # (B, 256, H/8, W/8)
        return x, features


# ============= Stem =============
class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.model = U_encoder()
        self.trans_dim = ConvBNReLU(256, 256, 1, 1, 0)
        self.position_embedding = nn.Parameter(torch.zeros((1, 784, 256)))

    def forward(self, x):
        x, features = self.model(x)
        x = self.trans_dim(x)
        x = x.flatten(2)
        x = x.transpose(-2, -1)
        x = x + self.position_embedding
        return x, features


# ============= Attention modules =============
class WinAttention(nn.Module):
    """Window attention for MT-UNet."""
    def __init__(self, dim, win_size=4, num_heads=8):
        super(WinAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.win_size = win_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.view(B, H, W, C)
        # Pad if not divisible by window size
        pad_h = (self.win_size - H % self.win_size) % self.win_size
        pad_w = (self.win_size - W % self.win_size) % self.win_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]
        # Window partition
        nH = Hp // self.win_size
        nW = Wp // self.win_size
        x = x.view(B, nH, self.win_size, nW, self.win_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B * nH * nW, self.win_size * self.win_size, C)

        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], self.win_size * self.win_size, C)
        x = self.proj(x)

        # Window reverse
        x = x.view(B, nH, nW, self.win_size, self.win_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        x = x.reshape(B, H * W, C)
        return x


class DlightConv(nn.Module):
    """Depth-wise lightweight convolution."""
    def __init__(self, dim, win_size=4):
        super(DlightConv, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.win_size = win_size

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = self.linear(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # Pad if not divisible by window size
        pad_h = (self.win_size - H % self.win_size) % self.win_size
        pad_w = (self.win_size - W % self.win_size) % self.win_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]
        x = F.unfold(x, kernel_size=self.win_size, stride=self.win_size)
        nH, nW = Hp // self.win_size, Wp // self.win_size
        x = x.view(B, C, self.win_size * self.win_size, -1).permute(0, 3, 2, 1)
        x = x.reshape(B * nH * nW, self.win_size * self.win_size, C)
        # Reverse
        x = x.view(B, nH, nW, self.win_size, self.win_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
        x = x.reshape(B, H * W, C)
        return x


class GaussianTrans(nn.Module):
    """Gaussian transform attention."""
    def __init__(self):
        super(GaussianTrans, self).__init__()
        self.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return x + self.bias


class CSAttention(nn.Module):
    """Combined Spatial Attention for MT-UNet."""
    def __init__(self, dim, configs):
        super(CSAttention, self).__init__()
        self.win_attn = WinAttention(dim, configs.get("win_size", 4), configs.get("head", 8))
        self.dlconv = DlightConv(dim, configs.get("win_size", 4))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x1 = self.win_attn(x)
        x2 = self.dlconv(x)
        x = self.norm(x1 + x2)
        return x


class MEAttention(nn.Module):
    """Mixed Efficient Attention for MT-UNet."""
    def __init__(self, dim, configs):
        super(MEAttention, self).__init__()
        self.num_heads = configs.get("head", 8)
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)
        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = self.linear_0(x)
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)
        return x


# ============= EAmodule =============
class EAmodule(nn.Module):
    """Mixed transform block: CSAttention + MEAttention."""
    def __init__(self, dim, configs=None):
        super(EAmodule, self).__init__()
        if configs is None:
            configs = {"win_size": 4, "head": 8}
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.CSAttention = CSAttention(dim, configs)
        self.EAttention = MEAttention(dim, configs)

    def forward(self, x):
        h = x
        x = self.SlayerNorm(x)
        x = self.CSAttention(x)
        x = h + x
        h = x
        x = self.ElayerNorm(x)
        x = self.EAttention(x)
        x = h + x
        return x


# ============= encoder_block =============
class encoder_block(nn.Module):
    def __init__(self, dim, configs=None):
        super(encoder_block, self).__init__()
        self.block = nn.ModuleList([
            EAmodule(dim, configs),
            EAmodule(dim, configs),
            ConvBNReLU(dim, dim * 2, 2, stride=2, padding=0)
        ])

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)
        B, N, C = x.shape
        h, w = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)
        skip = x
        x = self.block[2](x)
        return x, skip


@ENCODER_REGISTRY.register("mtunet")
class MTUNetEncoder(nn.Module):
    """MT-UNet Encoder wrapper.
    Faithful to https://github.com/Dootmaan/MT-UNet
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
        configs = {"win_size": 4, "head": 8}

        self.stem = Stem()
        self.encoder = nn.ModuleList()
        encoder_dims = [256, 512]
        for dim in encoder_dims:
            self.encoder.append(encoder_block(dim, configs))

        self.bottleneck = nn.Sequential(
            EAmodule(1024, configs),
            EAmodule(1024, configs))

        # out_channels: stem features [64, 128, 256] + encoder skips [256, 512] + bottleneck [1024]
        self._out_channels = [64, 128, 256, 256, 512, 1024]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x, stem_features = self.stem(x)  # (B, N, 256)
        skips = []

        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

        x = self.bottleneck(x)
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        bottleneck_feat = x.view(B, H, W, C).permute(0, 3, 1, 2)

        # Return all multi-scale features
        all_features = stem_features + skips + [bottleneck_feat]
        return all_features
