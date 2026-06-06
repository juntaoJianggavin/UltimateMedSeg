"""DAEFormer decoder module.

Extracted from networks/transformer/daeformer_model.py for modular reuse.
Faithful to the original: CrossAttentionBlock for fusing encoder features
+ bilinear upsampling + prediction head.
"""
# Source: https://github.com/xmindflow/DAEFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange
from einops.layers.torch import Rearrange
from medseg.registry import DECODER_REGISTRY


# ── Shared building blocks (duplicated for self-containment) ─────────────────

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x, H, W):
        return self.fc2(self.act(self.dwconv(self.fc1(x), H, W)))


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)

    def forward(self, x, H, W):
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)))
        return self.fc2(self.norm2(ax))


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x, H, W):
        return self.fc2(self.act(self.fc1(x)))


class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels,
                 head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels:
                                 (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels:
                                      (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels:
                           (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            av = (context.transpose(1, 2) @ query).reshape(
                n, head_value_channels, h, w)
            attended_values.append(av)
        return self.reprojection(torch.cat(attended_values, dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0,
                 proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.normalize(q.transpose(-2, -1), dim=-1)
        k = F.normalize(k.transpose(-2, -1), dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width,
                 head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width
        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    def forward(self, x1, x2):
        B, N, D = x1.size()
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels:
                                 (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels:
                                      (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels:
                           (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = context.transpose(1, 2) @ query
            attended_values.append(attended_value)
        agg = torch.cat(attended_values, dim=1).reshape(
            B, D, self.height, self.width)
        rep = self.reprojection(agg).reshape(B, 2 * D, N).permute(0, 2, 1)
        return self.norm(rep)


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_dim, key_dim, value_dim, height, width,
                 head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.attn = Cross_Attention(key_dim, value_dim, height, width,
                                    head_count=head_count)
        self.norm2 = nn.LayerNorm(in_dim * 2)
        if token_mlp == "mix":
            self.mlp = MixFFN(in_dim * 2, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip(in_dim * 2, int(in_dim * 4))
        else:
            self.mlp = MLP_FFN(in_dim * 2, int(in_dim * 4))

    def forward(self, x1, x2):
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)
        attn = self.attn(norm_1, norm_2)
        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        return tx + self.mlp(self.norm2(tx), self.H, self.W)


# ── DAEFormer Decoder ───────────────────────────────────────────────────────

@DECODER_REGISTRY.register("daeformer")
class DAEFormerDecoder(nn.Module):
    """DAEFormer cross-attention decoder.

    Standard interface: ``forward(bottleneck_feat, skip_features)``
    where skip_features = [x1(64ch,56x56), x2(128ch,28x28)].
    bottleneck_feat = x3 (320ch, 14x14).

    Architecture:
        cross_attn_3 (self-attn on x3) -> upsample 2x ->
        cross_attn_2 (fuse x2 + x3_up) -> upsample 8x -> 1x1 pred
    """

    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 image_size=224, num_classes=2,
                 in_dim=None, key_dim=None, value_dim=None,
                 token_mlp="mix_skip", head_count=1,
                 **kwargs):
        super().__init__()
        # Adapt to actual encoder channels
        # skip_features[-1] = encoder_channels[-1] (last skip), bottleneck = bottleneck_channels
        skip_ch = encoder_channels[-1] if len(encoder_channels) > 0 else 128
        bneck_ch = bottleneck_channels
        # The CrossAttentionBlock reprojection doubles channels, so we need
        # in_dim[2] == in_dim[1] for the concat to work. Project bottleneck if needed.
        if in_dim is None:
            in_dim = [skip_ch, skip_ch, skip_ch]
        if key_dim is None:
            key_dim = list(in_dim)

        H, W = image_size, image_size
        # Determine spatial sizes based on encoder strides
        n_stages = len(encoder_channels)
        if n_stages >= 4:
            h2, w2 = H // 16, W // 16  # skip at stride /16
            h3, w3 = H // 32, W // 32  # bottleneck at stride /32
        elif n_stages >= 3:
            h2, w2 = H // 8, W // 8
            h3, w3 = H // 16, W // 16
        else:
            h2, w2 = H // 4, W // 4
            h3, w3 = H // 8, W // 8

        # Project bottleneck to skip_ch if dimensions differ
        self.bneck_proj = (nn.Conv2d(bneck_ch, skip_ch, 1)
                           if bneck_ch != skip_ch else nn.Identity())

        self.cross_attn_3 = CrossAttentionBlock(
            in_dim=in_dim[2], key_dim=key_dim[2], value_dim=in_dim[2],
            height=h3, width=w3, head_count=head_count, token_mlp=token_mlp)

        cross_2_in = in_dim[1] + in_dim[2] * 2
        self.cross_attn_2 = CrossAttentionBlock(
            in_dim=cross_2_in, key_dim=key_dim[1], value_dim=cross_2_in,
            height=h2, width=w2, head_count=head_count, token_mlp=token_mlp)

        self.d3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=False),
            nn.GELU())
        self.d1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear',
                        align_corners=False),
            nn.GELU())

        self.linear_pred = nn.Conv2d(cross_2_in * 2, num_classes,
                                     kernel_size=1)
        self._out_channels = num_classes

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # Project bottleneck to match skip channels if needed
        x3 = self.bneck_proj(bottleneck_feat)
        x2 = skip_features[-1]
        B = x3.shape[0]
        _, _, h3, w3 = x3.shape
        _, _, h2, w2 = x2.shape

        x3_seq = x3.flatten(2).transpose(1, 2)
        # Rebuild cross_attn_3 spatial dims if changed (different encoder)
        if (h3, w3) != getattr(self, '_cached_h3w3', (None, None)):
            self._cached_h3w3 = (h3, w3)
            self.cross_attn_3.H = h3
            self.cross_attn_3.W = w3
            self.cross_attn_3.attn.height = h3
            self.cross_attn_3.attn.width = w3
        if (h2, w2) != getattr(self, '_cached_h2w2', (None, None)):
            self._cached_h2w2 = (h2, w2)
            self.cross_attn_2.H = h2
            self.cross_attn_2.W = w2
            self.cross_attn_2.attn.height = h2
            self.cross_attn_2.attn.width = w2
        x3_cross = self.cross_attn_3(x3_seq, x3_seq)

        x3_4d = x3_cross.view(B, h3, w3, -1).permute(0, 3, 1, 2)
        x3_up = self.d3(x3_4d)
        # Align spatial size to x2 if needed
        if x3_up.shape[2:] != x2.shape[2:]:
            x3_up = F.interpolate(x3_up, size=x2.shape[2:], mode='bilinear',
                                  align_corners=False)

        x2_seq = x2.flatten(2).transpose(1, 2)
        x3_up_seq = x3_up.flatten(2).transpose(1, 2)
        fused_seq = torch.cat([x2_seq, x3_up_seq], dim=-1)
        fused_cross = self.cross_attn_2(fused_seq, fused_seq)

        fused_4d = fused_cross.view(B, h2, w2, -1).permute(0, 3, 1, 2)
        out = self.d1(fused_4d)
        return self.linear_pred(out)
