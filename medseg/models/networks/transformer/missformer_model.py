"""MISSFormer – self-contained port from github.com/ZhifangDeng/MISSFormer.

MISSFormer: An Effective Transformer for Medical Image Segmentation.
Architecture: MiT (Mix Transformer) encoder + Bridge blocks (multi-scale
              self-attention) + SegU-decoder (PatchExpand + TransformerBlock).
"""
# Source: https://github.com/ZhifangDeng/MISSFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange


# ---------------------------------------------------------------------------
# MiT encoder building blocks
# ---------------------------------------------------------------------------
class _DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        return self.dwconv(tx).flatten(2).transpose(1, 2)


class _EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x_atten)


class _M_EfficientSelfAtten(nn.Module):
    """Multi-scale efficient self-attention with scale reduction."""

    def __init__(self, dim, head, reduction_ratio, spatial_sizes=None,
                 ch_mults=(1, 2, 5, 8)):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        if reduction_ratio is not None and spatial_sizes is not None:
            self.scale_reduce = _ScaleReduce(dim, reduction_ratio, spatial_sizes,
                                             ch_mults)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x_atten)


class _ScaleReduce(nn.Module):
    """Spatial reduction for multi-scale attention (faithful to original).

    Splits the concatenated multi-scale sequence by per-scale token counts
    (h*w*ch_mult for each scale), applies strided Conv2d to reduce spatial
    resolution for all scales except the smallest, then re-concatenates.
    """

    def __init__(self, dim, reduction_ratio, spatial_sizes,
                 ch_mults=(1, 2, 5, 8)):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        self.spatial_sizes = spatial_sizes
        self.ch_mults = list(ch_mults)
        n = len(spatial_sizes)
        if n == 4:
            ks0 = min(reduction_ratio[3], spatial_sizes[0][0], spatial_sizes[0][1])
            ks1 = min(reduction_ratio[2], spatial_sizes[1][0], spatial_sizes[1][1])
            ks2 = min(reduction_ratio[1], spatial_sizes[2][0], spatial_sizes[2][1])
            self.sr0 = nn.Conv2d(dim, dim, ks0, ks0)
            self.sr1 = nn.Conv2d(dim, dim, ks1, ks1)
            self.sr2 = nn.Conv2d(dim, dim, ks2, ks2)
        elif n == 3:
            ks0 = min(reduction_ratio[2], spatial_sizes[0][0], spatial_sizes[0][1])
            ks1 = min(reduction_ratio[1], spatial_sizes[1][0], spatial_sizes[1][1])
            self.sr0 = nn.Conv2d(dim, dim, ks0, ks0)
            self.sr1 = nn.Conv2d(dim, dim, ks1, ks1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        ss = self.spatial_sizes
        mults = self.ch_mults
        n = len(ss)
        if n == 4:
            # Token counts per scale (h*w*mult)
            n0 = ss[0][0] * ss[0][1] * mults[0]
            n1 = ss[1][0] * ss[1][1] * mults[1]
            n2 = ss[2][0] * ss[2][1] * mults[2]
            # Reshape to 2D: effective spatial grid is (h, w*mult) because
            # flattening to C=dims absorbs extra channels into the W dimension.
            # Faithful to original: reshape interprets 1568 tokens as 56x28
            # (not 28x28), etc.
            tem0 = x[:, :n0, :].reshape(B, ss[0][0], ss[0][1] * mults[0], C).permute(0, 3, 1, 2)
            tem1 = x[:, n0:n0+n1, :].reshape(B, ss[1][0], ss[1][1] * mults[1], C).permute(0, 3, 1, 2)
            tem2 = x[:, n0+n1:n0+n1+n2, :].reshape(B, ss[2][0], ss[2][1] * mults[2], C).permute(0, 3, 1, 2)
            tem3 = x[:, n0+n1+n2:, :]
            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)
            return self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], dim=1))
        else:
            n0 = ss[0][0] * ss[0][1] * mults[0]
            n1 = ss[1][0] * ss[1][1] * mults[1]
            tem0 = x[:, :n0, :].reshape(B, ss[0][0], ss[0][1] * mults[0], C).permute(0, 3, 1, 2)
            tem1 = x[:, n0:n0+n1, :].reshape(B, ss[1][0], ss[1][1] * mults[1], C).permute(0, 3, 1, 2)
            tem2 = x[:, n0+n1:, :]
            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            return self.norm(torch.cat([sr_0, sr_1, tem2], dim=1))


class _MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = _DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x, H, W):
        return self.fc2(self.act(self.dwconv(self.fc1(x), H, W)))


class _MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = _DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)

    def forward(self, x, H, W):
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        return self.fc2(ax)


class _OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=3,
                 in_ch=3, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        return self.norm(fx), H, W


class _TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix_skip'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix_skip':
            self.mlp = _MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = _MixFFN(dim, int(dim * 4))

    def forward(self, x, H, W):
        tx = x + self.attn(self.norm1(x), H, W)
        return tx + self.mlp(self.norm2(tx), H, W)


# ---------------------------------------------------------------------------
# MiT (Mix Transformer) encoder
# ---------------------------------------------------------------------------
class _MiT(nn.Module):
    def __init__(self, img_size=224, in_ch=3, dims=(64, 128, 320, 512),
                 layers=(2, 2, 2, 2), token_mlp_mode='mix_skip'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding = [3, 1, 1, 1]
        heads = [1, 2, 5, 8]
        reduction_ratios = [8, 4, 2, 1]
        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(4):
            in_c = in_ch if i == 0 else dims[i - 1]
            self.patch_embeds.append(
                _OverlapPatchEmbeddings(img_size // (2 ** (i + 2)) if i > 0 else img_size,
                                        patch_sizes[i], strides[i], padding[i],
                                        in_c, dims[i]))
            block_list = nn.ModuleList()
            for _ in range(layers[i]):
                block_list.append(_TransformerBlock(
                    dims[i], heads[i], reduction_ratios[i], token_mlp_mode))
            self.blocks.append(block_list)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims])

    def forward(self, x):
        outputs = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            B, _, C = x.shape
            x = self.norms[i](x)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs


# ---------------------------------------------------------------------------
# Bridge blocks – faithful to original BridegeBlock_4 / BridgeLayer_4
# (multi-scale joint self-attention across ALL concatenated scales)
# ---------------------------------------------------------------------------
class _BridgeLayer4(nn.Module):
    """Faithful to original BridgeLayer_4.

    Original flattens ALL encoder features to C=dims[0] channels, absorbing
    extra channels into the token count (e.g. c2 with 128ch at 28x28 becomes
    1568 tokens of 64ch).  Per-scale MixFFN_skip uses native channel dims
    (dims, dims*2, dims*5, dims*8) after reinterpreting the split tokens.
    """

    def __init__(self, dims, head, reduction_ratios, spatial_sizes,
                 channel_dims=(64, 128, 320, 512)):
        super().__init__()
        self.spatial_sizes = spatial_sizes
        self.dims = dims
        self.channel_dims = list(channel_dims)
        # Channel multipliers relative to base dims
        self.ch_mults = [ch // dims for ch in channel_dims]
        # Token counts after flattening to C=dims: h*w*mult
        self.token_counts = [h * w * m
                             for (h, w), m in zip(spatial_sizes, self.ch_mults)]
        self.cum_indices = [0]
        for tc in self.token_counts:
            self.cum_indices.append(self.cum_indices[-1] + tc)

        self.norm1 = nn.LayerNorm(dims)
        self.attn = _M_EfficientSelfAtten(dims, head, reduction_ratios,
                                          spatial_sizes, self.ch_mults)
        self.norm2 = nn.LayerNorm(dims)
        # Per-scale MixFFN_skip with native channel dims (original)
        self.mixffn1 = _MixFFN_skip(dims, dims * 4)
        self.mixffn2 = _MixFFN_skip(dims * 2, dims * 8)
        self.mixffn3 = _MixFFN_skip(dims * 5, dims * 20)
        self.mixffn4 = _MixFFN_skip(dims * 8, dims * 32)

    def forward(self, inputs):
        C = self.dims  # base channel count (e.g. 64)
        ch_mults = [ch // C for ch in self.channel_dims]

        if isinstance(inputs, list):
            # First call: flatten 4D encoder features all to C=dims channels,
            # absorbing extra channels into the token (sequence) dimension.
            # This is faithful to original BridgeLayer_4 behavior.
            flat = []
            for feat in inputs:
                B, Ch, _, _ = feat.shape
                f = feat.permute(0, 2, 3, 1).reshape(B, -1, C)
                flat.append(f)
            combined = torch.cat(flat, dim=1)  # (B, sum(N_i), C)
        else:
            combined = inputs
            B, _, C = combined.shape

        # Multi-scale self-attention + residual
        tx1 = combined + self.attn(self.norm1(combined))
        tx = self.norm2(tx1)

        # Split back to per-scale tokens (still C=dims channels each)
        splits = []
        for i in range(len(self.spatial_sizes)):
            start = self.cum_indices[i]
            end = self.cum_indices[i + 1]
            splits.append(tx[:, start:end, :])

        # Reshape each split to native channel dims (C*mult) for per-scale MixFFN,
        # matching original: tem.reshape(B, -1, C*mult)
        h0, w0 = self.spatial_sizes[0]
        h1, w1 = self.spatial_sizes[1]
        h2, w2 = self.spatial_sizes[2]
        h3, w3 = self.spatial_sizes[3]
        tem1 = splits[0].reshape(B, -1, C * ch_mults[0])
        tem2 = splits[1].reshape(B, -1, C * ch_mults[1])
        tem3 = splits[2].reshape(B, -1, C * ch_mults[2])
        tem4 = splits[3].reshape(B, -1, C * ch_mults[3])

        # Per-scale MixFFN_skip with skip connection
        m1 = self.mixffn1(tem1, h0, w0).reshape(B, -1, C)
        m2 = self.mixffn2(tem2, h1, w1).reshape(B, -1, C)
        m3 = self.mixffn3(tem3, h2, w2).reshape(B, -1, C)
        m4 = self.mixffn4(tem4, h3, w3).reshape(B, -1, C)

        t_combined = torch.cat([m1, m2, m3, m4], dim=1)
        return tx1 + t_combined


class _BridegeBlock4(nn.Module):
    """Faithful to original BridegeBlock_4: 4 BridgeLayer_4 stages,
    then split and reshape back to 4D spatial features with native channels."""

    def __init__(self, dims, head, reduction_ratios, spatial_sizes,
                 channel_dims=(64, 128, 320, 512)):
        super().__init__()
        self.spatial_sizes = spatial_sizes
        self.dims = dims
        self.channel_dims = list(channel_dims)
        ch_mults = [ch // dims for ch in channel_dims]
        self.token_counts = [h * w * m
                             for (h, w), m in zip(spatial_sizes, ch_mults)]
        self.cum_indices = [0]
        for tc in self.token_counts:
            self.cum_indices.append(self.cum_indices[-1] + tc)

        self.bridge_layer1 = _BridgeLayer4(dims, head, reduction_ratios,
                                           spatial_sizes, channel_dims)
        self.bridge_layer2 = _BridgeLayer4(dims, head, reduction_ratios,
                                           spatial_sizes, channel_dims)
        self.bridge_layer3 = _BridgeLayer4(dims, head, reduction_ratios,
                                           spatial_sizes, channel_dims)
        self.bridge_layer4 = _BridgeLayer4(dims, head, reduction_ratios,
                                           spatial_sizes, channel_dims)

    def forward(self, x):
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)

        # Split flat tensor back to 4D features with native channel dims
        B, _, C = bridge4.shape
        outs = []
        for i in range(len(self.spatial_sizes)):
            start = self.cum_indices[i]
            end = self.cum_indices[i + 1]
            h, w = self.spatial_sizes[i]
            ch = self.channel_dims[i]
            sk = bridge4[:, start:end, :].reshape(B, h, w, ch).permute(0, 3, 1, 2)
            outs.append(sk)
        return outs


# ---------------------------------------------------------------------------
# Decoder (PatchExpand + TransformerBlock)
# ---------------------------------------------------------------------------
class _PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 \
            else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=2, p2=2, c=C // 4)
        return self.norm(x.reshape(B, -1, C // 4))


class _FinalPatchExpandX4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        return self.norm(x.reshape(B, -1, self.output_dim))


class _SegUDecoder(nn.Module):
    """Faithful to original SegU_decoder.

    Args:
        input_size: (H, W) for PatchExpand / FinalPatchExpand_X4.
        out_dim: output channel dimension for transformer blocks.
        concat_in_dim: total input channels for concat_linear (= x1_ch + x2_ch).
        heads / reduction_ratios: for TransformerBlock.
        n_class: output classes (only used when is_last=True).
        is_last: whether this is the final decoder stage.
    """

    def __init__(self, input_size, out_dim, concat_in_dim, heads,
                 reduction_ratios, n_class=9, norm_layer=nn.LayerNorm,
                 is_last=False):
        super().__init__()
        self.is_last = is_last
        if not is_last:
            self.concat_linear = nn.Linear(concat_in_dim, out_dim)
            self.layer_up = _PatchExpand(
                input_resolution=input_size, dim=out_dim,
                dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(concat_in_dim, out_dim)
            self.layer_up = _FinalPatchExpandX4(
                input_resolution=input_size, dim=out_dim,
                dim_scale=4, norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = _TransformerBlock(out_dim, heads, reduction_ratios)
        self.layer_former_2 = _TransformerBlock(out_dim, heads, reduction_ratios)

        def init_weights(self_mod):
            for m in self_mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            if self.last_layer is not None:
                out = self.last_layer(
                    self.layer_up(tran_layer_2)
                    .view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


# ---------------------------------------------------------------------------
# MISSFormer
# ---------------------------------------------------------------------------
class MISSFormer(nn.Module):
    """MISSFormer: MiT encoder + Bridge blocks + SegU decoder.

    Faithful to original ZhifangDeng/MISSFormer.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        self.img_size = img_size
        dims = [64, 128, 320, 512]
        heads = [1, 2, 5, 8]
        reduction_ratios = [8, 4, 2, 1]

        # Encoder (MiT)
        self.encoder = _MiT(img_size, in_channels, dims, (2, 2, 2, 2),
                            'mix_skip')

        # Spatial sizes at each encoder stage: img//4, img//8, img//16, img//32
        base = img_size // 4  # 56 for 224
        spatial_sizes = [(base, base), (base // 2, base // 2),
                         (base // 4, base // 4), (base // 8, base // 8)]

        # Bridge (multi-scale joint self-attention)
        self.bridge = _BridegeBlock4(dims[0], heads[0], [64, 32, 16, 8],
                                     spatial_sizes, dims)

        # Decoder — bridge outputs as skip connections
        # Token counts after bridge:
        #   bridge_out[3]: 7*7=49 tokens of 512ch
        #   bridge_out[2]: 14*14=196 tokens of 320ch
        #   bridge_out[1]: 28*28=784 tokens of 128ch
        #   bridge_out[0]: 56*56=3136 tokens of 64ch
        #
        # Decoder chain (PatchExpand 2x upsamples token count):
        #   d4: in=49 tokens → concat with skip@196 → upsample → 196 tokens
        #   d3: in=196 tokens → concat with skip@784 → upsample → 784 tokens
        #   d2: in=784 tokens → concat with skip@3136 → upsample → 3136 tokens
        #   d1: in=3136 tokens → concat with skip@3136 → FinalPatchExpandX4 → 50176
        # Decoder output channel dims (after concat_linear)
        dec_out_dims = [dims[1], dims[1], dims[0], dims[0]]  # 128, 128, 64, 64
        # Total concat input channels (= x1_ch + skip_ch for each decoder)
        # Note: x1_ch = previous decoder out_dim // 2 (PatchExpand halves channels)
        concat_in_dims = [dims[3] + dims[2],       # d4: 512+320=832
                          dims[1] // 2 + dims[1],   # d3: 64+128=192
                          dims[1] // 2 + dims[0],   # d2: 64+64=128
                          dims[0] // 2 + dims[0]]   # d1: 32+64=96
        dec_heads = [heads[3], 4, heads[1], heads[0]]
        dec_rr = [reduction_ratios[3], reduction_ratios[2],
                  reduction_ratios[1], reduction_ratios[0]]
        dec_sizes = [
            (base // 4, base // 4),   # 14x14  (d4 input spatial)
            (base // 2, base // 2),   # 28x28  (d3 input spatial)
            (base, base),             # 56x56  (d2 input spatial)
            (base, base),             # 56x56  (d1 input spatial)
        ]

        self.decoder4 = _SegUDecoder(
            dec_sizes[0], dec_out_dims[0], concat_in_dims[0],
            dec_heads[0], dec_rr[0], num_classes)
        self.decoder3 = _SegUDecoder(
            dec_sizes[1], dec_out_dims[1], concat_in_dims[1],
            dec_heads[1], dec_rr[1], num_classes)
        self.decoder2 = _SegUDecoder(
            dec_sizes[2], dec_out_dims[2], concat_in_dims[2],
            dec_heads[2], dec_rr[2], num_classes)
        self.decoder1 = _SegUDecoder(
            dec_sizes[3], dec_out_dims[3], concat_in_dims[3],
            dec_heads[3], dec_rr[3], num_classes, is_last=True)

    def forward(self, x):
        # Encode
        enc_features = self.encoder(x)
        # Bridge (multi-scale joint self-attention)
        bridge_out = self.bridge(enc_features)

        # Decode — bridge outputs as skip connections
        # Token counts: bridge_out[3]=49, [2]=196, [1]=784, [0]=3136
        # SegU_decoder concat requires matching token counts, so we
        # interpolate the smaller input to match the skip connection.
        B = x.shape[0]

        # Helper: flatten 4D to (B, L, C)
        def _flat(feat):
            return feat.permute(0, 2, 3, 1).reshape(B, -1, feat.shape[1])

        # Helper: align token counts by spatial interpolation
        def _align(src, target_h, target_w):
            B_s, L, C = src.shape
            src_h = src_w = int(L ** 0.5)
            if src_h * src_w == L and (src_h != target_h or src_w != target_w):
                src_4d = src.view(B_s, src_h, src_w, C).permute(0, 3, 1, 2)
                src_4d = F.interpolate(src_4d, size=(target_h, target_w),
                                       mode='bilinear', align_corners=True)
                return src_4d.permute(0, 2, 3, 1).reshape(B_s, -1, C)
            return src

        # d4: bridge[3] → align to bridge[2] spatial → concat → upsample
        x1 = _flat(bridge_out[3])
        skip2 = bridge_out[2].permute(0, 2, 3, 1)  # (B, 14, 14, 320)
        x1 = _align(x1, skip2.shape[1], skip2.shape[2])
        d4 = self.decoder4(x1, skip2)

        # d3: d4_out → align to bridge[1] spatial → concat → upsample
        skip1 = bridge_out[1].permute(0, 2, 3, 1)  # (B, 28, 28, 128)
        d4_aligned = _align(d4, skip1.shape[1], skip1.shape[2])
        d3 = self.decoder3(d4_aligned, skip1)

        # d2: d3_out → align to bridge[0] spatial → concat → upsample
        skip0 = bridge_out[0].permute(0, 2, 3, 1)  # (B, 56, 56, 64)
        d3_aligned = _align(d3, skip0.shape[1], skip0.shape[2])
        d2 = self.decoder2(d3_aligned, skip0)

        # d1: d2_out → align to bridge[0] spatial → concat → FinalPatchExpandX4
        d2_aligned = _align(d2, skip0.shape[1], skip0.shape[2])
        out = self.decoder1(d2_aligned, skip0)

        # Ensure output matches input size
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear',
                                align_corners=True)
        return out
