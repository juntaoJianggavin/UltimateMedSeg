"""HiFormer – self-contained port from github.com/amirhossein-kz/HiFormer.

HiFormer: Hierarchical Multi-scale Representations Using Transformers for
Medical Image Segmentation (ISBI 2022).

Architecture: Dual-branch (CNN + Swin Transformer) encoder with DLF
              (Double-Level Fusion) module + convolutional decoder.

NOTE: The original loads pretrained Swin/ResNet weights. This self-contained
      port initializes from scratch for portability.
"""
# Source: https://github.com/amirhossein-kz/HiFormer

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Swin Transformer building blocks
# ---------------------------------------------------------------------------
def _window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C)


def _window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        attn = attn + rpb.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class _SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        H, W = input_resolution
        if min(H, W) <= window_size:
            self.shift_size = 0
            self.window_size = min(H, W)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, (self.window_size, self.window_size),
                                     num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = _window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x
        x_windows = _window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = _window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x
        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class _BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            _SwinBlock(dim, input_resolution, num_heads, window_size,
                       0 if (i % 2 == 0) else window_size // 2, mlp_ratio)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class _PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


# ---------------------------------------------------------------------------
# Swin Transformer backbone (from scratch, no pretrained)
# ---------------------------------------------------------------------------
class _SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=(2, 2, 6), num_heads=(3, 6, 12), window_size=7,
                 mlp_ratio=4.):
        super().__init__()
        patches_res = img_size // patch_size
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, patch_size, patch_size),
            nn.LayerNorm([embed_dim, patches_res, patches_res]))
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(depths)):
            dim_i = embed_dim * (2 ** i)
            res_i = patches_res // (2 ** i)
            self.layers.append(_BasicLayer(
                dim_i, (res_i, res_i), depths[i], num_heads[i],
                window_size, mlp_ratio))
            if i < len(depths) - 1:
                self.downsamples.append(
                    _PatchMerging((res_i, res_i), dim_i))

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        return x


# ---------------------------------------------------------------------------
# ResNet backbone (mimics torchvision ResNet34 layer structure)
# ---------------------------------------------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(x)
        return self.relu(out + residual)


def _make_res_layer(in_ch, out_ch, blocks, stride=1):
    layers = [_ResBlock(in_ch, out_ch, stride)]
    for _ in range(1, blocks):
        layers.append(_ResBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


class _ResNet34Backbone(nn.Module):
    """Mimics torchvision resnet34 children()[:7] structure."""
    def __init__(self, in_channels=3):
        super().__init__()
        # layers[0:5] = stem + layer1+layer2 + layer3 + layer4 equivalent
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        self.layer1 = _make_res_layer(64, 64, 3)         # 56x56
        self.layer2 = _make_res_layer(64, 128, 4, stride=2)   # 28x28
        self.layer3 = _make_res_layer(128, 256, 6, stride=2)  # 14x14

    def forward_stem(self, x):
        """Run first 5 ResNet children (stem + layer1) → 64ch at 56x56."""
        x = self.stem(x)
        x = self.layer1(x)
        return x

    def forward_p2(self, x):
        """ResNet layer5 equivalent → 28x28"""
        return self.layer2(x)

    def forward_p3(self, x):
        """ResNet layer6 equivalent → 14x14"""
        return self.layer3(x)


# ---------------------------------------------------------------------------
# PyramidFeatures – interleaved CNN + Swin encoder (faithful to original)
# ---------------------------------------------------------------------------
class _PyramidFeatures(nn.Module):
    """Faithful to original PyramidFeatures: interleaves CNN (ResNet) and
    Swin Transformer with skip connections at 3 pyramid levels."""
    def __init__(self, img_size=224, in_channels=3,
                 swin_pyramid_fm=(96, 192, 384),
                 cnn_pyramid_fm=(64, 128, 256)):
        super().__init__()
        self.cnn = _ResNet34Backbone(in_channels)
        self.swin = _SwinTransformer(
            img_size=img_size, patch_size=4, in_chans=in_channels,
            embed_dim=swin_pyramid_fm[0], depths=(2, 2, 6),
            num_heads=(3, 6, 12), window_size=7)

        ps = img_size // 4  # 56
        # Level 1: 56x56
        self.p1_ch = nn.Conv2d(cnn_pyramid_fm[0], swin_pyramid_fm[0], 1)
        self.norm_1 = nn.LayerNorm(swin_pyramid_fm[0])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)
        self.p1_pm = _PatchMerging((ps, ps), swin_pyramid_fm[0])

        # Level 2: 28x28
        self.p2_ch = nn.Conv2d(cnn_pyramid_fm[1], swin_pyramid_fm[1], 1)
        self.p2_pm = _PatchMerging((ps // 2, ps // 2), swin_pyramid_fm[1])

        # Level 3: 14x14
        self.p3_ch = nn.Conv2d(cnn_pyramid_fm[2], swin_pyramid_fm[2], 1)
        self.norm_2 = nn.LayerNorm(swin_pyramid_fm[2])
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Shared stem: resnet_layers[0:5] → 56x56 feature map
        fm1 = self.cnn.forward_stem(x)

        # Level 1: 56x56 (96 dim)
        fm1_ch = self.p1_ch(fm1)
        fm1_seq = rearrange(fm1_ch, 'b c h w -> b (h w) c')
        sw1 = self.swin.layers[0](fm1_seq)
        sw1_skip = fm1_seq + sw1
        norm1 = self.norm_1(sw1_skip)
        cls1 = rearrange(self.avgpool_1(norm1.transpose(1, 2)), 'b c 1 -> b 1 c')
        fm1_sw1 = self.p1_pm(sw1_skip)  # → 28x28, 192 dim

        # Level 2: 28x28 (192 dim)
        fm1_sw2 = self.swin.layers[1](fm1_sw1)
        fm2 = self.cnn.forward_p2(fm1)
        fm2_ch = self.p2_ch(fm2)
        fm2_seq = rearrange(fm2_ch, 'b c h w -> b (h w) c')
        fm2_sw2_skip = fm2_seq + fm1_sw2
        fm2_sw2 = self.p2_pm(fm2_sw2_skip)  # → 14x14, 384 dim

        # Level 3: 14x14 (384 dim)
        fm2_sw3 = self.swin.layers[2](fm2_sw2)
        fm3 = self.cnn.forward_p3(fm2)
        fm3_ch = self.p3_ch(fm3)
        fm3_seq = rearrange(fm3_ch, 'b c h w -> b (h w) c')
        fm3_sw3_skip = fm3_seq + fm2_sw3
        norm2 = self.norm_2(fm3_sw3_skip)
        cls2 = rearrange(self.avgpool_2(norm2.transpose(1, 2)), 'b c 1 -> b 1 c')

        return [torch.cat((cls1, sw1_skip), dim=1),
                torch.cat((cls2, fm3_sw3_skip), dim=1)]


# ---------------------------------------------------------------------------
# Attention (from original HiFormer utils.py)
# ---------------------------------------------------------------------------
class _Attention(nn.Module):
    """Multi-head attention with configurable dim_head (original uses 64)."""
    def __init__(self, dim, factor, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim * factor),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = rearrange(torch.matmul(attn, v), 'b h n d -> b n (h d)')
        return self.to_out(out)


# ---------------------------------------------------------------------------
# MultiScaleBlock (faithful to original)
# ---------------------------------------------------------------------------
class _MultiScaleBlock(nn.Module):
    """Independent per-branch self-attention + MLP (faithful to original)."""
    def __init__(self, embed_dim, num_patches, depth, num_heads=(6, 12),
                 mlp_ratio=(2., 2., 1.), qkv_bias=True, drop_path=None):
        super().__init__()
        self.attns = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(2):
            d = depth[i]
            dim = embed_dim[i]
            if d > 0:
                attn_layers = nn.ModuleList()
                for _ in range(d):
                    attn_layers.append(_Attention(
                        dim, factor=1, heads=num_heads[i],
                        dim_head=64, dropout=0.))
                self.attns.append(attn_layers)
                self.mlps.append(_Mlp(dim, int(dim * mlp_ratio[i])))
                self.norms.append(nn.LayerNorm(dim))
            else:
                self.attns.append(nn.Identity())
                self.mlps.append(nn.Identity())
                self.norms.append(nn.Identity())

    def forward(self, xs):
        outs = []
        for i, x in enumerate(xs):
            if isinstance(self.attns[i], nn.ModuleList):
                for attn in self.attns[i]:
                    x = x + attn(x)
                x = x + self.mlps[i](self.norms[i](x))
            outs.append(x)
        return outs


# ---------------------------------------------------------------------------
# All2Cross (DLF Module – faithful to original)
# ---------------------------------------------------------------------------
class _All2Cross(nn.Module):
    """DLF module: PyramidFeatures + positional embeddings + MultiScaleBlocks."""
    def __init__(self, img_size=224, in_channels=3,
                 embed_dim=(96, 384),
                 swin_pyramid_fm=(96, 192, 384),
                 cnn_pyramid_fm=(64, 128, 256),
                 depth=((1, 2, 0),),
                 num_heads=(6, 12),
                 mlp_ratio=(2., 2., 1.),
                 qkv_bias=True):
        super().__init__()
        self.pyramid = _PyramidFeatures(
            img_size=img_size, in_channels=in_channels,
            swin_pyramid_fm=swin_pyramid_fm,
            cnn_pyramid_fm=cnn_pyramid_fm)
        ps = 4  # patch_size
        n_p1 = (img_size // ps) ** 2       # 3136 for 224
        n_p2 = (img_size // ps // 4) ** 2  # 196 for 224
        num_patches = (n_p1, n_p2)
        self.num_branches = 2
        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i]))
            for i in range(self.num_branches)])
        self.blocks = nn.ModuleList()
        for block_config in depth:
            blk = _MultiScaleBlock(
                embed_dim, num_patches, block_config,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            self.blocks.append(blk)
        self.norm = nn.ModuleList([nn.LayerNorm(embed_dim[i])
                                   for i in range(self.num_branches)])

    def forward(self, x):
        xs = self.pyramid(x)
        # xs already contain CLS + spatial tokens from PyramidFeatures
        out_xs = []
        for i, x in enumerate(xs):
            # Add full positional embeddings (CLS + spatial positions)
            x = x + self.pos_embed[i]
            out_xs.append(x)
        for blk in self.blocks:
            out_xs = blk(out_xs)
        return [self.norm[i](x) for i, x in enumerate(out_xs)]


# ---------------------------------------------------------------------------
# Decoder (faithful to original ConvUpsample + SegmentationHead)
# ---------------------------------------------------------------------------
class _ConvUpsample(nn.Module):
    def __init__(self, in_chans=384, out_chans=(128,), upsample=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = list(out_chans)
        layers = []
        c_in = in_chans
        for i, c_out in enumerate(self.out_chans):
            if i > 0:
                c_in = self.out_chans[i - 1]
            layers.append(nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False))
            layers.append(nn.GroupNorm(32, c_out))
            layers.append(nn.ReLU(inplace=False))
            if upsample:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear',
                                          align_corners=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# HiFormer (faithful to original)
# ---------------------------------------------------------------------------
class HiFormer(nn.Module):
    """HiFormer: CNN + Swin Transformer with DLF cross-attention fusion.

    Faithful to github.com/amirhossein-kz/HiFormer (HiFormer-B config).

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        self.orig_img_size = img_size
        self.patch_size = [4, 16]

        # HiFormer-B defaults
        swin_pyramid_fm = (96, 192, 384)
        cnn_pyramid_fm = (64, 128, 256)  # resnet34
        embed_dim = (96, 384)
        depth = ((1, 2, 0),)
        num_heads = (6, 12)
        mlp_ratio = (2., 2., 1.)

        # Swin window=7 with patch=4 and 2 patch-merges (3 stages) requires
        # img_size to be a multiple of patch * 2^(stages-1) * window = 4*4*7 = 112
        # so that every pyramid level is divisible into 7x7 windows.
        # Round img_size up to the next multiple of 112 and pad/crop in forward
        # to preserve pretrained-weight compatible architecture.
        self._swin_window = 7
        self._swin_patch = 4
        self._swin_stages = 3  # 1 patch_embed + 2 patch_merges
        self._size_multiple = (
            self._swin_patch * (2 ** (self._swin_stages - 1)) * self._swin_window
        )
        self.img_size = (
            (img_size + self._size_multiple - 1) // self._size_multiple
        ) * self._size_multiple

        self.all2cross = _All2Cross(
            img_size=self.img_size, in_channels=in_channels,
            embed_dim=embed_dim,
            swin_pyramid_fm=swin_pyramid_fm,
            cnn_pyramid_fm=cnn_pyramid_fm,
            depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True)

        # Decoder – faithful: ConvUp_s upsample=True, ConvUp_l upsample=False
        self.conv_up_s = _ConvUpsample(384, [128, 128], upsample=True)
        self.conv_up_l = _ConvUpsample(96, [128], upsample=False)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(128, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))
        self.seg_head = nn.Conv2d(16, num_classes, 3, 1, 1)

    def forward(self, x):
        orig_h, orig_w = x.shape[-2:]
        target = self.img_size
        pad_h = target - orig_h
        pad_w = target - orig_w
        if pad_h > 0 or pad_w > 0:
            # Reflect padding requires pad < dim; fall back to replicate then
            # reflect for tiny inputs to stay safe.
            mode = 'reflect' if pad_h < orig_h and pad_w < orig_w else 'replicate'
            x = F.pad(x, (0, max(pad_w, 0), 0, max(pad_h, 0)), mode=mode)

        xs = self.all2cross(x)
        # Strip CLS tokens
        embeddings = [t[:, 1:] for t in xs]
        reshaped = []
        for i, embed in enumerate(embeddings):
            h = w = self.img_size // self.patch_size[i]
            embed = rearrange(embed, 'b (h w) d -> b d h w', h=h, w=w)
            embed = self.conv_up_l(embed) if i == 0 else self.conv_up_s(embed)
            reshaped.append(embed)
        C = reshaped[0] + reshaped[1]
        C = self.conv_pred(C)
        out = self.seg_head(C)
        if pad_h > 0 or pad_w > 0:
            out = out[..., :orig_h, :orig_w]
        return out
