"""nnFormer (2D adaptation) - self-contained port.

Reference: github.com/282857341/nnFormer (NeurIPS 2022).
A pure-transformer U-shaped network with hierarchical Swin-style local
self-attention. The original work targets 3D volumetric segmentation; this
port flattens the architecture to 2D and follows the standard interface
used elsewhere in this repository.

Encoder: _PatchEmbed (stride=4) -> 4 stages of SwinTransformerBlocks
(depths=[2,2,2,2]) with _PatchMerging in between.
Decoder: mirrors the encoder with _PatchExpand for 2x upsampling at every
stage; the skip connection from the encoder is fused with a 1x1 conv
(implemented as a per-token Linear over channels) instead of the
``FinalPatchExpand`` used by Swin-UNet. Final logits are bilinearly
interpolated back to the original input resolution.

Standard interface:
    model = NNFormer2D(in_channels=3, num_classes=2, img_size=224)
    out = model(x)  # -> (B, num_classes, H, W)
"""
# Source: https://github.com/282857341/nnFormer

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.utils.timm_compat import DropPath, to_2tuple, trunc_normal_


# ---------------------------------------------------------------------------
# Building blocks (Swin-style)
# ---------------------------------------------------------------------------
class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


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


class _WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],
                                             indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
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
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class _SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size
        self.norm1 = norm_layer(dim)
        self.attn = _WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = (DropPath(drop_path)
                          if drop_path > 0. else nn.Identity())
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
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
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = _window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C)
        shifted_x = _window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class _PatchExpand(nn.Module):
    """2x patch expansion (mirrors _PatchMerging).

    Doubles spatial resolution and halves channel dimension.
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = self.expand(x)  # (B, L, 2C)
        x = x.view(B, H, W, 2 * C)
        # pixel-shuffle style: split 2C -> 2x2 spatial blocks of C/2
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, C // 2)
        x = x.view(B, -1, C // 2)
        return self.norm(x)


class _PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Encoder / decoder stages
# ---------------------------------------------------------------------------
class _EncoderStage(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            _SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=(drop_path[i] if isinstance(drop_path, list)
                           else drop_path),
                norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = (downsample(input_resolution, dim=dim,
                                       norm_layer=norm_layer)
                           if downsample is not None else None)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class _DecoderStage(nn.Module):
    """Decoder stage: _PatchExpand -> concat skip -> 1x1 conv -> SwinBlocks."""
    def __init__(self, dim_in, input_resolution_in, depth, num_heads,
                 window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # _PatchExpand: (B, H*W, dim_in) -> (B, 2H*2W, dim_in//2)
        self.upsample = _PatchExpand(
            input_resolution=input_resolution_in, dim=dim_in,
            norm_layer=norm_layer)
        out_dim = dim_in // 2
        out_resolution = (input_resolution_in[0] * 2,
                          input_resolution_in[1] * 2)
        # 1x1 conv fusion of [upsampled, skip] -> out_dim. We implement the
        # 1x1 conv as a per-token Linear in BLC space (equivalent op).
        self.fuse = nn.Linear(2 * out_dim, out_dim, bias=False)
        self.fuse_norm = norm_layer(out_dim)
        self.input_resolution = out_resolution
        self.blocks = nn.ModuleList([
            _SwinTransformerBlock(
                dim=out_dim, input_resolution=out_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=(drop_path[i] if isinstance(drop_path, list)
                           else drop_path),
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=-1)
        x = self.fuse_norm(self.fuse(x))
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------
class NNFormer2D(nn.Module):
    """2D adaptation of nnFormer (NeurIPS 2022).

    Args:
        in_channels (int): Input channels (default 3).
        num_classes (int): Number of segmentation classes (default 2).
        img_size (int): Reference input size used to build the network
            (default 224). Inputs of arbitrary spatial size are accepted
            at forward time and padded internally; the model is built
            against the smallest multiple of
            ``patch_size * 2^(num_stages-1) * window_size`` that is
            >= ``img_size``.
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 embed_dim=96, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size=7,
                 patch_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True, **kwargs):
        super().__init__()
        depths = list(depths)
        num_heads = list(num_heads)
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.window_size = window_size
        self.patch_size = patch_size

        # input must be a multiple of patch_size * 2^(num_stages-1) * window
        divisor = patch_size * (2 ** (self.num_stages - 1)) * window_size
        if img_size % divisor == 0:
            padded_img_size = img_size
        else:
            padded_img_size = ((img_size + divisor - 1) // divisor) * divisor
        self.orig_img_size = img_size
        self.padded_img_size = padded_img_size
        self.divisor = divisor

        # ------------- patch embedding -------------
        self.patch_embed = _PatchEmbed(
            img_size=padded_img_size, patch_size=patch_size,
            in_chans=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth schedule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ------------- encoder -------------
        self.encoder_stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = _EncoderStage(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(patches_resolution[0] // (2 ** i),
                                  patches_resolution[1] // (2 ** i)),
                depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=(_PatchMerging
                            if i < self.num_stages - 1 else None))
            self.encoder_stages.append(stage)

        self.encoder_norm = norm_layer(
            int(embed_dim * 2 ** (self.num_stages - 1)))

        # ------------- decoder -------------
        # Decoder mirrors encoder: we have (num_stages - 1) upsample steps,
        # each fusing with the skip from the matching encoder stage.
        self.decoder_stages = nn.ModuleList()
        for j in range(self.num_stages - 1):
            # encoder stage index that produced the skip we'll fuse with
            enc_idx = self.num_stages - 2 - j
            dim_in = int(embed_dim * 2 ** (enc_idx + 1))
            input_res_in = (patches_resolution[0] // (2 ** (enc_idx + 1)),
                            patches_resolution[1] // (2 ** (enc_idx + 1)))
            stage = _DecoderStage(
                dim_in=dim_in,
                input_resolution_in=input_res_in,
                depth=depths[enc_idx], num_heads=num_heads[enc_idx],
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:enc_idx]):
                              sum(depths[:enc_idx + 1])],
                norm_layer=norm_layer)
            self.decoder_stages.append(stage)

        self.decoder_norm = norm_layer(embed_dim)
        # final 1x1 conv to logits at H/4 x W/4 (after the last decoder stage)
        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        skips = []
        for stage in self.encoder_stages:
            x, skip = stage(x)
            skips.append(skip)
        # x is currently the downsampled tensor of the deepest stage's
        # downsample (None for last stage), which equals skip for last stage.
        x = self.encoder_norm(x)
        return x, skips

    def forward_decoder(self, x, skips):
        # decoder consumes the deepest feature and fuses with skips
        # skips[num_stages-1] is the deepest stage output (= x after norm).
        # We walk upward fusing skips[num_stages-2], skips[num_stages-3], ...
        for j, stage in enumerate(self.decoder_stages):
            skip_idx = self.num_stages - 2 - j
            x = stage(x, skips[skip_idx])
        x = self.decoder_norm(x)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        # pad input to match padded_img_size (constructor-time-fixed)
        target = self.padded_img_size
        pad_h = max(0, target - H)
        pad_w = max(0, target - W)
        # additionally, ensure we are a multiple of divisor if input is larger
        if target - H < 0 and (H % self.divisor != 0):
            pad_h = ((H + self.divisor - 1) // self.divisor
                     ) * self.divisor - H
        if target - W < 0 and (W % self.divisor != 0):
            pad_w = ((W + self.divisor - 1) // self.divisor
                     ) * self.divisor - W
        if pad_h > 0 or pad_w > 0:
            mode = "reflect"
            if pad_h >= H or pad_w >= W:
                mode = "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        Hp, Wp = x.shape[-2], x.shape[-1]

        # If the (possibly padded) input does not match the resolution the
        # encoder was constructed for, rebuild the resolution-dependent
        # buffers on the fly by tracking per-stage resolutions.
        # The current Swin blocks have fixed input_resolution buffers
        # baked-in for the constructor-time padded_img_size. To handle
        # padded inputs larger than that, we instead pad to a multiple of
        # divisor and then interpolate the patch tokens back to the
        # padded_img_size grid before running the encoder. This keeps the
        # model constant-shape and dependency-free.
        if (Hp, Wp) != (self.padded_img_size, self.padded_img_size):
            x = F.interpolate(
                x, size=(self.padded_img_size, self.padded_img_size),
                mode='bilinear', align_corners=False)

        feats, skips = self.forward_features(x)
        x = self.forward_decoder(feats, skips)
        # x is (B, (H/4)*(W/4), embed_dim) at the patches_resolution grid
        H0, W0 = self.patches_resolution
        B2, L, C2 = x.shape
        x = x.transpose(1, 2).contiguous().view(B2, C2, H0, W0)
        x = self.head(x)
        # upsample back to the original (un-padded) input size
        out = F.interpolate(x, size=(H, W), mode='bilinear',
                            align_corners=False)
        return out
