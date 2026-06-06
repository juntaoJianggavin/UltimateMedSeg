"""DAEFormer – self-contained port from github.com/xmindflow/DAEFormer.

Combines DAEFormer.py and segformer.py into a single file.
Official repo: https://github.com/xmindflow/DAEFormer

Architecture Overview:
  - Encoder (MiT): 3-stage with dims=[128, 320, 512], DualTransformerBlock
  - Decoder: MyDecoderLayer with PatchExpand/FinalPatchExpand_X4 + CrossAttentionBlock
    + DualTransformerBlock for skip fusion
  - FinalPatchExpand_X4 for 4x upsampling at last stage
"""
# Source: https://github.com/xmindflow/DAEFormer

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F


# ── Building blocks (from segformer.py) ──────────────────────────────────────
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
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        return self.fc2(ax)


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x, H, W):
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        return self.fc2(ax)


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1,
                 in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        return self.norm(fx), H, W


# ── DAEFormer-specific attention modules ─────────────────────────────────────
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
        agg = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
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
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx


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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DualTransformerBlock(nn.Module):
    def __init__(self, in_dim, key_dim, value_dim, head_count=1,
                 token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim,
                                       value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x, H, W):
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)
        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)
        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)
        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)
        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)
        mx = add3 + mlp2
        return mx


# ── Encoder (MiT) ───────────────────────────────────────────────────────────
class MiT(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers,
                 head_count=1, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3]
        strides = [4, 2, 2]
        padding_sizes = [3, 1, 1]
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0],
            3, in_dim[0])
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1],
            in_dim[0], in_dim[1])
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2],
            in_dim[1], in_dim[2])
        self.block1 = nn.ModuleList([
            DualTransformerBlock(in_dim[0], key_dim[0], value_dim[0],
                                 head_count, token_mlp)
            for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])
        self.block2 = nn.ModuleList([
            DualTransformerBlock(in_dim[1], key_dim[1], value_dim[1],
                                 head_count, token_mlp)
            for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(in_dim[1])
        self.block3 = nn.ModuleList([
            DualTransformerBlock(in_dim[2], key_dim[2], value_dim[2],
                                 head_count, token_mlp)
            for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(in_dim[2])

    def forward(self, x):
        B = x.shape[0]
        outs = []
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs


# ── Decoder components ───────────────────────────────────────────────────────
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
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
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())
        return x


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, head_count, token_mlp_mode,
                 n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims, key_dim, value_dim, input_size[0], input_size[1],
                head_count, token_mlp_mode)
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            self.layer_up = PatchExpand(
                input_resolution=input_size, dim=out_dim, dim_scale=2,
                norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims * 2, key_dim, value_dim, input_size[0], input_size[1],
                head_count, token_mlp_mode)
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4,
                norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)
        self.layer_former_1 = DualTransformerBlock(
            out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = DualTransformerBlock(
            out_dim, key_dim, value_dim, head_count, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
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
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2).view(
                        b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


# ── Top-level DAEFormer ──────────────────────────────────────────────────────
class _DAEFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip",
                 img_size=224):
        super().__init__()
        # Official dims: [128, 320, 512]
        dims = [128, 320, 512]
        key_dim = [128, 320, 512]
        value_dim = [128, 320, 512]
        layers = [2, 2, 2]
        self.backbone = MiT(
            image_size=img_size, in_dim=dims, key_dim=key_dim,
            value_dim=value_dim, layers=layers,
            head_count=head_count, token_mlp=token_mlp_mode)
        d_base_feat_size = img_size // 32  # 224->7, 256->8, 64->2, etc.
        in_out_chan = [
            [64, 128, 128, 128, 160],   # decoder_0 (last): [dim, out_dim, key_dim, value_dim, x2_dim]
            [320, 320, 320, 320, 256],   # decoder_1
            [512, 512, 512, 512, 512],   # decoder_2
        ]
        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2], head_count, token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1], head_count, token_mlp_mode, n_class=num_classes)
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0], head_count, token_mlp_mode, n_class=num_classes,
            is_last=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        output_enc = self.backbone(x)
        b, c, _, _ = output_enc[2].shape
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))
        return tmp_0


class DAEFormer(nn.Module):
    """DAEFormer wrapper with standard UltimateMedSeg interface."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        self.model = _DAEFormer(num_classes=num_classes, img_size=img_size)

    def forward(self, x):
        return self.model(x)
