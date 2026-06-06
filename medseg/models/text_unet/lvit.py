# Reference: https://github.com/HUANGLIZI/LViT
# Paper:     https://arxiv.org/abs/2206.14718
"""LViT: Language meets Vision Transformer in Medical Image Segmentation.

Implemented from the paper (Li et al., "LViT: Language meets Vision
Transformer in Medical Image Segmentation", IEEE TMI, 2023) and the
architecture overview in the README of the official repository:
    https://github.com/HUANGLIZI/LViT
        nets/LViT.py (CNN U-Net branch),
        nets/Vit.py  (ViT branch + text-injected stage-1 patch embed),
        nets/pixlevel.py (PixLevelModule attention on skip connections).
Modules are re-derived from the paper figures and method equations.

Architecture (Sec. 3 of the paper):

    image  ─► CNN U-Net branch  ───► reconstruct + skip ─► UNet decoder
                  │                       ▲
                  ▼                       │
              ViT branch (4 stages, with text injection at stage-1)
                  ▲
                  │ text_module1..4 (1-D conv pyramid that downscales the
                  │   BERT [B, L=10, 768] feature to {64,128,256,512} ch)
                  │
    text   ─► BERT-base token embedding ─► (B, L=10, 768)

Strict no-fallback policy:
    * timm is a hard import — DropPath is part of the paper's drop-path
      regularisation; silently falling back to nn.Identity when
      drop_path_rate > 0 changes the network.
    * forward(image, text=None) raises — text injection at stage-1 is the
      defining feature of LViT.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, Dropout
from torch.nn.modules.utils import _pair

from timm.models.layers import DropPath  # type: ignore


# ============================================================================
# Activation helpers (1:1 from nets/LViT.py / nets/UNet.py)
# ============================================================================


def _get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    return nn.ReLU()


def _make_n_conv(in_channels, out_channels, nb_conv, activation="ReLU"):
    layers = [ConvBatchNorm(in_channels, out_channels, activation)]
    for _ in range(nb_conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)."""

    def __init__(self, in_channels, out_channels, activation="ReLU"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = _get_activation(activation)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution."""

    def __init__(self, in_channels, out_channels, nb_conv, activation="ReLU"):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.n_convs = _make_n_conv(in_channels, out_channels, nb_conv, activation)

    def forward(self, x):
        return self.n_convs(self.maxpool(x))


# ============================================================================
# PixLevelModule (1:1 from nets/pixlevel.py)
# ============================================================================


class PixLevelModule(nn.Module):
    """Pixel-level attention used by the U-Net upsample blocks."""

    def __init__(self, in_channels):
        super().__init__()
        self.middle_layer_size_ratio = 2
        self.conv_avg = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * self.middle_layer_size_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(3 * self.middle_layer_size_ratio, 1),
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_avg = self.conv_avg(x)
        x_avg = self.relu_avg(x_avg)
        x_avg = torch.mean(x_avg, dim=1).unsqueeze(dim=1)

        x_max = self.conv_max(x)
        x_max = self.relu_max(x_max)
        x_max = torch.max(x_max, dim=1).values.unsqueeze(dim=1)

        x_out = x_max + x_avg
        x_output = torch.cat((x_avg, x_max, x_out), dim=1)
        x_output = x_output.transpose(1, 3)
        x_output = self.bottleneck(x_output)
        x_output = x_output.transpose(1, 3)
        return x_output * x


class UpblockAttention(nn.Module):
    """Upsample → PixLevelModule on skip → concat → conv."""

    def __init__(self, in_channels, out_channels, nb_conv, activation="ReLU"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pix_module = PixLevelModule(in_channels // 2)
        self.n_convs = _make_n_conv(in_channels, out_channels, nb_conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pix_module(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.n_convs(x)


# ============================================================================
# ViT branch (1:1 from nets/Vit.py)
# ============================================================================


class _Reconstruct(nn.Module):
    """Reshape token sequence back to a feature map and upscale."""

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
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


class _Embeddings(nn.Module):
    """Patch + position embedding for a single ViT stage."""

    def __init__(self, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels, in_channels,
                                       kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings)


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(self.act_layer(x))
        x = self.fc2(x)
        x = self.dropout(self.act_layer(x))
        return x


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0,
                 attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(in_dim=dim, hidden_dim=self.mlp_hidden_dim, out_dim=dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _ConvTransBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class _VisionTransformer(nn.Module):
    """Single-stage ViT branch with text injection at the lowest level."""

    def __init__(self, img_size, channel_num, patch_size, embed_dim,
                 depth=1, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 num_classes=1, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0):
        super().__init__()
        self.embeddings = _Embeddings(patch_size=patch_size, img_size=img_size,
                                      in_channels=channel_num)
        self.depth = depth
        self.dim = embed_dim
        norm_layer = nn.LayerNorm
        self.norm = norm_layer(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.encoder_blocks = nn.Sequential(*[
            _Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                   qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                   drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.ctbn = _ConvTransBN(in_channels=embed_dim, out_channels=embed_dim // 2)
        self.ctbn2 = _ConvTransBN(in_channels=embed_dim * 2, out_channels=embed_dim)
        self.ctbn3 = _ConvTransBN(in_channels=10, out_channels=196)

    def forward(self, x, skip_x, text, reconstruct=False):
        if not reconstruct:
            x = self.embeddings(x)
            if self.dim == 64:
                x = x + self.ctbn3(text)
            x = self.encoder_blocks(x)
        else:
            x = self.encoder_blocks(x)

        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x
        if not reconstruct:
            x = x.transpose(1, 2)
            x = self.ctbn(x)
            x = x.transpose(1, 2)
            return torch.cat([x, skip_x], dim=2)
        # reconstruct
        skip_x = skip_x.transpose(1, 2)
        skip_x = self.ctbn2(skip_x)
        skip_x = skip_x.transpose(1, 2)
        return x + skip_x


# ============================================================================
# Full LViT
# ============================================================================


class LViT(nn.Module):
    """LViT model.  Takes an image + text embedding (BERT [B, L=10, 768]) and
    returns a multi-class segmentation map.

    Args:
        in_channels: input image channels (3).
        num_classes: output channels.  num_classes==1 -> sigmoid binary
            (faithful to upstream).
        img_size: must be 224 for the upstream stage scheme.
        base_channel: base channel of the U-Net, upstream uses 64.
        text_len: number of text tokens, upstream uses 10.
        text_embed_dim: BERT hidden size, upstream uses 768.
    """

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 224,
        base_channel: int = 64,
        text_len: int = 10,
        text_embed_dim: int = 768,
    ):
        super().__init__()
        assert img_size == 224, (
            "Upstream LViT hard-codes the patch / spatial scheme for 224×224. "
            "Run the model at 224×224 input."
        )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.text_len = text_len
        self.text_embed_dim = text_embed_dim

        in_ch = base_channel
        # ---- CNN branch ----------------------------------------------------
        self.inc = ConvBatchNorm(in_channels, in_ch)
        self.down1 = DownBlock(in_ch, in_ch * 2, nb_conv=2)
        self.down2 = DownBlock(in_ch * 2, in_ch * 4, nb_conv=2)
        self.down3 = DownBlock(in_ch * 4, in_ch * 8, nb_conv=2)
        self.down4 = DownBlock(in_ch * 8, in_ch * 8, nb_conv=2)
        self.up4 = UpblockAttention(in_ch * 16, in_ch * 4, nb_conv=2)
        self.up3 = UpblockAttention(in_ch * 8, in_ch * 2, nb_conv=2)
        self.up2 = UpblockAttention(in_ch * 4, in_ch, nb_conv=2)
        self.up1 = UpblockAttention(in_ch * 2, in_ch, nb_conv=2)
        self.outc = nn.Conv2d(in_ch, num_classes, kernel_size=(1, 1), stride=(1, 1))

        # ---- ViT branch ----------------------------------------------------
        self.down_vit = _VisionTransformer(img_size=224, channel_num=64,
                                           patch_size=16, embed_dim=64)
        self.down_vit1 = _VisionTransformer(img_size=112, channel_num=128,
                                            patch_size=8, embed_dim=128)
        self.down_vit2 = _VisionTransformer(img_size=56, channel_num=256,
                                            patch_size=4, embed_dim=256)
        self.down_vit3 = _VisionTransformer(img_size=28, channel_num=512,
                                            patch_size=2, embed_dim=512)
        self.up_vit = _VisionTransformer(img_size=224, channel_num=64,
                                         patch_size=16, embed_dim=64)
        self.up_vit1 = _VisionTransformer(img_size=112, channel_num=128,
                                          patch_size=8, embed_dim=128)
        self.up_vit2 = _VisionTransformer(img_size=56, channel_num=256,
                                          patch_size=4, embed_dim=256)
        self.up_vit3 = _VisionTransformer(img_size=28, channel_num=512,
                                          patch_size=2, embed_dim=512)

        # ---- Reconstruction modules ----------------------------------------
        self.reconstruct1 = _Reconstruct(64, 64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = _Reconstruct(128, 128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = _Reconstruct(256, 256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = _Reconstruct(512, 512, kernel_size=1, scale_factor=(2, 2))

        # ---- Text pyramid (upstream's text_module4..1) ---------------------
        self.text_module4 = nn.Conv1d(text_embed_dim, 512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)

        self.last_activation = nn.Sigmoid()

    # ------------------------------------------------------------------
    def forward(self, image, text=None, **kwargs):
        """Forward.

        Args:
            image: (B, C, 224, 224)
            text:  (B, text_len=10, text_embed_dim=768) BERT token features.
                   ``None`` raises — text injection at stage-1 is the
                   defining mechanism of LViT and a zero-tensor would
                   reduce the model to a vanilla CNN+ViT hybrid.

        Returns:
            (B, num_classes, 224, 224) logits / probability map.
        """
        B = image.shape[0]
        if text is None:
            raise ValueError(
                "LViT.forward requires `text` of shape "
                f"(B, {self.text_len}, {self.text_embed_dim}) (BERT token "
                "features). The paper injects text at stage-1 of the ViT "
                "branch; running without text is not LViT."
            )
        if text.shape[-2:] != (self.text_len, self.text_embed_dim):
            raise ValueError(
                f"LViT.forward `text` must be (B, {self.text_len}, "
                f"{self.text_embed_dim}), got {tuple(text.shape)}"
            )

        x = image.float()
        x1 = self.inc(x)

        # Text pyramid (downsample BERT embedding along channel axis).
        # ``text`` is (B, L, C); upstream applies a Conv1d that operates over
        # the channel axis, so we transpose channel/length pairs accordingly.
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)

        # Down ViT path (text injected only at stage-1).
        y1 = self.down_vit(x1, x1, text1)
        x2 = self.down1(x1)
        y2 = self.down_vit1(x2, y1, text2)
        x3 = self.down2(x2)
        y3 = self.down_vit2(x3, y2, text3)
        x4 = self.down3(x3)
        y4 = self.down_vit3(x4, y3, text4)
        x5 = self.down4(x4)

        # Up ViT path (reconstruct=True).
        y4 = self.up_vit3(y4, y4, text4, True)
        y3 = self.up_vit2(y3, y4, text3, True)
        y2 = self.up_vit1(y2, y3, text2, True)
        y1 = self.up_vit(y1, y2, text1, True)

        # Add reconstructed ViT tokens back to CNN features.
        x1 = self.reconstruct1(y1) + x1
        x2 = self.reconstruct2(y2) + x2
        x3 = self.reconstruct3(y3) + x3
        x4 = self.reconstruct4(y4) + x4

        # Decoder.
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        if self.num_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)
        return logits
