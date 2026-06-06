"""SAMUS: SAM-based UltraSound segmentation network.

Reference:
    Xian Lin, Yangyang Xiang, Li Zhang, Xin Yang, Zengqiang Yan, Li Yu.
    "SAMUS: Adapting Segment Anything Model for Clinically-Friendly and
    Generalizable Ultrasound Image Segmentation." 2024.
    Upstream code: https://github.com/xianlin7/SAMUS

Architecture overview:
    - Lightweight SAM variant tailored for medical ultrasound, working at a
      small default input resolution of 256x256.
    - Encoder: SAM ViT-B (768-dim, 12-layer, patch size 16) is unavailable
      via timm out of the box, so we reuse the structurally-equivalent
      ``vit_base_patch16_224`` and inject a learnable *position-bias adapter*
      that lets the backbone adapt to non-224 inputs (the original SAMUS uses
      a similar adapter that shifts the SAM pos-embed for low-resolution
      ultrasound frames).
    - Decoder: a lightweight CNN that mirrors SAMUS's upsampling head, with
      four ConvTranspose stages (768 -> 384 -> 192 -> 96 -> num_classes) and
      BatchNorm + GELU between them. A final bilinear interpolation snaps the
      logits back to the input resolution if any padding is in effect.

Self-contained: only torch and timm are required.
"""
# Source: https://github.com/xianlin7/SAMUS

from __future__ import annotations

import os

# Limit huggingface_hub retry/timeout budgets so a network outage does not
# stall model construction for minutes. Must be set before importing timm.
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '3')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '5')

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from .sam_base import SAMBase, load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interpolate_pos_embed(pos_embed: torch.Tensor, num_prefix: int,
                           new_hw: tuple) -> torch.Tensor:
    """Bicubic-resample a 1-D positional embedding to a new (H, W) grid.

    ``pos_embed`` has shape (1, num_prefix + N, C) where the first
    ``num_prefix`` tokens (e.g. the CLS token) are kept untouched and the
    remaining N entries form a square grid that is interpolated to the
    requested (H, W) shape.
    """
    prefix = pos_embed[:, :num_prefix]
    grid = pos_embed[:, num_prefix:]
    N = grid.shape[1]
    C = grid.shape[-1]
    old = int(round(math.sqrt(N)))
    if old * old != N:
        raise ValueError(
            'pos_embed grid is not square: N=%d (sqrt=%.3f)' % (N, math.sqrt(N))
        )
    new_h, new_w = new_hw
    grid = grid.reshape(1, old, old, C).permute(0, 3, 1, 2)
    grid = F.interpolate(
        grid, size=(new_h, new_w), mode='bicubic', align_corners=False,
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_h * new_w, C)
    return torch.cat([prefix, grid], dim=1)


class _PositionBiasAdapter(nn.Module):
    """Small MLP that injects a learnable bias into the patch tokens.

    SAMUS notes that SAM's positional embeddings are tuned for 1024x1024
    inputs and degrade on 256x256 ultrasound frames. The adapter is a
    lightweight residual that lets the encoder shift its position-aware
    features without retraining the backbone.
    """

    def __init__(self, dim: int, mlp_ratio: float = 0.25):
        super().__init__()
        hidden = max(int(dim * mlp_ratio), 8)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        # Zero-init the output projection so the adapter starts as a no-op.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class _ViTBackbone(nn.Module):
    """SAM-style ViT-B/16 backbone built on top of timm's vit_base_patch16_224.

    We extract the patch-embed Conv2d, cls token, pos-embed, transformer
    blocks and final LayerNorm from the timm model, and run a custom forward
    that supports arbitrary input sizes by interpolating the positional
    embedding on the fly. A position-bias adapter is added on top of the
    pos-embedded tokens before they enter the transformer stack.
    """

    PATCH_SIZE = 16
    EMBED_DIM = 768

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        vit = load_with_ssl_fallback(
            timm.create_model,
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
        )

        # Patch embedding: keep only the Conv2d projection so we are free to
        # feed inputs of arbitrary (H, W) that are multiples of patch_size.
        self.proj = vit.patch_embed.proj
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed  # (1, 1 + 14*14, 768)
        self.pos_drop = getattr(vit, 'pos_drop', nn.Identity())
        self.blocks = vit.blocks
        self.norm = vit.norm

        # Trainable position-bias adapter inserted right after pos-embedding.
        self.pos_bias_adapter = _PositionBiasAdapter(self.EMBED_DIM)

    def forward(self, x: torch.Tensor):
        """Return ``(tokens_grid, Hp, Wp)`` where ``tokens_grid`` has shape
        ``(B, EMBED_DIM, Hp, Wp)`` and ``Hp = H / PATCH_SIZE`` (likewise Wp).
        ``x`` must already be padded to a multiple of ``PATCH_SIZE``.
        """
        B = x.shape[0]
        x = self.proj(x)  # (B, C, Hp, Wp)
        Hp, Wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)  # (B, Hp*Wp, C)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + Hp*Wp, C)

        pos = _interpolate_pos_embed(self.pos_embed, num_prefix=1,
                                     new_hw=(Hp, Wp))
        x = x + pos
        x = self.pos_bias_adapter(x)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Drop the CLS token and reshape back to a 2-D feature grid.
        x = x[:, 1:]
        x = x.transpose(1, 2).reshape(B, self.EMBED_DIM, Hp, Wp).contiguous()
        return x, Hp, Wp


class _UpBlock(nn.Module):
    """One stage of the lightweight CNN decoder.

    ConvTranspose2d (stride 2, kernel 2) -> BatchNorm2d -> GELU.
    The last decoder stage skips BN/GELU since it produces the logits.
    """

    def __init__(self, in_c: int, out_c: int, last: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        if last:
            self.bn = nn.Identity()
            self.act = nn.Identity()
        else:
            self.bn = nn.BatchNorm2d(out_c)
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.up(x)))


class _LightCNNDecoder(nn.Module):
    """4-stage transposed-conv decoder, total upsampling 16x."""

    def __init__(self, num_classes: int, dims=(768, 384, 192, 96)):
        super().__init__()
        c0, c1, c2, c3 = dims
        self.stage1 = _UpBlock(c0, c1)              # 1/16 -> 1/8
        self.stage2 = _UpBlock(c1, c2)              # 1/8  -> 1/4
        self.stage3 = _UpBlock(c2, c3)              # 1/4  -> 1/2
        self.stage4 = _UpBlock(c3, num_classes,     # 1/2  -> 1/1
                               last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class SAMUS(SAMBase):
    """SAMUS: SAM-style ViT-B encoder with a lightweight CNN decoder.

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation classes.
        img_size: nominal input spatial size (defaults to 256 to match the
            SAMUS paper; the forward pass also accepts other resolutions, padded
            internally to the patch size).
        pretrained: whether to load ImageNet-pretrained ViT-B/16 weights via
            timm. Falls back to random init when the download is unreachable.
    """

    _PATCH = 16

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 img_size: int = 256, pretrained: bool = True, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            pretrained=pretrained,
            **kwargs,
        )

        self.image_encoder = _ViTBackbone(in_channels=in_channels,
                                          pretrained=self._pretrained)
        # SAMUS is prompt-free; expose None so SAMBase.apply_freeze can
        # introspect uniformly across SAM-family models.
        self.prompt_encoder = None
        self.mask_decoder = _LightCNNDecoder(num_classes=num_classes,
                                             dims=(768, 384, 192, 96))

        self.apply_freeze()

    # ------------------------------------------------------------------
    def _pad_to_multiple(self, x: torch.Tensor, mult: int):
        H, W = x.shape[-2:]
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, self._PATCH)

        feat, _, _ = self.image_encoder(x_pad)
        logits = self.mask_decoder(feat)

        # The decoder upsamples exactly 16x, so logits should already match
        # x_pad's spatial size; bilinear is a safety net for odd cases.
        if logits.shape[-2:] != x_pad.shape[-2:]:
            logits = F.interpolate(
                logits, size=x_pad.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits
