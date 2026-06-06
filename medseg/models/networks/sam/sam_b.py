"""SAM ViT-B: canonical Meta SAM-family backbone for medical segmentation.

Reference:
    Alexander Kirillov et al., "Segment Anything," 2023.
    https://github.com/facebookresearch/segment-anything

This module assembles a SAM-style ViT-B image encoder (768-dim, 12-block,
patch-16) on top of timm's ``vit_base_patch16_224`` via
:func:`load_with_ssl_fallback`. The original Meta SAM exposes three modules:
``image_encoder``, ``prompt_encoder`` and ``mask_decoder``. In this prompt-free
segmentation variant we keep the ViT-B image encoder, set
``prompt_encoder = None`` and replace the original mask decoder with a
lightweight four-stage transposed-conv stack (768 -> 256 -> 128 -> 64
-> num_classes) with BatchNorm + GELU, exactly as in the spec.

The backbone is strict-size with respect to its patch grid: inputs are
zero-padded so their height and width are multiples of 16, and the logits are
cropped back to the original spatial size.
"""
# Source: https://github.com/facebookresearch/segment-anything

from __future__ import annotations

import os

# Bound HF Hub timeouts so an offline / blocked environment can't stall model
# construction. Must be set before timm imports huggingface_hub internally.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interpolate_pos_embed(pos_embed: torch.Tensor, num_prefix: int,
                           new_hw: tuple) -> torch.Tensor:
    """Bicubic-resample a 1-D positional embedding to a new (H, W) grid."""
    prefix = pos_embed[:, :num_prefix]
    grid = pos_embed[:, num_prefix:]
    N = grid.shape[1]
    C = grid.shape[-1]
    old = int(round(math.sqrt(N)))
    if old * old != N:
        raise ValueError(
            "pos_embed grid is not square: N=%d (sqrt=%.3f)" % (N, math.sqrt(N))
        )
    new_h, new_w = new_hw
    grid = grid.reshape(1, old, old, C).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new_h, new_w), mode="bicubic",
                         align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_h * new_w, C)
    return torch.cat([prefix, grid], dim=1)


# ---------------------------------------------------------------------------
# Image encoder: SAM ViT-B/16 (built on timm's vit_base_patch16_224)
# ---------------------------------------------------------------------------
class _SAMImageEncoderViTB(nn.Module):
    """Canonical SAM-style ViT-B/16 image encoder.

    timm does not ship Meta's SAM weights, so we reuse the structurally
    identical ``vit_base_patch16_224`` (12 layers, 768-dim, patch 16) and
    interpolate the position embedding on the fly. The forward returns a
    spatial feature map of shape ``(B, 768, H/16, W/16)``.
    """

    PATCH_SIZE = 16
    EMBED_DIM = 768

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        import timm

        def _create(pretrained: bool):
            return timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=0,
                in_chans=in_channels,
            )

        vit = load_with_ssl_fallback(_create, pretrained=pretrained)

        # Keep only the Conv2d patch projection so we can feed arbitrary
        # (H, W) inputs that are multiples of PATCH_SIZE.
        self.proj = vit.patch_embed.proj
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed  # (1, 1 + 14*14, 768)
        self.pos_drop = getattr(vit, "pos_drop", nn.Identity())
        self.blocks = vit.blocks         # 12 transformer blocks
        self.norm = vit.norm             # final LayerNorm

        self.num_prefix_tokens = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x)                          # (B, 768, Hp, Wp)
        Hp, Wp = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)          # (B, Hp*Wp, 768)

        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)            # (B, 1 + Hp*Wp, 768)

        pos = _interpolate_pos_embed(self.pos_embed, num_prefix=1,
                                     new_hw=(Hp, Wp))
        x = x + pos
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Drop the CLS token and reshape to a 2-D feature grid.
        x = x[:, 1:]
        x = x.transpose(1, 2).reshape(B, self.EMBED_DIM, Hp, Wp).contiguous()
        return x


# ---------------------------------------------------------------------------
# Mask decoder: four 2x ConvTranspose stages (BN + GELU between stages).
# ---------------------------------------------------------------------------
class _UpBlock(nn.Module):
    """ConvTranspose2d (stride 2, kernel 2) + BatchNorm2d + GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.up(x)))


class _SAMMaskDecoder(nn.Module):
    """Four 2x ConvTranspose stages: 768 -> 256 -> 128 -> 64 -> num_classes.

    Each intermediate stage applies BatchNorm + GELU; the final stage emits
    raw logits (no normalization / activation).
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.up1 = _UpBlock(embed_dim, 256)   # /16 -> /8
        self.up2 = _UpBlock(256, 128)         # /8  -> /4
        self.up3 = _UpBlock(128, 64)          # /4  -> /2
        self.up4 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class SAMViTBase(SAMBase):
    """Canonical Meta SAM ViT-B segmentation network (prompt-free).

    The model exposes the three canonical SAM submodules
    (``image_encoder``, ``prompt_encoder``, ``mask_decoder``) so freeze
    configuration on :class:`SAMBase` applies uniformly. ``prompt_encoder``
    is ``None`` because this variant runs without box/point prompts.
    """

    PATCH = 16
    EMBED_DIM = 768

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        pretrained: bool = True,
        pretrained_path: str = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        unfreeze_last_n_blocks: int = 0,
        inference_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            freeze_image_encoder=freeze_image_encoder,
            freeze_prompt_encoder=freeze_prompt_encoder,
            freeze_mask_decoder=freeze_mask_decoder,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            inference_only=inference_only,
        )

        # 1) Image encoder — SAM ViT-B/16 via timm.
        self.image_encoder = _SAMImageEncoderViTB(
            in_channels=in_channels,
            pretrained=self._pretrained,
        )

        # 2) Prompt encoder — not used in this prompt-free variant.
        self.prompt_encoder = None

        # 3) Mask decoder — four 2x ConvTranspose stages.
        self.mask_decoder = _SAMMaskDecoder(
            embed_dim=self.EMBED_DIM, num_classes=num_classes,
        )

        # Optional local checkpoint override.
        if pretrained_path:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                missing, unexpected = self.load_state_dict(state, strict=False)
                if missing or unexpected:
                    warnings.warn(
                        "SAMViTBase: loaded %s with missing=%d unexpected=%d" % (
                            pretrained_path, len(missing), len(unexpected),
                        )
                    )
            except Exception as e:  # pragma: no cover - defensive
                warnings.warn(
                    "SAMViTBase: failed to load pretrained_path=%s (%s)" % (
                        pretrained_path, e,
                    )
                )

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

        # Backbone is strict patch-aligned: pad to a multiple of 16.
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, self.PATCH)

        feat = self.image_encoder(x_pad)          # (B, 768, H'/16, W'/16)
        logits = self.mask_decoder(feat)          # (B, num_classes, H', W')

        # The decoder upsamples exactly 16x, so logits should already match
        # x_pad's spatial size; bilinear is a defensive safety net.
        if logits.shape[-2:] != x_pad.shape[-2:]:
            logits = F.interpolate(
                logits, size=x_pad.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        # Crop back to the original input spatial size.
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits


# Public alias matching the arch_key for downstream registries.
Sam_b = SAMViTBase
