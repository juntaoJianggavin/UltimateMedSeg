"""MedSAM – Segment Anything in Medical Images (Nature Communications 2024).

Reference: Jun Ma et al., "Segment Anything in Medical Images",
Nature Communications 2024. Code: https://github.com/bowang-lab/MedSAM

This is a *prompt-free* generalist segmentation variant: we keep MedSAM's
SAM ViT-B image encoder (loaded via timm; falls back to random init if the
pretrained download fails) and bypass the prompt encoder + mask decoder in
favour of a tiny conv decoder that upsamples the patch tokens back to the
full input resolution. This keeps the model self-contained (torch + timm
only) and trainable on any dataset without box/point prompts.

Standard interface:
    model = MedSAM(in_channels=3, num_classes=2, img_size=224)
    out = model(x)  # -> (B, num_classes, H, W)
"""
# Source: https://github.com/bowang-lab/MedSAM

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback


# ── lightweight conv decoder block ───────────────────────────────────────────
class _UpBlock(nn.Module):
    """ConvTranspose2d ×2 upsample + BN + GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.up(x)))


# ── encoder factory ──────────────────────────────────────────────────────────
def _build_vit_encoder(img_size: int, in_channels: int, pretrained: bool = True):
    """Build a SAM-like ViT-B image encoder via timm.

    timm's ``vit_base_patch16_224`` matches the SAM image encoder structure
    (12 layers, 768 dim, patch 16). With ``dynamic_img_size=True`` it can
    process arbitrary multiples of 16, which is what we need for 224/256/512.
    """
    import timm

    def _create(pretrained: bool):
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            dynamic_img_size=True,
            img_size=img_size,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )

    return load_with_ssl_fallback(_create, pretrained=pretrained)


# ── mask decoder (lightweight conv stack) ────────────────────────────────────
class _MedSAMMaskDecoder(nn.Module):
    """Four 2x ConvTranspose stages: 768 -> 256 -> 128 -> 64 -> num_classes."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.up1 = _UpBlock(embed_dim, 256)   # /16 → /8
        self.up2 = _UpBlock(256, 128)         # /8  → /4
        self.up3 = _UpBlock(128, 64)          # /4  → /2
        self.up4 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up1(x)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        return h


# ── main model ───────────────────────────────────────────────────────────────
class MedSAM(SAMBase):
    """MedSAM: SAM ViT-B encoder + lightweight conv decoder.

    Args:
        in_channels: number of input channels (default 3).
        num_classes: number of output segmentation classes (default 2).
        img_size:    nominal input resolution; the network is fully convolutional
                     w.r.t. multiples of 16 so other sizes are also supported via
                     dynamic positional-embedding interpolation.
        vit_variant: only "base" is implemented (kept for API parity).
    """

    PATCH = 16
    EMBED_DIM = 768

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        vit_variant: str = "base",
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            **kwargs,
        )
        if vit_variant != "base":
            warnings.warn(
                f"MedSAM: vit_variant='{vit_variant}' not implemented, "
                "falling back to ViT-B."
            )

        # SAM ViT-B image encoder.
        self.image_encoder = _build_vit_encoder(
            img_size=img_size,
            in_channels=in_channels,
            pretrained=self._pretrained,
        )
        self.num_prefix_tokens = int(getattr(self.image_encoder, "num_prefix_tokens", 1))

        # MedSAM is prompt-free; expose a None attribute so SAMBase.apply_freeze
        # can introspect uniformly across SAM-family models.
        self.prompt_encoder = None

        # Lightweight conv mask decoder.
        self.mask_decoder = _MedSAMMaskDecoder(self.EMBED_DIM, num_classes)

        self.apply_freeze()

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens reshaped to (B, C, H/16, W/16)."""
        B, _, H, W = x.shape
        tokens = self.image_encoder.forward_features(x)
        # Drop the cls/prefix token(s).
        if self.num_prefix_tokens > 0:
            tokens = tokens[:, self.num_prefix_tokens:, :]
        # (B, N, C) → (B, C, H/16, W/16)
        Hp, Wp = H // self.PATCH, W // self.PATCH
        expected = Hp * Wp
        if tokens.shape[1] != expected:
            # Some timm versions return ``(B, C, H', W')`` already; handle both.
            if tokens.dim() == 4:
                return tokens
            raise RuntimeError(
                f"MedSAM: unexpected token count {tokens.shape[1]} (expected {expected})."
            )
        feat = tokens.transpose(1, 2).reshape(B, self.EMBED_DIM, Hp, Wp).contiguous()
        return feat

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Pad to a multiple of PATCH (16); record original size for cropping.
        pad_h = (self.PATCH - H % self.PATCH) % self.PATCH
        pad_w = (self.PATCH - W % self.PATCH) % self.PATCH
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        feat = self._encode(x)            # (B, 768, H'/16, W'/16)
        h = self.mask_decoder(feat)       # (B, num_classes, H', W')

        # If padding changed the spatial size, bilinear-resize to the padded
        # input then crop back to the original H×W.
        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if pad_h or pad_w:
            h = h[..., :H, :W]
        return h


# Public alias matching the file name / arch key for downstream registries.
Medsam = MedSAM
