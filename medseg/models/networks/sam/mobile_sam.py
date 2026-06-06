"""MobileSAM – lightweight SAM variant with a TinyViT image encoder.

Reference:
    Chaoning Zhang, Dongshen Han, Yu Qiao, Jung Uk Kim, Sung-Ho Bae,
    Seungkyu Lee, Choong Seon Hong. "Faster Segment Anything: Towards
    Lightweight SAM for Mobile Applications" (2023).
    Upstream code: https://github.com/ChaoningZhang/MobileSAM

This medical-segmentation adaptation is *prompt-free*: the backbone is a
TinyViT-5M (or 11M as a fallback) image encoder loaded through timm, and a
lightweight transposed-convolution mask decoder maps the /32 feature map
back up to the input resolution. The model is self-contained — only torch
and timm are required — and supports inputs of arbitrary spatial size by
padding to the nearest multiple of 32 and cropping the logits at the end.

Standard interface:
    model = MobileSAM(in_channels=3, num_classes=2, img_size=224)
    out = model(x)  # -> (B, num_classes, H, W)
"""
# Source: https://github.com/ChaoningZhang/MobileSAM

from __future__ import annotations

import os

# Limit huggingface_hub retry/timeout budgets so a network outage does not
# stall model construction for minutes. Must be set before importing timm.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback


# ---------------------------------------------------------------------------
# TinyViT encoder wrapper
# ---------------------------------------------------------------------------
class _TinyViTEncoder(nn.Module):
    """Wraps a timm TinyViT model and exposes:

    * a forward that returns the /32 spatial feature map ``(B, C, H/32, W/32)``;
    * a flat ``blocks`` ``ModuleList`` so ``SAMBase.apply_freeze`` can apply
      ``unfreeze_last_n_blocks`` uniformly across SAM-family models.

    The encoder accepts any input spatial size that is a multiple of 32.
    """

    REDUCTION = 32

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        import timm

        model_name, embed_dim = "tiny_vit_5m_224", 320

        def _create(pretrained: bool):
            return timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
                in_chans=in_channels,
            )

        tv = load_with_ssl_fallback(_create, pretrained=pretrained)

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.backbone = tv

        # Build a flat list of transformer blocks so SAMBase.apply_freeze's
        # ``unfreeze_last_n_blocks`` knob works. We aggregate the per-stage
        # ``blocks`` Sequential's into a single ModuleList. The first stage of
        # TinyViT is a ConvLayer; its blocks are still nn.Modules and can be
        # safely unfrozen too.
        flat: List[nn.Module] = []
        if hasattr(tv, "stages"):
            for stage in tv.stages:
                stage_blocks = getattr(stage, "blocks", None)
                if stage_blocks is not None:
                    for blk in stage_blocks:
                        flat.append(blk)
        # Use ``add_module`` rather than assigning a fresh ModuleList of the
        # same parameters (which would double-count them in .parameters()).
        # Instead we keep an attribute holding *references* to the existing
        # block modules. SAMBase only iterates ``getattr(image_encoder, 'blocks')``
        # to flip ``requires_grad``, so a plain Python list of module refs is
        # sufficient and avoids parameter duplication.
        self._block_refs = flat

    # Expose as an attribute SAMBase can iterate: a plain list of modules. We
    # intentionally do NOT register this as a ModuleList to avoid
    # double-registration of parameters that already live under ``backbone``.
    @property
    def blocks(self):
        return self._block_refs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, embed_dim, H/32, W/32)`` features."""
        feat = self.backbone.forward_features(x)
        # timm TinyViT.forward_features returns (B, C, H', W'); some versions
        # may return (B, N, C) — handle both.
        if feat.dim() == 3:
            B, N, C = feat.shape
            import math

            side = int(round(math.sqrt(N)))
            if side * side != N:
                raise RuntimeError(
                    f"MobileSAM: cannot reshape TinyViT tokens of length {N} to a square grid."
                )
            feat = feat.transpose(1, 2).reshape(B, C, side, side).contiguous()
        return feat


# ---------------------------------------------------------------------------
# Lightweight mask decoder
# ---------------------------------------------------------------------------
class _UpBlock(nn.Module):
    """ConvTranspose2d (stride 2, kernel 2) + BatchNorm + GELU."""

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


class _MobileSAMMaskDecoder(nn.Module):
    """Four 2x ConvTranspose stages: 320 -> 128 -> 64 -> 32 -> num_classes.

    Net upsampling factor is 16x. Combined with the TinyViT /32 stride, the
    decoder produces logits at /2 of the (padded) input; the outer ``forward``
    bilinearly snaps them up to the original resolution and crops away any
    right/bottom padding.
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.up1 = _UpBlock(embed_dim, 128)   # /32 -> /16
        self.up2 = _UpBlock(128, 64)          # /16 -> /8
        self.up3 = _UpBlock(64, 32)           # /8  -> /4
        self.up4 = _UpBlock(32, num_classes,  # /4  -> /2
                            last=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class MobileSAM(SAMBase):
    """MobileSAM: TinyViT image encoder + lightweight conv mask decoder.

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation classes.
        img_size: nominal input spatial size (defaults to 224 to match the
            TinyViT-5M training resolution; arbitrary sizes are supported via
            padding to a multiple of 32).
        pretrained: load timm ImageNet weights for TinyViT when available;
            falls back to random init on download failure.
        pretrained_path: optional path to a local checkpoint (forwarded to
            ``SAMBase``; the encoder itself loads weights via timm).
        freeze_image_encoder / freeze_prompt_encoder / freeze_mask_decoder:
            standard SAMBase freezing knobs.
        unfreeze_last_n_blocks: re-enable gradients on the last N transformer
            blocks of the TinyViT encoder (useful when ``freeze_image_encoder``
            is True).
        inference_only: put the model in ``eval()`` mode and freeze every
            parameter.
    """

    PATCH = 32

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
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

        # ── encoder ─────────────────────────────────────────────────────────
        self.image_encoder = _TinyViTEncoder(
            in_channels=in_channels,
            pretrained=self._pretrained,
        )

        # ── prompt encoder (MobileSAM here is prompt-free for medical seg) ──
        self.prompt_encoder = None

        # ── mask decoder ────────────────────────────────────────────────────
        self.mask_decoder = _MobileSAMMaskDecoder(
            embed_dim=self.image_encoder.embed_dim,
            num_classes=num_classes,
        )

        # Optional local checkpoint (best-effort, never fatal).
        if pretrained_path:
            self._maybe_load_local_checkpoint(pretrained_path)

        self.apply_freeze()

    # ------------------------------------------------------------------
    def _maybe_load_local_checkpoint(self, path: str) -> None:
        if not os.path.isfile(path):
            warnings.warn(f"MobileSAM: pretrained_path '{path}' not found; skipping.")
            return
        try:
            sd = torch.load(path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            if missing or unexpected:
                warnings.warn(
                    f"MobileSAM: loaded '{path}' with "
                    f"{len(missing)} missing / {len(unexpected)} unexpected keys."
                )
        except Exception as e:  # pragma: no cover - best-effort
            warnings.warn(f"MobileSAM: failed to load checkpoint '{path}': {e}")

    # ------------------------------------------------------------------
    def _pad_to_multiple(self, x: torch.Tensor, mult: int):
        H, W = x.shape[-2:]
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, self.PATCH)

        feat = self.image_encoder(x_pad)        # (B, embed_dim, H'/32, W'/32)
        logits = self.mask_decoder(feat)        # (B, num_classes, H'/2, W'/2)

        # Snap to padded input size, then crop back to the original (H, W).
        if logits.shape[-2:] != x_pad.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x_pad.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits


# Public alias matching the arch_key for downstream registries.
Mobile_SAM = MobileSAM
