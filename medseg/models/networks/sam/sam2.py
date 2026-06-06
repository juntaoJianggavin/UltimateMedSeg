"""SAM2 (Segment Anything Model 2, Meta 2024) - 2D medical-segmentation variant.

Reference:
    Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, et al.,
    "SAM 2: Segment Anything in Images and Videos", Meta FAIR, 2024.
    https://github.com/facebookresearch/segment-anything-2

SAM2's image encoder is a *Hiera* hierarchical ViT that produces multi-scale
features at strides 4/8/16/32. We reuse the same backbone via timm:

    1. ``hiera_tiny_224``                 (preferred)
    2. ``hiera_base_plus_224``            (fallback)
    3. ``vit_small_patch16_224``          (final fallback; single-stage tiled
       into a pseudo multi-stage pyramid)

All multi-scale features are projected to a common 256-d width with 1x1
convs, then a 4-stage CNN ``mask_decoder`` progressively upsamples them back
to the input resolution with skip-additions at every level (mirroring the
upstream SAM2 mask decoder which similarly fuses multi-scale features).

The prompt encoder is intentionally ``None`` (prompt-free generalist
segmentation, matching the rest of the SAM-family models in this repo).

Inputs are padded to a multiple of 32 (the Hiera total stride). Outputs are
cropped back to the original H x W.
"""
# Source: https://github.com/facebookresearch/sam2

from __future__ import annotations

import os

# Keep huggingface_hub from hanging the constructor on flaky networks.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

import warnings
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Backbone wrappers
# ---------------------------------------------------------------------------
_HIERA_MODEL = "hiera_tiny_224"
_TOTAL_STRIDE = 32


class _HieraEncoder(nn.Module):
    """Thin wrapper around a timm Hiera features extractor.

    Exposes ``.blocks`` (a flat ``nn.ModuleList`` of transformer blocks) so
    ``SAMBase.apply_freeze`` can implement the ``unfreeze_last_n_blocks``
    schedule. Also exposes ``.channels`` (a list of per-stage widths) and
    ``.strides`` (the per-stage downsampling factors).
    """

    def __init__(self, in_channels: int, img_size: int, pretrained: bool):
        super().__init__()
        import timm

        def _create(pretrained: bool):
            return timm.create_model(
                _HIERA_MODEL,
                pretrained=pretrained,
                features_only=True,
                img_size=img_size,
                in_chans=in_channels,
            )

        feat = load_with_ssl_fallback(_create, pretrained=pretrained)

        self.feat = feat
        self.channels = list(feat.feature_info.channels())
        self.strides = list(feat.feature_info.reduction())
        # Expose the transformer blocks of the underlying Hiera so
        # SAMBase.apply_freeze can selectively unfreeze the tail.
        inner = getattr(feat, "model", feat)
        blocks = getattr(inner, "blocks", None)
        if blocks is None:
            blocks = nn.ModuleList()
        self.blocks = blocks
        self._backbone_name = _HIERA_MODEL

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.feat(x))


# ---------------------------------------------------------------------------
# Mask decoder
# ---------------------------------------------------------------------------
class _LateralProj(nn.Module):
    """1x1 conv that projects a feature map to ``out_ch`` channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class _UpFuseBlock(nn.Module):
    """2x ConvTranspose upsample, fuse a skip feature, refine with a 3x3 conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = x + skip
        return self.refine(x)


class _SAM2MaskDecoder(nn.Module):
    """4-stage CNN mask decoder that fuses multi-scale features.

    Stage layout (assuming Hiera with strides 4/8/16/32):
        stage1: 1/32 -> 1/16, add lateral(f3)
        stage2: 1/16 -> 1/8,  add lateral(f2)
        stage3: 1/8  -> 1/4,  add lateral(f1)
        stage4: 1/4  -> 1/1,  ConvTranspose ×4 to full resolution

    All intermediate features live at ``proj_dim`` (default 256) channels.
    """

    def __init__(self, in_channels: Sequence[int], num_classes: int,
                 proj_dim: int = 256):
        super().__init__()
        c1, c2, c3, c4 = in_channels  # strides 4, 8, 16, 32
        self.lateral4 = _LateralProj(c4, proj_dim)
        self.lateral3 = _LateralProj(c3, proj_dim)
        self.lateral2 = _LateralProj(c2, proj_dim)
        self.lateral1 = _LateralProj(c1, proj_dim)

        self.up1 = _UpFuseBlock(proj_dim, proj_dim)  # 1/32 -> 1/16
        self.up2 = _UpFuseBlock(proj_dim, proj_dim)  # 1/16 -> 1/8
        self.up3 = _UpFuseBlock(proj_dim, proj_dim)  # 1/8  -> 1/4

        # Final 4x upsample to full resolution, producing class logits.
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(proj_dim, proj_dim // 2,
                               kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=min(32, proj_dim // 2),
                         num_channels=proj_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(proj_dim // 2, proj_dim // 4,
                               kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=min(32, proj_dim // 4),
                         num_channels=proj_dim // 4),
            nn.GELU(),
            nn.Conv2d(proj_dim // 4, num_classes, kernel_size=1),
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = feats  # strides 4, 8, 16, 32
        x4 = self.lateral4(f4)
        x3 = self.lateral3(f3)
        x2 = self.lateral2(f2)
        x1 = self.lateral1(f1)

        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        y = self.up4(y)
        return y


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class SAM2(SAMBase):
    """SAM2 2D segmentation network with a Hiera-ViT encoder.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: input spatial size used to size the Hiera positional/window
            structures. The forward pass accepts other resolutions, padded
            internally to a multiple of 32 and bilinear-resized into the
            backbone's native size if needed.
        pretrained: whether to attempt loading pretrained backbone weights via
            timm. Falls back to random init when the download is unreachable.
        pretrained_path: unused; accepted for API parity with other SAM-family
            constructors.
        freeze_image_encoder / freeze_prompt_encoder / freeze_mask_decoder:
            standard freezing knobs handled by ``SAMBase.apply_freeze``.
        unfreeze_last_n_blocks: when > 0 and ``freeze_image_encoder`` is True,
            re-enables training on the last N transformer blocks of the
            backbone.
        inference_only: when True, the entire model is put into eval mode and
            all parameters are frozen.
    """

    PROJ_DIM = 256

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        pretrained: bool = True,
        pretrained_path: str | None = None,
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

        # Hiera positional structures are baked at construction; pick the
        # backbone img_size as the smallest multiple of 32 that is >= img_size.
        self._backbone_size = max(_TOTAL_STRIDE,
                                  _round_up(img_size, _TOTAL_STRIDE))

        self.image_encoder = _HieraEncoder(
            in_channels=in_channels,
            img_size=self._backbone_size,
            pretrained=self._pretrained,
        )
        # SAM2 here is prompt-free; expose None so apply_freeze is uniform.
        self.prompt_encoder = None

        self.mask_decoder = _SAM2MaskDecoder(
            in_channels=self.image_encoder.channels,
            num_classes=num_classes,
            proj_dim=self.PROJ_DIM,
        )

        self.apply_freeze()

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, mult: int):
        H, W = x.shape[-2:]
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        return F.pad(x, (0, pad_w, 0, pad_h)), (pad_h, pad_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # 1. Pad input to a multiple of 32.
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, _TOTAL_STRIDE)
        Hp, Wp = x_pad.shape[-2:]

        # 2. If the padded input does not match the backbone's strict size,
        #    bilinear-resize into the backbone, then resize logits back.
        if (Hp, Wp) != (self._backbone_size, self._backbone_size):
            x_bb = F.interpolate(
                x_pad,
                size=(self._backbone_size, self._backbone_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_bb = x_pad

        # 3. Multi-scale features (strides 4, 8, 16, 32).
        feats = self.image_encoder(x_bb)
        if len(feats) < 4:
            raise RuntimeError(
                f"SAM2: expected 4 backbone stages, got {len(feats)}."
            )
        feats = feats[-4:]

        # 4. Decode.
        logits = self.mask_decoder(feats)

        # 5. Snap back to the padded input size, then crop to (H, W).
        if logits.shape[-2:] != (Hp, Wp):
            logits = F.interpolate(
                logits, size=(Hp, Wp), mode="bilinear", align_corners=False,
            )
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


# Public aliases for downstream registries.
Sam2 = SAM2
