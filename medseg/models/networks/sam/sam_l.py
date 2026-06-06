"""SAMViTLarge — SAM-style segmentation model with a ViT-L/16 image encoder.

Encoder:
    timm ``vit_large_patch16_224`` (1024-dim, 24 transformer blocks, patch 16).
    Created with ``dynamic_img_size=True`` so it can ingest arbitrary input
    sizes that are multiples of the patch size (we pad to the nearest patch
    multiple in ``forward`` and crop the output back).

Mask decoder:
    Four ConvTranspose2d stages with BatchNorm + GELU between them,
    producing 16x upsampling:

        1024 -> 384 -> 192 -> 96 -> num_classes

The model is prompt-free (``prompt_encoder = None``) so it can be trained on
any 2-D segmentation dataset without box/point prompts. It inherits from
``SAMBase`` for uniform freeze / inference-only handling.
"""
# Source: https://github.com/facebookresearch/segment-anything

from __future__ import annotations

import os

# Bound huggingface_hub timeouts so an offline run does not stall in timm.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Helper building blocks (underscore-prefixed; not part of the public API).
# ---------------------------------------------------------------------------
class _UpBlock(nn.Module):
    """ConvTranspose2d (stride 2) followed by optional BN + GELU."""

    def __init__(self, in_ch: int, out_ch: int, last: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        if last:
            self.bn = nn.Identity()
            self.act = nn.Identity()
        else:
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.up(x)))


class _MaskDecoder(nn.Module):
    """4-stage transposed-conv decoder: 1024 -> 384 -> 192 -> 96 -> C."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.up1 = _UpBlock(embed_dim, 384)                  # /16 -> /8
        self.up2 = _UpBlock(384, 192)                        # /8  -> /4
        self.up3 = _UpBlock(192, 96)                         # /4  -> /2
        self.up4 = _UpBlock(96, num_classes, last=True)      # /2  -> /1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


def _build_vit_l_encoder(img_size: int, in_channels: int, pretrained: bool):
    """Instantiate timm's vit_large_patch16_224 with dynamic-size support."""
    import timm

    def _create(pretrained: bool):
        return timm.create_model(
            "vit_large_patch16_224",
            pretrained=pretrained,
            dynamic_img_size=True,
            img_size=img_size,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )

    return load_with_ssl_fallback(_create, pretrained=pretrained)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class SAMViTLarge(SAMBase):
    """SAM-style segmentation network with a ViT-L/16 image encoder.

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation classes.
        img_size:    nominal input spatial size; ``forward`` also accepts any
            other size and pads internally to the nearest multiple of 16.
        pretrained:  load ImageNet ViT-L/16 weights via timm (falls back to
            random init if offline).
        pretrained_path: optional path to a local checkpoint (.pt/.pth).
        freeze_image_encoder / freeze_prompt_encoder / freeze_mask_decoder:
            standard SAMBase freeze controls.
        unfreeze_last_n_blocks: re-enable grads on the last N encoder blocks
            even when the encoder is frozen.
        inference_only: full eval/no-grad mode.
    """

    _PATCH = 16
    _EMBED_DIM = 1024

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

        # Image encoder: timm ViT-L/16 (1024-dim, 24 blocks).
        self.image_encoder = _build_vit_l_encoder(
            img_size=img_size,
            in_channels=in_channels,
            pretrained=self._pretrained,
        )
        self._num_prefix_tokens = int(
            getattr(self.image_encoder, "num_prefix_tokens", 1)
        )

        # Prompt-free generalist seg variant.
        self.prompt_encoder = None

        # 4x ConvTranspose decoder: 1024 -> 384 -> 192 -> 96 -> num_classes.
        self.mask_decoder = _MaskDecoder(self._EMBED_DIM, num_classes)

        # Optional local checkpoint override.
        if pretrained_path:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                missing, unexpected = self.load_state_dict(state, strict=False)
                if missing or unexpected:
                    warnings.warn(
                        "SAMViTLarge: loaded %s with missing=%d unexpected=%d"
                        % (pretrained_path, len(missing), len(unexpected))
                    )
            except Exception as e:
                warnings.warn(
                    "SAMViTLarge: failed to load %s (%s)" % (pretrained_path, e)
                )

        self.apply_freeze()

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ViT-L encoder and return a (B, C, Hp, Wp) feature grid.

        ``x`` must already be padded to a multiple of ``_PATCH``.
        """
        B, _, H, W = x.shape
        Hp, Wp = H // self._PATCH, W // self._PATCH

        tokens = self.image_encoder.forward_features(x)

        # timm may return either (B, N, C) tokens or an already-shaped (B, C, H, W)
        # feature map depending on version / pooling settings.
        if tokens.dim() == 4:
            return tokens

        if self._num_prefix_tokens > 0 and tokens.shape[1] == Hp * Wp + self._num_prefix_tokens:
            tokens = tokens[:, self._num_prefix_tokens:, :]

        if tokens.shape[1] != Hp * Wp:
            raise RuntimeError(
                "SAMViTLarge: unexpected token count %d (expected %d)"
                % (tokens.shape[1], Hp * Wp)
            )

        feat = (
            tokens.transpose(1, 2)
            .reshape(B, self._EMBED_DIM, Hp, Wp)
            .contiguous()
        )
        return feat

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Pad input to the nearest multiple of the patch size, remembering the
        # original size so we can crop the final logits.
        pad_h = (self._PATCH - H % self._PATCH) % self._PATCH
        pad_w = (self._PATCH - W % self._PATCH) % self._PATCH
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        feat = self._encode(x)                # (B, 1024, H'/16, W'/16)
        logits = self.mask_decoder(feat)      # (B, num_classes, H', W')

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits


# Convenience alias matching the file name / arch key.
SamL = SAMViTLarge
