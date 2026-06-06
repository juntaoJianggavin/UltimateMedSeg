"""AutoSAM (2024) — auto-prompted SAM segmentation network.

Reference:
    Tal Shaharabany et al., "AutoSAM: Adapting SAM to Medical Images by
    Overloading the Prompt Encoder," 2024.

AutoSAM replaces SAM's hand-crafted box/point prompt encoder with a small
image-conditioned prompt encoder (a tiny CNN) that produces a learned prompt
embedding from the input image itself. The mask decoder cross-attends the
learned prompt to the image features and upsamples them through a four-stage
ConvTranspose2d stack to produce dense per-class logits.

Layout (mirrors the canonical SAM submodule split so :class:`SAMBase`
freeze configuration applies uniformly):

* ``image_encoder``  — timm ``vit_base_patch16_224`` (768-dim, 12-block,
  patch 16). Built strict-size at ``img_size``; inputs that do not match are
  zero-padded (or resized if larger) and the output is cropped back.
* ``prompt_encoder`` — 3-stage CNN (3->32->64->128) + global average pool
  + ``Linear(128, 768)`` producing one image-conditioned prompt token.
* ``mask_decoder``   — cross-attention layer (prompt queries image tokens)
  to inject prompt information into the spatial features, followed by four
  2x ConvTranspose stages (768 -> 256 -> 128 -> 64 -> num_classes) that
  upsample the /16 feature grid back to full resolution.
"""
# Source: https://github.com/talshaharabany/AutoSAM

from __future__ import annotations

import os

# Bound HF Hub timeouts so an offline / blocked environment can't stall model
# construction. Must be set before timm imports huggingface_hub internally.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Image encoder: SAM-style ViT-B/16 via timm.
# ---------------------------------------------------------------------------
def _build_vit_encoder(img_size: int, in_channels: int, pretrained: bool):
    """Build timm's ``vit_base_patch16_224`` at the requested ``img_size``.

    The encoder is *strict-size*: it expects inputs of exactly
    ``(img_size, img_size)``. The caller is responsible for padding / cropping
    inputs to satisfy that constraint.
    """
    import timm

    def _create(pretrained: bool):
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )

    return load_with_ssl_fallback(_create, pretrained=pretrained)


# ---------------------------------------------------------------------------
# Prompt encoder: image-conditioned learned prompt embedding.
# ---------------------------------------------------------------------------
class _AutoPromptEncoder(nn.Module):
    """Small CNN: 3 conv stages (in_ch -> 32 -> 64 -> 128) + GAP + Linear -> embed_dim.

    Produces one prompt token of shape ``(B, embed_dim)`` conditioned on the
    full input image — i.e. the model "prompts itself" instead of relying on
    user-supplied points / boxes.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)               # (B, 128)
        return self.fc(x)                         # (B, embed_dim)


# ---------------------------------------------------------------------------
# Mask decoder: cross-attention + 4-stage ConvTranspose upsampler.
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


class _AutoSAMMaskDecoder(nn.Module):
    """Cross-attend the learned prompt token to the /16 image feature map and
    upsample through four 2x ConvTranspose stages.

    Cross attention: the prompt token serves as the query and the flattened
    image tokens as keys/values. The attended prompt is broadcast and added
    back to every image token (LayerNorm + residual) so the spatial features
    are modulated by the image-conditioned prompt before upsampling.
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 2, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        self.up1 = _UpBlock(embed_dim, 256)       # /16 -> /8
        self.up2 = _UpBlock(256, 128)             # /8  -> /4
        self.up3 = _UpBlock(128, 64)              # /4  -> /2
        self.up4 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, feat: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat:   (B, C, Hp, Wp) image feature map from the ViT.
            prompt: (B, C) image-conditioned prompt token.
        Returns:
            (B, num_classes, Hp*16, Wp*16) per-class logits.
        """
        B, C, Hp, Wp = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()   # (B, N, C)
        q = self.norm_q(prompt.unsqueeze(1))                    # (B, 1, C)
        kv = self.norm_kv(tokens)                               # (B, N, C)

        attended, _ = self.cross_attn(q, kv, kv)                # (B, 1, C)
        # Broadcast the attended prompt over every spatial token (residual).
        tokens = self.norm_out(tokens + attended)               # (B, N, C)

        feat = tokens.transpose(1, 2).reshape(B, C, Hp, Wp).contiguous()

        feat = self.up1(feat)
        feat = self.up2(feat)
        feat = self.up3(feat)
        feat = self.up4(feat)
        return feat


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class AutoSAM(SAMBase):
    """AutoSAM — auto-prompted SAM segmentation network.

    The canonical SAM submodule names (``image_encoder``, ``prompt_encoder``,
    ``mask_decoder``) are preserved so that :class:`SAMBase`'s freeze
    configuration (``freeze_image_encoder``, ``freeze_prompt_encoder``,
    ``freeze_mask_decoder``, ``unfreeze_last_n_blocks``, ``inference_only``)
    applies uniformly across the SAM-family models.
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

        # 1) Image encoder — timm vit_base_patch16_224, strict-size at img_size.
        self.image_encoder = _build_vit_encoder(
            img_size=img_size,
            in_channels=in_channels,
            pretrained=self._pretrained,
        )
        self.num_prefix_tokens = int(
            getattr(self.image_encoder, "num_prefix_tokens", 1)
        )

        # 2) Prompt encoder — image-conditioned learned prompt embedding.
        self.prompt_encoder = _AutoPromptEncoder(
            in_channels=in_channels, embed_dim=self.EMBED_DIM,
        )

        # 3) Mask decoder — cross-attention + four 2x ConvTranspose stages.
        self.mask_decoder = _AutoSAMMaskDecoder(
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
                        "AutoSAM: loaded %s with missing=%d unexpected=%d" % (
                            pretrained_path, len(missing), len(unexpected),
                        )
                    )
            except Exception as e:  # pragma: no cover - defensive
                warnings.warn(
                    "AutoSAM: failed to load pretrained_path=%s (%s)" % (
                        pretrained_path, e,
                    )
                )

        self.apply_freeze()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fit_to_encoder(self, x: torch.Tensor):
        """Pad (or resize) ``x`` so spatial dims equal ``self.img_size``.

        timm's ``vit_base_patch16_224`` is strict-size with respect to the
        ``img_size`` it was constructed with. We zero-pad inputs that are
        smaller and bilinear-resize inputs that are larger; in both cases we
        return enough metadata to undo the transformation on the output side.
        """
        H, W = x.shape[-2:]
        S = self.img_size
        if H == S and W == S:
            return x, ("none", H, W)
        if H <= S and W <= S:
            pad_h = S - H
            pad_w = S - W
            x = F.pad(x, (0, pad_w, 0, pad_h))
            return x, ("pad", H, W)
        # At least one dim exceeds img_size: bilinear-resize to (S, S).
        x = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False)
        return x, ("resize", H, W)

    def _restore_output(self, logits: torch.Tensor, info) -> torch.Tensor:
        mode, H, W = info
        if mode == "none":
            return logits
        if mode == "pad":
            # Logits are at the padded resolution: crop the bottom-right pad.
            return logits[..., :H, :W]
        # mode == "resize"
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False,
            )
        return logits

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ViT and reshape patch tokens to (B, C, H/16, W/16)."""
        B, _, H, W = x.shape
        tokens = self.image_encoder.forward_features(x)
        if tokens.dim() == 4:
            return tokens
        if self.num_prefix_tokens > 0:
            tokens = tokens[:, self.num_prefix_tokens:, :]
        Hp, Wp = H // self.PATCH, W // self.PATCH
        if tokens.shape[1] != Hp * Wp:
            raise RuntimeError(
                "AutoSAM: unexpected token count %d (expected %d)." % (
                    tokens.shape[1], Hp * Wp,
                )
            )
        return tokens.transpose(1, 2).reshape(B, self.EMBED_DIM, Hp, Wp).contiguous()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]

        # Strict-size backbone: pad / resize to (img_size, img_size).
        x_enc, info = self._fit_to_encoder(x)

        # Image-conditioned prompt token (uses the encoder-aligned input so
        # the prompt sees what the encoder sees).
        prompt = self.prompt_encoder(x_enc)               # (B, EMBED_DIM)

        # ViT features at /16.
        feat = self._encode_image(x_enc)                  # (B, 768, S/16, S/16)

        # Cross-attend prompt to image and upsample to encoder resolution.
        logits = self.mask_decoder(feat, prompt)          # (B, num_classes, S, S)

        # Safety: defensively match encoder spatial size before restoring.
        if logits.shape[-2:] != x_enc.shape[-2:]:
            logits = F.interpolate(
                logits, size=x_enc.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        return self._restore_output(logits, info)


# Public alias matching the arch_key for downstream registries.
Auto_sam = AutoSAM
