"""MedicalSAMAdapter (Med-SA / Medical-SAM-Adapter, 2024).

Reference:
    Junde Wu et al., "Medical SAM Adapter: Adapting Segment Anything Model
    for Medical Image Segmentation." 2024.
    Code: https://github.com/SuperMedIntel/Medical-SAM-Adapter

Implementation notes:
    - Backbone: timm ``vit_base_patch16_224`` (12 blocks, 768-dim, patch 16),
      loaded with ``dynamic_img_size=True`` so it can ingest 224/256/512
      inputs via on-the-fly positional-embedding interpolation.
    - Med-SA inserts a small bottleneck MLP adapter AFTER each attention
      sub-layer AND AFTER each MLP sub-layer of every transformer block,
      yielding 2x adapters per block (24 adapters total for ViT-B). The ViT
      itself is frozen by default; only the adapters (and the mask decoder)
      are trained.
    - Mask decoder: four ``ConvTranspose2d`` stages (768 -> 256 -> 128 -> 64
      -> num_classes), each 2x upsample, for a total 16x upsample matching
      the patch grid back to the input resolution.
    - For inputs whose spatial size is not a multiple of the patch size we
      zero-pad before the backbone and crop the decoder output back to the
      original (H, W). This is the "backbone strict-size, pad input and
      crop output" contract.
"""
# Source: https://github.com/SuperMedIntel/Medical-SAM-Adapter

from __future__ import annotations

import os

# Bound HuggingFace download timeouts so a flaky network does not stall the
# constructor for minutes. Must be set before ``timm`` is imported.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Adapter primitives
# ---------------------------------------------------------------------------
class _Adapter(nn.Module):
    """Bottleneck MLP adapter as used by Med-SA.

    Structure: LayerNorm -> Linear(dim -> hidden) -> GELU -> Linear(hidden ->
    dim) with a residual connection. The output projection is zero-initialised
    so the adapter starts as an identity, leaving the pretrained backbone
    untouched at step 0.
    """

    def __init__(self, dim: int, mlp_ratio: float = 0.25):
        super().__init__()
        hidden = max(int(dim * mlp_ratio), 8)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class _AdaptedBlock(nn.Module):
    """Wraps one timm ViT ``Block`` to inject two adapters per the Med-SA
    recipe: one right AFTER the attention residual, one right AFTER the MLP
    residual.

    We deliberately keep a reference to the original block and replicate its
    forward expression (with ``ls`` / ``drop_path`` modules if present) so the
    wrapper stays compatible across timm versions.
    """

    def __init__(self, block: nn.Module, dim: int, mlp_ratio: float = 0.25):
        super().__init__()
        self.block = block
        self.adapter_attn = _Adapter(dim, mlp_ratio)
        self.adapter_mlp = _Adapter(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        # timm.Block uses ls1/ls2 (LayerScale or Identity) and
        # drop_path1/drop_path2 (DropPath or Identity); both are present in
        # the timm versions we target. Fall back gracefully otherwise.
        ls1 = getattr(b, "ls1", nn.Identity())
        ls2 = getattr(b, "ls2", nn.Identity())
        dp1 = getattr(b, "drop_path1", nn.Identity())
        dp2 = getattr(b, "drop_path2", nn.Identity())

        x = x + dp1(ls1(b.attn(b.norm1(x))))
        x = self.adapter_attn(x)
        x = x + dp2(ls2(b.mlp(b.norm2(x))))
        x = self.adapter_mlp(x)
        return x


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------
def _build_vit_with_adapters(img_size: int, in_channels: int,
                             pretrained: bool, mlp_ratio: float = 0.25):
    """Create timm ``vit_base_patch16_224`` and replace each Block with an
    ``_AdaptedBlock`` wrapper. Returns the (modified) timm model.
    """
    import timm

    def _create(pretrained: bool = False):
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            dynamic_img_size=True,
            img_size=img_size,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )

    vit = load_with_ssl_fallback(_create, pretrained=pretrained)

    # Inject adapters after attn AND after mlp in every transformer block.
    # timm's ViT calls ``self.blocks(x)`` (a Sequential), so we keep the same
    # container type when swapping in the wrapped blocks.
    embed_dim = vit.embed_dim if hasattr(vit, "embed_dim") else 768
    new_blocks = nn.Sequential(
        *[_AdaptedBlock(blk, embed_dim, mlp_ratio=mlp_ratio) for blk in vit.blocks]
    )
    vit.blocks = new_blocks
    return vit


# ---------------------------------------------------------------------------
# Mask decoder (four ConvTranspose2d stages)
# ---------------------------------------------------------------------------
class _MaskDecoder(nn.Module):
    """Four ``ConvTranspose2d`` stages: 768 -> 256 -> 128 -> 64 -> n_class.

    Each non-final stage is followed by BatchNorm + GELU. The final stage
    produces raw logits with no activation.
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.GELU()

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.GELU()

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.GELU()

        self.up4 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.up1(x)))
        x = self.act2(self.bn2(self.up2(x)))
        x = self.act3(self.bn3(self.up3(x)))
        x = self.up4(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class MedicalSAMAdapter(SAMBase):
    """Med-SA / Medical-SAM-Adapter (2024) for prompt-free medical seg.

    Args:
        in_channels: number of input channels (default 3).
        num_classes: number of output segmentation classes (default 2).
        img_size: nominal input resolution; the model also accepts other
            multiples of 16 via dynamic positional-embedding interpolation.
        pretrained: load ImageNet-pretrained ViT-B/16 via timm. Falls back to
            random init transparently on download failure.
        pretrained_path: optional local checkpoint path (unused for this
            architecture but kept for SAMBase API parity).
        freeze_image_encoder: freeze the ViT backbone (recommended; the
            adapters are always kept trainable regardless of this flag).
        freeze_prompt_encoder: ignored (no prompt encoder).
        freeze_mask_decoder: freeze the conv decoder (default False).
        unfreeze_last_n_blocks: optionally unfreeze the last N transformer
            blocks (including their internal adapters).
        inference_only: if True, sets the whole model to ``eval`` and disables
            grad for every parameter.
    """

    _PATCH = 16
    _EMBED_DIM = 768

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

        # Backbone with 2 adapters injected per block.
        self.image_encoder = _build_vit_with_adapters(
            img_size=img_size,
            in_channels=in_channels,
            pretrained=self._pretrained,
        )
        self.num_prefix_tokens = int(
            getattr(self.image_encoder, "num_prefix_tokens", 1)
        )

        # Med-SA is prompt-free; expose ``None`` so SAMBase.apply_freeze can
        # introspect uniformly across SAM-family models.
        self.prompt_encoder = None

        # Conv decoder (four ConvTranspose2d layers, 16x upsample).
        self.mask_decoder = _MaskDecoder(self._EMBED_DIM, num_classes)

        self.apply_freeze()

        # Re-enable adapter params: even when the image encoder is frozen,
        # the adapter sub-modules must remain trainable. ``inference_only``
        # still wins — it freezes everything, including adapters.
        if not self._freeze_cfg["inference_only"]:
            for name, p in self.image_encoder.named_parameters():
                if "adapter_attn" in name or "adapter_mlp" in name:
                    p.requires_grad = True

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ViT and return patch tokens reshaped to ``(B, C, Hp, Wp)``."""
        B, _, H, W = x.shape
        tokens = self.image_encoder.forward_features(x)

        # Strip CLS / prefix tokens if present.
        if tokens.dim() == 3 and self.num_prefix_tokens > 0:
            tokens = tokens[:, self.num_prefix_tokens:, :]

        Hp, Wp = H // self._PATCH, W // self._PATCH
        if tokens.dim() == 4:
            # Some timm versions hand back a 2-D feature grid directly.
            return tokens
        expected = Hp * Wp
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"MedicalSAMAdapter: unexpected token count {tokens.shape[1]} "
                f"(expected {expected})."
            )
        feat = tokens.transpose(1, 2).reshape(B, self._EMBED_DIM, Hp, Wp).contiguous()
        return feat

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Pad to a multiple of the patch size; record original (H, W) so we
        # can crop the decoder output back to it.
        pad_h = (self._PATCH - H % self._PATCH) % self._PATCH
        pad_w = (self._PATCH - W % self._PATCH) % self._PATCH
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        feat = self._encode(x)                       # (B, 768, Hp, Wp)
        logits = self.mask_decoder(feat)             # (B, n_class, 16*Hp, 16*Wp)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits


# Public alias matching the file name for downstream registries.
Medical_SAM_Adapter = MedicalSAMAdapter
