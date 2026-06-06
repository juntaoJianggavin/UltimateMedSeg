"""SAMed — LoRA-adapted SAM ViT-B for medical segmentation.

Reference:
    Kaidong Zhang and Dong Liu, "Customized Segment Anything Model for Medical
    Image Segmentation," 2023.
    https://arxiv.org/abs/2304.13785

SAMed freezes the ViT-B/16 image encoder and injects low-rank adaptation
(LoRA) updates into the QKV projection of every transformer block. The
original ViT weights stay frozen at training time; only the LoRA matrices
(plus the mask decoder) receive gradients. As in the canonical SAM-family
medical setup we build on timm's structurally identical
``vit_base_patch16_224``, set ``prompt_encoder = None`` (prompt-free), and
replace SAM's mask decoder with a lightweight stack of four 2x
``ConvTranspose2d`` stages (768 -> 256 -> 128 -> 64 -> num_classes) with
BatchNorm + GELU between stages.

The backbone is strict patch-aligned: inputs are zero-padded so their height
and width are multiples of 16, and the logits are cropped back to the
original spatial size.
"""
# Source: https://github.com/hitachinsk/SAMed

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
# LoRA wrapper for the QKV projection.
# ---------------------------------------------------------------------------
class _LoRALinear(nn.Module):
    """LoRA wrapper around a (frozen) ``nn.Linear``.

    Forward computes ``base(x) + (x @ A^T) @ B^T``, where ``A`` is
    ``(rank, in_features)`` and ``B`` is ``(out_features, rank)``. Only the
    LoRA matrices ``A`` and ``B`` receive gradients; the wrapped ``base``
    linear is frozen.

    For SAMed we apply this to the fused QKV projection of every block, which
    is the standard placement from the LoRA paper (Hu et al., 2021): the
    update is restricted to the Q and V slices of the output, while the K
    slice is left untouched (B has zero rows for K). This matches the
    behaviour described in the SAMed paper.
    """

    def __init__(self, base: nn.Linear, rank: int = 4,
                 alpha: float = 1.0, apply_to=("q", "v")):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA target must be nn.Linear, got %s" % type(base))
        out_features = base.out_features
        in_features = base.in_features
        if out_features % 3 != 0:
            raise ValueError(
                "Expected fused QKV linear with out_features divisible by 3, "
                "got %d" % out_features
            )

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.in_features = in_features
        self.out_features = out_features

        head_dim = out_features // 3
        self.head_dim = head_dim
        self._apply_to = tuple(apply_to)

        # One (A, B) pair per targeted projection slice (Q, K, V).
        self._slot_index = {"q": 0, "k": 1, "v": 2}
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in self._apply_to:
            if name not in self._slot_index:
                raise ValueError("Unknown LoRA slot %r" % name)
            A = nn.Parameter(torch.empty(self.rank, in_features))
            B = nn.Parameter(torch.zeros(head_dim, self.rank))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.lora_A[name] = A
            self.lora_B[name] = B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self._apply_to:
            return out

        # Build a sparse delta over the full out_features dimension by writing
        # each (A, B) update into its Q/K/V slice. This avoids materialising a
        # dense (out_features, in_features) tensor.
        # x shape: (..., in_features)
        delta = torch.zeros_like(out)
        for name in self._apply_to:
            A = self.lora_A[name]                       # (rank, in_features)
            B = self.lora_B[name]                       # (head_dim, rank)
            inter = F.linear(x, A)                      # (..., rank)
            update = F.linear(inter, B) * self.scaling  # (..., head_dim)
            idx = self._slot_index[name]
            start = idx * self.head_dim
            stop = start + self.head_dim
            delta[..., start:stop] = delta[..., start:stop] + update
        return out + delta


# ---------------------------------------------------------------------------
# Image encoder: SAM ViT-B/16 with LoRA-wrapped QKV in every block.
# ---------------------------------------------------------------------------
class _SAMedImageEncoderViTB(nn.Module):
    """SAM-style ViT-B/16 image encoder with LoRA QKV adapters.

    The forward returns a spatial feature map of shape
    ``(B, 768, H/16, W/16)``.
    """

    PATCH_SIZE = 16
    EMBED_DIM = 768

    def __init__(self, in_channels: int = 3, pretrained: bool = True,
                 lora_rank: int = 4):
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

        self.proj = vit.patch_embed.proj
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed  # (1, 1 + 14*14, 768)
        self.pos_drop = getattr(vit, "pos_drop", nn.Identity())
        self.blocks = vit.blocks         # 12 transformer blocks
        self.norm = vit.norm             # final LayerNorm

        self.num_prefix_tokens = 1
        self.lora_rank = int(lora_rank)

        # Inject LoRA wrappers around each block's fused QKV linear.
        for blk in self.blocks:
            attn = getattr(blk, "attn", None)
            if attn is None or not hasattr(attn, "qkv"):
                continue
            attn.qkv = _LoRALinear(attn.qkv, rank=self.lora_rank)

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


class _SAMedMaskDecoder(nn.Module):
    """Four 2x ConvTranspose stages: 768 -> 256 -> 128 -> 64 -> num_classes."""

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
class SAMed(SAMBase):
    """SAMed: SAM ViT-B/16 with LoRA adapters in every block's QKV.

    Exposes the three canonical SAM submodules (``image_encoder``,
    ``prompt_encoder``, ``mask_decoder``) so freeze configuration on
    :class:`SAMBase` applies uniformly. ``prompt_encoder`` is ``None`` for
    this prompt-free segmentation variant.

    When the image encoder is frozen (the default, matching the SAMed
    recipe), the LoRA parameters injected into each block's QKV remain
    trainable because their wrapping ``_LoRALinear`` is re-enabled after the
    base freeze pass; this preserves the SAMed training contract.
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
        lora_rank: int = 4,
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

        self.lora_rank = int(lora_rank)

        # 1) Image encoder — SAM ViT-B/16 via timm, with LoRA QKV adapters.
        self.image_encoder = _SAMedImageEncoderViTB(
            in_channels=in_channels,
            pretrained=self._pretrained,
            lora_rank=self.lora_rank,
        )

        # 2) Prompt encoder — not used in this prompt-free variant.
        self.prompt_encoder = None

        # 3) Mask decoder — four 2x ConvTranspose stages.
        self.mask_decoder = _SAMedMaskDecoder(
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
                        "SAMed: loaded %s with missing=%d unexpected=%d" % (
                            pretrained_path, len(missing), len(unexpected),
                        )
                    )
            except Exception as e:  # pragma: no cover - defensive
                warnings.warn(
                    "SAMed: failed to load pretrained_path=%s (%s)" % (
                        pretrained_path, e,
                    )
                )

        self.apply_freeze()

        # SAMed contract: ViT weights are frozen, but LoRA matrices train.
        # apply_freeze() walked the whole image_encoder, so re-enable the
        # LoRA A/B parameters here unless inference_only was requested.
        if not self._freeze_cfg["inference_only"]:
            for m in self.image_encoder.modules():
                if isinstance(m, _LoRALinear):
                    for p in m.lora_A.parameters():
                        p.requires_grad = True
                    for p in m.lora_B.parameters():
                        p.requires_grad = True

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

        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits
