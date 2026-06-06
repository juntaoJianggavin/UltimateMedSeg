"""SAM-Med2D — OpenGVLab 2024.

Reference:
    Junlong Cheng et al., "SAM-Med2D: A Comprehensive Foundation Model for
    Generalized 2D Medical Image Segmentation", 2024.
    Upstream code: https://github.com/OpenGVLab/SAM-Med2D

Architecture overview:
    - Image encoder: ViT-B/16 (timm ``vit_base_patch16_224``) with a small
      bottleneck *adapter* (Linear 768->64 -> GELU -> Linear 64->768 + residual)
      injected inside every transformer block. With the backbone frozen by
      default, only the adapters + mask decoder receive gradients, which is
      what makes SAM-Med2D a parameter-efficient medical fine-tune of SAM.
    - Mask decoder: four ConvTranspose2d stages
      (768 -> 256 -> 128 -> 64 -> num_classes), each with BatchNorm + GELU
      except the last (which produces the logits).

The model is prompt-free: a ``prompt_encoder`` slot is exposed as ``None`` for
API parity with the rest of the SAM-family wrappers.

Self-contained: only torch + timm are required. If the ImageNet pretrained
download for ``vit_base_patch16_224`` is unreachable, the encoder transparently
falls back to random initialisation (a warning is emitted).
"""
# Source: https://github.com/OpenGVLab/SAM-Med2D

from __future__ import annotations

import os

# Limit huggingface_hub timeouts so a network outage cannot stall construction.
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '3')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '5')

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam_base import SAMBase, load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _Adapter(nn.Module):
    """Bottleneck residual adapter.

    Linear(dim -> hidden) -> GELU -> Linear(hidden -> dim), added back to the
    input via a skip connection. The output projection is zero-initialised so
    the adapter starts as a near-identity and does not disrupt the pretrained
    backbone's behaviour on the first forward pass.
    """

    def __init__(self, dim: int = 768, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class _BlockWithAdapter(nn.Module):
    """Wrap a ViT transformer block and append an _Adapter to its output."""

    def __init__(self, block: nn.Module, dim: int = 768, hidden: int = 64):
        super().__init__()
        self.block = block
        self.adapter = _Adapter(dim=dim, hidden=hidden)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.block(x, *args, **kwargs)
        return self.adapter(x)


def _build_vit(img_size: int, in_channels: int, pretrained: bool):
    """Build the SAM-style ViT-B/16 backbone via timm.

    ``dynamic_img_size=True`` lets the model accept any spatial size that is a
    multiple of the patch size (16), with positional embeddings interpolated
    on-the-fly. This is what enables forward passes at 224 / 256 / 512.
    """

    import timm

    def _create(pretrained: bool = False):
        return timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            dynamic_img_size=True,
            img_size=img_size,
            in_chans=in_channels,
            num_classes=0,
            global_pool='',
        )

    return load_with_ssl_fallback(_create, pretrained=pretrained)


class _SAMMed2DEncoder(nn.Module):
    """SAM-Med2D image encoder: timm ViT-B/16 with per-block adapters.

    The wrapped ``blocks`` attribute is also exposed at the encoder level so
    :class:`SAMBase.apply_freeze` can find it when ``unfreeze_last_n_blocks``
    is requested.
    """

    PATCH = 16
    EMBED_DIM = 768

    def __init__(self, img_size: int, in_channels: int, pretrained: bool,
                 use_adapter: bool = True, adapter_hidden: int = 64):
        super().__init__()
        self.vit = _build_vit(img_size, in_channels, pretrained)
        self.num_prefix_tokens = int(getattr(self.vit, 'num_prefix_tokens', 1))
        self.use_adapter = use_adapter

        if use_adapter:
            wrapped = nn.Sequential(*[
                _BlockWithAdapter(blk, self.EMBED_DIM, adapter_hidden)
                for blk in self.vit.blocks
            ])
            self.vit.blocks = wrapped

        # Re-expose for SAMBase.apply_freeze (which looks for `blocks` / `norm`).
        self.blocks = self.vit.blocks
        if hasattr(self.vit, 'norm') and isinstance(self.vit.norm, nn.Module):
            self.norm = self.vit.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        tokens = self.vit.forward_features(x)

        # timm ViT returns either ``(B, N, C)`` (with prefix CLS tokens) or
        # ``(B, C, Hp, Wp)`` depending on the version / pooling config.
        if tokens.dim() == 3:
            if self.num_prefix_tokens > 0:
                tokens = tokens[:, self.num_prefix_tokens:, :]
            Hp = H // self.PATCH
            Wp = W // self.PATCH
            if tokens.shape[1] != Hp * Wp:
                raise RuntimeError(
                    f'SAM-Med2D: unexpected token count {tokens.shape[1]} '
                    f'(expected {Hp * Wp} for {H}x{W} input).'
                )
            feat = tokens.transpose(1, 2).reshape(
                B, self.EMBED_DIM, Hp, Wp,
            ).contiguous()
        else:
            feat = tokens
        return feat


class _SAMMed2DMaskDecoder(nn.Module):
    """4-stage transposed-conv decoder: 768 -> 256 -> 128 -> 64 -> num_classes.

    Each stage doubles the spatial size (kernel 2, stride 2), giving an overall
    16x upsampling that exactly inverts the ViT patchification.
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.up4 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.up1(x)))
        x = self.act(self.bn2(self.up2(x)))
        x = self.act(self.bn3(self.up3(x)))
        x = self.up4(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class SAMMed2D(SAMBase):
    """SAM-Med2D: SAM ViT-B/16 with per-block adapters + a conv mask decoder.

    Args:
        in_channels: number of input image channels (default 3).
        num_classes: number of segmentation classes (default 2).
        img_size: nominal input spatial size. The forward pass also accepts
            other resolutions (padded internally to a multiple of the patch
            size); validated at 224, 256, and 512.
        pretrained: whether to load ImageNet-pretrained ViT-B/16 weights.
        pretrained_path: optional path to a custom SAM-Med2D checkpoint.
        freeze_image_encoder: freeze the ViT backbone (default True). Adapters
            remain trainable so the model still learns when this is True.
        freeze_prompt_encoder: present for API parity (SAM-Med2D is prompt-free).
        freeze_mask_decoder: freeze the conv mask decoder (default False).
        unfreeze_last_n_blocks: if > 0, unfreeze the final N ViT blocks (and
            the encoder norm) even when the encoder is otherwise frozen.
        inference_only: if True, freeze everything and set eval mode.
        use_adapter: insert per-block adapters (default True).
    """

    PATCH = 16
    EMBED_DIM = 768

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        pretrained: bool = True,
        pretrained_path=None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        unfreeze_last_n_blocks: int = 0,
        inference_only: bool = False,
        use_adapter: bool = True,
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
        self.use_adapter = bool(use_adapter)

        # Image encoder: ViT-B/16 with adapters.
        self.image_encoder = _SAMMed2DEncoder(
            img_size=img_size,
            in_channels=in_channels,
            pretrained=self._pretrained,
            use_adapter=self.use_adapter,
        )

        # SAM-Med2D in this prompt-free variant has no prompt encoder; expose
        # ``None`` so SAMBase.apply_freeze can introspect uniformly.
        self.prompt_encoder = None

        # 4-stage transposed-conv mask decoder.
        self.mask_decoder = _SAMMed2DMaskDecoder(self.EMBED_DIM, num_classes)

        # Optional custom checkpoint (e.g. an OpenGVLab SAM-Med2D weight dump).
        if pretrained_path:
            try:
                sd = torch.load(pretrained_path, map_location='cpu')
                if isinstance(sd, dict):
                    for k in ('state_dict', 'model', 'model_state_dict'):
                        if k in sd:
                            sd = sd[k]
                            break
                missing, unexpected = self.load_state_dict(sd, strict=False)
                if missing or unexpected:
                    warnings.warn(
                        f'SAM-Med2D: loaded {pretrained_path} with '
                        f'{len(missing)} missing and {len(unexpected)} '
                        'unexpected keys.'
                    )
            except Exception as e:
                warnings.warn(
                    f'SAM-Med2D: failed to load checkpoint {pretrained_path}: {e}'
                )

        self.apply_freeze()

    # ------------------------------------------------------------------
    def apply_freeze(self):
        """Freeze per SAMBase, then unfreeze the adapters.

        SAMBase.apply_freeze sets ``requires_grad = False`` on the entire image
        encoder when ``freeze_image_encoder=True``. The whole point of
        SAM-Med2D's adapters is that they stay trainable, so we flip them back
        on here (unless ``inference_only`` is in force).
        """
        super().apply_freeze()
        cfg = self._freeze_cfg
        if (cfg['image_encoder'] and self.use_adapter
                and not cfg['inference_only']):
            for name, p in self.image_encoder.named_parameters():
                if 'adapter' in name:
                    p.requires_grad = True

    # ------------------------------------------------------------------
    def _pad_to_multiple(self, x: torch.Tensor, mult: int):
        H, W = x.shape[-2:]
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        return F.pad(x, (0, pad_w, 0, pad_h)), (pad_h, pad_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, self.PATCH)

        feat = self.image_encoder(x_pad)        # (B, 768, H'/16, W'/16)
        logits = self.mask_decoder(feat)        # (B, num_classes, H', W')

        # The decoder upsamples exactly 16x, so logits already match x_pad's
        # spatial size; bilinear is a safety net for unusual cases.
        if logits.shape[-2:] != x_pad.shape[-2:]:
            logits = F.interpolate(
                logits, size=x_pad.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits


# Public alias for downstream registries / arch_key conventions.
SamMed2D = SAMMed2D
