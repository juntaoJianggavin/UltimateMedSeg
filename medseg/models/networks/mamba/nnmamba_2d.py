"""nnMamba 2D: nnUNet-style encoder/decoder with Mamba SSM blocks inserted between stages.

2D adaptation of nnMamba (Gong et al., 2024):
  https://github.com/lhaof/nnMamba

Architecture:
  - Encoder: stack of nnU-Net-style ConvDropoutNormNonlin stages, each consisting
    of two Conv->InstanceNorm->LeakyReLU blocks; the first conv of each non-stem
    stage performs the 2x downsample (stride=2).
  - After every encoder stage a MambaLayer operates on (B, H*W, C) patch-tokens,
    providing long-range modelling between stages.
  - Decoder mirrors the encoder with ConvTranspose2d upsamplers and the usual
    skip-concatenation followed by two conv blocks per resolution.
  - 1x1 segmentation head returns logits at the full input resolution.

Self-contained — depends only on torch (timm is permitted but not required).
Requires ``mamba_ssm`` — the official nnMamba source
(https://github.com/lhaof/nnMamba) hard-depends on mamba_ssm.
"""
# Source: https://github.com/lhaof/nnMamba

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSL-failure fallback for any future pretrained-weight loading
# ---------------------------------------------------------------------------

def _load_pretrained_with_fallback(load_fn, *args, **kwargs):
    import ssl, urllib.request
    try:
        return load_fn(*args, **kwargs)
    except Exception:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Pretrained weight download failed: {type(e).__name__}: {e}. "
                "No silent fallback to random init — either: (a) ensure network "
                "access, (b) provide a local checkpoint via pretrained_path, "
                "or (c) explicitly pass pretrained=False for random init."
            ) from e
        finally:
            ssl._create_default_https_context = prev


# ---------------------------------------------------------------------------
# Mamba SSM wrapper (hard dependency on mamba_ssm, matching official source)
# ---------------------------------------------------------------------------

class _MambaSSM(nn.Module):
    """Mamba SSM wrapper. Hard-depends on ``mamba_ssm`` (matches official nnMamba source).

    Interface: (B, L, D) -> (B, L, D)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model

        try:
            from mamba_ssm import Mamba  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "nnMamba requires the `mamba_ssm` CUDA package. "
                "Install from https://github.com/state-spaces/mamba. "
                "The official nnMamba source hard-depends on mamba_ssm; "
                "no pure-PyTorch fallback is provided."
            ) from e
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(x)


# Backward-compatible alias
_MambaSSMFallback = _MambaSSM


# ---------------------------------------------------------------------------
# MambaLayer: patch-token Mamba on (B, C, H, W) -> (B, H*W, C) -> Mamba -> back
# ---------------------------------------------------------------------------

class _MambaLayer(nn.Module):
    """Patch-token Mamba layer for 2D features.

    (B, C, H, W) -> flatten spatial to (B, H*W, C) -> LayerNorm -> Mamba ->
    reshape back to (B, C, H, W).
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = _MambaSSM(dim, d_state=d_state,
                               d_conv=d_conv, expand=expand)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C, H, W = x.shape
        assert C == self.dim, f"channel mismatch: {C} vs {self.dim}"
        x_flat = x.reshape(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.norm(x_flat)
        x_out = self.mamba(x_norm)
        return x_out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# nnU-Net-style conv blocks
# ---------------------------------------------------------------------------

class _ConvDropoutNormNonlin(nn.Module):
    """nnU-Net default conv block: Conv2d -> Dropout2d -> InstanceNorm2d -> LeakyReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 dropout_p=0.0, neg_slope=1e-2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.dropout = (nn.Dropout2d(p=dropout_p, inplace=True)
                        if dropout_p > 0 else nn.Identity())
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.nonlin = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)

    def forward(self, x):
        return self.nonlin(self.norm(self.dropout(self.conv(x))))


class _StackedConvLayers(nn.Module):
    """Two-conv nnU-Net stage; first conv carries the stride."""

    def __init__(self, in_ch, out_ch, num_convs=2, kernel_size=3,
                 stride=1, dropout_p=0.0):
        super().__init__()
        layers = [_ConvDropoutNormNonlin(in_ch, out_ch, kernel_size,
                                        stride=stride, dropout_p=dropout_p)]
        for _ in range(num_convs - 1):
            layers.append(_ConvDropoutNormNonlin(out_ch, out_ch, kernel_size,
                                                stride=1, dropout_p=dropout_p))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


# ---------------------------------------------------------------------------
# nnMamba 2D
# ---------------------------------------------------------------------------

class NnMamba2D(nn.Module):
    """nnMamba 2D — nnU-Net encoder/decoder with patch-token Mamba between stages.

    Args:
        in_channels: Input channel count.
        num_classes: Output channel count (segmentation classes).
        img_size: Spatial size of the input (used only to size the model; the
            network actually accepts any size divisible by ``2**(num_stages-1)``).
        base_features: Base channel count; doubled at each downsampling stage and
            capped at ``max_features``.
        num_stages: Number of encoder stages (stem + downsampling stages).
        max_features: Channel cap for deeper stages (nnU-Net default 320).
        num_convs_per_stage: Number of conv blocks per encoder/decoder stage.
        mamba_d_state / mamba_d_conv / mamba_expand: Mamba SSM hyperparameters.
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 img_size: int = 224,
                 base_features: int = 32,
                 num_stages: int = 5,
                 max_features: int = 320,
                 num_convs_per_stage: int = 2,
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_stages = num_stages

        features = [min(base_features * (2 ** i), max_features)
                    for i in range(num_stages)]
        self.features = features

        # Encoder: stem stage (stride 1) + (num_stages-1) strided stages.
        self.encoder_stages = nn.ModuleList()
        self.encoder_mambas = nn.ModuleList()

        prev_ch = in_channels
        for s in range(num_stages):
            stride = 1 if s == 0 else 2
            self.encoder_stages.append(
                _StackedConvLayers(prev_ch, features[s],
                                  num_convs=num_convs_per_stage,
                                  kernel_size=3, stride=stride))
            prev_ch = features[s]
            # Mamba SSM block inserted between stages (after each conv stage).
            self.encoder_mambas.append(
                _MambaLayer(features[s],
                           d_state=mamba_d_state,
                           d_conv=mamba_d_conv,
                           expand=mamba_expand))

        # Decoder: mirror with ConvTranspose2d + skip concat + stacked convs.
        self.up_layers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for s in range(num_stages - 1, 0, -1):
            in_ch = features[s]
            out_ch = features[s - 1]
            self.up_layers.append(
                nn.ConvTranspose2d(in_ch, out_ch,
                                   kernel_size=2, stride=2, bias=True))
            self.decoder_stages.append(
                _StackedConvLayers(out_ch * 2, out_ch,
                                  num_convs=num_convs_per_stage,
                                  kernel_size=3, stride=1))

        # Final 1x1 segmentation head.
        self.seg_head = nn.Conv2d(features[0], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=1e-2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_h, in_w = x.shape[-2], x.shape[-1]

        # Encoder with inter-stage Mamba.
        skips = []
        for stage, mamba in zip(self.encoder_stages, self.encoder_mambas):
            x = stage(x)
            x = mamba(x)
            skips.append(x)

        # Decoder with skip concatenation.
        x = skips[-1]
        for i, (up, dec) in enumerate(zip(self.up_layers, self.decoder_stages)):
            x = up(x)
            skip = skips[-(i + 2)]
            # Defensive size-match in case of odd sizes.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                  mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        out = self.seg_head(x)
        if out.shape[-2:] != (in_h, in_w):
            out = F.interpolate(out, size=(in_h, in_w),
                                mode='bilinear', align_corners=False)
        return out


__all__ = ['NnMamba2D']
