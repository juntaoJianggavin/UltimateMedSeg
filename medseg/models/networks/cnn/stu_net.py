"""STU-Net 2D – Scalable Transferable U-Net (MICCAI 2024).

A self-contained 2D port of the official 3D STU-Net by Huang et al.
(``uni-medical/STU-Net``). STU-Net replaces nnU-Net's plain conv blocks
with pre-activation-free *residual* conv blocks
(Conv->IN->LeakyReLU->Conv->IN + identity->LeakyReLU) and keeps the
nnU-Net encoder-decoder topology and planning conventions (InstanceNorm
+ LeakyReLU, stride fused into the first conv of each downsampling
stage, ConvTranspose2d upsamples, skip concatenation).

This 2D variant uses 6 encoder stages with widths
``[32, 64, 128, 256, 512, 512]`` matching the nnU-Net planner default,
and 2 residual blocks per stage. Deep supervision is disabled; the
forward pass returns a single full-resolution logit map.

Reference: Huang et al., "STU-Net: Scalable and Transferable Medical
Image Segmentation Models", MICCAI 2024.
Repo: https://github.com/uni-medical/STU-Net
"""
# Source: https://github.com/uni-medical/STU-Net

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: SSL fallback for any future pretrained-weight download.
# ---------------------------------------------------------------------------
def _load_with_ssl_fallback(load_fn, *args, **kwargs):
    import ssl, warnings
    try:
        return load_fn(*args, **kwargs)
    except Exception:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception:
            warnings.warn('Pretrained download failed; using random init.')
            return load_fn(*args, **{**kwargs, 'pretrained': False})
        finally:
            ssl._create_default_https_context = prev


# ---------------------------------------------------------------------------
# Residual block (2D adaptation of the official BasicResBlock).
# ---------------------------------------------------------------------------
class _BasicResBlock(nn.Module):
    """Two 3x3 convs with IN + LeakyReLU and an identity / 1x1 shortcut.

    The first conv carries the optional stride (used for downsampling at
    the first block of each encoder stage). A 1x1 projection on the
    shortcut is used whenever the input/output shapes differ.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_1x1conv: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=True,
        )
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=kernel_size,
            stride=1, padding=padding, bias=True,
        )
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv or stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=stride, bias=True,
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        sc = self.shortcut(x) if self.shortcut is not None else x
        return self.act2(y + sc)


class _ResStage(nn.Module):
    """A stack of ``depth`` BasicResBlocks; the first carries the stride."""

    def __init__(self, in_ch: int, out_ch: int, depth: int = 2, stride: int = 1):
        super().__init__()
        blocks = [_BasicResBlock(in_ch, out_ch, stride=stride, use_1x1conv=True)]
        for _ in range(depth - 1):
            blocks.append(_BasicResBlock(out_ch, out_ch, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class STUNet(nn.Module):
    """STU-Net 2D segmentation network.

    Args:
        in_channels: input image channels.
        num_classes: output segmentation classes.
        img_size: nominal input resolution (the network is fully
            convolutional; this is recorded for compatibility).
        base_features: stem width (doubled at each encoder stage, capped
            at ``max_features``).
        num_stages: number of encoder stages. The first stage keeps the
            input resolution; each subsequent stage downsamples by 2.
        max_features: hard cap on per-stage feature width.
        depth: number of residual blocks per stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        base_features: int = 32,
        num_stages: int = 6,
        max_features: int = 512,
        depth: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_stages = num_stages
        # Total downsampling factor – pad inputs to a multiple of this.
        self.size_divisor = 2 ** (num_stages - 1)

        # Per-stage feature widths.
        feats: List[int] = [
            min(base_features * (2 ** i), max_features) for i in range(num_stages)
        ]
        self.feats = feats

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        prev_c = in_channels
        for s in range(num_stages):
            stride = 1 if s == 0 else 2
            self.encoders.append(
                _ResStage(prev_c, feats[s], depth=depth, stride=stride)
            )
            prev_c = feats[s]

        # ── Decoder ────────────────────────────────────────────────────────
        # (num_stages - 1) up-steps; each does ConvTranspose2d -> concat -> ResStage.
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for s in range(num_stages - 1, 0, -1):
            in_c = feats[s]
            skip_c = feats[s - 1]
            out_c = feats[s - 1]
            self.upsamples.append(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, bias=True)
            )
            # After concat with the skip we have (out_c + skip_c) channels.
            self.decoders.append(
                _ResStage(out_c + skip_c, out_c, depth=depth, stride=1)
            )

        # Segmentation head – single 1x1 logit map at full resolution.
        self.seg_head = nn.Conv2d(feats[0], num_classes, kernel_size=1, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        # nnU-Net / STU-Net both use He-normal with a=1e-2 (matches the
        # LeakyReLU negative slope) and zero biases.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=1e-2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_hw = x.shape[-2:]

        # Pad to a multiple of size_divisor so all encoder strides line up.
        H, W = in_hw
        pad_h = (self.size_divisor - H % self.size_divisor) % self.size_divisor
        pad_w = (self.size_divisor - W % self.size_divisor) % self.size_divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder: store skip feature at every stage.
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        # Decoder: start from the bottleneck and walk up, concatenating skips.
        h = skips[-1]
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            skip = skips[self.num_stages - 2 - i]
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(
                    h, size=skip.shape[-2:], mode='bilinear', align_corners=False,
                )
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.seg_head(h)
        # Crop back to the original input size if we padded.
        if out.shape[-2:] != in_hw:
            out = out[..., : in_hw[0], : in_hw[1]]
        return out
