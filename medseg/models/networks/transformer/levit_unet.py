"""LeViT-UNet (Pattern Recognition 2023) - self-contained port.

Reference: Xu et al., "LeViT-UNet: Make Faster Encoders with Transformer for
Medical Image Segmentation." github.com/apple1986/LeViT_UNet

Standard interface:
    model = LeViTUNet(in_channels=3, num_classes=2, img_size=224)
    out = model(x)  # -> (B, num_classes, H, W)

The encoder is a LeViT backbone created via ``timm.create_model`` with
``features_only=True``. LeViT carries fixed-shape attention bias tables that
strictly require a 224x224 input, so this module bilinearly resamples the
incoming feature map to (224, 224) before invoking the backbone and resamples
the final logits back to the requested (H, W). A lightweight CNN stem provides
high-resolution skip features at the original padded resolution.
"""
# Source: https://github.com/apple1986/LeViT_UNet

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


_LEVIT_INPUT = 224  # LeViT has fixed positional bias for a 224x224 input.


def _conv_bn_relu(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int | None = None) -> nn.Sequential:
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class _DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_relu(in_c, out_c),
            _conv_bn_relu(out_c, out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):
    """ConvTranspose2d upsample followed by a DoubleConv after concatenation."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = _DoubleConv(out_c + skip_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LeViTUNet(nn.Module):
    """LeViT-UNet: LeViT encoder + UNet-style decoder with skip connections."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        levit_model: str = "levit_192",
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = int(img_size)
        self.levit_model = levit_model

        # ---- LeViT encoder (timm) -----------------------------------------
        self.levit = load_with_ssl_fallback(
            timm.create_model,
            levit_model,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
        )
        levit_ch = list(self.levit.feature_info.channels())  # e.g. [192, 288, 384]
        assert len(levit_ch) >= 3, f"Expected >=3 LeViT feature maps, got {len(levit_ch)}"
        c1, c2, c3 = levit_ch[0], levit_ch[1], levit_ch[2]

        # Fuse the three LeViT feature maps at the coarsest-but-largest grid
        # (14x14 for a 224 input) so the decoder can start from a single tensor.
        fused_ch = 256
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 + c2 + c3, fused_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(fused_ch),
            nn.ReLU(inplace=True),
        )

        # ---- CNN stem providing high-res skip features --------------------
        # Operates on the padded input (multiple of 16). Produces skips at
        # strides 2, 4, 8 with channel widths 32, 64, 128 respectively.
        stem_c = (32, 64, 128, 192)
        self.stem0 = _DoubleConv(in_channels, stem_c[0])               # stride 1
        self.down1 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(stem_c[0], stem_c[1]))  # /2
        self.down2 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(stem_c[1], stem_c[2]))  # /4
        self.down3 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(stem_c[2], stem_c[3]))  # /8

        # ---- Decoder ------------------------------------------------------
        # The decoder starts from the fused LeViT feature (resampled to /16 of
        # the padded input) and progressively upsamples through /8, /4, /2, /1
        # while fusing the CNN-stem skip features at each level.
        self.dec3 = _Up(fused_ch, stem_c[3], 192)   # /16 -> /8  (skip: s3)
        self.dec2 = _Up(192, stem_c[2], 128)        # /8  -> /4  (skip: s2)
        self.dec1 = _Up(128, stem_c[1], 64)         # /4  -> /2  (skip: s1)
        self.dec0 = _Up(64, stem_c[0], 32)          # /2  -> /1  (skip: s0)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

        # ---- Init non-pretrained layers -----------------------------------
        for m in [self.fuse, self.stem0, self.down1, self.down2, self.down3,
                  self.dec3, self.dec2, self.dec1, self.dec0, self.head]:
            for mm in m.modules():
                if isinstance(mm, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(mm.weight, mode="fan_out", nonlinearity="relu")
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
                elif isinstance(mm, nn.BatchNorm2d):
                    nn.init.ones_(mm.weight)
                    nn.init.zeros_(mm.bias)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, mult: int = 16) -> tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        new_h = int(math.ceil(h / mult) * mult)
        new_w = int(math.ceil(w / mult) * mult)
        pad_h = new_h - h
        pad_w = new_w - w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Pad to multiple of 16 for clean stride arithmetic.
        x_pad, pad_h, pad_w = self._pad_to_multiple(x, 16)

        # CNN stem on padded input.
        s0 = self.stem0(x_pad)        # 1/1
        s1 = self.down1(s0)           # 1/2
        s2 = self.down2(s1)           # 1/4
        s3 = self.down3(s2)           # 1/8

        # LeViT backbone strictly requires a 224x224 input.
        x_levit = x_pad if (x_pad.shape[-2] == _LEVIT_INPUT and x_pad.shape[-1] == _LEVIT_INPUT) \
            else F.interpolate(x_pad, size=(_LEVIT_INPUT, _LEVIT_INPUT),
                               mode="bilinear", align_corners=False)
        f1, f2, f3 = self.levit(x_levit)[:3]

        # Align deeper LeViT maps to f1's spatial grid then fuse.
        target_hw = f1.shape[-2:]
        if f2.shape[-2:] != target_hw:
            f2 = F.interpolate(f2, size=target_hw, mode="bilinear", align_corners=False)
        if f3.shape[-2:] != target_hw:
            f3 = F.interpolate(f3, size=target_hw, mode="bilinear", align_corners=False)
        bn = self.fuse(torch.cat([f1, f2, f3], dim=1))  # (B, 256, 14, 14) at 224 path

        # Resample fused bottleneck to the padded /16 grid.
        pH, pW = x_pad.shape[-2], x_pad.shape[-1]
        bn = F.interpolate(bn, size=(pH // 16, pW // 16),
                           mode="bilinear", align_corners=False)

        d = self.dec3(bn, s3)  # /16 -> /8
        d = self.dec2(d, s2)   # /8  -> /4
        d = self.dec1(d, s1)   # /4  -> /2
        d = self.dec0(d, s0)   # /2  -> /1

        out = self.head(d)

        # Remove padding then resize to the original H, W (cheap if same size).
        if pad_h or pad_w:
            out = out[..., :x_pad.shape[-2] - pad_h, :x_pad.shape[-1] - pad_w]
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out


# Convenience aliases ---------------------------------------------------------
def levit_unet(**kwargs) -> LeViTUNet:
    return LeViTUNet(**kwargs)


__all__ = ["LeViTUNet", "levit_unet"]
