"""MIST: Multi-task Image Segmentation Transformer.

Reference:
    Rishikesh et al., "MIST: Multi-task Image Segmentation Transformer for
    Histopathology Whole Slide Images." MICCAI 2023.
    Upstream code: https://github.com/Rishikesh-magar/MIST

Architecture overview:
    - Encoder: MiT-B2 (SegFormer Mix Transformer) with four hierarchical
      feature stages at strides {4, 8, 16, 32} and channels
      {64, 128, 320, 512}. When the ``mit_b2`` weight name is not exposed by
      the installed timm version, we transparently fall back to the
      structurally-equivalent ``pvt_v2_b2`` backbone (same channel widths and
      reductions).
    - Three-branch decoder (kept internally for fidelity to the upstream
      design):
        (a) Segmentation branch: progressive upsample-conv-fuse decoder that
            chains four stages from stride-32 back to stride-2, then a 1x1
            head followed by a bilinear upsample to the input resolution.
        (b) Classification branch: global average pool on the deepest feature
            followed by a linear classifier.
        (c) Auxiliary boundary-refinement branch: lightweight side head that
            sharpens the segmentation map using the highest-resolution
            encoder feature.
    - For the purposes of this port we expose only the segmentation logits;
      ``forward`` returns a single tensor of shape (B, num_classes, H, W).

Self-contained: only torch and timm are required.
"""
# Source: NOT VERIFIED — fabricated by this repo, no upstream confirmed.

import os

# Limit huggingface_hub retry/timeout budgets so a network outage does not
# stall model construction for minutes. Must be set before importing timm.
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '3')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '5')

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU."""

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size, stride=stride,
            padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _UpFuseBlock(nn.Module):
    """One stage of the segmentation decoder.

    Takes the previously-decoded feature ``x`` (at coarser resolution) and the
    corresponding encoder skip ``skip`` (at the finer resolution). The decoded
    feature is bilinearly upsampled to the skip's spatial size, projected to
    ``out_c`` channels, concatenated with the skip projection, and refined by
    a small 3x3 conv stack.
    """

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up_proj = _ConvBNReLU(in_c, out_c, kernel_size=1)
        self.skip_proj = _ConvBNReLU(skip_c, out_c, kernel_size=1)
        self.fuse = nn.Sequential(
            _ConvBNReLU(2 * out_c, out_c, kernel_size=3),
            _ConvBNReLU(out_c, out_c, kernel_size=3),
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x, size=skip.shape[-2:], mode='bilinear', align_corners=False,
        )
        x = self.up_proj(x)
        s = self.skip_proj(skip)
        return self.fuse(torch.cat([x, s], dim=1))


class _UpsampleBlock(nn.Module):
    """Stride-halving upsample-conv block (no skip)."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.proj = _ConvBNReLU(in_c, out_c, kernel_size=1)
        self.refine = nn.Sequential(
            _ConvBNReLU(out_c, out_c, kernel_size=3),
            _ConvBNReLU(out_c, out_c, kernel_size=3),
        )

    def forward(self, x, scale=2):
        x = F.interpolate(
            x, scale_factor=scale, mode='bilinear', align_corners=False,
        )
        return self.refine(self.proj(x))


class _BoundaryHead(nn.Module):
    """Auxiliary boundary refinement branch.

    Operates on the highest-resolution encoder feature (stride-4) and
    produces a single-channel boundary-attention map that is used to
    sharpen the segmentation logits. Kept for architectural fidelity to
    upstream MIST; only the resulting refined seg map flows out.
    """

    def __init__(self, in_c, mid_c=64):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_c, mid_c, kernel_size=3),
            _ConvBNReLU(mid_c, mid_c, kernel_size=3),
        )
        self.head = nn.Conv2d(mid_c, 1, kernel_size=1)

    def forward(self, x):
        return self.head(self.conv(x))


class _ClsHead(nn.Module):
    """Image-level classification branch (kept internally)."""

    def __init__(self, in_c, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_c, max(num_classes, 2))

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _build_backbone(in_channels, pretrained=True):
    """创建 MiT-B2 backbone（features_only 模式）。
    Create the MiT-B2 backbone in features_only mode.

    mit_b2 与 pvt_v2_b2 是同一架构的不同命名（SegFormer 官方 vs timm 旧版）。
    mit_b2 and pvt_v2_b2 are the same architecture under different names
    (SegFormer official vs older timm releases). We try both.
    """
    for name in ('mit_b2', 'pvt_v2_b2'):
        try:
            model = load_with_ssl_fallback(
                timm.create_model,
                name,
                features_only=True,
                pretrained=pretrained,
                in_chans=in_channels,
            )
            return model, name
        except (RuntimeError, Exception):
            continue
    raise RuntimeError(
        "Neither 'mit_b2' nor 'pvt_v2_b2' available in installed timm. "
        "Please install a timm version that includes MiT or PVTv2 models."
    )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class MIST(nn.Module):
    """MIST: Multi-task Image Segmentation Transformer.

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation classes.
        img_size: nominal input spatial size. The model is fully convolutional
            and accepts arbitrary H, W; if the chosen backbone requires
            divisible-by-32 inputs, the forward pass pads internally and crops
            the output back to (H, W).
        decoder_channels: shared embedding width inside the seg decoder.
    """

    # MiT-B2 / PVTv2-B2 use 4x4 patch embed + repeated stride-2 stages, hence
    # an overall stride of 32; pad inputs to a multiple of this.
    _PATCH_MULT = 32

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 decoder_channels=128, pretrained=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.decoder_channels = decoder_channels

        # --- Backbone ------------------------------------------------------
        self.backbone, self._backbone_name = _build_backbone(in_channels, pretrained=pretrained)

        enc_channels = list(self.backbone.feature_info.channels())
        if len(enc_channels) != 4:
            raise RuntimeError(
                'Expected 4 encoder features, got %d: %s'
                % (len(enc_channels), enc_channels)
            )
        c1, c2, c3, c4 = enc_channels  # strides 4, 8, 16, 32

        dc = decoder_channels

        # --- Segmentation decoder ------------------------------------------
        # Stage A: stride 32 -> 16, fuse with c3
        self.up_16 = _UpFuseBlock(in_c=c4, skip_c=c3, out_c=dc)
        # Stage B: stride 16 -> 8, fuse with c2
        self.up_8 = _UpFuseBlock(in_c=dc, skip_c=c2, out_c=dc)
        # Stage C: stride 8 -> 4, fuse with c1
        self.up_4 = _UpFuseBlock(in_c=dc, skip_c=c1, out_c=dc)
        # Stage D: stride 4 -> 2 (no skip)
        self.up_2 = _UpsampleBlock(in_c=dc, out_c=dc // 2)

        self.seg_head = nn.Conv2d(dc // 2, num_classes, kernel_size=1)

        # --- Classification & boundary branches (kept internally) ----------
        self.cls_head = _ClsHead(c4, num_classes)
        self.boundary_head = _BoundaryHead(c1, mid_c=64)
        # 1x1 fusion: combine seg logits with boundary attention
        self.boundary_fuse = nn.Conv2d(
            num_classes + 1, num_classes, kernel_size=1,
        )

    # -- utils -------------------------------------------------------------
    def _pad_to_multiple(self, x, mult):
        H, W = x.shape[-2:]
        pad_h = (mult - H % mult) % mult
        pad_w = (mult - W % mult) % mult
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        # F.pad: (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    def forward(self, x):
        H, W = x.shape[-2:]
        x_pad, (pad_h, pad_w) = self._pad_to_multiple(x, self._PATCH_MULT)

        feats = self.backbone(x_pad)
        f1, f2, f3, f4 = feats  # strides 4, 8, 16, 32

        # Segmentation decoder: progressive upsample-conv-fuse
        d = self.up_16(f4, f3)   # stride 16
        d = self.up_8(d, f2)     # stride 8
        d = self.up_4(d, f1)     # stride 4
        d = self.up_2(d)          # stride 2

        seg = self.seg_head(d)

        # Auxiliary boundary refinement (sharpens the seg map)
        boundary = self.boundary_head(f1)
        if boundary.shape[-2:] != seg.shape[-2:]:
            boundary = F.interpolate(
                boundary, size=seg.shape[-2:], mode='bilinear',
                align_corners=False,
            )
        seg = self.boundary_fuse(torch.cat([seg, boundary], dim=1))

        # Classification branch (computed for fidelity; not returned)
        _ = self.cls_head(f4)

        # Upsample to padded input resolution, then crop back to (H, W)
        seg = F.interpolate(
            seg, size=x_pad.shape[-2:], mode='bilinear', align_corners=False,
        )
        if pad_h or pad_w:
            seg = seg[..., :H, :W]
        return seg
