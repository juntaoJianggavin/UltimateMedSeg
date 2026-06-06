"""SSFormer: Stepwise Strengthened Transformer for Polyp Segmentation.

Reference:
    Jinfeng Wang, Qiming Huang, Feilong Tang, Jia Meng, Jionglong Su,
    Sifan Song. "Stepwise Feature Fusion: Local Guides Global."
    Machine Intelligence Research (MIR), 2022 (a.k.a. SSFormer).
    Upstream code: https://github.com/Qiming-Huang/ssformer

Architecture overview:
    - Encoder: MiT-B2 (Mix Transformer) producing 4 hierarchical features at
      strides {4, 8, 16, 32} with channels {64, 128, 320, 512}. When the
      ``mit_b2`` weight name is not exposed by the installed timm version, we
      fall back to the structurally-equivalent ``pvt_v2_b2`` backbone (same
      channel widths and reductions).
    - PLD (Pyramid / Progressive Local Decoder): each encoder stage is
      projected to a common channel width via 1x1 conv, then bilinearly
      upsampled to the highest-resolution stage (stride 4), concatenated, and
      refined with a stack of 3x3 convolutions.
    - 1x1 segmentation head, followed by a final bilinear upsample back to the
      input image resolution.

Self-contained: only torch and timm are required.
"""
# Source: https://github.com/Qiming-Huang/ssformer

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


class _PLD(nn.Module):
    """Pyramid Local Decoder.

    Projects each encoder feature to a common channel width with a 1x1 conv,
    bilinearly upsamples every stage to the top (highest-resolution) stage,
    concatenates them, and refines with a 3x3 conv stack. A final 1x1 head
    produces the segmentation logits.
    """

    def __init__(self, in_channels_list, decoder_channels, num_classes):
        super().__init__()
        # 1x1 projections to a common embedding dim
        self.reduces = nn.ModuleList([
            nn.Conv2d(c, decoder_channels, kernel_size=1, bias=False)
            for c in in_channels_list
        ])
        n = len(in_channels_list)
        # 3x3 conv stack on the concatenated multi-stage features
        self.fuse = nn.Sequential(
            _ConvBNReLU(n * decoder_channels, decoder_channels, kernel_size=3),
            _ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3),
        )
        self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, feats):
        # feats: list ordered shallow->deep (largest spatial first)
        ref_size = feats[0].shape[-2:]
        ups = []
        for proj, f in zip(self.reduces, feats):
            y = proj(f)
            if y.shape[-2:] != ref_size:
                y = F.interpolate(
                    y, size=ref_size, mode='bilinear', align_corners=False,
                )
            ups.append(y)
        x = torch.cat(ups, dim=1)
        x = self.fuse(x)
        return self.head(x)


def _build_backbone(in_channels, pretrained=True):
    """创建 MiT-B2 backbone（features_only 模式）。
    Create the MiT-B2 backbone in features_only mode.

    mit_b2 与 pvt_v2_b2 是同一架构的不同命名。
    mit_b2 and pvt_v2_b2 are the same architecture under different names.
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
class SSFormer(nn.Module):
    """SSFormer: MiT-B2 encoder + PLD decoder.

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation classes.
        img_size: nominal input spatial size. The model is fully convolutional
            and accepts arbitrary H, W, but if the chosen backbone requires
            divisible-by-32 inputs, the forward pass pads internally and crops
            the output back to (H, W).
        decoder_channels: shared embedding width inside the PLD decoder.
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

        # --- Decoder + head -----------------------------------------------
        self.decoder = _PLD(enc_channels, decoder_channels, num_classes)

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
        # ordered shallow->deep (largest spatial first), as required by _PLD
        logits = self.decoder(feats)

        # upsample to the (padded) input resolution then crop back
        logits = F.interpolate(
            logits, size=x_pad.shape[-2:], mode='bilinear', align_corners=False,
        )
        if pad_h or pad_w:
            logits = logits[..., :H, :W]
        return logits
