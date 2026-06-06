"""nnU-Net 2D – self-contained port of MIC-DKFZ/nnUNet's plain-conv UNet.

A faithful 2D implementation of the "plain conv" UNet used by nnU-Net's
default planner: an encoder-decoder with InstanceNorm + LeakyReLU
activations, doubled features per stage capped at ``max_features`` (320
or 512 in the original), stride-2 pooling fused into the first Conv of
each downsampling stage, and a mirrored ConvTranspose2d decoder with
skip concatenation. Deep supervision is intentionally omitted – the
wrapper exposes a single full-resolution logit map matching the input.

Reference: Isensee et al., "nnU-Net: a self-configuring method for deep
learning-based biomedical image segmentation", Nat. Methods 2021.
Repo: https://github.com/MIC-DKFZ/nnUNet
"""
# Source: https://github.com/MIC-DKFZ/nnUNet

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _ConvDropoutNormNonlin(nn.Module):
    """nnU-Net's basic conv block: Conv -> Dropout -> InstanceNorm -> LeakyReLU.

    Dropout is disabled by default (p=0.0) to match the standard 2D plan.
    """

    def __init__(self, in_ch, out_ch, stride=1, dropout_p=0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride,
            padding=1, bias=True,
        )
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.norm = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.nonlin = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x):
        return self.nonlin(self.norm(self.dropout(self.conv(x))))


class _StackedConvLayers(nn.Module):
    """Two ConvDropoutNormNonlin blocks; the first carries the stage stride."""

    def __init__(self, in_ch, out_ch, first_stride=1, dropout_p=0.0):
        super().__init__()
        self.blocks = nn.Sequential(
            _ConvDropoutNormNonlin(in_ch, out_ch, stride=first_stride,
                                   dropout_p=dropout_p),
            _ConvDropoutNormNonlin(out_ch, out_ch, stride=1,
                                   dropout_p=dropout_p),
        )

    def forward(self, x):
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class NNUNet2D(nn.Module):
    """nnU-Net 2D plain-conv UNet.

    Args:
        in_channels: input image channels.
        num_classes: output segmentation classes (channels in the logit map).
        img_size: nominal input resolution (kept for the wrapper API; the
            network itself is fully convolutional).
        base_features: stem width (doubled at each encoder stage).
        num_stages: total encoder stages (first does not downsample;
            remaining ``num_stages-1`` stages downsample by stride 2).
        max_features: hard cap on per-stage feature width.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        base_features: int = 32,
        num_stages: int = 6,
        max_features: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_stages = num_stages

        # Per-stage feature widths: doubled, capped at max_features.
        feats = [min(base_features * (2 ** i), max_features)
                 for i in range(num_stages)]

        # ── Encoder ────────────────────────────────────────────────────────
        # Stage 0 keeps spatial resolution (stride=1); subsequent stages
        # downsample via stride-2 in the first conv of the block.
        self.encoders = nn.ModuleList()
        prev_c = in_channels
        for s in range(num_stages):
            stride = 1 if s == 0 else 2
            self.encoders.append(
                _StackedConvLayers(prev_c, feats[s], first_stride=stride)
            )
            prev_c = feats[s]

        # ── Decoder ────────────────────────────────────────────────────────
        # One transpose-conv + stacked-conv per "up" step. There are
        # (num_stages - 1) decoder steps mirroring the downsampling stages.
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for s in range(num_stages - 1, 0, -1):
            in_c = feats[s]
            skip_c = feats[s - 1]
            out_c = feats[s - 1]
            self.upsamples.append(
                nn.ConvTranspose2d(
                    in_c, out_c, kernel_size=2, stride=2, bias=True,
                )
            )
            # After concat with the skip we have (out_c + skip_c) channels.
            self.decoders.append(
                _StackedConvLayers(out_c + skip_c, out_c, first_stride=1)
            )

        # ── Segmentation head ──────────────────────────────────────────────
        self.seg_head = nn.Conv2d(feats[0], num_classes, kernel_size=1, bias=True)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        # nnU-Net initialises convs with He-normal (a=1e-2 to match the
        # LeakyReLU negative slope) and zero-init biases.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=1e-2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_hw = x.shape[-2:]

        # Encoder: collect skips at every stage.
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        # Decoder: start from the bottleneck (deepest skip) and progressively
        # upsample, concatenating with each shallower encoder feature.
        h = skips[-1]
        # decoder index 0 corresponds to undoing stage (num_stages-1);
        # the skip to concatenate is skips[num_stages-2], etc.
        for i, (up, dec) in enumerate(zip(self.upsamples, self.decoders)):
            skip = skips[self.num_stages - 2 - i]
            h = up(h)
            # Safety: align spatial dims if input dimensions weren't a
            # clean multiple of 2**(num_stages-1) – we crop/pad up to skip.
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:],
                                  mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.seg_head(h)
        if out.shape[-2:] != in_hw:
            out = F.interpolate(out, size=in_hw, mode='bilinear',
                                align_corners=False)
        return out
