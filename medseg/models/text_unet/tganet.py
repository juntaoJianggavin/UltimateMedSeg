# Reference: https://github.com/nikhilroxtomar/TGANet
# Paper:     https://arxiv.org/abs/2205.04280
"""TGANet: Text-guided Attention for Improved Polyp Segmentation.

Re-implemented from the paper (Tomar et al., "TGANet: Text-guided
Attention for Improved Polyp Segmentation", MICCAI 2022) and the README
of the official repository:
    https://github.com/nikhilroxtomar/TGANet  (model.py)
The building blocks are derived from the paper's Sec. 3 formulas
(text classifier, label attention, multi-scale aggregation, dilated
ASPP-style encoder).

The architecture consumes:
    image:  (B, 3, H, W)
    label:  (B, 5, 300)            # 5 fixed phrases x 300-d GloVe embeddings
                                   # 5 = (2 num-polyp candidates) + (3 size candidates)
and produces:
    logits:        (B, num_classes, H, W)
    num_polyps:    (B, 2)          # auxiliary text-classification logit
    polyp_sizes:   (B, 3)          # auxiliary text-classification logit

Framework-glue diffs (algorithm layer unchanged):

1. Backbone uses ``torchvision.models.resnet50`` rather than the
   repo-local copy of ResNet.
2. ``num_classes`` is exposed at the constructor; ``num_classes==1``
   preserves the binary sigmoid case.

Strict no-fallback policy:
    * forward(image, text=None) raises — label attention and the
      embedding-feature-fusion module are core to TGANet; running with a
      zero label is not TGANet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Building blocks (1:1 from upstream model.py)
# ============================================================================


class _Conv2d(nn.Module):
    """Upstream: ``conv2d``.  Conv-BN-(ReLU)."""

    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    """Upstream: ``channel_attention``."""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Upstream: ``spatial_attention``."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class DilatedConv(nn.Module):
    """Upstream: ``dilated_conv``."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.c1 = nn.Sequential(_Conv2d(in_c, out_c, kernel_size=1, padding=0),
                                ChannelAttention(out_c))
        self.c2 = nn.Sequential(_Conv2d(in_c, out_c, kernel_size=(3, 3),
                                        padding=6, dilation=6),
                                ChannelAttention(out_c))
        self.c3 = nn.Sequential(_Conv2d(in_c, out_c, kernel_size=(3, 3),
                                        padding=12, dilation=12),
                                ChannelAttention(out_c))
        self.c4 = nn.Sequential(_Conv2d(in_c, out_c, kernel_size=(3, 3),
                                        padding=18, dilation=18),
                                ChannelAttention(out_c))
        self.c5 = _Conv2d(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = _Conv2d(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = SpatialAttention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], dim=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc + xs)
        x = self.sa(x)
        return x


class LabelAttention(nn.Module):
    """Upstream: ``label_attention``.  Channel attention conditioned on
    the (B, 128)-d label embedding produced by ``embedding_feature_fusion``."""

    def __init__(self, in_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c[1], in_c[0], kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_c[0], in_c[0], kernel_size=1, padding=0, bias=False),
        )

    def forward(self, feats, label):
        b, c = label.shape
        label = label.reshape(b, c, 1, 1)
        ch_attn = self.c1(label)
        ch_map = torch.sigmoid(ch_attn)
        feats = feats * ch_map
        ch_attn = ch_attn.reshape(ch_attn.shape[0], ch_attn.shape[1])
        return ch_attn, feats


class DecoderBlock(nn.Module):
    """Upstream: ``decoder_block``."""

    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = _Conv2d(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = _Conv2d(out_c, out_c, act=False)
        self.c3 = _Conv2d(out_c, out_c, act=False)
        self.c4 = _Conv2d(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        x = self.ca(x)
        x = self.sa(x)
        return x


class OutputBlock(nn.Module):
    """Upstream: ``output_block``."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        return self.c1(self.up(x))


class TextClassifier(nn.Module):
    """Upstream: ``text_classifier``.

    Two heads on the deepest visual feature: number-of-polyps (2 logits)
    and polyp-size (3 logits).
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_c, in_c // 8, bias=False), nn.ReLU(),
            nn.Linear(in_c // 8, out_c[0], bias=False),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_c, in_c // 8, bias=False), nn.ReLU(),
            nn.Linear(in_c // 8, out_c[1], bias=False),
        )

    def forward(self, feats):
        pool = self.avg_pool(feats).view(feats.shape[0], feats.shape[1])
        num_polyps = self.fc1(pool)
        polyp_sizes = self.fc2(pool)
        return num_polyps, polyp_sizes


class EmbeddingFeatureFusion(nn.Module):
    """Upstream: ``embedding_feature_fusion``.

    Combines the predicted text-classification logits with the (5, 300)
    label embedding to produce a (B, out_c) label feature.
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d((in_c[0] + in_c[1]) * in_c[2], out_c, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.ReLU(),
        )

    def forward(self, num_polyps, polyp_sizes, label):
        num_polyps_prob = torch.softmax(num_polyps, dim=1)
        polyp_sizes_prob = torch.softmax(polyp_sizes, dim=1)
        prob = torch.cat([num_polyps_prob, polyp_sizes_prob], dim=1)
        prob = prob.view(prob.shape[0], prob.shape[1], 1)
        x = label * prob
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x


class MultiscaleFeatureAggregation(nn.Module):
    """Upstream: ``multiscale_feature_aggregation``."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.c11 = _Conv2d(in_c[0], out_c, kernel_size=1, padding=0)
        self.c12 = _Conv2d(in_c[1], out_c, kernel_size=1, padding=0)
        self.c13 = _Conv2d(in_c[2], out_c, kernel_size=1, padding=0)
        self.c14 = _Conv2d(out_c * 3, out_c, kernel_size=1, padding=0)
        self.c2 = _Conv2d(out_c, out_c, act=False)
        self.c3 = _Conv2d(out_c, out_c, act=False)

    def forward(self, x1, x2, x3):
        x1 = self.up_4x4(x1)
        x2 = self.up_2x2(x2)
        x1 = self.c11(x1)
        x2 = self.c12(x2)
        x3 = self.c13(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.c14(x)
        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)
        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)
        return x


# ============================================================================
# Full TGANet (= upstream ``TGAPolypSeg``)
# ============================================================================


class TGANet(nn.Module):
    """TGANet polyp / lesion segmentor with text-guided attention.

    Args:
        in_channels: input image channels (3 by default).
        num_classes: output channels (1 = sigmoid binary; >1 = softmax).
        img_size: kept for framework-compatibility, unused internally.
        n_label_phrases: number of text phrases used to look up
            embeddings.  Upstream uses 5 (= 2 + 3).
        label_embed_dim: dimensionality of each phrase embedding.
            Upstream uses 300 (Glove-300).
    """

    is_text_guided = True            # picked up by training loop

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        n_label_phrases: int = 5,
        label_embed_dim: int = 300,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.n_label_phrases = n_label_phrases
        self.label_embed_dim = label_embed_dim

        # ---- Backbone: ResNet50 (torchvision) -------------------------------
        from torchvision.models import resnet50
        backbone = resnet50(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                       padding=3, bias=False)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # ---- Text classifier + label fusion ---------------------------------
        self.text_classifier = TextClassifier(1024, [2, 3])
        self.label_fc = EmbeddingFeatureFusion([2, 3, label_embed_dim], 128)

        # ---- Dilated conv encoder neck --------------------------------------
        self.s1 = DilatedConv(64, 128)
        self.s2 = DilatedConv(256, 128)
        self.s3 = DilatedConv(512, 128)
        self.s4 = DilatedConv(1024, 128)

        # ---- Decoder --------------------------------------------------------
        self.d1 = DecoderBlock(128, 128, scale=2)
        self.a1 = LabelAttention([128, 128])
        self.d2 = DecoderBlock(128, 128, scale=2)
        self.a2 = LabelAttention([128, 128])
        self.d3 = DecoderBlock(128, 128, scale=2)
        self.a3 = LabelAttention([128, 128])

        self.ag = MultiscaleFeatureAggregation([128, 128, 128], 128)
        self.y1 = OutputBlock(128, num_classes)

    # ------------------------------------------------------------------
    def forward(self, image, text=None, **kwargs):
        """Forward.

        Args:
            image: (B, C, H, W) input.
            text:  (B, n_label_phrases, label_embed_dim) phrase embeddings.
                   Passing ``None`` raises — label attention conditioned on
                   the phrase embeddings is the defining mechanism of
                   TGANet.  A 2-D (B, n_label_phrases*label_embed_dim)
                   tensor is accepted and reshaped.

        Returns:
            dict with keys ``logits`` (B, num_classes, H, W),
            ``num_polyps`` (B, 2), ``polyp_sizes`` (B, 3).
        """
        B = image.shape[0]
        if text is None:
            raise ValueError(
                "TGANet.forward requires `text` of shape "
                f"(B, {self.n_label_phrases}, {self.label_embed_dim}) — "
                "the GloVe-300 phrase embeddings for the polyp size / "
                "count anchors. Running with a zero text vector disables "
                "label attention and the embedding-feature-fusion module."
            )
        if text.dim() == 2:
            # accept (B, n_label_phrases*label_embed_dim) flattened embeddings
            text = text.view(B, self.n_label_phrases, self.label_embed_dim)
        if text.shape[1:] != (self.n_label_phrases, self.label_embed_dim):
            raise ValueError(
                f"TGANet `text` must be (B, {self.n_label_phrases}, "
                f"{self.label_embed_dim}); got {tuple(text.shape)}"
            )

        # ---- Backbone -------------------------------------------------------
        x1 = self.layer0(image)        # (B,   64, H/2,  W/2)
        x2 = self.layer1(x1)           # (B,  256, H/4,  W/4)
        x3 = self.layer2(x2)           # (B,  512, H/8,  W/8)
        x4 = self.layer3(x3)           # (B, 1024, H/16, W/16)

        num_polyps, polyp_sizes = self.text_classifier(x4)
        f0 = self.label_fc(num_polyps, polyp_sizes, text)

        s1 = self.s1(x1)
        s2 = self.s2(x2)
        s3 = self.s3(x3)
        s4 = self.s4(x4)

        d1 = self.d1(s4, s3)
        f1, a1 = self.a1(d1, f0)
        d2 = self.d2(a1, s2)
        f = f0 + f1
        f2, a2 = self.a2(d2, f)
        d3 = self.d3(a2, s1)
        f = f0 + f1 + f2
        f3, a3 = self.a3(d3, f)

        ag = self.ag(a1, a2, a3)
        y1 = self.y1(ag)

        # Upsample to input resolution for framework compatibility.
        if y1.shape[2:] != image.shape[2:]:
            y1 = F.interpolate(y1, size=image.shape[2:],
                               mode="bilinear", align_corners=False)

        return {
            "logits": y1,
            "num_polyps": num_polyps,
            "polyp_sizes": polyp_sizes,
        }
