"""ESFPNet: Efficient Sparse Feature Pyramid Network (2023).

Self-contained port for medical / polyp segmentation. The original work
(Chang et al., "ESFPNet: efficient deep learning architecture for real-time
lesion segmentation in autofluorescence bronchoscopic video", 2023) pairs a
Mix-Transformer (MiT-B*) hierarchical encoder with an Efficient Sparse
Feature Pyramid (ESFP) decoder that consists of light-weight Linear
Prediction (LP) heads, 1x1 ConvBN-ReLU fuse modules and bilinear upsamples.

This file implements the same logical decoder (LP + linear fuse + multi-scale
concat -> 1x1 head) on top of a timm-provided hierarchical Transformer
backbone with the same per-stage channel widths as MiT-B2
([64, 128, 320, 512]) and the same stride pattern ([4, 8, 16, 32]). Timm does
not currently ship a stand-alone ``mit_b2``; ``pvt_v2_b2`` is the closest
publicly available hierarchical-Transformer backbone that matches every
relevant geometry (channel widths, reductions and overlap patch embedding
philosophy) and is therefore used as the encoder. All pretrained loads are
wrapped with an SSL-fallback helper so headless / restricted environments
still produce a usable random-init model rather than crashing.

Public class: ``ESFPNet``.
"""
# Source: https://github.com/dumyCq/ESFPNet

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Linear Prediction (LP) head from the official ESFPNet repo.
# Light wrapper: flatten -> Linear -> reshape back to NCHW.
# ---------------------------------------------------------------------------
class _LP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# 1x1 Conv + BN + ReLU fuse module (stand-in for mmcv.cnn.ConvModule).
# ---------------------------------------------------------------------------
class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Hierarchical Transformer encoder via timm. MiT-B2 channels = pvt_v2_b2
# channels = [64, 128, 320, 512] with strides [4, 8, 16, 32].
# ---------------------------------------------------------------------------
def _build_backbone(in_channels: int = 3, pretrained: bool = True):
    import timm

    def _create(pretrained: bool = True):
        return timm.create_model(
            'pvt_v2_b2',
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
        )

    return load_with_ssl_fallback(_create, pretrained=pretrained)


# ---------------------------------------------------------------------------
# Public model.
# ---------------------------------------------------------------------------
class ESFPNet(nn.Module):
    """Efficient Sparse Feature Pyramid Network.

    Args:
        in_channels: number of input channels (default 3).
        num_classes: number of output segmentation classes (default 2).
        img_size: nominal training resolution. Forward accepts any size and
            will pad up to a multiple of 32 internally when the backbone
            requires it, then crop back.
        pretrained: whether to attempt loading ImageNet weights for the
            backbone. Falls back to random init on network failure.
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 img_size: int = 224,
                 pretrained: bool = True,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # Hierarchical Transformer encoder. Channel widths match MiT-B2.
        self.backbone = _build_backbone(in_channels=in_channels,
                                        pretrained=pretrained)
        embed_dims = list(self.backbone.feature_info.channels())
        # Some backbones may yield >4 stages; we only need the 4 stride-4/8/16/32 ones.
        assert len(embed_dims) >= 4, 'Backbone must expose at least 4 feature stages'
        embed_dims = embed_dims[:4]
        self.embed_dims = embed_dims

        # Determine alignment requirement (pvt_v2/MiT use 4x4 patch + 3 strided
        # downsamples -> stride 32, so input H/W should be divisible by 32).
        self._stride = 32

        # --- ESFP decoder ---------------------------------------------------
        c1, c2, c3, c4 = embed_dims

        # Per-stage Linear Prediction heads (same width as input stage).
        self.LP_1 = _LP(c1, c1)
        self.LP_2 = _LP(c2, c2)
        self.LP_3 = _LP(c3, c3)
        self.LP_4 = _LP(c4, c4)

        # 1x1 fuse modules combining adjacent levels (top-down path).
        self.linear_fuse34 = _ConvBNReLU(c3 + c4, c3, kernel_size=1)
        self.linear_fuse23 = _ConvBNReLU(c2 + c3, c2, kernel_size=1)
        self.linear_fuse12 = _ConvBNReLU(c1 + c2, c1, kernel_size=1)

        # Refined Linear Prediction heads after fusion.
        self.LP_12 = _LP(c1, c1)
        self.LP_23 = _LP(c2, c2)
        self.LP_34 = _LP(c3, c3)

        # Final 1x1 prediction across concatenated multi-scale features.
        self.linear_pred = nn.Conv2d(c1 + c2 + c3 + c4, num_classes,
                                     kernel_size=1)

    # ------------------------------------------------------------------
    def _pad_to_stride(self, x: torch.Tensor):
        _, _, h, w = x.shape
        s = self._stride
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h == 0 and pad_w == 0:
            return x, (h, w)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        x, (h_in, w_in) = self._pad_to_stride(x)

        feats = self.backbone(x)
        out_1, out_2, out_3, out_4 = feats[0], feats[1], feats[2], feats[3]

        # First-pass LP heads.
        lp_1 = self.LP_1(out_1)
        lp_2 = self.LP_2(out_2)
        lp_3 = self.LP_3(out_3)
        lp_4 = self.LP_4(out_4)

        # Top-down fuse with bilinear upsampling and LP refinement.
        lp_34 = self.LP_34(self.linear_fuse34(torch.cat([
            lp_3,
            F.interpolate(lp_4, size=lp_3.shape[-2:], mode='bilinear',
                          align_corners=False),
        ], dim=1)))

        lp_23 = self.LP_23(self.linear_fuse23(torch.cat([
            lp_2,
            F.interpolate(lp_34, size=lp_2.shape[-2:], mode='bilinear',
                          align_corners=False),
        ], dim=1)))

        lp_12 = self.LP_12(self.linear_fuse12(torch.cat([
            lp_1,
            F.interpolate(lp_23, size=lp_1.shape[-2:], mode='bilinear',
                          align_corners=False),
        ], dim=1)))

        # Bring all refined maps to the stride-4 resolution and concat.
        target_size = lp_12.shape[-2:]
        lp4_resized = F.interpolate(lp_4, size=target_size, mode='bilinear',
                                    align_corners=False)
        lp3_resized = F.interpolate(lp_34, size=target_size, mode='bilinear',
                                    align_corners=False)
        lp2_resized = F.interpolate(lp_23, size=target_size, mode='bilinear',
                                    align_corners=False)

        out = self.linear_pred(torch.cat(
            [lp_12, lp2_resized, lp3_resized, lp4_resized], dim=1))

        # Upsample logits back to original input resolution and crop padding.
        out = F.interpolate(out, size=(h_in, w_in), mode='bilinear',
                            align_corners=False)
        if (h_in, w_in) != (orig_h, orig_w):
            out = out[..., :orig_h, :orig_w]
        return out
