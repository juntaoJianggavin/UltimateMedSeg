"""KiU-Net Decoder – representative dual-branch (under-/over-complete) decoder.

Inspired by KiU-Net (Valanarasu et al., MICCAI 2020 / TMI 2021) which pairs a
classic U-Net (down-sampling) branch with a Ki-Net (up-sampling, over-complete)
branch. Their cross-residual fusion encourages the network to capture both
coarse contextual and fine boundary detail.

For each decoder stage at the resolution of the next shallower encoder skip:
    1. Up-branch:   Conv -> ReLU -> bilinear-upsample to skip size
    2. Down-branch: Conv -> ReLU -> max-pool (then upsampled back to skip size)
    Both project the bottleneck/decoder tensor to the encoder skip's channel
    count, are concatenated together with the encoder skip (via the externally
    supplied ``skip_connection`` module on the concatenated dual-branch tensor),
    and a 3x3 conv mixes the fused features back to skip_ch.

If ``skip_connection`` is ``None`` (or the externally injected peer module
errors), the stage falls back to a plain single-branch UNet-style decode
(up-branch only, plain channel concat with skip).
"""
# Source: https://github.com/jeya-maria-jose/KiU-Net-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class _DualBranchStage(nn.Module):
    """A single KiU-Net decoder stage.

    Takes the current decoder tensor (``in_ch`` channels) and a target skip
    spatial size, producing the two branch tensors (each ``skip_ch`` channels)
    at the skip's spatial size.
    """

    def __init__(self, in_ch: int, skip_ch: int):
        super().__init__()
        # Up-branch: Conv -> BN -> ReLU; bilinear-upsample done in forward.
        self.up_proj = nn.Sequential(
            nn.Conv2d(in_ch, skip_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(inplace=True),
        )
        # Down-branch: Conv -> BN -> ReLU; max-pool done in forward.
        self.down_proj = nn.Sequential(
            nn.Conv2d(in_ch, skip_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(inplace=True),
        )
        self.down_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor, skip_size):
        # Up-branch: bilinear upsample to skip spatial size.
        up = self.up_proj(x)
        if up.shape[2:] != skip_size:
            up = F.interpolate(up, size=skip_size, mode='bilinear',
                               align_corners=False)
        # Down-branch: max-pool (over-complete style going coarser first),
        # then bring back up to the skip resolution so we can concat.
        down = self.down_proj(x)
        down = self.down_pool(down)
        if down.shape[2:] != skip_size:
            down = F.interpolate(down, size=skip_size, mode='bilinear',
                                 align_corners=False)
        return up, down


@DECODER_REGISTRY.register("kiunet")
class KiUNetDecoder(nn.Module):
    """KiU-Net style dual-branch decoder.

    Args:
        encoder_channels: Encoder skip channels (shallow -> deep). Matches the
            framework convention used by other decoders in this repo. May
            optionally include the bottleneck channel as the deepest entry;
            the decoder pairs stages with the actual ``skip_features`` length
            at forward time.
        bottleneck_channels: Channel count of the bottleneck feature handed to
            ``forward``.
        skip_connection: External skip fusion module. The decoder fuses the
            concatenated [up_branch, down_branch] dual-branch tensor with the
            encoder skip via this module. Falls back to plain UNet single-branch
            concat when ``None`` or when the peer raises.
        img_size: Input spatial size (kept for API symmetry; the final layer
            interpolates to the shallowest skip's spatial scale).
    """

    # Framework hands skips externally; we call ``self.skip_connection``.
    has_internal_skip = False

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection: nn.Module = None, img_size: int = 224,
                 **kwargs):
        super().__init__()
        self.skip_connection = skip_connection
        self.img_size = img_size
        self.encoder_channels = list(encoder_channels)
        self.bottleneck_channels = bottleneck_channels

        # We walk skips from deepest to shallowest.
        skip_channels_deep_first = list(reversed(encoder_channels))

        self.stages = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        # ``stage_in_channels`` lets us pick the right entry point when the
        # framework supplies fewer skips than encoder_channels (e.g. the
        # deepest "encoder channel" is actually the bottleneck level).
        self.stage_in_channels: List[int] = []
        self.stage_skip_channels: List[int] = []

        in_ch = bottleneck_channels
        for skip_ch in skip_channels_deep_first:
            self.stage_in_channels.append(in_ch)
            self.stage_skip_channels.append(skip_ch)
            self.stages.append(_DualBranchStage(in_ch, skip_ch))

            # Fused channel count after concat(up, down) -> 2*skip_ch, then
            # combined with the skip via ``skip_connection``.
            dual_ch = 2 * skip_ch
            if skip_connection is not None and hasattr(skip_connection,
                                                      "get_out_channels"):
                merged_ch = skip_connection.get_out_channels(dual_ch, skip_ch)
            else:
                merged_ch = dual_ch + skip_ch  # plain concat fallback

            self.fuse_convs.append(nn.Sequential(
                nn.Conv2d(merged_ch, skip_ch, kernel_size=3, padding=1,
                          bias=False),
                nn.BatchNorm2d(skip_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = skip_ch

        # Per spec: out_channels = shallowest encoder channel.
        self._out_channels = (encoder_channels[0] if encoder_channels
                              else bottleneck_channels)

        # 3x3 conv at the top encoder scale (after final interpolation).
        self.final_conv = nn.Sequential(
            nn.Conv2d(self._out_channels, self._out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(self._out_channels),
            nn.ReLU(inplace=True),
        )

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def _single_branch_fallback(self, x: torch.Tensor, skip: torch.Tensor,
                                stage: _DualBranchStage,
                                fuse_conv: nn.Module) -> torch.Tensor:
        """Plain UNet single-branch decode: up-branch only, plain concat."""
        up = stage.up_proj(x)
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='bilinear',
                               align_corners=False)
        # Build a tensor with the channel count fuse_conv expects (constructed
        # for the dual-branch path: 2*skip_ch (+ skip_ch from concat) or
        # whatever ``skip_connection.get_out_channels`` told us). We mimic the
        # dual branch by duplicating ``up`` for the down-branch slot.
        dual = torch.cat([up, up], dim=1)
        if self.skip_connection is not None:
            fused = self.skip_connection(dual, skip)
        else:
            fused = torch.cat([dual, skip], dim=1)
        expected = fuse_conv[0].in_channels
        if fused.shape[1] != expected:
            adapt = nn.Conv2d(fused.shape[1], expected, kernel_size=1,
                              bias=False).to(fused.device)
            fused = adapt(fused)
        return fuse_conv(fused)

    def forward(self, bottleneck_feat: torch.Tensor,
                skip_features: List[torch.Tensor]) -> torch.Tensor:
        # Skips are framework-supplied shallow -> deep; walk deep -> shallow.
        skips = list(reversed(skip_features))

        # Pair each skip with the matching stage. If the user constructed the
        # decoder with one extra encoder_channel (the bottleneck level), we
        # offset so the deepest skip lines up with the matching skip_ch stage.
        num_stages = len(self.stages)
        num_skips = len(skips)
        offset = max(0, num_stages - num_skips)

        x = bottleneck_feat
        for i, skip in enumerate(skips):
            si = offset + i
            if si >= num_stages:
                break
            stage = self.stages[si]
            fuse_conv = self.fuse_convs[si]
            up, down = stage(x, skip.shape[2:])
            dual = torch.cat([up, down], dim=1)
            if self.skip_connection is not None:
                fused = self.skip_connection(dual, skip)
            else:
                fused = torch.cat([dual, skip], dim=1)
            x = fuse_conv(fused)

        # Final interpolation up to the shallowest encoder/skip scale.
        if skips:
            top_size = (skips[-1].shape[2], skips[-1].shape[3])
            if x.shape[2:] != top_size:
                x = F.interpolate(x, size=top_size, mode='bilinear',
                                  align_corners=False)
        # Ensure channel count matches final_conv (it was built for
        # ``self._out_channels``; rare mismatch when no skips are provided).
        expected = self.final_conv[0].in_channels
        if x.shape[1] != expected:
            adapt = nn.Conv2d(x.shape[1], expected, kernel_size=1,
                              bias=False).to(x.device)
            x = adapt(x)
        x = self.final_conv(x)
        return x
