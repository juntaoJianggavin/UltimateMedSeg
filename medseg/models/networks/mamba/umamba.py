"""U-Mamba 2D: UNet with Mamba State Space Model for Medical Image Segmentation.

Faithful reimplementation from:
  https://github.com/bowang-lab/U-Mamba  (MICCAI 2024)

Architecture follows the original nnU-Net-based implementation:
  - Encoder: stem (stride-1 stage) + strided stages
  - Decoder: UpsampleLayer + concat skip + BasicResBlock, seg_layers for deep supervision
  - UMambaBot: UNetResEncoder + MambaLayer at bottleneck + UNetResDecoder
  - UMambaEnc: ResidualMambaEncoder (alternating MambaLayer) + UNetResDecoder

Requires ``mamba_ssm`` (and ``causal-conv1d``) — the official U-Mamba source
(https://github.com/bowang-lab/U-Mamba) hard-depends on the CUDA-accelerated
selective scan kernel. A hand-rolled SSM approximation would be numerically
different and silently change model behaviour, so no fallback is provided.
"""
# Source: https://github.com/bowang-lab/U-Mamba

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union


# ---------------------------------------------------------------------------
# Mamba SSM wrapper (hard dependency on mamba_ssm, matching official source)
# ---------------------------------------------------------------------------

class MambaSSM(nn.Module):
    """Mamba SSM wrapper. Hard-depends on ``mamba_ssm`` (matches official source).

    The official U-Mamba and LightM-UNet repos both require
    ``pip install mamba-ssm`` and ``pip install causal-conv1d``.
    No pure-PyTorch fallback is provided because a hand-rolled SSM
    approximation would be numerically different from the official
    selective scan and would silently change model behaviour.

    Interface: (B, L, D) -> (B, L, D)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.d_state = d_state
        self.d_conv = d_conv

        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise RuntimeError(
                "MambaBlock requires the `mamba_ssm` CUDA package. "
                "Install from https://github.com/state-spaces/mamba. "
                "The official U-Mamba source hard-depends on mamba_ssm; "
                "no pure-PyTorch fallback is provided."
            ) from e
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand)

    def forward(self, x):
        """x: (B, L, D) -> (B, L, D)"""
        return self.mamba(x)


# Backward-compatible alias for existing imports
MambaSSMFallback = MambaSSM


# ---------------------------------------------------------------------------
# MambaLayer for 2D features (faithful to source, with channel_token mode)
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    """Mamba layer for 2D features with optional channel_token mode.

    Faithful to the original U-Mamba implementation:
      - forward_patch_token: spatial tokens (H*W) with channel features (default)
      - forward_channel_token: channel tokens (C) with spatial features (H*W)
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = MambaSSM(dim, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        """(B, C, H, W) -> flatten spatial -> Mamba -> reshape."""
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)  # (B, N, C)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        return x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)

    def forward_channel_token(self, x):
        """(B, C, H, W) -> channels as tokens -> Mamba -> reshape."""
        B, n_tokens = x.shape[:2]  # n_tokens = C
        d_model = x.shape[2:].numel()  # d_model = H*W
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)  # (B, C, H*W)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        return x_mamba.reshape(B, n_tokens, *img_dims)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


# ---------------------------------------------------------------------------
# Building blocks (faithful to source UMambaBot_2d.py / UMambaEnc_2d.py)
# ---------------------------------------------------------------------------

class UpsampleLayer(nn.Module):
    """Upsample by interpolation + 1x1 conv (faithful to source)."""
    def __init__(self, input_channels, output_channels, pool_op_kernel_size,
                 mode='nearest'):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class BasicResBlock(nn.Module):
    """Residual block faithful to source: conv->norm->act, conv->norm, +skip->act.

    Uses InstanceNorm2d + LeakyReLU (standard nnU-Net configuration).
    """
    def __init__(self, input_channels, output_channels, kernel_size=3,
                 padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.norm1 = nn.InstanceNorm2d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size,
                               padding=padding, bias=False)
        self.norm2 = nn.InstanceNorm2d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels,
                                   kernel_size=1, stride=stride, bias=False)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class BasicBlockD(nn.Module):
    """Additional residual block (faithful to dynamic_network_architectures).

    Same-resolution residual block: conv->norm->act, conv->norm, +skip->act.
    """
    def __init__(self, channels, kernel_size=3, conv_bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding,
                               bias=conv_bias)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding,
                               bias=conv_bias)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act2(y + x)


# ---------------------------------------------------------------------------
# Encoder: stem + strided stages (faithful to source UNetResEncoder)
# ---------------------------------------------------------------------------

class UNetResEncoder(nn.Module):
    """UNet encoder with stem + strided stages (faithful to source).

    Stem processes input at stride=1 with n_blocks_per_stage[0] blocks.
    Stages process with given strides. Stem output is NOT included in skips.

    Attributes stored for decoder reference:
        output_channels, strides, kernel_sizes, conv_pad_sizes, conv_bias
    """
    def __init__(self, input_channels, n_stages, features_per_stage,
                 kernel_sizes=None, strides=None, n_blocks_per_stage=None,
                 conv_bias=False):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if kernel_sizes is None:
            kernel_sizes = [3] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if strides is None:
            strides = [1] + [2] * (n_stages - 1)
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages

        self.output_channels = list(features_per_stage)
        self.strides = list(strides)
        self.kernel_sizes = list(kernel_sizes)
        self.conv_pad_sizes = [k // 2 for k in kernel_sizes]
        self.conv_bias = conv_bias

        # Stem: stride-1, uses features_per_stage[0]
        stem_ch = features_per_stage[0]
        ks0 = kernel_sizes[0]
        pad0 = ks0 // 2
        self.stem = nn.Sequential(
            BasicResBlock(input_channels, stem_ch, ks0, pad0,
                          stride=1, use_1x1conv=True),
            *[BasicBlockD(stem_ch, ks0, conv_bias)
              for _ in range(n_blocks_per_stage[0] - 1)]
        )

        # Stages 1..n_stages-1 (stem is stage 0, its output is NOT in skips)
        self.stages = nn.ModuleList()
        ch = stem_ch
        for s in range(1, n_stages):
            ks = kernel_sizes[s]
            pad = ks // 2
            stage = nn.Sequential(
                BasicResBlock(ch, features_per_stage[s], ks, pad,
                              stride=strides[s], use_1x1conv=True),
                *[BasicBlockD(features_per_stage[s], ks, conv_bias)
                  for _ in range(n_blocks_per_stage[s] - 1)]
            )
            self.stages.append(stage)
            ch = features_per_stage[s]

    def forward(self, x):
        x = self.stem(x)
        ret = []
        for stage in self.stages:
            x = stage(x)
            ret.append(x)
        return ret


# ---------------------------------------------------------------------------
# Decoder: faithful to source UNetResDecoder
# ---------------------------------------------------------------------------

class UNetResDecoder(nn.Module):
    """UNet decoder with built-in seg_layers and deep supervision (faithful to source).

    Key source-faithful details:
      - seg_layers (1x1 conv -> num_classes) at each decoder stage
      - Last decoder stage does NOT concatenate with skip
      - Returns reversed seg_outputs for deep supervision
      - Without deep_supervision, returns only final (shallowest) prediction
    """
    def __init__(self, encoder, num_classes, n_conv_per_stage=None,
                 deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        n_stages_encoder = len(encoder.output_channels)
        if n_conv_per_stage is None:
            n_conv_per_stage = [2] * (n_stages_encoder - 1)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        stages = []
        upsample_layers = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]
            ks = encoder.kernel_sizes[-(s + 1)]
            pad = ks // 2

            upsample_layers.append(UpsampleLayer(
                input_features_below, input_features_skip,
                pool_op_kernel_size=stride_for_upsampling, mode='nearest'))

            # Last decoder stage: no skip concatenation
            in_ch = (2 * input_features_skip
                     if s < n_stages_encoder - 1
                     else input_features_skip)
            stages.append(nn.Sequential(
                BasicResBlock(in_ch, input_features_skip, ks, pad,
                              stride=1, use_1x1conv=True),
                *[BasicBlockD(input_features_skip, ks, encoder.conv_bias)
                  for _ in range(n_conv_per_stage[s - 1] - 1)]
            ))
            seg_layers.append(nn.Conv2d(input_features_skip, num_classes,
                                        1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """skips: list of encoder stage outputs (shallow to deep)."""
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            # Last decoder stage does NOT concatenate with skip (faithful to source)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # Reverse: deepest prediction last -> shallowest prediction first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision or not self.training:
            return seg_outputs[0]
        else:
            return seg_outputs


# ---------------------------------------------------------------------------
# ResidualMambaEncoder (faithful to source: alternating MambaLayers)
# ---------------------------------------------------------------------------

class ResidualMambaEncoder(nn.Module):
    """Encoder with alternating MambaLayers at each stage (faithful to source).

    Key source-faithful details:
      - Alternating pattern: bool(s%2) ^ bool(n_stages%2) guarantees last stage has Mamba
      - channel_token mode when spatial tokens <= channel count
      - Mamba applied directly (no residual addition from stage)
    """
    def __init__(self, input_size, input_channels, n_stages, features_per_stage,
                 kernel_sizes=None, strides=None, n_blocks_per_stage=None,
                 conv_bias=False, mamba_d_state=16, mamba_d_conv=4,
                 mamba_expand=2):
        super().__init__()
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if kernel_sizes is None:
            kernel_sizes = [3] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if strides is None:
            strides = [1] + [2] * (n_stages - 1)
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages

        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.output_channels = list(features_per_stage)
        self.strides = list(strides)
        self.kernel_sizes = list(kernel_sizes)
        self.conv_pad_sizes = [k // 2 for k in kernel_sizes]
        self.conv_bias = conv_bias

        # Compute feature map sizes and channel_token flags for all n_stages
        # (index 0 = stem level, 1..n_stages-1 = conv stage levels)
        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = list(input_size)
        for s in range(n_stages):
            feature_map_size = [sz // strides[s] for sz in feature_map_size]
            feature_map_sizes.append(list(feature_map_size))
            if int(np.prod(feature_map_size)) <= features_per_stage[s]:
                do_channel_token[s] = True

        # Stem (stage 0)
        stem_ch = features_per_stage[0]
        ks0 = kernel_sizes[0]
        pad0 = ks0 // 2
        self.stem = nn.Sequential(
            BasicResBlock(input_channels, stem_ch, ks0, pad0,
                          stride=1, use_1x1conv=True),
            *[BasicBlockD(stem_ch, ks0, conv_bias)
              for _ in range(n_blocks_per_stage[0] - 1)]
        )

        # Alternating Mamba layers for ALL stages (index 0=stem, 1..n_stages-1)
        # Pattern: bool(s%2) ^ bool(n_stages%2) guarantees last stage has Mamba
        mamba_layers = []
        for s in range(n_stages):
            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_dim = (int(np.prod(feature_map_sizes[s]))
                             if do_channel_token[s]
                             else features_per_stage[s])
                mamba_layers.append(MambaLayer(
                    dim=mamba_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    channel_token=do_channel_token[s]))
            else:
                mamba_layers.append(nn.Identity())
        self.mamba_layers = nn.ModuleList(mamba_layers)

        # Conv stages 1..n_stages-1 (stem is stage 0, not included here)
        stages = []
        ch = stem_ch
        for s in range(1, n_stages):
            ks = kernel_sizes[s]
            pad = ks // 2
            stage = nn.Sequential(
                BasicResBlock(ch, features_per_stage[s], ks, pad,
                              stride=strides[s], use_1x1conv=True),
                *[BasicBlockD(features_per_stage[s], ks, conv_bias)
                  for _ in range(n_blocks_per_stage[s] - 1)]
            )
            stages.append(stage)
            ch = features_per_stage[s]
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.stem(x)
        x = self.mamba_layers[0](x)  # Mamba on stem output
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s + 1](x)  # Mamba on stage output
            ret.append(x)
        return ret


# ---------------------------------------------------------------------------
# UMambaBot: UNet encoder + MambaLayer at bottleneck + UNet decoder
# ---------------------------------------------------------------------------

class UMambaBot(nn.Module):
    """U-Mamba Bot: Standard UNet with MambaLayer at bottleneck.

    Faithful to source: encoder(x) -> mamba at bottleneck -> decoder(skips).
    Deep supervision is handled by the decoder (seg_layers at each stage).

    Args:
        in_channels: Input channels.
        num_classes: Output classes.
        img_size: Input image size.
        features: Channel counts per encoder stage.
        n_blocks_per_stage: ResBlocks per stage (modified per source convention).
        strides: Stride per encoder stage (default: [1, 2, 2, ...]).
        kernel_sizes: Kernel size per stage (default: [3, 3, ...]).
        n_conv_per_stage_decoder: ResBlocks per decoder stage.
        mamba_d_state: Mamba SSM state dimension.
        mamba_d_conv: Mamba SSM conv width.
        mamba_expand: Mamba SSM expansion factor.
        deep_supervision: If True, return multi-scale predictions.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 features=None, n_blocks_per_stage=None,
                 strides=None, kernel_sizes=None,
                 n_conv_per_stage_decoder=None,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 deep_supervision=False, **kwargs):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]
        n_stages = len(features)

        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        n_blocks_per_stage = list(n_blocks_per_stage)

        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        n_conv_per_stage_decoder = list(n_conv_per_stage_decoder)

        # Source convention: reduce blocks in deeper encoder/decoder stages
        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        self.encoder = UNetResEncoder(
            in_channels, n_stages, features,
            kernel_sizes=kernel_sizes, strides=strides,
            n_blocks_per_stage=n_blocks_per_stage)

        self.mamba_layer = MambaLayer(
            dim=features[-1], d_state=mamba_d_state,
            d_conv=mamba_d_conv, expand=mamba_expand)

        self.decoder = UNetResDecoder(
            self.encoder, num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        # Apply Mamba at bottleneck (faithful to source: replaces skips[-1])
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)


# ---------------------------------------------------------------------------
# UMambaEnc: ResidualMambaEncoder + UNet decoder
# ---------------------------------------------------------------------------

class UMambaEnc(nn.Module):
    """U-Mamba Enc: ResidualMambaEncoder + UNet decoder.

    Faithful to source: alternating MambaLayers in encoder, deep supervision in decoder.

    Args:
        in_channels: Input channels.
        num_classes: Output classes.
        img_size: Input image size.
        features: Channel counts per encoder stage.
        n_blocks_per_stage: ResBlocks per stage (modified per source convention).
        strides: Stride per encoder stage.
        kernel_sizes: Kernel size per stage.
        n_conv_per_stage_decoder: ResBlocks per decoder stage.
        mamba_d_state: Mamba SSM state dimension.
        mamba_d_conv: Mamba SSM conv width.
        mamba_expand: Mamba SSM expansion factor.
        deep_supervision: If True, return multi-scale predictions.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 features=None, n_blocks_per_stage=None,
                 strides=None, kernel_sizes=None,
                 n_conv_per_stage_decoder=None,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 deep_supervision=False, **kwargs):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]
        n_stages = len(features)

        if isinstance(img_size, int):
            input_size = (img_size, img_size)
        else:
            input_size = tuple(img_size)

        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        n_blocks_per_stage = list(n_blocks_per_stage)

        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        n_conv_per_stage_decoder = list(n_conv_per_stage_decoder)

        # Source convention: reduce blocks in deeper stages
        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        self.encoder = ResidualMambaEncoder(
            input_size, in_channels, n_stages, features,
            kernel_sizes=kernel_sizes, strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand)

        self.decoder = UNetResDecoder(
            self.encoder, num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
