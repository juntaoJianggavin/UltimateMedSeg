"""MISSFormer Decoder: faithful port from https://github.com/ZhifangDeng/MISSFormer

Reference: Huang et al., "MISSFormer: An Effective Transformer for 2D Medical Image Segmentation"
File: MISSFormer.py

Decoder components: PatchExpand, FinalPatchExpand_X4, MyDecoderLayer
Uses TransformerBlock from missformer_encoder for processing.

Has its own internal skip connection mechanism (concat + linear + transformer).
External skip_connection parameter is IGNORED.
"""
# Source: https://github.com/ZhifangDeng/MISSFormer

import torch
import torch.nn as nn
from typing import List
from einops import rearrange

from medseg.registry import DECODER_REGISTRY
from medseg.models.encoders.missformer_encoder import TransformerBlock


class PatchExpand(nn.Module):
    """Patch expanding layer for 2x upsampling (from original MISSFormer)."""

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x, dynamic_hw=None):
        if dynamic_hw is not None:
            H, W = dynamic_hw
        else:
            H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        if L != H * W:
            # Infer actual H, W from token count (assume square)
            import math
            side = int(math.isqrt(L))
            if side * side == L:
                H = W = side
            else:
                raise RuntimeError(
                    f"PatchExpand: token count {L} incompatible with "
                    f"expected ({H}x{W}).")

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())
        return x


class FinalPatchExpand_X4(nn.Module):
    """Final 4x patch expanding for full resolution recovery."""

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x, dynamic_hw=None):
        if dynamic_hw is not None:
            H, W = dynamic_hw
        else:
            H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        if L != H * W:
            import math
            side = int(math.isqrt(L))
            if side * side == L:
                H = W = side
            else:
                raise RuntimeError(
                    f"FinalPatchExpand_X4: token count {L} incompatible with "
                    f"expected ({H}x{W}).")

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                       p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())
        return x


class MyDecoderLayer(nn.Module):
    """MISSFormer decoder layer: concat skip + linear + 2x TransformerBlock + PatchExpand.

    Faithful to original MyDecoderLayer from MISSFormer.py.
    """

    def __init__(self, input_size, in_ch, out_ch, skip_ch, head, reduction_ratio,
                 token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        if not is_last:
            self.concat_linear = nn.Linear(in_ch + skip_ch, out_ch)
            self.layer_up = PatchExpand(
                input_resolution=input_size, dim=out_ch, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(in_ch + skip_ch, out_ch)
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_ch, dim_scale=4, norm_layer=norm_layer)
            self.last_layer = True  # flag for forward

        self.layer_former_1 = TransformerBlock(out_ch, head, reduction_ratio, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_ch, head, reduction_ratio, token_mlp_mode)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1, x2=None, dynamic_hw=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            cat_linear_x = self.concat_linear(cat_x)
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.layer_up(tran_layer_2, dynamic_hw=(h, w))
            else:
                out = self.layer_up(tran_layer_2, dynamic_hw=(h, w))
        else:
            out = self.layer_up(x1, dynamic_hw=dynamic_hw)
        return out


@DECODER_REGISTRY.register("missformer")
class MISSFormerDecoder(nn.Module):
    """MISSFormer Transformer decoder.

    Faithful to the original MISSFormer decoder architecture.
    Architecture:
        decoder_3: PatchExpand(bottleneck) ->
        decoder_2: concat + linear + 2x TransformerBlock + PatchExpand ->
        decoder_1: concat + linear + 2x TransformerBlock + PatchExpand ->
        decoder_0: concat + linear + 2x TransformerBlock + FinalPatchExpand_X4

    External skip_connection is IGNORED.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None,
                 img_size: int = 224,
                 token_mlp_mode: str = "mix_skip",
                 **kwargs):
        super().__init__()
        # MISSFormer's internal heads/reduction_ratios are hard-coded for
        # specific channel dims [64, 128, 320, 512]. Use these expected dims
        # for the decoder layers and adapt incoming encoder features via
        # per-stage 1x1 conv channel adapters.
        self._expected_dims = (64, 128, 320, 512)

        # Per-stage 1x1 channel adapters (only built when encoder channels
        # differ from internal expected dims).
        self._channel_adapters = nn.ModuleList()
        for actual_ch, exp_ch in zip(encoder_channels, self._expected_dims):
            if actual_ch != exp_ch:
                self._channel_adapters.append(
                    nn.Conv2d(actual_ch, exp_ch, kernel_size=1, bias=False))
            else:
                self._channel_adapters.append(nn.Identity())

        # Adapter for the bottleneck feature (uses the deepest expected dim).
        expected_bottleneck = self._expected_dims[len(encoder_channels) - 1] \
            if len(encoder_channels) <= len(self._expected_dims) else self._expected_dims[-1]
        # If there's a dedicated bottleneck slot beyond skip dims, use 512.
        expected_bottleneck = 512
        if bottleneck_channels != expected_bottleneck:
            self._bottleneck_adapter = nn.Conv2d(
                bottleneck_channels, expected_bottleneck, kernel_size=1, bias=False)
        else:
            self._bottleneck_adapter = nn.Identity()

        # Use expected dims (not raw encoder_channels) to build the rest of
        # the decoder, so internal heads/reduction_ratios match.
        adapted_encoder_channels = list(self._expected_dims[:len(encoder_channels)])
        bottleneck_channels = expected_bottleneck

        # encoder_channels = [64, 128, 320] for skip features
        # bottleneck_channels = 512
        dims = list(adapted_encoder_channels) + [bottleneck_channels]
        n_stages = len(dims)  # e.g. 4 stages

        # Compute base feat size from img_size and number of stages
        # Each stage downsamples by 2x from patch_embed stride, so total is 4 * 2^(n-1)
        d_base = img_size // (4 * 2 ** (n_stages - 1))
        if d_base < 1:
            d_base = 1

        # Decoder heads and reduction ratios (mirror encoder)
        heads = [1, 2, 5, 8][:n_stages]
        reduction_ratios = [8, 4, 2, 1][:n_stages]

        # Build decoder layers (from deepest to shallowest)
        self.decoders = nn.ModuleList()

        # decoder_N-1: just PatchExpand from bottleneck (no skip concat)
        # The PatchExpand on bottleneck_channels gives bottleneck_channels // 2
        pe_out = bottleneck_channels // 2
        first_pe = PatchExpand(
            input_resolution=(d_base, d_base),
            dim=bottleneck_channels, dim_scale=2)
        self.first_expand = first_pe
        self.first_expand_out = pe_out

        # Remaining decoder layers
        prev_ch = pe_out
        for i in range(n_stages - 2, -1, -1):
            skip_ch = dims[i]
            out_ch = dims[i]
            h_size = d_base * (2 ** (n_stages - 1 - i))
            is_last = (i == 0)
            head = heads[i]
            rr = reduction_ratios[i]

            layer = MyDecoderLayer(
                input_size=(h_size, h_size),
                in_ch=prev_ch,
                out_ch=out_ch,
                skip_ch=skip_ch,
                head=head,
                reduction_ratio=rr,
                token_mlp_mode=token_mlp_mode,
                is_last=is_last)
            self.decoders.append(layer)
            if not is_last:
                prev_ch = out_ch // 2  # PatchExpand halves channels
            else:
                prev_ch = out_ch  # FinalPatchExpand keeps channels

        self._out_channels = dims[0]
        self.img_size = img_size

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        import math as _math
        # Apply per-stage channel adapters so internal decoder receives the
        # expected channel dims [64, 128, 320, 512].
        skip_features = [adapter(s) for adapter, s in zip(self._channel_adapters, skip_features)]
        bottleneck_feat = self._bottleneck_adapter(bottleneck_feat)

        B = bottleneck_feat.shape[0]
        # Actual bottleneck spatial size (may differ from design-time d_base
        # when the encoder has a different downsampling ratio, e.g. MedNeXt /8)
        bh, bw = bottleneck_feat.shape[2], bottleneck_feat.shape[3]

        # Convert bottleneck from (B, C, H, W) to (B, L, C)
        x = bottleneck_feat.flatten(2).transpose(1, 2)

        # First expand: PatchExpand on bottleneck
        x = self.first_expand(x, dynamic_hw=(bh, bw))
        cur_h, cur_w = bh * 2, bw * 2  # PatchExpand doubles spatial

        # Process decoder layers with skip connections
        for i, dec_layer in enumerate(self.decoders):
            skip_idx = len(skip_features) - 1 - i
            if 0 <= skip_idx < len(skip_features):
                skip = skip_features[skip_idx]
                # Convert skip to (B, H, W, C) format expected by MyDecoderLayer
                skip_bhwc = skip.permute(0, 2, 3, 1)  # (B, H, W, C)
                x = dec_layer(x, skip_bhwc)
                # After PatchExpand with skip's h,w as input, output is 2*h, 2*w
                cur_h = skip.shape[2] * 2
                cur_w = skip.shape[3] * 2
            else:
                x = dec_layer(x, dynamic_hw=(cur_h, cur_w))
                cur_h, cur_w = cur_h * 2, cur_w * 2

        # Convert from (B, L, C) to (B, C, H, W)
        import math
        L = x.shape[1]
        side = int(_math.isqrt(L))
        if side * side == L:
            H_out = W_out = side
        else:
            H_out = cur_h
            W_out = cur_w
        x = x.view(B, H_out, W_out, -1).permute(0, 3, 1, 2).contiguous()

        # Upsample to original input size if needed
        if H_out != self.img_size or W_out != self.img_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False)
        return x
