"""Cross Attention skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("cross_attn")
class CrossAttnSkip(nn.Module):
    """Cross-attention skip: decoder attends to skip features via cross attention.

    For very high-resolution feature maps (>max_tokens), falls back to
    simple concatenation to avoid O(n^2) memory.
    """
    _MAX_TOKENS = 8192

    def __init__(self, num_heads=4, **kwargs):
        super().__init__()
        self.num_heads = num_heads

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def forward(self, decoder_feat, skip_feat):
        B, Cd, H, W = decoder_feat.shape
        Cs = skip_feat.shape[1]
        N = H * W

        # Fall back to concat if spatial size too large
        if N > self._MAX_TOKENS:
            return torch.cat([decoder_feat, skip_feat], dim=1)

        # Flatten spatial dims
        d_flat = decoder_feat.reshape(B, Cd, N).permute(0, 2, 1)  # B, N, Cd
        s_flat = skip_feat.reshape(B, Cs, N).permute(0, 2, 1)    # B, N, Cs

        # Project to common dim (use min of channels)
        dim = min(Cd, Cs)
        scale = dim ** -0.5
        q = d_flat[..., :dim]  # B, N, dim
        k = s_flat[..., :dim]  # B, N, dim
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # B, N, N
        attn = F.softmax(attn, dim=-1)
        v = s_flat  # B, N, Cs
        attended = torch.bmm(attn, v)  # B, N, Cs
        attended = attended.permute(0, 2, 1).reshape(B, Cs, H, W)

        return torch.cat([decoder_feat, attended], dim=1)
