"""UCTransNet Skip — Channel-wise Cross Transformer (AAAI 2022).

Adapted from: https://github.com/McGregorWwww/UCTransNet
Paper: UCTransNet: Rethinking the Skip Connections in U-Net from a
       Channel-Wise Perspective (AAAI 2022, Wang et al.)

The original CTrans module operates on ALL 4 encoder features
simultaneously using:
  - **CCT** (Channel-wise Cross fusion Transformer): multi-head
    cross-attention where each scale queries against the concatenated
    key-value of all scales.
  - **CCA** (Channel-wise Cross Attention): per-scale attention between
    decoder features and encoder skip features, followed by concat.

Adapted to the framework's per-pair skip interface:
  1. Project ``decoder_feat`` and ``skip_feat`` to a unified channel dim.
  2. **CCT**: Cross-attention — decoder queries, skip provides key/value.
     ``Q=Wq(decoder), K=Wk(skip), V=Wv(skip) -> softmax(QK^T/sqrt(d))V``
     with residual connection and FFN (matching the original Block_ViT).
  3. **CCA**: Concatenate decoder and refined skip -> 3×3 conv fusion.
  4. Final 1×1 conv to produce output channels.

Output channel count: ``decoder_ch + skip_ch`` (concatenation pattern).
"""
# Source: https://github.com/McGregorWwww/UCTransNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


class _CrossChannelAttention(nn.Module):
    """CCT-style cross-channel multi-head attention.

    Decoder features serve as queries; skip features serve as key/value.
    """

    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_seq, skip_seq):
        """
        decoder_seq: (B, L, C) — queries
        skip_seq: (B, L, C) — keys and values
        """
        B, L, C = decoder_seq.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(decoder_seq).view(B, L, H, D).transpose(1, 2)
        K = self.k_proj(skip_seq).view(B, L, H, D).transpose(1, 2)
        V = self.v_proj(skip_seq).view(B, L, H, D).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, H, L, D)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        return self.out_proj(out)


class _CCTBlock(nn.Module):
    """CCT transformer block: cross-attention + FFN with residuals."""

    def __init__(self, channels, num_heads=4, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.attn_norm_d = nn.LayerNorm(channels)
        self.attn_norm_s = nn.LayerNorm(channels)
        self.cross_attn = _CrossChannelAttention(channels, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(channels)
        hidden = channels * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, decoder_seq, skip_seq):
        # Cross-attention with residual
        d_norm = self.attn_norm_d(decoder_seq)
        s_norm = self.attn_norm_s(skip_seq)
        d = decoder_seq + self.cross_attn(d_norm, s_norm)
        # FFN with residual
        d = d + self.ffn(self.ffn_norm(d))
        return d


@SKIP_REGISTRY.register("uctrans")
class UCTransSkip(nn.Module):
    """UCTransNet CTrans skip connection (per-pair adaptation).

    Uses CCT (cross-channel transformer attention) between decoder and
    skip features, followed by CCA (concatenation + conv fusion).

    Args:
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden channel multiplier.
        dropout: Dropout rate.
    """

    def __init__(self, num_heads: int = 4, mlp_ratio: int = 4,
                 dropout: float = 0.0, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self._cache: dict = {}

    def get_out_channels(self, decoder_ch: int, skip_ch: int) -> int:
        return decoder_ch + skip_ch

    def _build(self, decoder_ch: int, skip_ch: int, device):
        key = (decoder_ch, skip_ch, str(device))
        if key in self._cache:
            return self._cache[key]

        unified = max(decoder_ch, skip_ch)
        # Ensure unified is divisible by num_heads
        while unified % self.num_heads != 0:
            unified += 1

        dec_proj = nn.Conv2d(decoder_ch, unified, 1, bias=False).to(device)
        skip_proj = nn.Conv2d(skip_ch, unified, 1, bias=False).to(device)

        # CCT transformer block
        cct = _CCTBlock(unified, num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        dropout=self.dropout).to(device)

        # CCA: concat fusion (unified + unified = 2*unified) -> conv
        cca_fuse = nn.Sequential(
            nn.Conv2d(unified * 2, decoder_ch + skip_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(decoder_ch + skip_ch),
            nn.ReLU(inplace=True),
        ).to(device)

        mod = nn.ModuleDict({
            "dec_proj": dec_proj,
            "skip_proj": skip_proj,
            "cct": cct,
            "cca_fuse": cca_fuse,
        })
        safe_name = (f"_uctrans_{decoder_ch}_{skip_ch}_"
                     f"{str(device).replace(':', '_')}")
        setattr(self, safe_name, mod)
        self._cache[key] = mod
        return mod

    def forward(self, decoder_feat: torch.Tensor,
                skip_feat: torch.Tensor) -> torch.Tensor:
        B, _, H, W = decoder_feat.shape

        # Spatial align skip to decoder if needed
        if skip_feat.shape[2:] != decoder_feat.shape[2:]:
            skip_feat = F.interpolate(
                skip_feat, size=(H, W),
                mode='bilinear', align_corners=False
            )

        dec_ch = decoder_feat.shape[1]
        skip_ch = skip_feat.shape[1]
        mod = self._build(dec_ch, skip_ch, decoder_feat.device)

        # Project to unified channels
        d = mod["dec_proj"](decoder_feat)  # (B, unified, H, W)
        s = mod["skip_proj"](skip_feat)    # (B, unified, H, W)

        # Flatten spatial -> sequence: (B, H*W, unified)
        L = H * W
        d_seq = d.flatten(2).transpose(1, 2)
        s_seq = s.flatten(2).transpose(1, 2)

        # CCT: cross-channel transformer attention
        d_refined = mod["cct"](d_seq, s_seq)  # (B, L, unified)

        # Reshape back to 2D
        unified = d.shape[1]
        d_2d = d_refined.transpose(1, 2).view(B, unified, H, W)

        # CCA: concatenation + conv fusion
        fused = torch.cat([d_2d, s], dim=1)  # (B, 2*unified, H, W)
        return mod["cca_fuse"](fused)  # (B, decoder_ch + skip_ch, H, W)
