"""Transformer cross-fusion skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("transformer_fusion")
class TransformerFusionSkip(nn.Module):
    """Transformer cross-fusion skip.

    Flattens decoder and skip features to (B, N, C) token sequences and
    applies a single transformer block: queries come from the decoder
    feature, while keys and values come from a linearly-projected skip
    feature (``s_proj``). The attended output is reshaped back to
    (B, C, H, W) and concatenated with the decoder feature on the channel
    axis, giving an output with ``decoder_ch + skip_ch`` channels.
    """

    def __init__(self, num_heads=4, mlp_ratio=2.0, dropout=0.0, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        # Lazy submodules keyed by (decoder_ch, skip_ch)
        self._blocks = nn.ModuleDict()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def _key(self, dc, sc):
        return f"{dc}_{sc}"

    @staticmethod
    def _pick_heads(embed_dim, requested):
        """Pick a head count that divides embed_dim, <= requested, >= 1."""
        heads = min(max(int(requested), 1), max(embed_dim, 1))
        while heads > 1 and embed_dim % heads != 0:
            heads -= 1
        return heads

    def _build(self, decoder_ch, skip_ch, device):
        key = self._key(decoder_ch, skip_ch)
        if key in self._blocks:
            return
        # Output of the transformer branch has ``skip_ch`` channels so that
        # ``cat(transformer_out, d)`` yields ``decoder_ch + skip_ch`` channels,
        # matching ``get_out_channels``.
        embed_dim = skip_ch
        heads = self._pick_heads(embed_dim, self.num_heads)

        # Q projection: maps decoder tokens (Cd) into the embed_dim space.
        q_proj = nn.Linear(decoder_ch, embed_dim)
        # K/V projection on skip stream (Cs -> embed_dim). Per the spec this is
        # the only place the skip stream is touched before attention.
        s_proj = nn.Linear(skip_ch, embed_dim)

        norm_q = nn.LayerNorm(embed_dim)
        norm_kv = nn.LayerNorm(embed_dim)
        attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=self.dropout,
            batch_first=True,
        )
        norm_mlp = nn.LayerNorm(embed_dim)
        hidden = max(int(embed_dim * self.mlp_ratio), 1)
        mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(self.dropout),
        )

        block = nn.ModuleDict({
            "q_proj": q_proj,
            "s_proj": s_proj,
            "norm_q": norm_q,
            "norm_kv": norm_kv,
            "attn": attn,
            "norm_mlp": norm_mlp,
            "mlp": mlp,
        })
        self._blocks[key] = block.to(device)

    def forward(self, decoder_feat, skip_feat):
        # Spatial align skip to decoder if needed
        if skip_feat.shape[-2:] != decoder_feat.shape[-2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        B, Cd, H, W = decoder_feat.shape
        Cs = skip_feat.shape[1]
        self._build(Cd, Cs, decoder_feat.device)
        block = self._blocks[self._key(Cd, Cs)]

        # Flatten to (B, N, C)
        d_tokens = decoder_feat.flatten(2).transpose(1, 2)  # (B, N, Cd)
        s_tokens = skip_feat.flatten(2).transpose(1, 2)     # (B, N, Cs)

        # Project Q from decoder into embed_dim, and K/V from skip via s_proj.
        q_in = block["q_proj"](d_tokens)   # (B, N, embed_dim)
        kv_in = block["s_proj"](s_tokens)  # (B, N, embed_dim)

        # Pre-norm cross-attention with a residual from the (projected) query.
        q = block["norm_q"](q_in)
        kv = block["norm_kv"](kv_in)
        attn_out, _ = block["attn"](q, kv, kv, need_weights=False)
        x = q_in + attn_out

        # MLP block with residual.
        x = x + block["mlp"](block["norm_mlp"](x))

        # Reshape back to (B, embed_dim, H, W) where embed_dim == skip_ch.
        transformer_out = x.transpose(1, 2).reshape(B, Cs, H, W)

        return torch.cat([transformer_out, decoder_feat], dim=1)
