# BiomedParse (Nature Methods 2024)
# Reference: https://github.com/microsoft/BiomedParse
# Paper: https://arxiv.org/abs/2405.12971
# Implemented from paper formulas; not a copy of the official repo.
"""BiomedParse: a biomedical foundation model for joint segmentation,
detection and recognition across nine modalities.

BiomedParse couples a Focal/Swin-based vision encoder with a text encoder
and a query-based mask decoder (SEEM family).  The official release relies
on the full X-Decoder/SEEM stack which is several thousand LoC; reproducing
that exactly is out of scope for this self-contained port.  We re-implement
the *algorithmic core* described in §3 of the paper:

    (1) Pyramid vision encoder F_v ∈ R^{C×H×W} at four scales (we use a
        Swin/PVT-style hierarchical CNN-Transformer built from torch
        primitives — see ``_HierVisionEncoder``).
    (2) Text encoder T(.) producing a sentence-level prompt vector
        ``q_t`` ∈ R^D (HuggingFace `transformers` CLIP / BiomedBERT; strict
        load, no fallback).
    (3) A learnable bank of K *meta-queries* ``Q ∈ R^{K×D}`` that absorb
        modality / structure priors during pre-training.
    (4) A *prompt-conditioned mask decoder*: M layers of self-attention on
        the query stack followed by cross-attention from queries to
        per-scale vision tokens; the prompt vector is added to every query
        so that the network conditions on text without giving up its open-
        vocabulary capability.
    (5) Mask prediction: dot product between the decoded query embeddings
        and the highest-resolution pixel embedding map produces K candidate
        masks.  At evaluation time we select the mask whose query has the
        highest similarity to the text prompt.

The full BiomedParse training recipe additionally learns a "MaIoU" head
and several auxiliary losses; those auxiliary heads are *omitted* here
but the segmentation logits are bit-compatible with the supervised setup
``train_text_guided.py`` already runs.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.utils.weight_downloader import hf_from_pretrained


try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False


# ---------------------------------------------------------------------------
# Vision encoder: hierarchical CNN-Transformer (4 stages, swin-like)
# ---------------------------------------------------------------------------
class _ConvStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 4),
            nn.LayerNorm([out_channels, 1, 1])
            if False
            else nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class _TransformerStage(nn.Module):
    """A stage of N transformer blocks operating on flattened tokens."""

    def __init__(self, dim: int, depth: int, num_heads: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, H, W) -> tokens (B, H*W, C)
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = self.blocks(t)
        t = self.norm(t)
        return t.transpose(1, 2).reshape(B, C, H, W)


class _DownSample(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, out_c, 2, 2),
            nn.GroupNorm(1, out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class _HierVisionEncoder(nn.Module):
    """4-stage hierarchical encoder.

    Stage channels follow Swin-T: (96, 192, 384, 768).
    Depth   per stage: (2, 2, 6, 2).
    Output spatial scales: /4, /8, /16, /32.
    """

    out_channels = (96, 192, 384, 768)

    def __init__(self, in_channels: int = 3, channels=(96, 192, 384, 768), depths=(2, 2, 6, 2), heads=(3, 6, 12, 24)):
        super().__init__()
        c0, c1, c2, c3 = channels
        d0, d1, d2, d3 = depths
        h0, h1, h2, h3 = heads
        self.stem = _ConvStem(in_channels, c0)
        self.stage0 = _TransformerStage(c0, d0, h0)
        self.down1 = _DownSample(c0, c1)
        self.stage1 = _TransformerStage(c1, d1, h1)
        self.down2 = _DownSample(c1, c2)
        self.stage2 = _TransformerStage(c2, d2, h2)
        self.down3 = _DownSample(c2, c3)
        self.stage3 = _TransformerStage(c3, d3, h3)

    def forward(self, x):
        x = self.stem(x)
        s0 = self.stage0(x)             # /4
        s1 = self.stage1(self.down1(s0))  # /8
        s2 = self.stage2(self.down2(s1))  # /16
        s3 = self.stage3(self.down3(s2))  # /32
        return s0, s1, s2, s3


# ---------------------------------------------------------------------------
# Text encoder (HF wrapped, with strict loading)
# ---------------------------------------------------------------------------
class _BiomedTextEncoder(nn.Module):
    """Wraps a HuggingFace BERT / CLIP-text into a fixed-size prompt vector.

    No fallback: if HF transformers is missing or the snapshot fails, the
    constructor raises.  When ``hf_name`` is None we instead build a small
    randomly-initialised BERT — this is *not* a fallback, it is an explicit
    "untrained" mode the user opts into with ``hf_name=None``.
    """

    def __init__(self, hf_name: Optional[str] = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        self._hf_name = hf_name
        if hf_name is not None:
            if not _HAS_HF:
                raise ImportError(
                    "transformers is required to load the text encoder for BiomedParse. "
                    "Install: pip install transformers"
                )
            self.encoder = hf_from_pretrained(AutoModel, hf_name)
            in_dim = self.encoder.config.hidden_size
        else:
            # explicit "no pretrained" mode — random init compact BERT
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            self.vocab = nn.Embedding(30522, embed_dim)
            self.pos = nn.Embedding(512, embed_dim)
            layer = TransformerEncoderLayer(embed_dim, 8, embed_dim * 4, activation="gelu", batch_first=True)
            self.encoder = TransformerEncoder(layer, num_layers=6)
            in_dim = embed_dim
        self.project = nn.Linear(in_dim, embed_dim)
        # the upstream paper freezes the text encoder during the first
        # pretraining stage; mirror that default behaviour
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if self._hf_name is None:
            B, L = input_ids.shape
            pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
            h = self.vocab(input_ids) + self.pos(pos_ids)
            kpm = (attention_mask == 0) if attention_mask is not None else None
            h = self.encoder(h, src_key_padding_mask=kpm)
        else:
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            h = out.last_hidden_state
        # mean-pool over non-pad positions
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = h.mean(1)
        return self.project(pooled), self.project(h)  # (B, D), (B, L, D)


# ---------------------------------------------------------------------------
# Prompt-conditioned mask decoder (SEEM-flavoured)
# ---------------------------------------------------------------------------
class _PromptMaskDecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_q3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q: torch.Tensor, mem: torch.Tensor):
        q = q + self.self_attn(self.norm_q1(q), self.norm_q1(q), self.norm_q1(q))[0]
        q = q + self.cross_attn(self.norm_q2(q), mem, mem)[0]
        q = q + self.ffn(self.norm_q3(q))
        return q


class _PromptMaskDecoder(nn.Module):
    def __init__(self, dim: int = 256, n_queries: int = 32, n_layers: int = 4, heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(n_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([_PromptMaskDecoderLayer(dim, heads) for _ in range(n_layers)])
        self.out_proj = nn.Linear(dim, dim)
        self.n_queries = n_queries
        self.dim = dim

    def forward(self, mem: torch.Tensor, prompt: torch.Tensor):
        """mem: (B, N, D) flattened multi-scale tokens.
        prompt: (B, D) text prompt embedding (broadcast and added to every query).
        Returns query embeddings (B, K, D).
        """
        B = mem.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1) + prompt.unsqueeze(1)
        for layer in self.layers:
            q = layer(q, mem)
        return self.out_proj(q)


# ---------------------------------------------------------------------------
# BiomedParse main module
# ---------------------------------------------------------------------------
class BiomedParse(nn.Module):
    """BiomedParse — text-prompted biomedical foundation segmenter."""

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        decoder_dim: int = 256,
        text_dim: int = 768,
        n_queries: int = 32,
        n_dec_layers: int = 4,
        n_heads: int = 8,
        text_hf_name: Optional[str] = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.decoder_dim = decoder_dim
        self.n_queries = n_queries

        self.visual = _HierVisionEncoder(in_channels=in_channels)

        # projections for every scale -> decoder_dim
        self.scale_proj = nn.ModuleList([
            nn.Conv2d(c, decoder_dim, 1) for c in self.visual.out_channels
        ])
        # learnable scale embeddings
        self.scale_embed = nn.Parameter(torch.zeros(len(self.visual.out_channels), decoder_dim))
        nn.init.trunc_normal_(self.scale_embed, std=0.02)

        self.text = _BiomedTextEncoder(hf_name=text_hf_name, embed_dim=text_dim)
        self.text_to_dec = nn.Linear(text_dim, decoder_dim)

        self.mask_decoder = _PromptMaskDecoder(
            dim=decoder_dim, n_queries=n_queries,
            n_layers=n_dec_layers, heads=n_heads,
        )

        # high-res pixel embedding (use stage0 -> decoder_dim, upsample x4)
        self.pixel_embed = nn.Sequential(
            nn.Conv2d(self.visual.out_channels[0], decoder_dim, 3, 1, 1),
            nn.GroupNorm(1, decoder_dim), nn.GELU(),
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 4, 4),
        )

        # text-to-query selection (paper: pick the query best matching the text)
        # We project queries and text into a shared similarity space.
        self.query_to_sim = nn.Linear(decoder_dim, decoder_dim)
        self.text_to_sim = nn.Linear(decoder_dim, decoder_dim)

        # Learnable embeddings used when no text prompt is supplied: one per
        # class (open-vocab semantics replaced by closed-set classification).
        self.class_prompt = nn.Parameter(torch.zeros(num_classes, decoder_dim))
        nn.init.trunc_normal_(self.class_prompt, std=0.02)

    # ------------------------------------------------------------------
    def _resolve_text(self, image: torch.Tensor, text: Any):
        B = image.shape[0]
        device = image.device
        # 自动处理字符串输入 / Auto-handle string input
        if isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and isinstance(text[0], str)):
            if not hasattr(self, "_auto_text"):
                from medseg.models.text_unet.auto_text_encoder import AutoTextEncoder
                self._auto_text = AutoTextEncoder("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", max_length=64, tokenizer_type="auto")
            text = self._auto_text(text, device=image.device, batch_size=image.shape[0])
        if text is None:
            # use class prompts directly
            prompts = self.class_prompt.unsqueeze(0).expand(B, -1, -1)  # (B, C, D_dec)
            return prompts
        if isinstance(text, dict) and "input_ids" in text:
            ids = text["input_ids"].to(device).long()
            mask = text.get("attention_mask")
            if mask is not None:
                mask = mask.to(device).long()
            sent, _ = self.text(ids, mask)
            p = self.text_to_dec(sent).unsqueeze(1).expand(-1, self.num_classes, -1)
            return p
        if isinstance(text, torch.Tensor):
            t = text.to(device)
            if t.dim() == 2 and t.shape == (self.num_classes, self.decoder_dim):
                return t.unsqueeze(0).expand(B, -1, -1)
            if t.dim() == 3 and t.shape[1:] == (self.num_classes, self.decoder_dim):
                return t
            raise ValueError("text tensor must be (C, D_dec) or (B, C, D_dec)")
        raise TypeError(f"Unsupported text type: {type(text)}")

    def forward(self, image: torch.Tensor, text: Any = None) -> torch.Tensor:
        """forward(image, text=None) -> (B, num_classes, H, W) logits."""
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        H, W = image.shape[-2:]
        feats = self.visual(image)

        # Flatten + project all scales + add scale embedding
        tokens = []
        for i, f in enumerate(feats):
            t = self.scale_proj[i](f)
            B, D, h, w = t.shape
            t = t.flatten(2).transpose(1, 2)              # (B, h*w, D)
            t = t + self.scale_embed[i].view(1, 1, -1)
            tokens.append(t)
        mem = torch.cat(tokens, dim=1)                     # (B, sum(h*w), D)

        # Per-class prompt -> per-class set of decoded queries
        prompts = self._resolve_text(image, text)          # (B, C, D)

        pix = self.pixel_embed(feats[0])                   # (B, D, H, W)

        all_logits = []
        for c in range(self.num_classes):
            q = self.mask_decoder(mem, prompts[:, c, :])   # (B, K, D)
            # similarity-based selection of the query
            q_sim = F.normalize(self.query_to_sim(q), dim=-1)
            p_sim = F.normalize(self.text_to_sim(prompts[:, c, :]), dim=-1).unsqueeze(1)
            sim = (q_sim * p_sim).sum(-1)                 # (B, K)
            weights = sim.softmax(dim=-1).unsqueeze(-1)   # soft-pick top query
            chosen = (q * weights).sum(dim=1)             # (B, D)
            # mask = chosen ⋅ pixel_embed
            m = torch.einsum("bd,bdhw->bhw", chosen, pix)
            all_logits.append(m.unsqueeze(1))

        logits = torch.cat(all_logits, dim=1)              # (B, C, H, W)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


__all__ = ["BiomedParse"]
