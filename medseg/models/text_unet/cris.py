# CRIS (CVPR 2022)
# Reference: https://github.com/DerrickWang005/CRIS.pytorch
# Paper: https://arxiv.org/abs/2111.15174
# Implemented from paper formulas; not a copy of the official repo.
"""CRIS: CLIP-Driven Referring Image Segmentation.

Algorithm (faithful to the paper, NOT copied from the repo):

    1. Visual encoder produces multi-scale features (CLIP-RN50 in the paper).
       Here we use torchvision ResNet50 stages C2/C3/C4 ∈ {/8, /16, /32}.
    2. Text encoder (CLIP text Transformer in the paper) yields word
       features ``F_t`` ∈ R^{T×C} and a sentence feature ``F_s`` ∈ R^C.
    3. Cross-Modal Neck fuses the deepest visual feature with the sentence
       embedding through 1×1 conv + ReLU, then progressively upsamples and
       merges with C3, C2 features (FPN-style).
    4. Vision-Language Decoder is a stack of Transformer decoder layers in
       which flattened visual tokens query the word features ``F_t``.
    5. A projector head produces per-pixel embeddings of dim ``C_proj``;
       the text-to-pixel decision is a cosine product
            M(i, j) = σ((W_s · F_s) · F_p(i, j)) / τ
       which doubles as the supervision signal for the **text-pixel
       contrastive loss** (binary cross-entropy on logits).

Framework glue (vs. an upstream port):
    * ``pretrained=True`` lazily loads HuggingFace ``openai/clip-vit-base-patch32``
      and copies its text encoder weights into our own text Transformer; if
      the load fails we *raise* — there is no silent fallback.
    * ``forward(image, text=None)`` returns logits ``(B, num_classes, H, W)``
      at the input resolution.  ``num_classes`` heads share the projector
      and have one ``W_s`` matrix per class (one class ↔ one referring
      expression / organ word in the medical setting).
    * When ``text=None`` we fall back to learnable class embeddings (per
      class) so the smoke pipeline can still drive the model.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.utils.weight_downloader import hf_from_pretrained


# ---------------------------------------------------------------------------
# optional HF dependency
# ---------------------------------------------------------------------------
try:
    from transformers import CLIPModel, CLIPTokenizer  # type: ignore
    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False


# ---------------------------------------------------------------------------
# Visual backbone (torchvision ResNet50, 1:1 stage signature with CLIP-RN50)
# ---------------------------------------------------------------------------
class _ResNet50Backbone(nn.Module):
    """ResNet-50 stages: C2(/8) C3(/16) C4(/32). Output channels (512, 1024, 2048)."""

    out_channels = (512, 1024, 2048)

    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        from torchvision.models import resnet50
        weights = None
        if pretrained:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT
        net = resnet50(weights=weights)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2  # /8, 512
        self.layer3 = net.layer3  # /16, 1024
        self.layer4 = net.layer4  # /32, 2048

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c2, c3, c4


# ---------------------------------------------------------------------------
# Text encoder (CLIP-style Transformer, optionally seeded from HF CLIP)
# ---------------------------------------------------------------------------
class _CLIPTextEncoder(nn.Module):
    """CLIP-style text transformer.

    `pretrained=True` -> load HF ``openai/clip-vit-base-patch32`` weights into
    `token_embedding`, `positional_embedding`, transformer, `ln_final`,
    `text_projection`.  Raises if HF transformers is missing or download fails.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        transformer_width: int = 512,
        transformer_layers: int = 12,
        transformer_heads: int = 8,
        vocab_size: int = 49408,
        context_length: int = 77,
        pretrained: bool = False,
        hf_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_width,
            nhead=transformer_heads,
            dim_feedforward=transformer_width * 4,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.ln_final = nn.LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        nn.init.trunc_normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)

        if pretrained:
            self._load_hf_weights(hf_name)

    def _load_hf_weights(self, hf_name: str) -> None:
        if not _HAS_HF:
            raise ImportError(
                "transformers is required for CRIS(pretrained=True). "
                "Install: pip install transformers"
            )
        # strict load — no silent fallback
        hf = hf_from_pretrained(CLIPModel, hf_name)
        text_model = hf.text_model
        with torch.no_grad():
            self.token_embedding.weight.copy_(text_model.embeddings.token_embedding.weight)
            self.positional_embedding.copy_(text_model.embeddings.position_embedding.weight)
            self.text_projection.copy_(hf.text_projection.weight.t())

    def _build_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((L, L), float("-inf"), device=device)
        mask.triu_(1)
        return mask

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """input_ids: (B, L). Returns word_feats (B, L, D_proj) and sent_feat (B, D_proj)."""
        B, L = input_ids.shape
        if L > self.context_length:
            raise ValueError(f"input_ids length {L} exceeds context_length {self.context_length}")
        x = self.token_embedding(input_ids) + self.positional_embedding[:L]
        causal = self._build_causal_mask(L, x.device)
        kpm = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, mask=causal, src_key_padding_mask=kpm)
        x = self.ln_final(x)
        # sentence pooling: argmax position of input_ids (per CLIP convention,
        # the eot_token has the largest id within a tokenised caption)
        eot = input_ids.argmax(dim=-1)
        sent = x[torch.arange(B, device=x.device), eot]
        sent = sent @ self.text_projection  # (B, embed_dim)
        words = x @ self.text_projection    # (B, L, embed_dim)
        return words, sent


# ---------------------------------------------------------------------------
# Cross-Modal Neck (FPN with text-fusion at the deepest stage)
# ---------------------------------------------------------------------------
class _CrossModalNeck(nn.Module):
    def __init__(self, in_channels=(512, 1024, 2048), out_channels: int = 512, text_dim: int = 512):
        super().__init__()
        c2, c3, c4 = in_channels
        self.text_proj = nn.Linear(text_dim, out_channels)
        self.fuse4 = nn.Sequential(
            nn.Conv2d(c4 + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.lateral3 = nn.Conv2d(c3, out_channels, 1)
        self.lateral2 = nn.Conv2d(c2, out_channels, 1)
        self.smooth3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, c2, c3, c4, sent_feat):
        # broadcast sentence feature over deepest map
        B, _, H, W = c4.shape
        s = self.text_proj(sent_feat).view(B, -1, 1, 1).expand(-1, -1, H, W)
        m4 = self.fuse4(torch.cat([c4, s], dim=1))
        m3 = self.smooth3(self.lateral3(c3) + F.interpolate(m4, size=c3.shape[-2:], mode="bilinear", align_corners=False))
        m2 = self.smooth2(self.lateral2(c2) + F.interpolate(m3, size=c2.shape[-2:], mode="bilinear", align_corners=False))
        return m2


# ---------------------------------------------------------------------------
# Vision-Language Decoder (transformer cross-attn: visual Q, text K/V)
# ---------------------------------------------------------------------------
class _VLDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, v_tokens: torch.Tensor, t_tokens: torch.Tensor, t_mask: Optional[torch.Tensor] = None):
        # v_tokens: (B, HW, D); t_tokens: (B, L, D); t_mask: (B, L) where 1=keep
        x = v_tokens
        y, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + y
        kpm = (t_mask == 0) if t_mask is not None else None
        y, _ = self.cross_attn(self.norm2(x), t_tokens, t_tokens, key_padding_mask=kpm)
        x = x + y
        x = x + self.ffn(self.norm3(x))
        return x


class _VLDecoder(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 3, max_hw: int = 4096):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_hw, d_model))
        self.layers = nn.ModuleList([_VLDecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.max_hw = max_hw

    def forward(self, v_tokens: torch.Tensor, t_tokens: torch.Tensor, t_mask: Optional[torch.Tensor] = None):
        B, N, D = v_tokens.shape
        if N > self.max_hw:
            raise ValueError(f"vision tokens {N} exceed pos-embed budget {self.max_hw}")
        v = v_tokens + self.pos_embed[:, :N]
        for layer in self.layers:
            v = layer(v, t_tokens, t_mask)
        return v


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class CRIS(nn.Module):
    """CRIS: CLIP-Driven Referring Image Segmentation.

    Args mirror paper hyper-parameters:
        in_channels:  input image channels.
        num_classes:  number of referring categories — one mask per class.
        img_size:     square input resolution (e.g. 416 in the paper).
        embed_dim:    CLIP-aligned projection dim (default 512).
        neck_dim:     channel width inside neck + VL decoder.
        n_dec_layers: number of VL-decoder layers.
        pretrained:   load HF CLIP text-encoder weights (strict, no fallback).
    """

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 416,
        embed_dim: int = 512,
        neck_dim: int = 512,
        n_dec_layers: int = 3,
        n_heads: int = 8,
        context_length: int = 77,
        pretrained: bool = False,
        text_pretrained_name: str = "openai/clip-vit-base-patch32",
        backbone_pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.context_length = context_length

        self.visual = _ResNet50Backbone(in_channels=in_channels, pretrained=backbone_pretrained)
        self.text = _CLIPTextEncoder(
            embed_dim=embed_dim,
            transformer_width=embed_dim,
            transformer_layers=12,
            transformer_heads=n_heads,
            context_length=context_length,
            pretrained=pretrained,
            hf_name=text_pretrained_name,
        )

        self.neck = _CrossModalNeck(
            in_channels=self.visual.out_channels,
            out_channels=neck_dim,
            text_dim=embed_dim,
        )
        self.vl_decoder = _VLDecoder(d_model=neck_dim, n_heads=n_heads, n_layers=n_dec_layers)

        # Projector (paper: per-pixel embedding head)
        self.projector = nn.Sequential(
            nn.Conv2d(neck_dim, neck_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(neck_dim), nn.ReLU(inplace=True),
            nn.Conv2d(neck_dim, embed_dim, 1),
        )
        # Per-class sentence projection W_s (broadcast to a tensor of size
        # (num_classes, embed_dim)).  When the caller doesn't pass text, we
        # also keep a learnable class embedding to drive the model.
        self.W_s = nn.Linear(embed_dim, embed_dim, bias=False)
        self.class_text_embedding = nn.Parameter(torch.zeros(num_classes, embed_dim))
        nn.init.trunc_normal_(self.class_text_embedding, std=0.02)
        # learnable temperature (CLIP-style)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    # ------------------------------------------------------------------
    def _resolve_text(self, image: torch.Tensor, text: Any):
        """Return (word_feats, sent_feats_per_class) where:
            word_feats: (B, L, D) used by the VL decoder
            sent_feats_per_class: (B, num_classes, D) — one sentence per class
        """
        B = image.shape[0]
        device = image.device
        # 自动处理字符串输入 / Auto-handle string input
        if isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and isinstance(text[0], str)):
            if not hasattr(self, "_auto_text"):
                from medseg.models.text_unet.auto_text_encoder import AutoTextEncoder
                self._auto_text = AutoTextEncoder("openai/clip-vit-base-patch32", max_length=77, tokenizer_type="clip")
            text = self._auto_text(text, device=image.device, batch_size=image.shape[0])
        if text is None:
            # learnable class embeddings broadcast across batch
            sent = self.class_text_embedding.unsqueeze(0).expand(B, -1, -1)
            word = sent  # use the same vectors as word features
            return word, sent, None

        if isinstance(text, dict) and "input_ids" in text:
            ids = text["input_ids"].to(device).long()
            mask = text.get("attention_mask")
            if mask is not None:
                mask = mask.to(device).long()
            words, sent = self.text(ids, mask)
            # single sentence per sample -> broadcast to num_classes
            sent = sent.unsqueeze(1).expand(-1, self.num_classes, -1)
            return words, sent, mask

        if isinstance(text, torch.Tensor):
            # accept pre-computed sentence embeddings (B, num_classes, D) or
            # (num_classes, D)
            t = text.to(device)
            if t.dim() == 2:
                if t.shape != (self.num_classes, self.embed_dim):
                    raise ValueError(
                        f"text tensor must be ({self.num_classes},{self.embed_dim}); got {tuple(t.shape)}"
                    )
                sent = t.unsqueeze(0).expand(B, -1, -1)
            elif t.dim() == 3:
                if t.shape != (B, self.num_classes, self.embed_dim):
                    raise ValueError(
                        f"text tensor must be ({B},{self.num_classes},{self.embed_dim}); got {tuple(t.shape)}"
                    )
                sent = t
            else:
                raise ValueError("text tensor must be 2-D or 3-D")
            word = sent.reshape(B, self.num_classes, self.embed_dim)
            return word, sent, None

        raise TypeError(f"Unsupported text type: {type(text)}")

    def forward(self, image: torch.Tensor, text: Any = None) -> torch.Tensor:
        """forward(image, text=None) -> (B, num_classes, H, W) mask logits.

        ``text`` may be:
            * None — use learnable class embeddings.
            * dict with ``input_ids``/``attention_mask`` — tokenised caption.
            * Tensor (num_classes, D) or (B, num_classes, D) — pre-computed
              CLIP sentence embeddings (one per class).
        """
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        H, W = image.shape[-2:]
        c2, c3, c4 = self.visual(image)
        word, sent_per_class, t_mask = self._resolve_text(image, text)

        # ------ Cross-Modal Neck (uses class-averaged sentence as the
        # broadcast signal — multi-class predictions still happen at the
        # contrastive head)
        sent_pool = sent_per_class.mean(dim=1)
        F_m = self.neck(c2, c3, c4, sent_pool)         # (B, neck_dim, H/8, W/8)

        # ------ Vision-Language Decoder
        B, D, Hm, Wm = F_m.shape
        v_tokens = F_m.flatten(2).transpose(1, 2)       # (B, Hm*Wm, D)
        v_tokens = self.vl_decoder(v_tokens, word, t_mask)
        F_m = v_tokens.transpose(1, 2).reshape(B, D, Hm, Wm)

        # ------ Projector + contrastive head
        F_p = self.projector(F_m)                       # (B, D, Hm, Wm)
        F_p = F.normalize(F_p, dim=1)

        sent_proj = self.W_s(sent_per_class)            # (B, num_classes, D)
        sent_proj = F.normalize(sent_proj, dim=-1)

        # cosine sim: (B, num_classes, Hm, Wm)
        sim = torch.einsum("bcd,bdhw->bchw", sent_proj, F_p)
        sim = sim * self.logit_scale.exp()

        # Upsample to input resolution (paper uses 4× upsample)
        logits = F.interpolate(sim, size=(H, W), mode="bilinear", align_corners=False)
        return logits


__all__ = ["CRIS"]
