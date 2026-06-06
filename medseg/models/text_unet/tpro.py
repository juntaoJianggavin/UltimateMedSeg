# TPRO (MICCAI 2023)
# Reference: https://github.com/shijun18/TPRO
# Paper: https://arxiv.org/abs/2308.09728
# Implemented from paper formulas; not a copy of the official repo.
"""TPRO: Text-Prompted Multi-modal medical image segmentation.

The paper describes a U-shape segmentation network in which **knowledge
prompts** (per-class descriptive text snippets) are encoded by a frozen
language model (PubMedBERT) and used to modulate visual features through
cross-attention at multiple resolutions.  The auxiliary contribution is
an **anatomy-prompt** vector built from generic anatomical adjectives;
the same fusion block consumes both prompt types.

The architectural core (re-implemented here from the paper formulas, not
the upstream repository):

    image ──► Swin-Tiny-like encoder produces feature pyramid {F_i} at
              scales /4, /8, /16, /32.
    prompt ──► BERT tokeniser+model produces (B, num_classes, D_t) class
               descriptors and (B, num_anatomy, D_t) anatomy descriptors.
    For each decoder scale s:
        Tokens T_s = flatten(F_s); apply
        T_s' = T_s + CrossAttn(Q=T_s, K=V=concat(class_prompt, anatomy_prompt))
        Reshape -> F_s'.
    The decoder is a standard U-Net concat + 3×3 conv upsampler.
    Output: 1×1 conv to num_classes.

Implementation notes:
    * Strict HF loading for the text encoder; passing ``text_hf_name=None``
      enters a randomly-initialised compact BERT (an explicit untrained
      mode, not a fallback).
    * When ``text=None``, the model uses learnable class prompts (one per
      class) plus a small bank of learnable anatomy prompts.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

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
# Swin-like encoder (4 stages, hierarchical token mixing)
# ---------------------------------------------------------------------------
class _ConvAct(nn.Module):
    def __init__(self, in_c: int, out_c: int, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.GroupNorm(1, out_c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _EncoderStage(nn.Module):
    """A stage = optional channel projection + N residual conv blocks."""

    def __init__(self, in_c: int, out_c: int, depth: int = 2, downsample_stride: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        if downsample_stride > 1 or in_c != out_c:
            layers.append(_ConvAct(in_c, out_c, downsample_stride, downsample_stride, 0))
        for _ in range(depth):
            layers.append(_ConvAct(out_c, out_c))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class _SwinLikeEncoder(nn.Module):
    """Hierarchical encoder yielding 4 pyramid scales.

    Output channels: (96, 192, 384, 768), scales (/4, /8, /16, /32).
    """

    out_channels = (96, 192, 384, 768)

    def __init__(self, in_channels: int = 3):
        super().__init__()
        c0, c1, c2, c3 = self.out_channels
        self.stage0 = _EncoderStage(in_channels, c0, depth=2, downsample_stride=4)  # /4
        self.stage1 = _EncoderStage(c0, c1, depth=2, downsample_stride=2)            # /8
        self.stage2 = _EncoderStage(c1, c2, depth=4, downsample_stride=2)            # /16
        self.stage3 = _EncoderStage(c2, c3, depth=2, downsample_stride=2)            # /32

    def forward(self, x):
        s0 = self.stage0(x)                      # /4,  c0
        s1 = self.stage1(s0)                     # /8,  c1
        s2 = self.stage2(s1)                     # /16, c2
        s3 = self.stage3(s2)                     # /32, c3
        return s0, s1, s2, s3


# ---------------------------------------------------------------------------
# Knowledge & anatomy prompt encoder (HF wrapper)
# ---------------------------------------------------------------------------
class _PromptEncoder(nn.Module):
    """Wraps a frozen HF text encoder; returns sentence-pool per prompt.

    Strict: requires HF; ``hf_name=None`` builds an explicit untrained BERT.
    """

    def __init__(self, hf_name: Optional[str] = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", text_dim: int = 768):
        super().__init__()
        self._hf_name = hf_name
        self.text_dim = text_dim
        if hf_name is not None:
            if not _HAS_HF:
                raise ImportError(
                    "transformers is required for TPRO. Install: pip install transformers"
                )
            self.encoder = hf_from_pretrained(AutoModel, hf_name)
            in_dim = self.encoder.config.hidden_size
        else:
            self.vocab = nn.Embedding(30522, text_dim)
            self.pos = nn.Embedding(512, text_dim)
            layer = nn.TransformerEncoderLayer(text_dim, 8, text_dim * 4, activation="gelu", batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=6)
            in_dim = text_dim
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(in_dim, text_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if self._hf_name is None:
            B, L = input_ids.shape
            pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
            h = self.vocab(input_ids) + self.pos(pos)
            kpm = (attention_mask == 0) if attention_mask is not None else None
            h = self.encoder(h, src_key_padding_mask=kpm)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            h = out.last_hidden_state
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            pooled = h.mean(1)
        return self.proj(pooled)  # (B, text_dim)


# ---------------------------------------------------------------------------
# Cross-attention fusion block: visual tokens query text prompts
# ---------------------------------------------------------------------------
class _PromptFusion(nn.Module):
    def __init__(self, vis_dim: int, text_dim: int, n_heads: int = 4):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, vis_dim)
        self.norm_v = nn.LayerNorm(vis_dim)
        self.norm_t = nn.LayerNorm(vis_dim)
        self.attn = nn.MultiheadAttention(vis_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(vis_dim, vis_dim * 2), nn.GELU(), nn.Linear(vis_dim * 2, vis_dim)
        )
        self.norm_out = nn.LayerNorm(vis_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, vis: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        """vis: (B, C, H, W); prompts: (B, P, D_text). Returns (B, C, H, W)."""
        B, C, H, W = vis.shape
        v = vis.flatten(2).transpose(1, 2)        # (B, HW, C)
        p = self.text_proj(prompts)               # (B, P, C)
        v2 = self.attn(self.norm_v(v), self.norm_t(p), self.norm_t(p))[0]
        v = v + self.alpha * v2
        v = v + self.ffn(self.norm_out(v))
        return v.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Decoder block: upsample -> concat -> conv
# ---------------------------------------------------------------------------
class _DecBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c, 2, 2)
        self.fuse = nn.Sequential(
            _ConvAct(in_c + skip_c, out_c),
            _ConvAct(out_c, out_c),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class TPRO(nn.Module):
    """TPRO: Text-Prompted multi-modal medical image segmentation."""

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 256,
        text_dim: int = 768,
        n_anatomy_prompts: int = 4,
        text_hf_name: Optional[str] = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        n_heads: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.n_anatomy_prompts = n_anatomy_prompts
        self.text_dim = text_dim

        self.visual = _SwinLikeEncoder(in_channels=in_channels)
        c0, c1, c2, c3 = self.visual.out_channels

        self.prompt_encoder = _PromptEncoder(hf_name=text_hf_name, text_dim=text_dim)

        # learnable class & anatomy prompts (used when text=None)
        self.class_prompt = nn.Parameter(torch.zeros(num_classes, text_dim))
        self.anatomy_prompt = nn.Parameter(torch.zeros(n_anatomy_prompts, text_dim))
        nn.init.trunc_normal_(self.class_prompt, std=0.02)
        nn.init.trunc_normal_(self.anatomy_prompt, std=0.02)

        # Fusion blocks at every encoder scale (paper §3.3)
        self.fuse0 = _PromptFusion(c0, text_dim, n_heads)
        self.fuse1 = _PromptFusion(c1, text_dim, n_heads)
        self.fuse2 = _PromptFusion(c2, text_dim, n_heads)
        self.fuse3 = _PromptFusion(c3, text_dim, n_heads)

        # U-Net decoder
        self.dec3 = _DecBlock(c3, c2, c2)
        self.dec2 = _DecBlock(c2, c1, c1)
        self.dec1 = _DecBlock(c1, c0, c0)
        self.up = nn.ConvTranspose2d(c0, c0, 4, 4)
        self.head = nn.Conv2d(c0, num_classes, 1)

    # ------------------------------------------------------------------
    def _resolve_prompts(self, image: torch.Tensor, text: Any) -> torch.Tensor:
        """Return (B, P, text_dim) — P = num_classes + n_anatomy_prompts."""
        B = image.shape[0]
        device = image.device
        # Anatomy bank: always learnable, broadcast across batch
        ana = self.anatomy_prompt.unsqueeze(0).expand(B, -1, -1)

        # 自动处理字符串输入 / Auto-handle string input
        if isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and isinstance(text[0], str)):
            if not hasattr(self, "_auto_text"):
                from medseg.models.text_unet.auto_text_encoder import AutoTextEncoder
                self._auto_text = AutoTextEncoder("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", max_length=64, tokenizer_type="auto")
            text = self._auto_text(text, device=image.device, batch_size=image.shape[0])
        if text is None:
            cls = self.class_prompt.unsqueeze(0).expand(B, -1, -1)
        elif isinstance(text, dict) and "input_ids" in text:
            ids = text["input_ids"].to(device).long()
            mask = text.get("attention_mask")
            if mask is not None:
                mask = mask.to(device).long()
            # If multiple prompts are concatenated as (B, C, L), encode each.
            if ids.dim() == 3:
                Bp, C, L = ids.shape
                ids = ids.view(Bp * C, L)
                m2 = mask.view(Bp * C, L) if mask is not None else None
                pooled = self.prompt_encoder(ids, m2).view(Bp, C, -1)
                cls = pooled
                if cls.shape[1] != self.num_classes:
                    raise ValueError(
                        f"text supplies {cls.shape[1]} prompts but num_classes={self.num_classes}"
                    )
            else:
                # single caption per sample -> broadcast to num_classes
                pooled = self.prompt_encoder(ids, mask)  # (B, D)
                cls = pooled.unsqueeze(1).expand(-1, self.num_classes, -1)
        elif isinstance(text, torch.Tensor):
            t = text.to(device)
            if t.dim() == 2:
                if t.shape != (self.num_classes, self.text_dim):
                    raise ValueError(
                        f"text tensor must be ({self.num_classes},{self.text_dim}); got {tuple(t.shape)}"
                    )
                cls = t.unsqueeze(0).expand(B, -1, -1)
            elif t.dim() == 3:
                if t.shape[1:] != (self.num_classes, self.text_dim):
                    raise ValueError(
                        f"text tensor must be (B,{self.num_classes},{self.text_dim}); got {tuple(t.shape)}"
                    )
                cls = t
            else:
                raise ValueError("text tensor must be 2-D or 3-D")
        else:
            raise TypeError(f"Unsupported text type: {type(text)}")

        return torch.cat([cls, ana], dim=1)         # (B, P, D)

    def forward(self, image: torch.Tensor, text: Any = None) -> torch.Tensor:
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        H, W = image.shape[-2:]
        s0, s1, s2, s3 = self.visual(image)
        prompts = self._resolve_prompts(image, text)

        s0 = self.fuse0(s0, prompts)
        s1 = self.fuse1(s1, prompts)
        s2 = self.fuse2(s2, prompts)
        s3 = self.fuse3(s3, prompts)

        d3 = self.dec3(s3, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        y = self.up(d1)
        logits = self.head(y)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


__all__ = ["TPRO"]
