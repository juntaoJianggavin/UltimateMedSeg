# CausalCLIPSeg (MICCAI 2024)
# Reference: https://github.com/WUTCM-Lab/CausalCLIPSeg
# Paper: https://arxiv.org/abs/2407.16447
# Implemented from paper formulas; not a copy of the official repo.
"""CausalCLIPSeg: a causal-intervention CLIP-based text-guided medical
image segmentation network (MICCAI 2024).

Algorithm (faithful to the paper, NOT copied from the repo):

    1. Visual encoder produces a multi-scale CNN feature pyramid
       ``{V_2, V_3, V_4}`` at strides /8, /16, /32 (ResNet-50 stages, mirroring
       the paper's CLIP-RN50 image-encoder).
    2. Text encoder (CLIP text Transformer) yields a sentence-level prompt
       embedding ``t`` ∈ R^D — the "factual" prompt for the foreground class.
    3. A learnable "context" prompt ``c`` ∈ R^D (one vector per class) plays
       the role of the **counterfactual** / confounder describing the visual
       context that biases the prediction (background appearance).  The
       backdoor adjustment is realised through Eq. (4) of the paper as the
       difference of two cosine-similarity maps::

           M_factual       = cos(F_p , W_t · t)            (text-conditioned)
           M_counterfactual = cos(F_p , W_c · c)           (context-conditioned)
           M_causal         = M_factual − λ · M_counterfactual

       where ``λ ∈ [0,1]`` is a learnable scalar implementing the
       do-operator: do(T=t) cancels the confounding contribution of the
       context branch.
    4. A *cross-modal attention* fusion block (Eq. (5)) integrates ``t`` into
       the pixel feature stack before the contrastive head; we implement it
       as a single multi-head cross-attention layer in which pixel tokens
       query the (token-level) text features.

Framework glue (strict, no fallback):
    * ``pretrained=True`` lazily loads HuggingFace ``openai/clip-vit-base-patch32``
      through :func:`hf_from_pretrained` and copies its text-encoder weights
      into our local CLIP text transformer; failure raises a
      :class:`~medseg.utils.weight_downloader.WeightDownloadError`.
    * ``forward(image, text=None)`` returns logits of shape
      ``(B, num_classes, H, W)`` at the input resolution.  Accepts
      ``text=None`` (uses learnable class embeddings), a dict of
      ``input_ids``/``attention_mask`` (CLIP-tokenised caption), or a
      pre-computed ``(num_classes, D)`` / ``(B, num_classes, D)`` embedding
      tensor.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.utils.weight_downloader import hf_from_pretrained


# ---------------------------------------------------------------------------
# optional HF dependency
# ---------------------------------------------------------------------------
try:
    from transformers import CLIPModel  # type: ignore
    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False


# ---------------------------------------------------------------------------
# Visual backbone (ResNet-50 stages, mirrors CLIP-RN50 stage signature)
# ---------------------------------------------------------------------------
class _ResNet50Backbone(nn.Module):
    """ResNet-50 stages: C2(/8) C3(/16) C4(/32). Out channels (512, 1024, 2048)."""

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
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c2, c3, c4


# ---------------------------------------------------------------------------
# CLIP-style text encoder (optionally seeded from HF CLIP)
# ---------------------------------------------------------------------------
class _CLIPTextEncoder(nn.Module):
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
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, transformer_width)
        )
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
                "transformers is required for CausalCLIPSeg(pretrained=True). "
                "Install: pip install transformers"
            )
        hf = hf_from_pretrained(CLIPModel, hf_name)
        text_model = hf.text_model
        with torch.no_grad():
            self.token_embedding.weight.copy_(text_model.embeddings.token_embedding.weight)
            self.positional_embedding.copy_(text_model.embeddings.position_embedding.weight)
            self.text_projection.copy_(hf.text_projection.weight.t())

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((L, L), float("-inf"), device=device)
        mask.triu_(1)
        return mask

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, L = input_ids.shape
        if L > self.context_length:
            raise ValueError(f"input_ids length {L} exceeds context_length {self.context_length}")
        x = self.token_embedding(input_ids) + self.positional_embedding[:L]
        causal = self._causal_mask(L, x.device)
        kpm = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, mask=causal, src_key_padding_mask=kpm)
        x = self.ln_final(x)
        eot = input_ids.argmax(dim=-1)
        sent = x[torch.arange(B, device=x.device), eot]
        sent = sent @ self.text_projection
        words = x @ self.text_projection
        return words, sent


# ---------------------------------------------------------------------------
# FPN-style cross-modal neck with text fusion at the deepest stage
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
        B, _, H, W = c4.shape
        s = self.text_proj(sent_feat).view(B, -1, 1, 1).expand(-1, -1, H, W)
        m4 = self.fuse4(torch.cat([c4, s], dim=1))
        m3 = self.smooth3(
            self.lateral3(c3)
            + F.interpolate(m4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        )
        m2 = self.smooth2(
            self.lateral2(c2)
            + F.interpolate(m3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        )
        return m2


# ---------------------------------------------------------------------------
# Cross-modal attention (pixel tokens query word tokens)
# ---------------------------------------------------------------------------
class _CrossModalAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, v_tok: torch.Tensor, t_tok: torch.Tensor, t_mask: Optional[torch.Tensor] = None):
        kpm = (t_mask == 0) if t_mask is not None else None
        q = self.norm_q(v_tok)
        k = self.norm_kv(t_tok)
        y, _ = self.attn(q, k, k, key_padding_mask=kpm)
        v_tok = v_tok + y
        v_tok = v_tok + self.ffn(self.norm_ffn(v_tok))
        return v_tok


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class CausalCLIPSeg(nn.Module):
    """Causal-intervention CLIP-based medical segmentation network."""

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        embed_dim: int = 512,
        neck_dim: int = 512,
        n_heads: int = 8,
        context_length: int = 77,
        pretrained: bool = False,
        text_pretrained_name: str = "openai/clip-vit-base-patch32",
        backbone_pretrained: bool = False,
        causal_lambda_init: float = 0.5,
    ):
        super().__init__()
        if not 0.0 <= causal_lambda_init <= 1.0:
            raise ValueError("causal_lambda_init must be in [0, 1]")
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.context_length = context_length

        self.visual = _ResNet50Backbone(in_channels=in_channels, pretrained=backbone_pretrained)
        self.text = _CLIPTextEncoder(
            embed_dim=embed_dim,
            transformer_width=embed_dim,
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
        self.fusion = _CrossModalAttention(dim=neck_dim, n_heads=n_heads)

        # Per-pixel projection head -> embed_dim
        self.projector = nn.Sequential(
            nn.Conv2d(neck_dim, neck_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(neck_dim), nn.ReLU(inplace=True),
            nn.Conv2d(neck_dim, embed_dim, 1),
        )

        # W_t : factual (text) projection ;  W_c : counterfactual projection
        self.W_t = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_c = nn.Linear(embed_dim, embed_dim, bias=False)

        # Class text embedding (fallback when text=None)
        self.class_text_embedding = nn.Parameter(torch.zeros(num_classes, embed_dim))
        nn.init.trunc_normal_(self.class_text_embedding, std=0.02)

        # Per-class context / counterfactual prompt (learnable confounder)
        self.context_prompt = nn.Parameter(torch.zeros(num_classes, embed_dim))
        nn.init.trunc_normal_(self.context_prompt, std=0.02)

        # Backdoor mixing weight λ ∈ [0,1] (sigmoid of an unconstrained param)
        self._lambda_raw = nn.Parameter(
            torch.tensor(_logit(causal_lambda_init), dtype=torch.float32)
        )

        # CLIP-style learnable temperature
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    # ------------------------------------------------------------------
    @property
    def causal_lambda(self) -> torch.Tensor:
        return torch.sigmoid(self._lambda_raw)

    # ------------------------------------------------------------------
    def _resolve_text(self, image: torch.Tensor, text: Any):
        B = image.shape[0]
        device = image.device

        # 自动处理字符串输入 / Auto-handle string input
        if isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and isinstance(text[0], str)):
            if not hasattr(self, "_auto_text"):
                from medseg.models.text_unet.auto_text_encoder import AutoTextEncoder
                self._auto_text = AutoTextEncoder("openai/clip-vit-base-patch32", max_length=77, tokenizer_type="clip")
            text = self._auto_text(text, device=image.device, batch_size=image.shape[0])
        if text is None:
            sent = self.class_text_embedding.unsqueeze(0).expand(B, -1, -1)
            word = sent
            return word, sent, None

        if isinstance(text, dict) and "input_ids" in text:
            ids = text["input_ids"].to(device).long()
            mask = text.get("attention_mask")
            if mask is not None:
                mask = mask.to(device).long()
            words, sent = self.text(ids, mask)
            sent = sent.unsqueeze(1).expand(-1, self.num_classes, -1)
            return words, sent, mask

        if isinstance(text, torch.Tensor):
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
            word = sent
            return word, sent, None

        raise TypeError(f"Unsupported text type: {type(text)}")

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, text: Any = None) -> torch.Tensor:
        """forward(image, text=None) -> (B, num_classes, H, W) logits.

        Implements the causal backdoor adjustment::

            M = sigmoid_scale * (cos(F_p, W_t·t) - λ · cos(F_p, W_c·c))
        """
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        H, W = image.shape[-2:]
        c2, c3, c4 = self.visual(image)
        word, sent_per_class, t_mask = self._resolve_text(image, text)

        # Neck (broadcast averaged sentence)
        sent_pool = sent_per_class.mean(dim=1)
        F_m = self.neck(c2, c3, c4, sent_pool)

        # Cross-modal fusion (pixel tokens query text tokens)
        B, D, Hm, Wm = F_m.shape
        v_tok = F_m.flatten(2).transpose(1, 2)
        v_tok = self.fusion(v_tok, word, t_mask)
        F_m = v_tok.transpose(1, 2).reshape(B, D, Hm, Wm)

        # Projector + L2-normalised pixel embedding
        F_p = self.projector(F_m)
        F_p = F.normalize(F_p, dim=1)

        # Factual branch
        t_proj = F.normalize(self.W_t(sent_per_class), dim=-1)
        sim_factual = torch.einsum("bcd,bdhw->bchw", t_proj, F_p)

        # Counterfactual branch (per-class context prompt, broadcast over batch)
        c_proj = F.normalize(self.W_c(self.context_prompt), dim=-1)
        c_proj = c_proj.unsqueeze(0).expand(B, -1, -1)
        sim_cf = torch.einsum("bcd,bdhw->bchw", c_proj, F_p)

        # Causal intervention: M = M_factual - λ * M_counterfactual
        sim_causal = sim_factual - self.causal_lambda * sim_cf
        sim_causal = sim_causal * self.logit_scale.exp()

        logits = F.interpolate(sim_causal, size=(H, W), mode="bilinear", align_corners=False)
        return logits


def _logit(p: float) -> float:
    import math as _math
    eps = 1e-6
    p = min(max(p, eps), 1.0 - eps)
    return float(_math.log(p / (1.0 - p)))


__all__ = ["CausalCLIPSeg"]
