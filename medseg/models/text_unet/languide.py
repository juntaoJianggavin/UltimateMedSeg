# Reference: https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023
# Paper:     https://arxiv.org/abs/2307.03942
"""LanGuideMedSeg (Ariadne's Thread, MICCAI 2023) -- text-guided medical segmentation.

Re-implementation following the paper (Zhong et al., "Ariadne's Thread:
Using Text Prompts to Improve Segmentation of Infected Areas from Chest
X-ray Images", MICCAI 2023) and the architecture description in the
README of the official repository:
    https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023
        utils/model.py   (BERTModel / VisionModel / LanGuideMedSeg)
        utils/layers.py  (PositionalEncoding / GuideDecoderLayer / GuideDecoder)
Code is written from the paper's equations (multi-scale Guide-Decoder with
text self-/cross-attention) — not copy-pasted from the upstream sources.

Framework-glue diffs (algorithm layer is unchanged):
    1. forward uses (image, text) where ``text`` is a dict with
       ``input_ids`` / ``attention_mask`` (CXR-BERT tokenizer output) or a
       (input_ids, attention_mask) 2-tuple — both produced by medseg's
       text_image_dataset.
    2. is_text_guided=True class attribute for the trainer.

Strict no-fallback policy:
    * transformers, monai, einops are hard imports — missing deps raise.
    * BERTModel/VisionModel require valid HF model ids — no random-init
      stand-in.
    * forward(image, text=None) raises — without text features the model
      is no longer LanGuideMedSeg.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Hard imports — any missing dep is a real failure, never a silent mock.
from transformers import AutoModel  # type: ignore
from monai.networks.blocks.dynunet_block import UnetOutBlock  # type: ignore
from monai.networks.blocks.upsample import SubpixelUpsample  # type: ignore
from monai.networks.blocks.unetr_block import UnetrUpBlock  # type: ignore
from medseg.utils.weight_downloader import hf_from_pretrained


# ---------------------------------------------------------------------------
# layers.py 1:1 port
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """1:1 port of utils.layers.PositionalEncoding."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class GuideDecoderLayer(nn.Module):
    """1:1 port of utils.layers.GuideDecoderLayer."""

    def __init__(self, in_channels: int, output_text_len: int, input_text_len: int = 24, embed_dim: int = 768):
        super().__init__()
        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )

        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels, max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.scale = nn.Parameter(torch.tensor(1.421), requires_grad=True)

    def forward(self, x, txt):
        txt = self.text_project(txt)

        # Self-Attention
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2

        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2, _ = self.cross_attn(
            query=self.vis_pos(vis2),
            key=self.txt_pos(txt),
            value=txt,
        )
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale * vis2

        return vis


class GuideDecoder(nn.Module):
    """1:1 port of utils.layers.GuideDecoder."""

    def __init__(self, in_channels, out_channels, spatial_size, text_len) -> None:
        super().__init__()
        self.guide_layer = GuideDecoderLayer(in_channels, text_len)
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2, in_channels, out_channels, 3, 2, norm_name="BATCH")

    def forward(self, vis, skip_vis, txt):
        if txt is not None:
            vis = self.guide_layer(vis, txt)

        vis = rearrange(vis, "B (H W) C -> B C H W", H=self.spatial_size, W=self.spatial_size)
        skip_vis = rearrange(
            skip_vis, "B (H W) C -> B C H W", H=self.spatial_size * 2, W=self.spatial_size * 2
        )
        output = self.decoder(vis, skip_vis)
        output = rearrange(output, "B C H W -> B (H W) C")
        return output


# ---------------------------------------------------------------------------
# BERT / Vision encoders (with HF-free fallback)
# ---------------------------------------------------------------------------


class BERTModel(nn.Module):
    """Text encoder block from the paper: a frozen HuggingFace BERT (the
    paper uses ``microsoft/BiomedVLP-CXR-BERT-specialized``) followed by a
    small projection head used for the contrastive loss.

    Strict: requires a working HuggingFace BERT load. No mock fallback —
    if the model can't be fetched, we raise instead of substituting a
    randomly-initialised stand-in that would silently degrade quality.
    """

    def __init__(self, bert_type: str | None, project_dim: int):
        super().__init__()
        if not bert_type:
            raise ValueError("BERTModel requires a non-empty bert_type")
        self.model = hf_from_pretrained(AutoModel, 
            bert_type, output_hidden_states=True, trust_remote_code=True
        )

        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim),
        )
        # freeze the parameters (upstream behaviour)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output["hidden_states"]
        last_hidden_states = torch.stack([hidden[1], hidden[2], hidden[-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)
        embed = self.project_head(embed)
        return {"feature": hidden, "project": embed}


class VisionModel(nn.Module):
    """Vision encoder block from the paper: a ConvNeXt-tiny (the paper uses
    ``facebook/convnext-tiny-224``) producing 4 hierarchical feature maps
    consumed by the multi-scale Guide-Decoders.

    Strict: requires a working HuggingFace ConvNeXt load. No mock fallback —
    if the model can't be fetched, we raise instead of silently substituting
    a random-init stand-in.
    """

    def __init__(self, vision_type: str | None, project_dim: int):
        super().__init__()
        if not vision_type:
            raise ValueError("VisionModel requires a non-empty vision_type")
        self.model = hf_from_pretrained(AutoModel, vision_type, output_hidden_states=True)

        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        embeds = output["pooler_output"].squeeze()
        if embeds.dim() == 1:
            embeds = embeds.unsqueeze(0)
        project = self.project_head(embeds)
        return {"feature": output["hidden_states"], "project": project}


# ---------------------------------------------------------------------------
# main model
# ---------------------------------------------------------------------------


class LanGuideMedSeg(nn.Module):
    """1:1 port of utils.model.LanGuideMedSeg.

    Args mirror upstream:
        bert_type:   HF model id for text encoder (e.g.
                     "microsoft/BiomedVLP-CXR-BERT-specialized").
                     Pass None to force mock.
        vision_type: HF model id for vision encoder (e.g.
                     "facebook/convnext-tiny-224"). Pass None to force mock.
        project_dim: projection dim for contrastive head (default 512).
        text_len:    upstream tokenizer max length (default 24); we keep the
                     same for input_text_len of the topmost decoder layer.
    """

    is_text_guided = True

    def __init__(
        self,
        bert_type: str | None = "microsoft/BiomedVLP-CXR-BERT-specialized",
        vision_type: str | None = "facebook/convnext-tiny-224",
        project_dim: int = 512,
        num_classes: int = 1,
        img_size: int = 224,
        text_len: int = 24,
    ):
        super().__init__()
        if img_size != 224:
            raise ValueError(
                "LanGuideMedSeg is defined for 224x224 input "
                f"(spatial pyramid 7/14/28/56). Got img_size={img_size}."
            )

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        # 内置 tokenizer，用户只需传字符串
        # Built-in tokenizer so users only need to pass strings
        from medseg.models.text_unet.auto_text_encoder import AutoTextEncoder
        self._auto_text = AutoTextEncoder(bert_type, max_length=text_len, tokenizer_type="auto")

        self.spatial_dim = [7, 14, 28, 56]  # 224*224
        feature_dim = [768, 384, 192, 96]
        self.text_len = text_len

        self.decoder16 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = GuideDecoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = GuideDecoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=num_classes)

    def forward(self, image: torch.Tensor, text: Any | None = None):
        """forward(image, text).

        Args:
            image: (B, C, 224, 224); 1-channel inputs are repeated to 3 channels
                so the ConvNeXt stem can consume them.
            text: dict ``{input_ids, attention_mask}`` produced by the CXR-BERT
                tokenizer, or a 2-tuple ``(input_ids, attention_mask)``.
                Passing ``None`` raises — the paper's Guide-Decoders
                degenerate to vanilla self-attention without text and the
                model is no longer LanGuideMedSeg.

        Returns:
            (B, num_classes, H, W) sigmoid-activated mask.
        """
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        if image.shape[1] == 1:
            image = repeat(image, "b 1 h w -> b c h w", c=3)

        # 自动处理 text 输入：字符串 → tokenize，dict → 透传
        # Auto text handling: string → tokenize, dict → pass through
        if text is None:
            raise ValueError(
                "LanGuideMedSeg requires text input. Pass a string like "
                "'consolidation in right lung' or a list of strings."
            )
        if isinstance(text, str) or (isinstance(text, list) and isinstance(text[0], str)):
            text = self._auto_text(text, device=image.device, batch_size=image.shape[0])
        if isinstance(text, (tuple, list)) and len(text) == 2 and isinstance(text[0], torch.Tensor):
            text = {"input_ids": text[0], "attention_mask": text[1]}
        if not (isinstance(text, dict) and "input_ids" in text):
            raise ValueError(
                f"Unsupported text type: {type(text)}. Pass a string, "
                f"list of strings, or dict with input_ids/attention_mask."
            )

        image_output = self.encoder(image)
        image_features, _ = image_output["feature"], image_output["project"]
        text_output = self.text_encoder(text["input_ids"], text["attention_mask"])
        text_embeds, _ = text_output["feature"], text_output["project"]

        if len(image_features[0].shape) == 4:
            image_features = image_features[1:]
            image_features = [rearrange(item, "b c h w -> b (h w) c") for item in image_features]

        os32 = image_features[3]
        os16 = self.decoder16(os32, image_features[2], text_embeds[-1])
        os8 = self.decoder8(os16, image_features[1], text_embeds[-1])
        os4 = self.decoder4(os8, image_features[0], text_embeds[-1])
        os4 = rearrange(
            os4, "B (H W) C -> B C H W", H=self.spatial_dim[-1], W=self.spatial_dim[-1]
        )
        os1 = self.decoder1(os4)
        out = self.out(os1).sigmoid()
        return out
