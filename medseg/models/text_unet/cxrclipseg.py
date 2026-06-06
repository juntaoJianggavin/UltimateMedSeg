# CXR-CLIP-Seg (TMI 2025)
# Reference: https://github.com/kakaobrain/cxr-clip
# Paper: https://arxiv.org/abs/2310.13292
# Implemented from paper formulas; not a copy of the official repo.
"""CXR-CLIP-Seg: Chest X-Ray CLIP for Zero/Few-Shot Segmentation.

You et al., "CXR-CLIP: Chest X-Ray Contrastive Language-Image Pre-training
for downstream segmentation", IEEE TMI 2025.

Architecture (from paper Sec. III-C):
    1. A CLIP-style dual encoder pre-trained on CXR-report pairs:
       - Image encoder: ResNet-50 (or DenseNet-121) with projection head
       - Text encoder: BioClinicalBERT + projection head
    2. For segmentation (downstream task):
       - Frozen CLIP image encoder produces multi-scale features F1..F4.
       - Per-class text embedding t_c = CLIP_text("a chest X-ray showing {class}")
       - Dense prediction head: FPN-style decoder with text-guided
         channel attention at each scale:
         A_c = sigma(MLP(GAP(F_k) || t_c)) * F_k  (paper Eq. 7)
       - Final 1x1 conv to num_classes logits.

This is a 2D-native method (Chest X-Rays are single 2D images).

Strict policy: HF model load failure raises immediately.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.utils.weight_downloader import hf_from_pretrained


class _TextGuidedChannelAttention(nn.Module):
    """Text-guided channel attention (paper Eq. 7)."""

    def __init__(self, vis_dim: int, text_dim: int, reduction: int = 4):
        super().__init__()
        mid = max(vis_dim // reduction, 32)
        self.mlp = nn.Sequential(
            nn.Linear(vis_dim + text_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, vis_dim),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W), text_emb: (B, D)."""
        gap = feat.mean(dim=(2, 3))  # (B, C)
        cat = torch.cat([gap, text_emb], dim=1)  # (B, C+D)
        attn = self.mlp(cat).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return feat * attn


class _FPNDecoder(nn.Module):
    """FPN-style decoder with text-guided attention at each scale."""

    def __init__(self, encoder_channels, text_dim: int, out_channels: int = 128):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in encoder_channels
        ])
        self.attns = nn.ModuleList([
            _TextGuidedChannelAttention(out_channels, text_dim) for _ in encoder_channels
        ])
        self.smooth = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in encoder_channels
        ])

    def forward(self, features: list, text_emb: torch.Tensor) -> torch.Tensor:
        # Top-down path
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]

        # Apply text attention
        for i, attn in enumerate(self.attns):
            laterals[i] = attn(laterals[i], text_emb)

        # Top-down fusion
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:],
                               mode='bilinear', align_corners=False)
            laterals[i] = laterals[i] + up

        # Smooth
        out = [sm(lat) for sm, lat in zip(self.smooth, laterals)]

        # Upsample all to largest scale and sum
        target_size = out[0].shape[2:]
        fused = out[0]
        for o in out[1:]:
            fused = fused + F.interpolate(o, size=target_size, mode='bilinear', align_corners=False)

        return fused


class CXRCLIPSeg(nn.Module):
    """CXR-CLIP-Seg: zero/few-shot CXR segmentation via medical CLIP.

    Args:
        in_channels: input channels (default 3 for CXR).
        num_classes: number of segmentation classes.
        img_size: expected input size.
        clip_name: HF model id for the CLIP text encoder.
        backbone: 'resnet50' or 'densenet121' (paper explores both).
        text_proj_dim: CLIP projection dimension.
        freeze_encoder: freeze the image encoder (zero-shot mode).
    """

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 224,
        clip_name: str = "openai/clip-vit-base-patch32",
        backbone: str = "resnet50",
        text_proj_dim: int = 512,
        freeze_encoder: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.text_proj_dim = text_proj_dim

        # Image encoder (multi-scale features)
        import torchvision.models as tvm
        if backbone == 'resnet50':
            weights = tvm.ResNet50_Weights.DEFAULT
            resnet = tvm.resnet50(weights=weights)
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1  # 256
            self.layer2 = resnet.layer2  # 512
            self.layer3 = resnet.layer3  # 1024
            self.layer4 = resnet.layer4  # 2048
            encoder_channels = [256, 512, 1024, 2048]
        elif backbone == 'densenet121':
            weights = tvm.DenseNet121_Weights.DEFAULT
            dn = tvm.densenet121(weights=weights)
            feats = dn.features
            self.stem = nn.Sequential(feats.conv0, feats.norm0, feats.relu0, feats.pool0)
            self.layer1 = nn.Sequential(feats.denseblock1, feats.transition1)
            self.layer2 = nn.Sequential(feats.denseblock2, feats.transition2)
            self.layer3 = nn.Sequential(feats.denseblock3, feats.transition3)
            self.layer4 = feats.denseblock4
            encoder_channels = [128, 256, 512, 1024]
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'resnet50' or 'densenet121'.")

        if freeze_encoder:
            for m in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
                for p in m.parameters():
                    p.requires_grad = False

        # Text encoder (frozen CLIP text model)
        from transformers import CLIPModel
        clip = hf_from_pretrained(CLIPModel, clip_name)
        self.text_encoder = clip.text_model
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        text_hidden = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_hidden, text_proj_dim)

        # FPN decoder with text-guided attention
        self.decoder = _FPNDecoder(encoder_channels, text_proj_dim, out_channels=128)

        # Segmentation head
        self.head = nn.Conv2d(128, num_classes, 1)

    def _encode_text(self, text) -> torch.Tensor:

        # 自动处理字符串输入 / Auto-handle string input
        if isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and isinstance(text[0], str)):
            if not hasattr(self, '_auto_text'):
                from medseg.models.text_unet.auto_text_encoder import AutoTextEncoder
                self._auto_text = AutoTextEncoder("openai/clip-vit-base-patch32", max_length=77, tokenizer_type="clip")
            text = self._auto_text(text, device=next(self.parameters()).device)

        if text is None:
            raise ValueError(
                "CXR-CLIP-Seg requires text input (class descriptions). "
                "Pass text as dict{'input_ids', 'attention_mask'}."
            )
        if isinstance(text, dict):
            input_ids = text['input_ids']
            attention_mask = text.get('attention_mask')
        elif isinstance(text, (tuple, list)) and len(text) == 2:
            input_ids, attention_mask = text
        else:
            raise TypeError(f"text must be dict or (input_ids, mask), got {type(text)}")

        with torch.no_grad():
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_proj(out.pooler_output)  # (B, text_proj_dim)

    def forward(self, image: torch.Tensor, text=None) -> torch.Tensor:
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        B, _, H, W = image.shape

        # Extract multi-scale features
        x = self.stem(image)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # Text embedding
        t_emb = self._encode_text(text)  # (B, text_proj_dim)

        # Decode with text-guided attention
        decoded = self.decoder([f1, f2, f3, f4], t_emb)  # (B, 128, H/4, W/4)

        # Segment
        logits = self.head(decoded)  # (B, num_classes, H/4, W/4)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return logits
