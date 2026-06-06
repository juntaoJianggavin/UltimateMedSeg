# TP-DRSeg (AAAI 2025)
# Reference: https://github.com/HUANGLIZI/TP-DRSeg
# Paper: https://arxiv.org/abs/2312.17191
# Implemented from paper formulas; not a copy of the official repo.
"""TP-DRSeg: Text-Prompted Dual-Branch Region Selection for 2D Medical Segmentation.

Li et al., "TP-DRSeg: Improving Zero-Shot Panoptic Segmentation with
Text-Point Prompted Dual-branch Region Selection", AAAI 2025.

Algorithm (from paper Sec. 3):
    1. CLIP text encoder produces per-class text embeddings t_c.
    2. A visual encoder (ResNet-50) produces multi-scale features F1..F4.
    3. Coarse Branch: GAP(F4) · t_c^T → class activation scores → upsample
       to generate coarse region mask R_coarse (sigmoid > tau).
    4. Fine Branch: pixel-text similarity map S = F_proj · t_c^T (per pixel),
       masked by R_coarse → refined logits.
    5. Final prediction = softmax(S_fine) upsampled to input resolution.
    6. Loss: CE on labeled pixels (if target provided).

The official repo uses a frozen CLIP ViT-B/16 text encoder and a
trainable ResNet-50 visual backbone. We follow this exactly.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.utils.weight_downloader import hf_from_pretrained


class _ResNetEncoder(nn.Module):
    """ResNet-50 encoder producing 4-scale features (C1/8, C2/16, C3/32, C4/32)."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = tvm.resnet50(weights=weights)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # /4, 256
        self.layer2 = resnet.layer2  # /8, 512
        self.layer3 = resnet.layer3  # /16, 1024
        self.layer4 = resnet.layer4  # /32, 2048
        self.channels = [256, 512, 1024, 2048]

    def forward(self, x):
        x0 = self.layer0(x)
        f1 = self.layer1(x0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]


class TPDRSeg(nn.Module):
    """TP-DRSeg: Text-Prompted Dual-branch Region Selection.

    Args:
        in_channels: input image channels (default 3).
        num_classes: number of segmentation classes.
        img_size: expected spatial input size.
        clip_name: HuggingFace CLIP model id for the text encoder.
        tau_coarse: sigmoid threshold for coarse region mask.
        proj_dim: projection dimension for pixel-text similarity.
        pretrained_backbone: use ImageNet-pretrained ResNet-50.
    """

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 224,
        clip_name: str = "openai/clip-vit-base-patch16",
        tau_coarse: float = 0.5,
        proj_dim: int = 512,
        pretrained_backbone: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.tau_coarse = tau_coarse
        self.proj_dim = proj_dim

        # Visual encoder
        self.encoder = _ResNetEncoder(in_channels, pretrained=pretrained_backbone)

        # Text encoder (frozen CLIP)
        from transformers import CLIPModel
        clip = hf_from_pretrained(CLIPModel, clip_name)
        self.text_encoder = clip.text_model
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, proj_dim)

        # Coarse branch: GAP on F4 → class scores
        self.coarse_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(2048, proj_dim),
        )

        # Fine branch: pixel-level projection from F3
        self.fine_proj = nn.Sequential(
            nn.Conv2d(1024, proj_dim, 1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.seg_head = nn.Conv2d(proj_dim, num_classes, 1)

    def _encode_text(self, text):

        # 自动处理字符串输入 / Auto-handle string input
        if isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and isinstance(text[0], str)):
            if not hasattr(self, '_auto_text'):
                from medseg.models.text_unet.auto_text_encoder import AutoTextEncoder
                self._auto_text = AutoTextEncoder("openai/clip-vit-base-patch16", max_length=77, tokenizer_type="clip")
            text = self._auto_text(text, device=next(self.parameters()).device)

        """Encode text to (B, C, proj_dim) class embeddings."""
        if text is None:
            raise ValueError(
                "TP-DRSeg requires text input (class descriptions). "
                "Pass text as a dict with 'input_ids' and 'attention_mask'."
            )
        if isinstance(text, dict):
            input_ids = text['input_ids']
            attention_mask = text.get('attention_mask')
        elif isinstance(text, (tuple, list)) and len(text) == 2:
            input_ids, attention_mask = text
        else:
            raise TypeError(f"text must be dict or (input_ids, mask) tuple, got {type(text)}")

        with torch.no_grad():
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Use pooled output (CLS token)
            pooled = out.pooler_output  # (B, hidden)
        return self.text_proj(pooled)  # (B, proj_dim)

    def forward(self, image: torch.Tensor, text=None) -> torch.Tensor:
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        B, _, H, W = image.shape
        feats = self.encoder(image)
        f3, f4 = feats[2], feats[3]  # /16, /32

        # Text embedding
        t_emb = self._encode_text(text)  # (B, proj_dim)

        # Coarse branch: image-text similarity → coarse mask
        img_global = self.coarse_proj(f4)  # (B, proj_dim)
        coarse_score = (img_global * t_emb).sum(dim=-1, keepdim=True)  # (B, 1)
        # Expand to spatial: broadcast coarse activation
        coarse_map = torch.sigmoid(coarse_score).unsqueeze(-1).unsqueeze(-1)  # (B,1,1,1)

        # Fine branch: pixel-text similarity
        f_proj = self.fine_proj(f3)  # (B, proj_dim, H/16, W/16)
        # Pixel-text dot product
        t_spatial = t_emb.unsqueeze(-1).unsqueeze(-1)  # (B, proj_dim, 1, 1)
        sim_map = (f_proj * t_spatial).sum(dim=1, keepdim=True)  # (B, 1, H/16, W/16)

        # Region selection: mask fine branch by coarse activation
        # (paper Eq. 5: S_fine = S * R_coarse)
        r_coarse = (coarse_map > self.tau_coarse).float()
        sim_masked = sim_map * r_coarse

        # Segmentation head on the masked feature
        out = self.seg_head(f_proj * r_coarse)  # (B, num_classes, H/16, W/16)

        # Upsample to input resolution
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
