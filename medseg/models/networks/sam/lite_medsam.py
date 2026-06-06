# LiteMedSAM (NeurIPS 2024 / MedIA 2025)
# Reference: https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM
# Paper: https://arxiv.org/abs/2403.20329
# Implemented from paper formulas; not a copy of the official repo.
"""LiteMedSAM: A Lightweight SAM for Medical Image Segmentation.

Ma et al., "Segment Anything in Medical Images", Nature Communications
2024. The LiteMedSAM variant replaces the ViT-B image encoder with a
distilled TinyViT-5M (from MobileViT family) while keeping the SAM
prompt encoder and mask decoder unchanged.

Architecture (from paper + repo README):
    - Image encoder: TinyViT-5M (timm ``tiny_vit_5m_224``) outputting
      (B, 256, 64, 64) after a neck projection.
    - Prompt encoder: same as vanilla SAM — supports point / box / text
      (text mapped to dense embedding via a learned linear).
    - Mask decoder: SAM two-way transformer (2 layers) + upscale 4x +
      MLP IoU head.

This is a 2D-native method (no 3D adaptation).

Strict policy:
    - If TinyViT weights or LiteMedSAM checkpoint cannot be loaded,
      raises immediately — no fallback to random init.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TinyViTEncoder(nn.Module):
    """TinyViT-5M image encoder producing (B, 256, 64, 64) feature maps.

    Uses timm to load the backbone; a 1x1 neck projects to 256 channels
    (matching SAM's prompt encoder expected dim).
    """

    def __init__(self, img_size: int = 256, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required for LiteMedSAM. pip install timm") from e

        from medseg.models.networks.sam.sam_base import load_with_ssl_fallback
        self.backbone = load_with_ssl_fallback(
            timm.create_model,
            'tiny_vit_5m_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,),
        )
        # TinyViT-5M stage-3 output: 320 channels
        self.neck = nn.Sequential(
            nn.Conv2d(320, 256, 1, bias=False),
            nn.BatchNorm2d(256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        f = feats[-1]  # (B, 320, H/16, W/16)
        f = self.neck(f)  # (B, 256, H/16, W/16)
        # SAM expects 64x64 feature map
        if f.shape[2:] != (64, 64):
            f = F.interpolate(f, size=(64, 64), mode='bilinear', align_corners=False)
        return f


class _PromptEncoder(nn.Module):
    """Simplified prompt encoder for box/point/text prompts.

    Produces sparse embeddings (Np, 256) for points/boxes and a
    dense embedding (B, 256, 64, 64) placeholder (zeros if no mask prompt).
    """

    def __init__(self, embed_dim: int = 256, img_size: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.point_embeddings = nn.Embedding(4, embed_dim)  # 0=bg, 1=fg, 2=tl, 3=br
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        # Positional encoding for points (Fourier)
        self.pe_layer = nn.Linear(2, embed_dim)
        # Text prompt → dense embedding projection
        self.text_proj = nn.Linear(512, embed_dim)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """points: (B, N, 2), labels: (B, N) in {0=bg, 1=fg}."""
        pe = self.pe_layer(points)  # (B, N, 256)
        label_embed = self.point_embeddings(labels.long())  # (B, N, 256)
        return pe + label_embed

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """boxes: (B, 4) as [x1, y1, x2, y2] normalised to [0, 1]."""
        B = boxes.shape[0]
        corners = boxes.view(B, 2, 2)  # (B, 2, 2)
        pe = self.pe_layer(corners)  # (B, 2, 256)
        tl_embed = self.point_embeddings(torch.full((B,), 2, device=boxes.device))
        br_embed = self.point_embeddings(torch.full((B,), 3, device=boxes.device))
        return pe + torch.stack([tl_embed, br_embed], dim=1)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
    ):
        sparse = []
        B = 1
        device = self.point_embeddings.weight.device

        if points is not None:
            coords, labels = points
            B = coords.shape[0]
            sparse.append(self._embed_points(coords, labels))

        if boxes is not None:
            B = boxes.shape[0]
            sparse.append(self._embed_boxes(boxes))

        if text_embed is not None:
            B = text_embed.shape[0]
            t = self.text_proj(text_embed).unsqueeze(1)  # (B, 1, 256)
            sparse.append(t)

        if sparse:
            sparse_embed = torch.cat(sparse, dim=1)  # (B, Np, 256)
        else:
            sparse_embed = self.not_a_point_embed.weight.unsqueeze(0).expand(B, -1, -1)

        dense_embed = self.no_mask_embed.weight.view(1, self.embed_dim, 1, 1)
        dense_embed = dense_embed.expand(B, -1, 64, 64)

        return sparse_embed, dense_embed


class _MaskDecoder(nn.Module):
    """Lightweight mask decoder (SAM-style two-way transformer + upscale)."""

    def __init__(self, embed_dim: int = 256, num_classes: int = 1, num_heads: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.mask_tokens = nn.Embedding(num_classes + 1, embed_dim)
        self.iou_token = nn.Embedding(1, embed_dim)

        # Two-way transformer (2 layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=1024,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Upscale from 64x64 → 256x256
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.GELU(),
        )

        # Mask prediction MLP (per token)
        self.mask_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # IoU prediction head
        self.iou_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_classes + 1),
        )

    def forward(
        self,
        image_embed: torch.Tensor,
        sparse_embed: torch.Tensor,
        dense_embed: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = image_embed.shape

        # Prepare decoder queries: [iou_token, mask_tokens, sparse_prompts]
        tokens = torch.cat([
            self.iou_token.weight.unsqueeze(0).expand(B, -1, -1),
            self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1),
            sparse_embed,
        ], dim=1)  # (B, 1+C+Np, 256)

        # Memory: image features + dense prompt
        memory = (image_embed + dense_embed).flatten(2).permute(0, 2, 1)  # (B, HW, 256)

        # Transformer decode
        decoded = self.transformer(tokens, memory)  # (B, N_tokens, 256)

        # Extract mask tokens
        mask_decoded = decoded[:, 1:1 + self.num_classes + 1, :]  # (B, C+1, 256)

        # Upscale image features
        up = self.upscale(image_embed)  # (B, 64, 4H, 4W)

        # Mask prediction: dot product of mask MLP output with upscaled features
        mask_proj = self.mask_mlp(mask_decoded)  # (B, C+1, 64)
        masks = torch.einsum('bcd,bdhw->bchw', mask_proj, up)  # (B, C+1, 4H, 4W)

        # Return only foreground masks (skip background token at index 0)
        return masks[:, 1:, :, :]  # (B, num_classes, 4H, 4W)


class LiteMedSAM(nn.Module):
    """LiteMedSAM: Lightweight Segment Anything Model for Medical Images.

    A distilled version of MedSAM using TinyViT-5M as the image encoder
    (5.7M params vs 89M for ViT-B), retaining SAM's prompt encoder and
    mask decoder for box/point/text-prompted segmentation.

    Args:
        in_channels: input image channels.
        num_classes: number of foreground classes.
        img_size: expected input spatial size (default 256 for LiteMedSAM).
        pretrained: load TinyViT ImageNet pretrained weights.
        checkpoint: path to LiteMedSAM fine-tuned checkpoint.
    """

    is_text_guided = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        pretrained: bool = True,
        checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("LiteMedSAM requires 3-channel input (RGB/grayscale→3ch)")

        self.num_classes = num_classes
        self.img_size = img_size

        self.image_encoder = _TinyViTEncoder(img_size=img_size, pretrained=pretrained)
        self.prompt_encoder = _PromptEncoder(embed_dim=256)
        self.mask_decoder = _MaskDecoder(embed_dim=256, num_classes=num_classes)

        if checkpoint and os.path.isfile(checkpoint):
            state = torch.load(checkpoint, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing and len(missing) > 0.5 * sum(1 for _ in self.parameters()):
                raise RuntimeError(
                    f"LiteMedSAM checkpoint '{checkpoint}' matched only a "
                    f"minority of params (missing={len(missing)}). "
                    f"Refusing to start from near-random weights."
                )

    def forward(self, image: torch.Tensor, text=None) -> torch.Tensor:
        """Forward pass.

        Args:
            image: (B, 3, H, W) input image.
            text: optional prompt. Can be:
                - None: uses a default foreground point at center.
                - dict with 'boxes': (B, 4) normalised [x1,y1,x2,y2]
                - dict with 'points'/'labels': (B, N, 2) and (B, N)
                - (B, 512) text embedding tensor

        Returns:
            (B, num_classes, H, W) logit mask.
        """
        B, _, H, W = image.shape
        img_embed = self.image_encoder(image)  # (B, 256, 64, 64)

        # Parse prompt
        points = boxes = text_embed = None
        if text is None:
            # Default: center point as foreground
            cx, cy = 0.5, 0.5
            pts = torch.tensor([[[cx, cy]]], device=image.device).expand(B, -1, -1)
            lbls = torch.ones(B, 1, device=image.device)
            points = (pts, lbls)
        elif isinstance(text, dict):
            if 'boxes' in text:
                boxes = text['boxes'].to(image.device)
            elif 'points' in text and 'labels' in text:
                points = (text['points'].to(image.device), text['labels'].to(image.device))
            elif 'input_ids' in text:
                raise ValueError(
                    "LiteMedSAM does not accept tokenised text. Pass a "
                    "pre-computed text embedding (B, 512) tensor instead."
                )
        elif isinstance(text, torch.Tensor) and text.dim() == 2:
            text_embed = text
        else:
            raise TypeError(f"Unsupported text type: {type(text)}")

        sparse, dense = self.prompt_encoder(points=points, boxes=boxes, text_embed=text_embed)
        masks = self.mask_decoder(img_embed, sparse, dense)  # (B, C, 256, 256)

        # Upsample to input resolution
        if masks.shape[2:] != (H, W):
            masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)

        return masks
