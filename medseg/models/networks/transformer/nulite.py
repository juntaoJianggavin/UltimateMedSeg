"""NuLite – lightweight nuclei segmentation with FastViT encoder.

Ported from: https://github.com/CosmoIknosLab/NuLite
Paper: NuLite – Lightweight and Fast Model for Nuclei Instance Segmentation
       and Classification (arXiv 2408.01797, 2024)

Architecture highlights
-----------------------
* FastViT (Apple, 2023) backbone as hierarchical feature encoder
* U-Net-style upsampling decoder with skip connections
* Lightweight Conv2D blocks + ConvTranspose2d upsampling
* Multi-task head (simplified to single segmentation head here)

Adapted for the project's standard interface:
    NuLite(in_channels, num_classes, img_size, pretrained, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── building blocks ──────────────────────────────────────────────────────────

class _Conv2DBlock(nn.Module):
    """Conv → BN → ReLU → Dropout."""

    def __init__(self, in_ch: int, out_ch: int, ksize: int = 3, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ksize, stride=1, padding=(ksize - 1) // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── FastViT encoder wrapper ──────────────────────────────────────────────────

class _FastViTEncoder(nn.Module):
    """Wrap a timm FastViT model to extract multi-scale features."""

    _EMBED_DIMS = {
        "fastvit_t8":  [48,  96, 192, 384],
        "fastvit_t12": [64, 128, 256, 512],
        "fastvit_s12": [64, 128, 256, 512],
        "fastvit_sa12":[64, 128, 256, 512],
        "fastvit_sa24":[64, 128, 256, 512],
        "fastvit_sa36":[64, 128, 256, 512],
        "fastvit_ma36":[76, 152, 304, 608],
    }

    def __init__(self, backbone: str, in_channels: int, pretrained: bool = True):
        super().__init__()
        if backbone not in self._EMBED_DIMS:
            raise ValueError(f"Unknown FastViT backbone '{backbone}'. "
                             f"Choose from {list(self._EMBED_DIMS)}")
        self.embed_dims = self._EMBED_DIMS[backbone]
        self.backbone = timm.create_model(
            f"{backbone}.apple_in1k",
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
        )

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)  # list of feature maps (typically 4)
        return feats


# ── decoder ──────────────────────────────────────────────────────────────────

class _Decoder(nn.Module):
    """U-Net upsampling decoder with skip connections.

    Follows the original NuLite ``create_upsampling_branch`` structure:
    bottleneck_up → decoder4 → decoder3 → decoder2 → decoder1,
    each stage uses Conv2D blocks + ConvTranspose2d for 2× upsampling.
    """

    def __init__(self, embed_dims: list, dropout: float = 0.0):
        super().__init__()
        # embed_dims = [c1, c2, c3, c4] from encoder stages 1-4
        # timm features_only may return 4 or 5 maps; we use the last 4
        c1, c2, c3, c4 = embed_dims

        # Bottleneck: c4 → c3, upsample 2×
        self.bottleneck_up = nn.Sequential(
            _Conv2DBlock(c4, c3, dropout=dropout),
            nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2),
        )
        # Decoder stage 4: cat(z3, up) → c2, upsample 2×
        self.dec4_up = nn.Sequential(
            _Conv2DBlock(c3 * 2, c3, dropout=dropout),
            _Conv2DBlock(c3, c2, dropout=dropout),
            nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2),
        )
        # Decoder stage 3: cat(z2, up) → c1, upsample 2×
        self.dec3_up = nn.Sequential(
            _Conv2DBlock(c2 * 2, c2, dropout=dropout),
            _Conv2DBlock(c2, c1, dropout=dropout),
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
        )
        # Decoder stage 2: cat(z1, up) → c1, upsample 2×
        self.dec2_up = nn.Sequential(
            _Conv2DBlock(c1 * 2, c1, dropout=dropout),
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
        )
        # Decoder stage 1: upsample 2×
        self.dec1_up = nn.Sequential(
            _Conv2DBlock(c1, c1, dropout=dropout),
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
        )

    def forward(self, feats: list):
        """
        Parameters
        ----------
        feats : list of Tensor
            Encoder feature maps [z1, z2, z3, z4] where z4 is deepest.
        """
        z1, z2, z3, z4 = feats

        b = self.bottleneck_up(z4)                  # c4 → c3, up 2×
        b = self.dec4_up(torch.cat([z3, b], dim=1))  # cat z3, c3→c2, up 2×
        b = self.dec3_up(torch.cat([z2, b], dim=1))  # cat z2, c2→c1, up 2×
        b = self.dec2_up(torch.cat([z1, b], dim=1))  # cat z1, c1, up 2×
        b = self.dec1_up(b)                          # c1, up 2×
        return b


# ── main model ───────────────────────────────────────────────────────────────

class NuLite(nn.Module):
    """NuLite: lightweight nuclei segmentation with FastViT encoder.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 3 for RGB pathology images).
    num_classes : int
        Number of output segmentation classes.
    img_size : int
        Input spatial resolution (kept for interface compatibility).
    pretrained : bool
        Whether to load ImageNet-pretrained FastViT weights.
    backbone : str
        FastViT variant. Default ``fastvit_t8`` (lightest).
    drop_rate : float
        Dropout rate inside decoder blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        pretrained: bool = True,
        backbone: str = "fastvit_t8",
        drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.encoder = _FastViTEncoder(backbone, in_channels, pretrained)
        embed_dims = self.encoder.embed_dims  # e.g. [64, 128, 256, 512]
        c1 = embed_dims[0]

        self.decoder = _Decoder(embed_dims, dropout=drop_rate)

        # decoder0: process raw input at same spatial resolution as decoder output
        # Original NuLite: Conv2DBlock(in_channels, c1) applied to raw input
        self.decoder0 = _Conv2DBlock(in_channels, c1, 3, dropout=drop_rate)

        # Segmentation head: cat(decoder0, decoder) → predict
        self.seg_head = nn.Sequential(
            _Conv2DBlock(c1 * 2, c1, dropout=drop_rate),
            nn.Conv2d(c1, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)

        # Use last 4 feature maps as skip connections
        if len(feats) > 4:
            feats = feats[-4:]
        elif len(feats) < 4:
            raise RuntimeError(f"Expected ≥4 feature maps from encoder, got {len(feats)}")

        dec_out = self.decoder(feats)  # (B, c1, H_dec, W_dec)

        # Process raw input through decoder0
        x_skip = self.decoder0(x)      # (B, c1, H, W)

        # Align spatial sizes (interpolate decoder0 output to match decoder output)
        if x_skip.shape[2:] != dec_out.shape[2:]:
            x_skip = F.interpolate(x_skip, size=dec_out.shape[2:], mode="bilinear", align_corners=False)

        fused = torch.cat([x_skip, dec_out], dim=1)
        out = self.seg_head(fused)
        return out
