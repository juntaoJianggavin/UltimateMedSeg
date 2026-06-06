"""HAM (Hamburger) Decoder - faithfully ported from SegNeXt.
Reference: https://github.com/Visual-Attention-Network/SegNeXt/blob/main/mmseg/models/decode_heads/ham_head.py

Uses NMF (Non-negative Matrix Factorization) for global context modeling.
Takes ALL multi-scale features. External skip_connection is IGNORED.
"""
# Source: https://github.com/Gsunshine/Enjoy-Hamburger

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from medseg.registry import DECODER_REGISTRY


class NMF2D(nn.Module):
    """NMF-based matrix decomposition - faithful port from SegNeXt."""
    def __init__(self, S=1, D=512, R=64, train_steps=6, eval_steps=7):
        super().__init__()
        self.S = S
        self.D = D
        self.R = R
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.inv_t = 1

    def _build_bases(self, B, S, D, R, device):
        bases = torch.rand((B * S, D, R), device=device)
        bases = F.normalize(bases, dim=1)
        return bases

    def local_step(self, x, bases, coef):
        # (B*S, D, N)^T @ (B*S, D, R) -> (B*S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B*S, N, R) @ [(B*S, D, R)^T @ (B*S, D, R)] -> (B*S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        # (B*S, D, N) @ (B*S, N, R) -> (B*S, D, R)
        numerator = torch.bmm(x, coef)
        # (B*S, D, R) @ [(B*S, N, R)^T @ (B*S, N, R)] -> (B*S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def local_inference(self, x, bases):
        # (B*S, D, N)^T @ (B*S, D, R) -> (B*S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef

    def forward(self, x):
        B, C, H, W = x.shape
        D = C // self.S
        N = H * W
        x_flat = x.view(B * self.S, D, N)

        bases = self._build_bases(B, self.S, D, self.R, x.device)
        bases, coef = self.local_inference(x_flat, bases)
        coef = self.compute_coef(x_flat, bases, coef)

        # Reconstruct: (B*S, D, R) @ (B*S, N, R)^T -> (B*S, D, N)
        x_flat = torch.bmm(bases, coef.transpose(1, 2))
        x = x_flat.view(B, C, H, W)
        return x


class Hamburger(nn.Module):
    """Hamburger module - faithful port from SegNeXt."""
    def __init__(self, ham_channels=512, S=1, R=64, train_steps=6, eval_steps=7):
        super().__init__()
        self.ham_in = nn.Sequential(
            nn.Conv2d(ham_channels, ham_channels, 1),
        )
        self.ham = NMF2D(S=S, D=ham_channels, R=R,
                          train_steps=train_steps, eval_steps=eval_steps)
        self.ham_out = nn.Sequential(
            nn.Conv2d(ham_channels, ham_channels, 1),
            nn.GroupNorm(32, ham_channels),
        )

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)
        return ham


@DECODER_REGISTRY.register("ham")
class HAMDecoder(nn.Module):
    """LightHamHead decoder - faithful port from SegNeXt.

    Takes ALL multi-scale features, resizes to highest resolution, concatenates,
    squeezes, applies Hamburger (NMF), then aligns.

    External skip_connection parameter is IGNORED.
    """
    has_internal_skip = True

    def __init__(self, encoder_channels: List[int], bottleneck_channels: int,
                 skip_connection=None, ham_channels: int = 512, embed_dim: int = 256,
                 nmf_R: int = 64, **kwargs):
        super().__init__()
        all_channels = list(encoder_channels) + [bottleneck_channels]

        # squeeze: reduce concatenated channels to ham_channels
        self.squeeze = nn.Sequential(
            nn.Conv2d(sum(all_channels), ham_channels, 1, bias=False),
            nn.BatchNorm2d(ham_channels),
            nn.ReLU(inplace=True),
        )

        self.hamburger = Hamburger(ham_channels, R=nmf_R)

        # align: reduce to output dimension
        self.align = nn.Sequential(
            nn.Conv2d(ham_channels, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self._out_channels = embed_dim

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, bottleneck_feat: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        all_features = list(skip_features) + [bottleneck_feat]
        target_size = all_features[0].shape[2:]

        # Resize all to highest resolution
        resized = []
        for feat in all_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized.append(feat)

        x = torch.cat(resized, dim=1)
        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        return x
