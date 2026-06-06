"""Attention Gate skip connection (Oktay 2018 — Attention U-Net)."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("attention_gate")
class AttentionGateSkip(nn.Module):
    """Attention Gate skip (Oktay 2018).

    g = decoder (gating signal), x = skip features.
    theta_x = Conv1x1(x), phi_g = Conv1x1(g), align spatial via bilinear,
    add + ReLU, Conv1x1, sigmoid -> attention map alpha.
    Output = concat(alpha * x, decoder).
    """

    def __init__(self, inter_channels=None, **kwargs):
        super().__init__()
        # If None, defaults to skip_ch // 2 (min 1) at build time.
        self.inter_channels = inter_channels
        # Lazily-built submodules keyed by (decoder_ch, skip_ch).
        self._theta_xs = nn.ModuleDict()
        self._phi_gs = nn.ModuleDict()
        self._psis = nn.ModuleDict()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + skip_ch

    def _key(self, dc, sc):
        return f"{dc}_{sc}"

    def _build(self, decoder_ch, skip_ch, device):
        key = self._key(decoder_ch, skip_ch)
        if key in self._psis:
            return
        inter = self.inter_channels if self.inter_channels is not None else max(skip_ch // 2, 1)

        theta_x = nn.Conv2d(skip_ch, inter, kernel_size=1, bias=False)
        phi_g = nn.Conv2d(decoder_ch, inter, kernel_size=1, bias=False)
        psi = nn.Sequential(
            nn.Conv2d(inter, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self._theta_xs[key] = theta_x.to(device)
        self._phi_gs[key] = phi_g.to(device)
        self._psis[key] = psi.to(device)

    def forward(self, decoder_feat, skip_feat):
        # Spatial align skip to decoder if needed.
        if skip_feat.shape[-2:] != decoder_feat.shape[-2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        decoder_ch = decoder_feat.shape[1]
        skip_ch = skip_feat.shape[1]
        self._build(decoder_ch, skip_ch, decoder_feat.device)
        key = self._key(decoder_ch, skip_ch)

        theta_x = self._theta_xs[key](skip_feat)
        phi_g = self._phi_gs[key](decoder_feat)

        # phi_g should match theta_x spatially; align via bilinear if differ.
        if phi_g.shape[-2:] != theta_x.shape[-2:]:
            phi_g = F.interpolate(
                phi_g, size=theta_x.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        f = F.relu(theta_x + phi_g, inplace=True)
        alpha = self._psis[key](f)  # (B, 1, H, W)

        attended = skip_feat * alpha
        return torch.cat([decoder_feat, attended], dim=1)
