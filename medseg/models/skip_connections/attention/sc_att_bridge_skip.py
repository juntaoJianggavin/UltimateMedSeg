"""SC-Att-Bridge skip (per-pair adaptation of MALUNet's SC_Att_Bridge).

Original (MALUNet, BIBM 2022): operates on a LIST of encoder skip features at
all scales — first applies per-feature spatial attention (7x7 conv on
avg+max channel-pool), then cross-feature channel attention (concat
avg-pool of all features, 1D conv across the channel axis, project back to
each feature's channel count). The two attention paths are residually
combined.

This module adapts the same Spatial + Channel attention design to the
framework's per-pair (decoder_feat, skip_feat) skip interface so it can be
used as a drop-in skip. Cross-feature channel attention reduces to a
2-feature mixing (decoder + skip) instead of N=5.

Distinct from `cab_skip` (which is SE-only on the skip feature) — this
module weights BOTH spatial and channel dimensions on BOTH features.
"""
# Source: https://github.com/JCruan519/MALUNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("sc_att_bridge")
class SCAttBridgeSkip(nn.Module):
    """Spatial-Channel Attention Bridge skip (per-pair).

    Topology (paraphrasing MALUNet's SCABridge for 2 features):
        residuals = [d, s]
        satts = [shared_conv2d(cat(avg(d), max(d))), shared_conv2d(cat(avg(s), max(s)))]
        feats = [satts[i] * x for i, x in enumerate(residuals)]
        r2 = list(feats)
        feats = [f + r for f, r in zip(feats, residuals)]
        catts = channel_att(feats)              # cross-feature 1D-conv mixing
        feats = [c * f for c, f in zip(catts, feats)]
        feats = [f + r for f, r in zip(feats, r2)]
        return cat(feats[0], feats[1])
    """

    def __init__(self, split_att: str = "fc", **kwargs):
        super().__init__()
        # 7x7 conv on channel-pooled features (shared across decoder & skip)
        self.shared_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=9, dilation=3),
            nn.Sigmoid(),
        )
        self.split_att = split_att
        self._channel_caches = {}  # (d_ch, s_ch) -> nn.Module

    def get_out_channels(self, decoder_ch: int, skip_ch: int) -> int:
        return decoder_ch + skip_ch

    def _channel_att(self, d_ch: int, s_ch: int, device):
        """Build / fetch the cross-feature channel-attention head (lazy)."""
        key = (d_ch, s_ch, device)
        if key in self._channel_caches:
            return self._channel_caches[key]

        total = d_ch + s_ch
        # 1D Conv across the 1-D "channel-as-spatial" axis of length 2 (two features pooled to 1×1)
        get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False).to(device)
        if self.split_att == "fc":
            head_d = nn.Linear(total, d_ch).to(device)
            head_s = nn.Linear(total, s_ch).to(device)
        else:
            head_d = nn.Conv1d(total, d_ch, 1).to(device)
            head_s = nn.Conv1d(total, s_ch, 1).to(device)
        sigmoid = nn.Sigmoid()
        mod = nn.ModuleDict({
            "get_all_att": get_all_att,
            "head_d": head_d,
            "head_s": head_s,
            "sigmoid": sigmoid,
        })
        # Keep a reference so optimizer's .parameters() picks them up after a forward pass
        setattr(self, f"_ch_{d_ch}_{s_ch}", mod)
        self._channel_caches[key] = mod
        return mod

    def _spatial_att(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.shared_spatial(torch.cat([avg_out, max_out], dim=1))

    def forward(self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        # Align spatial dims (decoder is usually upsampled to skip size by the framework,
        # but guard anyway).
        if skip_feat.shape[2:] != decoder_feat.shape[2:]:
            skip_feat = F.interpolate(skip_feat, size=decoder_feat.shape[2:],
                                      mode="bilinear", align_corners=False)

        d, s = decoder_feat, skip_feat
        residuals = [d, s]

        # Spatial attention path
        s_att_d = self._spatial_att(d)
        s_att_s = self._spatial_att(s)
        feats = [s_att_d * d, s_att_s * s]
        r2 = list(feats)
        feats = [f + r for f, r in zip(feats, residuals)]

        # Cross-feature channel attention
        d_ch, s_ch = d.shape[1], s.shape[1]
        mod = self._channel_att(d_ch, s_ch, d.device)
        # Pool both features to (B, C, 1, 1) then concat → (B, d+s, 1, 1)
        pooled = torch.cat(
            [F.adaptive_avg_pool2d(feats[0], 1), F.adaptive_avg_pool2d(feats[1], 1)],
            dim=1,
        )  # (B, d+s, 1, 1)
        att = pooled.squeeze(-1).transpose(-1, -2)  # (B, 1, d+s)
        att = mod["get_all_att"](att)               # (B, 1, d+s)
        if self.split_att != "fc":
            att = att.transpose(-1, -2)             # (B, d+s, 1)

        if self.split_att == "fc":
            a_d = mod["sigmoid"](mod["head_d"](att))  # (B, 1, d_ch)
            a_s = mod["sigmoid"](mod["head_s"](att))  # (B, 1, s_ch)
            a_d = a_d.transpose(-1, -2).unsqueeze(-1).expand_as(feats[0])  # (B, d_ch, H, W)
            a_s = a_s.transpose(-1, -2).unsqueeze(-1).expand_as(feats[1])
        else:
            a_d = mod["sigmoid"](mod["head_d"](att)).unsqueeze(-1).expand_as(feats[0])
            a_s = mod["sigmoid"](mod["head_s"](att)).unsqueeze(-1).expand_as(feats[1])

        feats = [a_d * feats[0], a_s * feats[1]]
        feats = [f + r for f, r in zip(feats, r2)]

        return torch.cat(feats, dim=1)
