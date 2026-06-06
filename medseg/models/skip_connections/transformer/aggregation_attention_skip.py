"""CASCADE AAM-style aggregation attention skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("aggregation_attention")
class AggregationAttentionSkip(nn.Module):
    """CASCADE Aggregation Attention Module (AAM)-style skip.

    Projects skip to decoder_ch via 1x1 conv, computes a soft attention
    gate from the decoder feature via sigmoid(Conv1x1(d)), multiplies the
    gate against the projected skip, then concatenates the gated skip
    with the decoder feature.

    Output channels = decoder_ch + decoder_ch = 2 * decoder_ch.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Lazily built submodules keyed by (decoder_ch, skip_ch)
        self._skip_projs = nn.ModuleDict()
        self._gates = nn.ModuleDict()

    def get_out_channels(self, decoder_ch, skip_ch):
        return decoder_ch + decoder_ch

    def _key(self, dc, sc):
        return f"{dc}_{sc}"

    def _build(self, decoder_ch, skip_ch, device):
        key = self._key(decoder_ch, skip_ch)
        if key in self._gates:
            return
        # 1x1 conv projecting skip features to decoder_ch
        skip_proj = nn.Conv2d(skip_ch, decoder_ch, kernel_size=1)
        # 1x1 conv on decoder feature -> sigmoid gate (per-channel, per-pixel)
        gate = nn.Sequential(
            nn.Conv2d(decoder_ch, decoder_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self._skip_projs[key] = skip_proj.to(device)
        self._gates[key] = gate.to(device)

    def forward(self, decoder_feat, skip_feat):
        # Spatial align skip to decoder if needed
        if skip_feat.shape[-2:] != decoder_feat.shape[-2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        decoder_ch = decoder_feat.shape[1]
        skip_ch = skip_feat.shape[1]
        self._build(decoder_ch, skip_ch, decoder_feat.device)
        key = self._key(decoder_ch, skip_ch)

        # Project skip to decoder_ch
        skip_proj = self._skip_projs[key](skip_feat)
        # Soft attention gate from decoder
        gate = self._gates[key](decoder_feat)
        # Apply gate to projected skip
        gated_skip = skip_proj * gate
        # Concatenate gated skip with decoder feature
        return torch.cat([gated_skip, decoder_feat], dim=1)
