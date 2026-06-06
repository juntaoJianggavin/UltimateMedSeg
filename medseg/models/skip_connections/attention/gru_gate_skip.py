"""GRU-style gated skip connection."""
# Source: INTERNAL — framework adaptation (this repo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import SKIP_REGISTRY


@SKIP_REGISTRY.register("gru_gate")
class GRUGateSkip(nn.Module):
    """GRU-style gated skip connection.

    Treat the decoder feature as the previous hidden state and the skip
    feature as the new input. Compute update / reset gates and a candidate
    hidden state in GRU fashion, then linearly interpolate.

        z         = sigmoid(Conv1x1(concat(d, s)))
        r         = sigmoid(Conv1x1(concat(d, s)))
        h_tilde   = tanh(Conv1x1(concat(r * s, d)))
        out       = (1 - z) * d + z * h_tilde

    Decoder/skip features are first projected to a unified channel count
    equal to ``max(decoder_ch, skip_ch)``. Spatial dims of ``skip_feat``
    are bilinearly resized to match ``decoder_feat`` when they differ.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Lazily-built submodules keyed by (decoder_ch, skip_ch)
        self._dec_projs = nn.ModuleDict()
        self._skip_projs = nn.ModuleDict()
        self._z_convs = nn.ModuleDict()
        self._r_convs = nn.ModuleDict()
        self._h_convs = nn.ModuleDict()
        self._out_projs = nn.ModuleDict()

    def get_out_channels(self, decoder_ch, skip_ch):
        return max(decoder_ch, skip_ch)

    def _key(self, dc, sc):
        return f"{dc}_{sc}"

    def _build(self, decoder_ch, skip_ch, device):
        key = self._key(decoder_ch, skip_ch)
        if key in self._z_convs:
            return
        unified = max(decoder_ch, skip_ch)

        dec_proj = (nn.Conv2d(decoder_ch, unified, kernel_size=1)
                    if decoder_ch != unified else nn.Identity())
        skip_proj = (nn.Conv2d(skip_ch, unified, kernel_size=1)
                     if skip_ch != unified else nn.Identity())

        z_conv = nn.Conv2d(unified * 2, unified, kernel_size=1)
        r_conv = nn.Conv2d(unified * 2, unified, kernel_size=1)
        h_conv = nn.Conv2d(unified * 2, unified, kernel_size=1)
        out_proj = nn.Conv2d(unified, unified, kernel_size=1)

        self._dec_projs[key] = dec_proj.to(device)
        self._skip_projs[key] = skip_proj.to(device)
        self._z_convs[key] = z_conv.to(device)
        self._r_convs[key] = r_conv.to(device)
        self._h_convs[key] = h_conv.to(device)
        self._out_projs[key] = out_proj.to(device)

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

        d = self._dec_projs[key](decoder_feat)
        s = self._skip_projs[key](skip_feat)

        ds = torch.cat([d, s], dim=1)
        z = torch.sigmoid(self._z_convs[key](ds))
        r = torch.sigmoid(self._r_convs[key](ds))

        h_in = torch.cat([r * s, d], dim=1)
        h_tilde = torch.tanh(self._h_convs[key](h_in))

        out = (1.0 - z) * d + z * h_tilde
        out = self._out_projs[key](out)
        return out
