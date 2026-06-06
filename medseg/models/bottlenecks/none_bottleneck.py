"""None (pass-through) bottleneck."""
# Source: INTERNAL — framework adaptation (this repo).

import torch.nn as nn
from medseg.registry import BOTTLENECK_REGISTRY


@BOTTLENECK_REGISTRY.register("none")
class NoneBottleneck(nn.Module):
    """Pass-through bottleneck: does nothing."""
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return x
