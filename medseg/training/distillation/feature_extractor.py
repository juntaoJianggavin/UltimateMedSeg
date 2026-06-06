"""Feature extractor helper for hook-based intermediate feature capture.

(Self-contained: no single canonical GitHub source.)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class FeatureExtractor(nn.Module):
    """Extract intermediate features from UNet models."""

    def __init__(self, model, feature_layers: Optional[List[str]] = None):
        super().__init__()
        self.model = model
        self.feature_layers = feature_layers or []
        self.features = {}

        # Register hooks for feature extraction
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        def make_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        for name, module in self.model.named_modules():
            for layer_name in self.feature_layers:
                if layer_name in name:
                    module.register_forward_hook(make_hook(name))

    def forward(self, x):
        """Forward pass and return both output and features."""
        output = self.model(x)
        return output, self.features.copy()
