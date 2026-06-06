"""Compound Loss - combines multiple losses."""

import torch.nn as nn
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("compound")
class CompoundLoss(nn.Module):
    """Compound loss: weighted combination of multiple registered losses.

    Usage in config:
        loss:
          name: compound
          params:
            losses:
              - name: ce
                weight: 1.0
              - name: dice
                weight: 1.0
    """
    def __init__(self, losses=None, **kwargs):
        super().__init__()
        if losses is None:
            losses = [{"name": "ce", "weight": 1.0}, {"name": "dice", "weight": 1.0}]

        self.loss_modules = nn.ModuleList()
        self.weights = []
        for loss_cfg in losses:
            name = loss_cfg["name"]
            weight = loss_cfg.get("weight", 1.0)
            params = loss_cfg.get("params", {})
            loss_cls = LOSS_REGISTRY.get(name)
            self.loss_modules.append(loss_cls(**params))
            self.weights.append(weight)

    def forward(self, pred, target):
        total = 0.0
        for loss_fn, w in zip(self.loss_modules, self.weights):
            total = total + w * loss_fn(pred, target)
        return total
