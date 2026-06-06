"""Deep supervision loss wrapper.

When a model returns a list [main_output, aux1, aux2, ...] during training
with deep_supervision=True, this loss wrapper computes weighted losses on all outputs.
"""

import torch.nn as nn
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("deep_supervision")
class DeepSupervisionLoss(nn.Module):
    """Wraps any base loss to support deep supervision multi-output.

    Convention:
      - Model output is a list: [main_pred, aux_pred_1, aux_pred_2, ...]
        where all predictions are at the same spatial size (input resolution).
      - Loss = base_loss(main_pred, target) + sum(aux_weight * base_loss(aux_i, target))

    Usage in YAML config:
        loss:
          name: deep_supervision
          params:
            base_loss:
              name: compound
              params:
                losses:
                  - name: ce
                    weight: 1.0
                  - name: dice
                    weight: 1.0
            aux_weight: 0.4
    """

    def __init__(self, base_loss=None, aux_weight=0.4, **kwargs):
        super().__init__()
        if base_loss is None:
            base_loss = {"name": "compound"}
        # Build the base loss from registry
        loss_cls = LOSS_REGISTRY.get(base_loss["name"])
        self.base_loss = loss_cls(**base_loss.get("params", {}))
        self.aux_weight = aux_weight

    def forward(self, pred, target):
        """Compute deep supervision loss.

        Args:
            pred: Either a single tensor (no DS) or a list [main, aux1, aux2, ...].
            target: Ground truth tensor.
        """
        if isinstance(pred, (list, tuple)):
            # Deep supervision mode
            main_loss = self.base_loss(pred[0], target)
            aux_loss = sum(self.base_loss(p, target) for p in pred[1:])
            n_aux = len(pred) - 1
            if n_aux > 0:
                return main_loss + self.aux_weight * aux_loss / n_aux
            return main_loss
        else:
            # Single output mode (no DS or inference)
            return self.base_loss(pred, target)
