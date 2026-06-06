"""Base class for semi-supervised segmentation methods.

Reference implementation: https://github.com/HiLab-git/SSL4MIS
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any


class BaseSemiMethod(ABC):
    """Abstract base class for all semi-supervised segmentation methods.

    Subclasses must implement:
        - ``build()``: set up auxiliary models (teacher, second model, etc.)
        - ``train_step()``: one training iteration on labeled + unlabeled data
        - ``get_eval_model()``: return the model used for validation

    Attributes:
        model: The primary (student) model.
        device: Torch device.
        consistency_weight: Maximum consistency loss weight.
        rampup_epochs: Number of epochs to ramp up consistency weight.
        img_size: Image spatial size (used for strong augmentation).
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        self.model = model
        self.device = device
        self.consistency_weight = consistency_weight
        self.rampup_epochs = rampup_epochs
        self.img_size = img_size

    @abstractmethod
    def build(self) -> None:
        """Build auxiliary components (teacher model, extra decoders, etc.)."""
        ...

    @abstractmethod
    def train_step(
        self,
        labeled_batch: Dict[str, Any],
        unlabeled_batch: Dict[str, Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        """Perform one training step.

        Must call ``optimizer.zero_grad()``, ``loss.backward()``, and
        ``optimizer.step()`` internally so that methods like CPS can manage
        multiple optimizers.

        Returns:
            dict with at least keys ``"loss"``, ``"sup_loss"``, ``"unsup_loss"``.
        """
        ...

    def update(self, epoch: int) -> None:
        """Called once per training step for EMA or other updates.

        Default implementation does nothing.
        """
        pass

    @abstractmethod
    def get_eval_model(self) -> nn.Module:
        """Return the model to use for validation / inference."""
        ...

    def extra_params(self):
        """Return extra parameters that need an optimizer (e.g. CPS model2).

        Default returns empty list.
        """
        return []

    def extra_optimizers(self, lr: float = 1e-4):
        """Return list of (optimizer, name) tuples for auxiliary components.

        Used by methods like SASSNet that have a discriminator requiring
        its own optimizer.  Default returns empty list.
        """
        return []
