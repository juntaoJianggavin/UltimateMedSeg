"""Inference utilities: model ensemble (logit averaging) + test-time augmentation.

Modules:
    ensemble.py : EnsembleModel for logit-averaging multiple segmentation models.
                  Supports both train-time ensemble (joint optimisation) and
                  inference-time ensemble (frozen sub-models loaded from
                  checkpoints).
    tta.py      : TTAWrapper applying geometric / photometric augmentations at
                  test time, then inverting masks back to the original frame
                  before merging logits.

Both wrappers are nn.Module so they compose freely:

    >>> ens  = EnsembleModel([m1, m2, m3], weights=[0.5, 0.3, 0.2])
    >>> tta  = TTAWrapper(ens, augmentations=["identity", "hflip", "vflip",
    ...                                       "rot90", "rot180", "rot270"])
    >>> logits = tta(images)        # (B, C, H, W)
"""

from medseg.inference.ensemble import (
    EnsembleModel,
    build_ensemble_from_config,
    load_ensemble_from_checkpoints,
)
from medseg.inference.tta import (
    TTAWrapper,
    AVAILABLE_TTAS,
    build_tta_from_config,
)

__all__ = [
    "EnsembleModel",
    "build_ensemble_from_config",
    "load_ensemble_from_checkpoints",
    "TTAWrapper",
    "AVAILABLE_TTAS",
    "build_tta_from_config",
]
