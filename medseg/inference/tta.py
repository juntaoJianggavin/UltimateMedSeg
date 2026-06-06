"""Test-Time Augmentation (TTA) wrapper for segmentation models.

Applies a set of geometric and/or photometric augmentations at inference
time, runs the wrapped model on each augmented copy, **inverts** the
spatial augmentations on the predicted logits, then merges all logits
(default: arithmetic mean) to produce a single robust prediction.

Built-in augmentations (selectable by name in ``augmentations`` list):

Geometric (mask is inverse-transformed):
    'identity'         no change
    'rot90'            90° clockwise rotation
    'rot180'           180° rotation
    'rot270'           270° rotation (= 90° counter-clockwise)
    'hflip'            horizontal flip
    'vflip'            vertical flip
    'transpose'        H↔W transpose (diagonal flip)

Photometric (do **not** require inverse on the mask):
    'brightness_up'    multiply by 1+brightness_delta (default +5%)
    'brightness_down'  multiply by 1-brightness_delta
    'contrast_up'      contrast scale 1+contrast_delta around per-channel mean
    'contrast_down'    contrast scale 1-contrast_delta
    'gamma_up'         gamma=1/1+gamma_delta (brighten)
    'gamma_down'       gamma=1+gamma_delta   (darken)

Default preset (``augmentations=None``) corresponds to the user's
specification: 3×90° rotations + 2 flips + identity + brightness ±:

    ['identity', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip',
     'brightness_up', 'brightness_down']

Merge modes (``merge``):
    'mean'      arithmetic mean of logits (default)
    'gmean'     geometric mean of softmax probs (returned as logits)
    'max'       per-pixel max logit
    'median'    per-pixel median logit

Usage::

    >>> tta = TTAWrapper(model, augmentations=['identity', 'hflip', 'rot90'])
    >>> logits = tta(images)                    # (B, C, H, W)

The wrapper is itself an ``nn.Module`` so it can be wrapped in turn by
:class:`EnsembleModel` (or vice versa).
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


AVAILABLE_TTAS: Tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "hflip",
    "vflip",
    "transpose",
    "brightness_up",
    "brightness_down",
    "contrast_up",
    "contrast_down",
    "gamma_up",
    "gamma_down",
)

DEFAULT_AUGS: Tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "hflip",
    "vflip",
    "brightness_up",
    "brightness_down",
)

GEOMETRIC_AUGS = {
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "hflip",
    "vflip",
    "transpose",
}


def _to_main_logits(out) -> torch.Tensor:
    if isinstance(out, (list, tuple)):
        return out[0]
    if isinstance(out, dict):
        for key in ("out", "logits", "pred", "main"):
            if key in out:
                return out[key]
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    return out


# ----------------------------------------------------------------------
# Forward / inverse spatial transforms
# ----------------------------------------------------------------------
def _apply_spatial(x: torch.Tensor, aug: str) -> torch.Tensor:
    if aug == "identity":
        return x
    if aug == "rot90":
        return torch.rot90(x, k=1, dims=(-2, -1))
    if aug == "rot180":
        return torch.rot90(x, k=2, dims=(-2, -1))
    if aug == "rot270":
        return torch.rot90(x, k=3, dims=(-2, -1))
    if aug == "hflip":
        return torch.flip(x, dims=(-1,))
    if aug == "vflip":
        return torch.flip(x, dims=(-2,))
    if aug == "transpose":
        return x.transpose(-2, -1).contiguous()
    return x  # photometric (no spatial change) handled separately


def _invert_spatial(y: torch.Tensor, aug: str) -> torch.Tensor:
    if aug == "identity":
        return y
    if aug == "rot90":
        # forward rotated by k=+1 → invert with k=-1 (= +3)
        return torch.rot90(y, k=-1, dims=(-2, -1))
    if aug == "rot180":
        return torch.rot90(y, k=-2, dims=(-2, -1))
    if aug == "rot270":
        return torch.rot90(y, k=-3, dims=(-2, -1))
    if aug == "hflip":
        return torch.flip(y, dims=(-1,))
    if aug == "vflip":
        return torch.flip(y, dims=(-2,))
    if aug == "transpose":
        return y.transpose(-2, -1).contiguous()
    return y


# ----------------------------------------------------------------------
# Photometric transforms (input only)
# ----------------------------------------------------------------------
def _apply_photometric(
    x: torch.Tensor,
    aug: str,
    brightness_delta: float = 0.05,
    contrast_delta: float = 0.10,
    gamma_delta: float = 0.10,
) -> torch.Tensor:
    """Photometric augmentations.

    Operates without assuming a fixed input range. ``brightness/contrast``
    are simple multiplicative perturbations; ``gamma`` is applied on the
    per-batch min-max-rescaled image, then mapped back. None of these
    operations need to be inverted on the predicted mask, since they only
    perturb intensities (not geometry).
    """
    if aug == "brightness_up":
        return x * (1.0 + brightness_delta)
    if aug == "brightness_down":
        return x * (1.0 - brightness_delta)
    if aug == "contrast_up":
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return (x - mean) * (1.0 + contrast_delta) + mean
    if aug == "contrast_down":
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return (x - mean) * (1.0 - contrast_delta) + mean
    if aug in ("gamma_up", "gamma_down"):
        # Rescale per-image to [0,1] for safe gamma, then back
        flat_min = x.amin(dim=(-2, -1), keepdim=True)
        flat_max = x.amax(dim=(-2, -1), keepdim=True)
        scale = (flat_max - flat_min).clamp(min=1e-6)
        x01 = (x - flat_min) / scale
        gamma = (1.0 / (1.0 + gamma_delta)) if aug == "gamma_up" else (1.0 + gamma_delta)
        x01 = x01.clamp(min=0.0, max=1.0).pow(gamma)
        return x01 * scale + flat_min
    return x


# ----------------------------------------------------------------------
class TTAWrapper(nn.Module):
    """Test-Time Augmentation wrapper.

    Args:
        model: any segmentation ``nn.Module`` returning logits ``(B, C, H, W)``.
        augmentations: sequence of names from :data:`AVAILABLE_TTAS`. Default
            is :data:`DEFAULT_AUGS` (3 rotations + 2 flips + identity +
            brightness ±).
        weights: optional per-augmentation weights; auto-normalised. Default
            equal weights.
        merge: ``'mean' | 'gmean' | 'max' | 'median'``.
        brightness_delta / contrast_delta / gamma_delta: photometric
            perturbation magnitudes.
        ignore_unknown: if ``True``, silently drop unknown augmentations
            instead of raising.
    """

    MERGE_MODES = ("mean", "gmean", "max", "median")

    def __init__(
        self,
        model: nn.Module,
        augmentations: Optional[Sequence[str]] = None,
        weights: Optional[Sequence[float]] = None,
        merge: str = "mean",
        brightness_delta: float = 0.05,
        contrast_delta: float = 0.10,
        gamma_delta: float = 0.10,
        ignore_unknown: bool = False,
    ):
        super().__init__()
        self.model = model

        if augmentations is None:
            augmentations = list(DEFAULT_AUGS)
        else:
            augmentations = list(augmentations)
        # Validate
        clean: List[str] = []
        for a in augmentations:
            if a not in AVAILABLE_TTAS:
                if ignore_unknown:
                    logger.warning(f"[tta] dropping unknown aug '{a}'")
                    continue
                raise ValueError(
                    f"Unknown augmentation '{a}'. Available: {AVAILABLE_TTAS}"
                )
            clean.append(a)
        if len(clean) == 0:
            raise ValueError("TTAWrapper needs at least one augmentation.")
        self.augmentations = clean

        if weights is None:
            weights = [1.0 / len(clean)] * len(clean)
        else:
            if len(weights) != len(clean):
                raise ValueError(
                    f"weights length {len(weights)} != #augs {len(clean)}"
                )
            s = float(sum(weights))
            if s <= 0:
                raise ValueError("Sum of TTA weights must be positive.")
            weights = [float(w) / s for w in weights]
        self.register_buffer(
            "weights", torch.tensor(weights, dtype=torch.float32)
        )

        if merge not in self.MERGE_MODES:
            raise ValueError(
                f"Unknown merge='{merge}'. Choose from {self.MERGE_MODES}"
            )
        self.merge = merge

        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta
        self.gamma_delta = gamma_delta

    # ------------------------------------------------------------------
    def _augment_input(self, x: torch.Tensor, aug: str) -> torch.Tensor:
        if aug in GEOMETRIC_AUGS:
            return _apply_spatial(x, aug)
        return _apply_photometric(
            x,
            aug,
            brightness_delta=self.brightness_delta,
            contrast_delta=self.contrast_delta,
            gamma_delta=self.gamma_delta,
        )

    def _invert_output(self, y: torch.Tensor, aug: str) -> torch.Tensor:
        if aug in GEOMETRIC_AUGS:
            return _invert_spatial(y, aug)
        return y  # photometric: no inverse on mask

    # ------------------------------------------------------------------
    def forward_per_aug(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run model on each augmented copy and invert geometry."""
        outs: List[torch.Tensor] = []
        for aug in self.augmentations:
            x_aug = self._augment_input(x, aug)
            y = _to_main_logits(self.model(x_aug))
            y_back = self._invert_output(y, aug)
            outs.append(y_back)
        return outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.forward_per_aug(x)

        # Sanity: all outputs must agree on (B, C, H, W); if a model resizes,
        # we let it through but log once.
        ref_shape = outs[0].shape
        for i, o in enumerate(outs[1:], start=1):
            if o.shape != ref_shape:
                logger.warning(
                    f"[tta] aug '{self.augmentations[i]}' produced shape "
                    f"{tuple(o.shape)} != ref {tuple(ref_shape)}; skipping."
                )
        outs = [o for o in outs if o.shape == ref_shape]

        stacked = torch.stack(outs, dim=0)  # (T, B, C, H, W)
        if self.merge == "mean":
            w = self.weights[: stacked.size(0)].view(-1, 1, 1, 1, 1).to(stacked.device)
            w = w / w.sum().clamp(min=1e-8)
            return (stacked * w).sum(dim=0)
        if self.merge == "gmean":
            probs = torch.softmax(stacked, dim=2).clamp(min=1e-8)
            log_p = torch.log(probs)
            w = self.weights[: stacked.size(0)].view(-1, 1, 1, 1, 1).to(stacked.device)
            w = w / w.sum().clamp(min=1e-8)
            return (log_p * w).sum(dim=0)  # log(geometric mean) = weighted mean of log
        if self.merge == "max":
            return stacked.max(dim=0).values
        if self.merge == "median":
            return stacked.median(dim=0).values
        raise ValueError(self.merge)  # pragma: no cover


# ----------------------------------------------------------------------
def build_tta_from_config(model: nn.Module, cfg: dict) -> TTAWrapper:
    """Build a TTAWrapper from a yaml-style config dict.

    Schema::

        tta:
          enabled: true
          augmentations: [identity, rot90, rot180, rot270, hflip, vflip,
                          brightness_up, brightness_down]
          merge: mean
          weights: null
          brightness_delta: 0.05
          contrast_delta: 0.10
          gamma_delta: 0.10
    """
    return TTAWrapper(
        model=model,
        augmentations=cfg.get("augmentations"),
        weights=cfg.get("weights"),
        merge=cfg.get("merge", "mean"),
        brightness_delta=cfg.get("brightness_delta", 0.05),
        contrast_delta=cfg.get("contrast_delta", 0.10),
        gamma_delta=cfg.get("gamma_delta", 0.10),
        ignore_unknown=cfg.get("ignore_unknown", False),
    )
