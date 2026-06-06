"""EnsembleModel: logit averaging over multiple segmentation models.

Supports two operation modes (controlled implicitly by ``requires_grad``):

* **Train-time ensemble**
    All sub-models are trainable; their logits are averaged with optional
    learnable / fixed weights, and the criterion is applied to the averaged
    logits. This realises the classical "co-training / mean of experts"
    objective in a single ``forward`` call. Use ``mode='train'`` (default).

* **Inference-time ensemble**
    Sub-models are frozen (loaded from checkpoints) and their logits are
    averaged on the fly to produce a single prediction. Use ``mode='infer'``
    or :func:`load_ensemble_from_checkpoints`.

Resolution handling
-------------------
Sub-models may produce logits at different spatial sizes. We always
upsample to the **largest** of the outputs via bilinear interpolation
before averaging, so the ensemble logit map matches the highest-resolution
member.

Auxiliary outputs (deep supervision)
------------------------------------
If a sub-model returns a ``(list | tuple)`` of logits (e.g. deep
supervision), only the first entry – assumed to be the main head – is
used by ``EnsembleModel.forward``. The other entries can be retrieved per
sub-model with :meth:`forward_per_model` for advanced training schemes.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _to_main_logits(out) -> torch.Tensor:
    """Pick the main segmentation logits from a model output."""
    if isinstance(out, (list, tuple)):
        return out[0]
    if isinstance(out, dict):
        for key in ("out", "logits", "pred", "main"):
            if key in out:
                return out[key]
        # fallback: first tensor value
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    return out


class EnsembleModel(nn.Module):
    """Logit-averaging ensemble of multiple segmentation models.

    Args:
        models: Iterable of ``nn.Module``. All must accept the same input
            tensor shape and return logits ``(B, C, H, W)`` (or a list/tuple
            whose first entry has that shape).
        weights: Optional per-model weights. ``None`` (default) means equal
            weights. Weights are auto-normalised to sum to 1.
        mode: ``'train'`` (sub-models stay trainable) or ``'infer'`` (sub-
            models are frozen). Affects ``requires_grad`` of sub-model
            parameters and ``model.eval()`` calls.
        average: ``'logit'`` averages raw logits (default), ``'softmax'``
            averages softmax probabilities (then takes ``log`` so the result
            still acts like logits w.r.t. ``argmax``), ``'sigmoid'`` is the
            binary analogue.
        upsample_to: ``'max'`` (default) upsample all members to the largest
            output size, ``'min'`` to the smallest, ``'first'`` to the first
            member's size.
    """

    AVERAGES = ("logit", "softmax", "sigmoid")
    UPSAMPLE_MODES = ("max", "min", "first")

    def __init__(
        self,
        models: Iterable[nn.Module],
        weights: Optional[Sequence[float]] = None,
        mode: str = "train",
        average: str = "logit",
        upsample_to: str = "max",
    ):
        super().__init__()
        models = list(models)
        if len(models) == 0:
            raise ValueError("EnsembleModel requires at least one sub-model.")
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError(
                    f"weights length {len(weights)} != #models {len(models)}"
                )
            s = float(sum(weights))
            if s <= 0:
                raise ValueError("Sum of weights must be positive.")
            weights = [float(w) / s for w in weights]
        self.register_buffer(
            "weights", torch.tensor(weights, dtype=torch.float32)
        )

        if average not in self.AVERAGES:
            raise ValueError(
                f"Unknown average='{average}'. Choose from {self.AVERAGES}"
            )
        self.average = average

        if upsample_to not in self.UPSAMPLE_MODES:
            raise ValueError(
                f"Unknown upsample_to='{upsample_to}'. Choose from "
                f"{self.UPSAMPLE_MODES}"
            )
        self.upsample_to = upsample_to

        self.mode_ = mode
        if mode == "infer":
            self.freeze_sub_models()

    # ------------------------------------------------------------------
    def freeze_sub_models(self) -> None:
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_sub_models(self) -> None:
        for m in self.models:
            for p in m.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    def _resolve_target_size(self, sizes: List[torch.Size]) -> torch.Size:
        if self.upsample_to == "first":
            return sizes[0]
        # max / min by H*W
        areas = [int(s[0]) * int(s[1]) for s in sizes]
        idx = areas.index(max(areas)) if self.upsample_to == "max" else areas.index(min(areas))
        return sizes[idx]

    # ------------------------------------------------------------------
    def forward_per_model(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return main logits from every sub-model (no averaging)."""
        outs: List[torch.Tensor] = []
        for m in self.models:
            outs.append(_to_main_logits(m(x)))
        return outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.forward_per_model(x)

        # Align spatial size
        sizes = [o.shape[-2:] for o in outs]
        unique_sizes = {tuple(s) for s in sizes}
        if len(unique_sizes) > 1:
            target = self._resolve_target_size(sizes)
            outs = [
                o
                if o.shape[-2:] == target
                else F.interpolate(o, size=target, mode="bilinear", align_corners=False)
                for o in outs
            ]

        # Channel-dim sanity (warn if mismatch, but allow broadcast if 1)
        ch_set = {o.shape[1] for o in outs}
        if len(ch_set) > 1:
            logger.warning(
                f"EnsembleModel: sub-models have different channel counts {ch_set}. "
                f"Logit averaging may be ill-defined; please verify."
            )

        # Average
        stacked = torch.stack(outs, dim=0)  # (M, B, C, H, W)
        if self.average == "logit":
            w = self.weights.view(-1, 1, 1, 1, 1).to(stacked.device)
            return (stacked * w).sum(dim=0)
        elif self.average == "softmax":
            probs = F.softmax(stacked, dim=2)  # softmax over class dim
            w = self.weights.view(-1, 1, 1, 1, 1).to(probs.device)
            mean_p = (probs * w).sum(dim=0).clamp(min=1e-8)
            return torch.log(mean_p)
        elif self.average == "sigmoid":
            probs = torch.sigmoid(stacked)
            w = self.weights.view(-1, 1, 1, 1, 1).to(probs.device)
            mean_p = (probs * w).sum(dim=0).clamp(min=1e-8, max=1 - 1e-8)
            return torch.log(mean_p / (1 - mean_p))   # invert sigmoid
        else:  # pragma: no cover
            raise ValueError(self.average)


# ----------------------------------------------------------------------
def build_ensemble_from_config(cfg: dict, build_one_fn) -> EnsembleModel:
    """Build an EnsembleModel from a yaml-style config dict.

    Expected schema::

        model:
          type: ensemble
          mode: train | infer       # default: train
          average: logit | softmax | sigmoid    # default: logit
          weights: [0.4, 0.3, 0.3]  # optional
          upsample_to: max          # optional
          members:
            - { encoder: { name: resnet50 }, decoder: { name: unet }, ... }
            - { encoder: { name: convnext_tiny }, decoder: { name: unet }, ... }

    ``build_one_fn(member_cfg)`` is the user-supplied callback that builds a
    single sub-model from one member dict (typically
    :func:`medseg.model_builder.build_model`).
    """
    members = cfg.get("members") or cfg.get("models")
    if not members:
        raise ValueError(
            "Ensemble config requires a non-empty 'members' (or 'models') list."
        )
    sub_models = [build_one_fn(_wrap_member(m)) for m in members]
    return EnsembleModel(
        models=sub_models,
        weights=cfg.get("weights"),
        mode=cfg.get("mode", "train"),
        average=cfg.get("average", "logit"),
        upsample_to=cfg.get("upsample_to", "max"),
    )


def _wrap_member(member: dict) -> dict:
    """Normalise a member dict so build_model can consume it.

    Accepts both ``{ encoder: ..., decoder: ..., ... }`` (model-only) or
    ``{ model: { ... } }`` style.
    """
    if "model" in member:
        return member
    return {"model": member}


# ----------------------------------------------------------------------
def load_ensemble_from_checkpoints(
    member_cfgs: List[dict],
    checkpoints: List[str],
    build_one_fn,
    weights: Optional[Sequence[float]] = None,
    average: str = "logit",
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> EnsembleModel:
    """Build an inference-time ensemble by loading frozen sub-models.

    Args:
        member_cfgs: One dict per sub-model (passed to ``build_one_fn``).
        checkpoints: One ``.pth`` path per sub-model. Same length as
            ``member_cfgs``. May contain ``None`` to skip loading.
        build_one_fn: e.g. ``medseg.model_builder.build_model``.
        weights: Optional per-model weights.
        average: ``'logit' | 'softmax' | 'sigmoid'``.
        map_location: passed to ``torch.load``.
        strict: passed to ``load_state_dict``.
    """
    if len(member_cfgs) != len(checkpoints):
        raise ValueError(
            f"member_cfgs ({len(member_cfgs)}) and checkpoints "
            f"({len(checkpoints)}) must have the same length."
        )

    models: List[nn.Module] = []
    for cfg, ckpt in zip(member_cfgs, checkpoints):
        m = build_one_fn(_wrap_member(cfg))
        if ckpt is not None:
            try:
                state = torch.load(ckpt, map_location=map_location)
                if isinstance(state, dict):
                    state = state.get("model_state_dict", state.get("state_dict", state))
                missing, unexpected = m.load_state_dict(state, strict=strict)
                if missing or unexpected:
                    logger.info(
                        f"[ensemble] {ckpt}: missing={len(missing)} "
                        f"unexpected={len(unexpected)}"
                    )
                else:
                    logger.info(f"[ensemble] loaded {ckpt}")
            except Exception as e:
                logger.error(f"[ensemble] failed loading {ckpt}: {e}")
                if strict:
                    raise
        models.append(m)

    return EnsembleModel(
        models=models,
        weights=weights,
        mode="infer",
        average=average,
    )
