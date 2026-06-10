"""Local cache helpers for timm ImageNet pretrained weights.

Default workflow (international): timm downloads from Hugging Face Hub at runtime,
or users pre-cache weights with ``huggingface-cli download`` / the optional
download script.

Optional backends (explicit opt-in only):

- ``MEDSEG_TIMM_SOURCE=modelscope`` — fetch via ModelScope when ``modelscope``
  is installed (same hub ids as HF, e.g. ``timm/resnet50.a1_in1k``).
- Local cache under ``$MEDSEG_WEIGHT_CACHE`` or ``~/.cache/medseg/weights/timm/``
  — used automatically when weights are already on disk (any download method).

HF mirror (opt-in): ``MEDSEG_HF_MIRROR=1`` or ``HF_ENDPOINT=https://hf-mirror.com``.
See :mod:`medseg.utils.hf_hub`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import timm

logger = logging.getLogger(__name__)

_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "model.bin")


def weight_cache_root() -> Path:
    root = os.environ.get("MEDSEG_WEIGHT_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "medseg" / "weights"


def timm_hf_hub_id(model_name: str) -> Optional[str]:
    """Return timm hub id such as ``timm/resnet50.a1_in1k``."""
    model = timm.create_model(model_name, pretrained=False)
    cfg = getattr(model, "default_cfg", None) or {}
    hub_id = cfg.get("hf_hub_id")
    return hub_id if isinstance(hub_id, str) and hub_id else None


def timm_cache_dir(hub_id: str) -> Path:
    safe = hub_id.replace("/", "--")
    return weight_cache_root() / "timm" / safe


def find_weight_file(directory: Path) -> Optional[Path]:
    """Pick the first usable checkpoint file under ``directory``."""
    directory = Path(directory)
    for name in _WEIGHT_NAMES:
        path = directory / name
        if path.is_file():
            return path
    for name in _WEIGHT_NAMES:
        matches = sorted(directory.rglob(name))
        if matches:
            return matches[0]
    return None


def cached_timm_weight_path(model_name: str) -> Optional[Path]:
    """Return a cached weight file for ``model_name``, if present."""
    hub_id = timm_hf_hub_id(model_name)
    if not hub_id:
        return None
    return find_weight_file(timm_cache_dir(hub_id))


def pretrained_kwargs_from_file(weight_path: Path) -> dict:
    return {
        "pretrained": True,
        "pretrained_cfg_overlay": {"file": str(weight_path)},
    }


def load_timm_cached_pretrained_kwargs(model_name: str) -> Optional[dict]:
    """Build create_model kwargs from an on-disk cache, without downloading."""
    path = cached_timm_weight_path(model_name)
    if path is None:
        return None
    logger.info("Using cached timm weights for %s: %s", model_name, path)
    return pretrained_kwargs_from_file(path)


def ensure_timm_pretrained_via_hf(
    model_name: str,
    *,
    force: bool = False,
) -> Path:
    """Download timm weights from Hugging Face Hub into the medseg cache."""
    if not force:
        cached = cached_timm_weight_path(model_name)
        if cached is not None:
            return cached

    hub_id = timm_hf_hub_id(model_name)
    if not hub_id:
        raise ValueError(f"timm model '{model_name}' has no hf_hub_id in default_cfg")

    from medseg.utils.hf_hub import configure_hf_hub, hf_hub_download_file

    configure_hf_hub()

    cache_dir = timm_cache_dir(hub_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading timm weights for %s from Hugging Face (%s) ...", model_name, hub_id)
    for filename in _WEIGHT_NAMES:
        try:
            path = hf_hub_download_file(
                hub_id,
                filename,
                local_dir=cache_dir,
            )
            if Path(path).is_file():
                return Path(path)
        except Exception:
            continue
    raise FileNotFoundError(f"No weight file found in Hugging Face repo {hub_id}")


def ensure_timm_pretrained_via_modelscope(
    model_name: str,
    *,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """Optional: download timm weights from ModelScope (requires ``modelscope`` pkg)."""
    if not force:
        cached = cached_timm_weight_path(model_name)
        if cached is not None:
            return cached

    hub_id = timm_hf_hub_id(model_name)
    if not hub_id:
        raise ValueError(f"timm model '{model_name}' has no hf_hub_id in default_cfg")

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "ModelScope backend requested but modelscope is not installed. "
            "Install with: pip install modelscope"
        ) from exc

    cache_dir = timm_cache_dir(hub_id)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading timm weights for %s via ModelScope (%s) ...", model_name, hub_id)
    local_dir = snapshot_download(hub_id, local_dir=str(cache_dir))
    weight = find_weight_file(Path(local_dir))
    if weight is None:
        raise FileNotFoundError(
            f"ModelScope download for {hub_id} finished but no weight file found under {local_dir}"
        )
    if verbose:
        logger.info("Saved to %s", weight)
    return weight
