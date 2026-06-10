"""Central Hugging Face Hub configuration and download helpers.

All HF downloads in medseg should go through this module so mirror / endpoint
settings apply consistently.

Environment variables (endpoint resolution order):

1. ``HF_ENDPOINT`` — standard ``huggingface_hub`` variable (highest priority)
2. ``MEDSEG_HF_ENDPOINT`` — project alias, used when ``HF_ENDPOINT`` is unset
3. ``MEDSEG_HF_MIRROR=1`` — opt-in shorthand for ``https://hf-mirror.com``

The project does **not** force a mirror by default (international-friendly).
For China-accessible mirrors, set one of the above before downloading or training::

    export MEDSEG_HF_MIRROR=1
    # or
    export HF_ENDPOINT=https://hf-mirror.com

Optional auth for gated repos: ``HF_TOKEN`` or ``HUGGINGFACE_TOKEN``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

logger = logging.getLogger(__name__)

HF_MIRROR_DEFAULT = "https://hf-mirror.com"
_CONFIGURED = False


def resolve_hf_endpoint() -> Optional[str]:
    """Return the HF API endpoint URL, or ``None`` for official huggingface.co."""
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if endpoint:
        return endpoint.rstrip("/")
    endpoint = os.environ.get("MEDSEG_HF_ENDPOINT", "").strip()
    if endpoint:
        return endpoint.rstrip("/")
    mirror_flag = os.environ.get("MEDSEG_HF_MIRROR", "").strip().lower()
    if mirror_flag in ("1", "true", "yes", "on"):
        return HF_MIRROR_DEFAULT
    return None


def configure_hf_hub(*, log: bool = True) -> Optional[str]:
    """Apply resolved endpoint to ``HF_ENDPOINT`` once per process."""
    global _CONFIGURED
    endpoint = resolve_hf_endpoint()
    if endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = endpoint
    if log and endpoint and not _CONFIGURED:
        logger.info("Hugging Face Hub endpoint: %s", endpoint)
    _CONFIGURED = True
    return endpoint or os.environ.get("HF_ENDPOINT")


def hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return token.strip() if token else None


def hf_hub_download_file(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    local_dir: Optional[Union[str, Path]] = None,
    local_dir_use_symlinks: Union[bool, str] = False,
    token: Optional[str] = None,
):
    """Download a single file from Hugging Face Hub (respects mirror settings)."""
    configure_hf_hub(log=False)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": repo_type,
        "token": token if token is not None else hf_token(),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
        kwargs["local_dir_use_symlinks"] = local_dir_use_symlinks

    return hf_hub_download(**kwargs)


def hf_snapshot_download(
    repo_id: str,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    local_dir: Optional[Union[str, Path]] = None,
    allow_patterns: Optional[Union[str, Sequence[str]]] = None,
    ignore_patterns: Optional[Union[str, Sequence[str]]] = None,
    token: Optional[str] = None,
):
    """Download a repository snapshot from Hugging Face Hub."""
    configure_hf_hub(log=False)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    kwargs = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "token": token if token is not None else hf_token(),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns is not None:
        kwargs["ignore_patterns"] = ignore_patterns

    return snapshot_download(**kwargs)


def download_repo_files(
    repo_id: str,
    filenames: Iterable[str],
    *,
    repo_type: str = "model",
    local_dir: Union[str, Path],
    revision: Optional[str] = None,
) -> list[Path]:
    """Download specific files from a repo into ``local_dir``."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename in filenames:
        path = hf_hub_download_file(
            repo_id,
            filename,
            repo_type=repo_type,
            revision=revision,
            local_dir=local_dir,
        )
        paths.append(Path(path))
    return paths


# Known HF dataset repos useful for medseg (not all are TransUNet npz format).
HF_DATASET_CATALOG = {
    "medseg7d": {
        "repo_id": "MaybeRichard/MedSeg-7D",
        "repo_type": "dataset",
        "description": "7 public 2D med-seg datasets incl. ACDC (PNG, not TransUNet npz)",
        "includes": ["ACDC", "BraTS2020", "CVC-ClinicDB", "..."],
    },
    "acdc_nifti": {
        "repo_id": "viennh2012/cardiac_cine_acdc",
        "repo_type": "dataset",
        "description": "ACDC cardiac cine-MRI (processed NIfTI)",
        "includes": ["ACDC"],
    },
    "m3d_seg": {
        "repo_id": "GoodBaiBai88/M3D-Seg",
        "repo_type": "dataset",
        "description": "25 3D CT seg datasets incl. BTCV (NIfTI zips, not TransUNet npz)",
        "includes": ["BTCV", "..."],
    },
}
