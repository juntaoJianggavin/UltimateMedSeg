"""MediSee 权重自动下载器。

总共需要 4 套权重，全部走 HuggingFace Hub:

1. ``microsoft/llava-med-v1.5-mistral-7b``  — MLLM 主体 (~14GB)
2. ``openai/clip-vit-large-patch14-336``    — vision tower (~1.7GB)
3. ``flaviagiammarino/medsam-vit-base``     — MedSAM ViT-B (~366MB)
   * 上游 README 推荐从 Google Drive 下载 ``medsam_vit_b.pth``，但 GDrive 无 API；
     这里改用 HF 上的等价权重 + 转换为 SAM 原生 state_dict 格式。
4. ``Carryyy/MediSee``                      — MediSee 自身 fine-tuned 权重

下载策略:
* 优先用 ``HF_HUB_CACHE`` / ``HF_HOME`` 指定的全局缓存；若未设置，落到
  ``~/.cache/medseg/medisee/<repo_basename>/``。
* ``snapshot_download`` 自带 resume/etag/lock，重复调用幂等；权重已在
  本地时直接返回路径，不再触发网络请求。
* MedSAM 需要 ``.pth`` 文件路径（``build_sam_vit_b(checkpoint=...)`` 要求），
  下载后从 HF safetensors / pytorch_model.bin 中重建 ``state_dict`` 写入
  本地 .pth。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Optional

# 默认 HF repo 标识
LLAVA_MED_REPO = "microsoft/llava-med-v1.5-mistral-7b"
CLIP_REPO = "openai/clip-vit-large-patch14-336"
# MedSAM ViT-B 权重源 (按优先级):
#   1) ``SansuiHan/medical_models``  — 直接托管原生 SAM 格式 ``medsam_vit_b.pth``
#      (375MB, 与 bowang-lab GDrive 一致, 可被 ``build_sam_vit_b`` 直接 ``torch.load`` 使用)
# Per project policy this loader does NOT fall back to a different MedSAM
# checkpoint when the canonical native one is unreachable — it raises.
MEDSAM_NATIVE_REPO = "SansuiHan/medical_models"
MEDSAM_NATIVE_FILENAME = "medsam_vit_b.pth"
MEDISEE_REPO = "Carryyy/MediSee"


def _default_cache_root() -> str:
    """返回 medseg 私有缓存目录, 用作 fallback."""
    home = os.path.expanduser("~")
    root = os.environ.get(
        "MEDSEG_CACHE_DIR",
        os.path.join(home, ".cache", "medseg", "medisee"),
    )
    os.makedirs(root, exist_ok=True)
    return root


def _try_hf_snapshot(repo_id: str, cache_dir: Optional[str] = None) -> str:
    """调用 huggingface_hub.snapshot_download, 返回本地目录."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for MediSee weight auto-download. "
            "Install it via `pip install huggingface_hub`."
        ) from e

    kwargs = dict(repo_id=repo_id)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    # 允许通过环境变量传 token（私有 repo / 限速场景）
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        kwargs["token"] = token
    local_dir = snapshot_download(**kwargs)
    return local_dir


def _try_hf_file(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> str:
    """调用 huggingface_hub.hf_hub_download 下载单个文件, 返回本地绝对路径."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for MediSee weight auto-download. "
            "Install it via `pip install huggingface_hub`."
        ) from e

    kwargs = dict(repo_id=repo_id, filename=filename)
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        kwargs["token"] = token
    return hf_hub_download(**kwargs)


@dataclass
class MediSeeWeightPaths:
    """聚合 MediSee 推理需要的全部本地路径."""

    llava_med_dir: str
    clip_dir: str
    medsam_ckpt: str
    medisee_dir: str


def _ensure_medsam_pth(cache_root: str, cache_dir: Optional[str] = None) -> str:
    """获取原生 SAM 格式 ``medsam_vit_b.pth``.

    上游 ``model/segment_anything/build_sam.py:build_sam_vit_b(checkpoint=path)``
    使用 ``torch.load(path)`` + ``sam.load_state_dict(state_dict, strict=False)``
    加载 SAM 原生 state_dict, 因此优先获取与 bowang-lab GDrive 等价的原生 .pth.

    Strict policy: only the canonical native ``medsam_vit_b.pth`` is
    accepted. If it cannot be fetched, raise — we do NOT substitute the HF
    ``flaviagiammarino/medsam-vit-base`` checkpoint with a best-effort
    reverse key mapping, since the resulting weights are not bit-equivalent
    and silently degrade downstream accuracy.
    """
    target_pth = os.path.join(cache_root, "medsam_vit_b.pth")
    if os.path.exists(target_pth) and os.path.getsize(target_pth) > 0:
        return target_pth

    try:
        native_path = _try_hf_file(
            MEDSAM_NATIVE_REPO, MEDSAM_NATIVE_FILENAME, cache_dir=cache_dir
        )
        shutil.copyfile(native_path, target_pth)
        return target_pth
    except Exception as primary_err:  # noqa: BLE001
        raise RuntimeError(
            "Failed to auto-fetch MedSAM native weights.\n"
            f"  source ({MEDSAM_NATIVE_REPO}/{MEDSAM_NATIVE_FILENAME}): "
            f"{type(primary_err).__name__}: {primary_err}\n"
            "Manually download medsam_vit_b.pth from "
            "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN "
            "or https://zenodo.org/records/10689643 and place it at "
            f"{target_pth}."
        ) from primary_err


def ensure_all_weights(
    cache_dir: Optional[str] = None,
    download_medisee: bool = True,
    download_llava: bool = True,
    download_clip: bool = True,
    download_medsam: bool = True,
) -> MediSeeWeightPaths:
    """确保 4 套权重已落到本地, 缺哪个下哪个.

    Parameters
    ----------
    cache_dir
        HF cache 根目录. None 时使用 HF 默认 (``HF_HOME`` / ``~/.cache/huggingface``).
    download_*
        细粒度开关, 用于调试/单独跳过某项下载. 默认全开.

    Returns
    -------
    MediSeeWeightPaths
        4 个本地路径: llava_med_dir / clip_dir / medsam_ckpt / medisee_dir.
    """
    medseg_cache = _default_cache_root()

    llava_med_dir = (
        _try_hf_snapshot(LLAVA_MED_REPO, cache_dir) if download_llava else ""
    )
    clip_dir = _try_hf_snapshot(CLIP_REPO, cache_dir) if download_clip else ""
    medisee_dir = (
        _try_hf_snapshot(MEDISEE_REPO, cache_dir) if download_medisee else ""
    )
    if download_medsam:
        medsam_ckpt = _ensure_medsam_pth(medseg_cache, cache_dir=cache_dir)
    else:
        medsam_ckpt = ""

    return MediSeeWeightPaths(
        llava_med_dir=llava_med_dir,
        clip_dir=clip_dir,
        medsam_ckpt=medsam_ckpt,
        medisee_dir=medisee_dir,
    )


__all__ = [
    "LLAVA_MED_REPO",
    "CLIP_REPO",
    "MEDSAM_NATIVE_REPO",
    "MEDSAM_NATIVE_FILENAME",
    "MEDISEE_REPO",
    "MediSeeWeightPaths",
    "ensure_all_weights",
]
