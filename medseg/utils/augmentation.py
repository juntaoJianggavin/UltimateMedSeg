"""Albumentations 集成 + YAML Pipeline 配置：统一的数据增强接口。
Albumentations integration + YAML Pipeline config: unified data augmentation interface.

支持三种增强模式 / Supports three augmentation modes:
  - ``basic``: 项目内置基础增强 (flip, rotate, scale)
  - ``albumentations``: Albumentations 库增强
  - ``pipeline``: YAML 可配置增强流水线（推荐）
  - ``none``: 不使用增强

用法 / Usage:
    from medseg.utils.augmentation import build_transforms

    train_transform = build_transforms(cfg, split="train")
    val_transform = build_transforms(cfg, split="val")

yaml 配置示例 (pipeline 模式) / yaml config example (pipeline mode):
    training:
      augmentation: pipeline
      aug_pipeline:
        - name: horizontal_flip
          params: {p: 0.5}
        - name: vertical_flip
          params: {p: 0.3}
        - name: random_rotate90
          params: {p: 0.3}
        - name: copy_paste
          params: {p: 0.3}
        - name: mosaic
          params: {p: 0.2}
        - name: photometric_distortion
          params: {p: 0.3}
        - name: grid_mask
          params: {p: 0.2}
        - name: random_erasing
          params: {p: 0.3}

yaml 配置示例 (albumentations 模式) / yaml config example (albumentations mode):
    training:
      augmentation: albumentations
      aug_params:
        p_flip: 0.5
        p_rotate: 0.3
        p_color: 0.3
        p_elastic: 0.2
        p_gridmask: 0.1
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

_log = logging.getLogger(__name__)


def _build_albu_train(img_size: int, params: dict):
    """构建 Albumentations 训练增强 pipeline。
    Build Albumentations training augmentation pipeline."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        raise ImportError(
            "albumentations is required. Install: "
            "pip install albumentations"
        )

    p_flip = params.get("p_flip", 0.5)
    p_rotate = params.get("p_rotate", 0.3)
    p_color = params.get("p_color", 0.3)
    p_elastic = params.get("p_elastic", 0.2)
    p_blur = params.get("p_blur", 0.1)
    p_noise = params.get("p_noise", 0.1)

    transform = A.Compose([
        A.Resize(img_size, img_size),
        # 空间变换 / Spatial transforms
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=p_flip * 0.5),
        A.RandomRotate90(p=p_rotate),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
            border_mode=0, p=p_rotate,
        ),
        A.ElasticTransform(alpha=50, sigma=10, p=p_elastic),
        # 颜色变换 / Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
            A.CLAHE(clip_limit=4.0),
        ], p=p_color),
        # 模糊/噪声 / Blur/noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
        ], p=p_blur),
        A.GaussNoise(var_limit=(5, 30), p=p_noise),
        # 标准化 / Normalize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return transform


def _build_albu_val(img_size: int):
    """构建 Albumentations 验证增强 pipeline（只 resize + normalize）。
    Build Albumentations validation pipeline (resize + normalize only)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class AlbuWrapper:
    """将 Albumentations transform 包装成项目 dataset 期望的 dict 接口。
    Wrap Albumentations transform to match project dataset's dict interface.

    项目 dataset 传 {"image": ndarray, "label": ndarray}，
    Albumentations 期望 image= + mask= 参数。
    Project datasets pass {"image": ndarray, "label": ndarray},
    Albumentations expects image= + mask= arguments.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        label = sample.get("label")

        if label is not None:
            result = self.transform(image=image, mask=label)
            return {"image": result["image"], "label": result["mask"]}
        else:
            result = self.transform(image=image)
            return {"image": result["image"]}


def build_transforms(cfg: dict, split: str = "train", dataset=None):
    """根据配置构建数据增强。/ Build data augmentation from config.

    Supports three modes:
    - ``basic``: Default project transforms (flip, rotate, scale, noise).
    - ``albumentations``: Albumentations library transforms.
    - ``pipeline``: YAML-configurable pipeline of registered augmentations.
    - ``none``: No augmentation (only resize + normalize).

    Args:
        cfg: 完整 yaml 配置 / Full yaml config.
        split: "train" / "val" / "test"
        dataset: Dataset reference (for sample-level augmentations like CopyPaste/Mosaic).

    Returns:
        transform 对象（兼容项目 dataset 的 dict 接口）。
        Transform object compatible with project dataset's dict interface.
    """
    train_cfg = cfg.get("training", {})
    aug_type = train_cfg.get("augmentation", "basic")
    img_size = cfg.get("data", {}).get("img_size", cfg.get("model", {}).get("img_size", 224))
    if img_size == "native":
        img_size = 224

    # Validation/test: no augmentation
    if aug_type == "none" or split in ("val", "test"):
        if aug_type == "albumentations":
            return AlbuWrapper(_build_albu_val(img_size))
        from medseg.datasets import get_val_transforms
        return get_val_transforms(img_size)

    # Albumentations mode
    if aug_type == "albumentations":
        params = train_cfg.get("aug_params", {})
        return AlbuWrapper(_build_albu_train(img_size, params))

    # Pipeline mode: YAML-configurable augmentations
    if aug_type == "pipeline":
        aug_pipeline = train_cfg.get("aug_pipeline", [])
        if aug_pipeline:
            from medseg.datasets.advanced_aug import build_augmentation_pipeline
            return build_augmentation_pipeline(aug_pipeline, img_size=img_size, dataset=dataset)
        _log.warning("augmentation=pipeline but aug_pipeline is empty, falling back to basic")

    # Default: basic transforms
    from medseg.datasets import get_train_transforms
    return get_train_transforms(img_size)
