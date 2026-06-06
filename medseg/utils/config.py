"""配置继承与合并工具。/ Config inheritance and merge utilities.

支持 yaml 中的 _base_ 字段实现配置继承，减少 2000+ yaml 的重复。
Supports _base_ field in yaml for config inheritance, reducing redundancy.

用法 / Usage:
    # child.yaml
    _base_: ../base_resnet50.yaml
    model:
      num_classes: 9       # 覆盖基础配置 / Override base config

    # 代码中 / In code:
    from medseg.utils.config import load_config
    cfg = load_config("configs/xxx/child.yaml")
    # cfg 已自动合并 _base_ 的内容 / cfg auto-merged with _base_

规则 / Rules:
    1. child 的值覆盖 base / Child values override base
    2. 支持多级继承（base 也可以有 _base_）/ Multi-level inheritance supported
    3. 列表不合并，直接覆盖 / Lists are replaced, not merged
    4. _base_ 路径相对于当前 yaml 文件所在目录 / _base_ path relative to current yaml dir
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个 dict（override 优先）。
    Deep merge two dicts (override takes precedence).

    列表直接替换，不做元素级合并。
    Lists are replaced entirely, not element-wise merged.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(path: str, _visited: Optional[set] = None) -> dict:
    """加载 yaml 配置，自动处理 _base_ 继承。
    Load yaml config with automatic _base_ inheritance.

    Args:
        path: yaml 文件路径 / Path to yaml file.
        _visited: 已访问的文件集合（防止循环继承）/ Visited files (prevent circular).

    Returns:
        合并后的完整配置 dict / Merged config dict.
    """
    path = os.path.abspath(path)

    if _visited is None:
        _visited = set()
    if path in _visited:
        raise ValueError(f"循环继承 / Circular inheritance detected: {path}")
    _visited.add(path)

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # 处理 _base_ / Handle _base_
    base_path = cfg.pop("_base_", None)
    if base_path is not None:
        # _base_ 路径相对于当前文件 / _base_ path relative to current file
        base_abs = os.path.join(os.path.dirname(path), base_path)
        if not os.path.exists(base_abs):
            raise FileNotFoundError(
                f"_base_ config not found: {base_path} "
                f"(resolved to {base_abs})"
            )
        base_cfg = load_config(base_abs, _visited)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def load_config_with_overrides(path: str, overrides: Optional[dict] = None) -> dict:
    """加载配置 + 命令行覆盖。
    Load config with CLI overrides.

    Args:
        path: yaml 文件路径。
        overrides: 点号分隔的覆盖，如 {"data.fold_idx": 2, "training.epochs": 100}。

    用法 / Usage:
        cfg = load_config_with_overrides("config.yaml", {"data.fold_idx": 2})
    """
    cfg = load_config(path)
    if overrides:
        for key, val in overrides.items():
            parts = key.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val
    return cfg
