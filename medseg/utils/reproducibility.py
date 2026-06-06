"""可复现性工具 / Reproducibility utilities.

提供完整的随机种子设置，确保实验可复现。
Provides complete random seed setup for reproducible experiments.

用法 / Usage:
    from medseg.utils.reproducibility import set_seed, worker_init_fn

    # 在训练开始前调用 / Call before training starts
    set_seed(cfg.get('training', {}).get('random_state', 42))

    # DataLoader 中使用 / Use in DataLoader
    DataLoader(..., worker_init_fn=worker_init_fn)

yaml 配置 / yaml config:
    training:
      random_state: 42        # 全局随机种子 / Global random seed
      deterministic: true     # 启用确定性模式 / Enable deterministic mode
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """设置全局随机种子，确保可复现。
    Set global random seed for full reproducibility.

    Args:
        seed: 随机种子值 / Random seed value.
        deterministic: 是否启用 CUDA 确定性模式 / Whether to enable CUDA deterministic mode.
                       启用后性能可能降低 5-10%，但保证结果完全一致。
                       May reduce performance by 5-10% but guarantees identical results.
    """
    # Python 内置 random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (所有 GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 环境变量（用于某些 hash-based 操作）
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        # cuDNN 确定性模式 / cuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 2.0+ 确定性算法 / PyTorch 2.0+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    else:
        # 非确定性模式：开启 cuDNN benchmark 以获得更好性能
        # Non-deterministic: enable cuDNN benchmark for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 初始化函数，确保多 worker 下可复现。
    DataLoader worker init function for reproducibility with num_workers > 0.

    用法 / Usage:
        DataLoader(..., worker_init_fn=worker_init_fn)

    原理：每个 worker 用 (base_seed + worker_id) 作为种子，
    避免所有 worker 产生相同的随机增强序列。
    Each worker uses (base_seed + worker_id) as its seed to avoid
    all workers producing identical augmentation sequences.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """创建一个带种子的 PyTorch Generator，用于 DataLoader 的 shuffle。
    Create a seeded PyTorch Generator for DataLoader shuffling.

    用法 / Usage:
        g = get_generator(42)
        DataLoader(..., generator=g)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
