"""AMP（混合精度）+ DDP（分布式数据并行）工具。
AMP (Automatic Mixed Precision) + DDP (Distributed Data Parallel) utilities.

用法 / Usage:
    # 单 GPU + AMP
    python train.py --config xxx.yaml --amp

    # 多 GPU DDP
    torchrun --nproc_per_node=4 train.py --config xxx.yaml --amp

    # 代码中使用 / In code:
    from medseg.utils.amp_ddp import setup_ddp, cleanup_ddp, AMPScaler, is_main_process

yaml 配置 / yaml config:
    training:
      amp: true              # 启用混合精度 / Enable mixed precision
      sync_bn: true          # DDP 时同步 BN / Sync BN in DDP
      find_unused: false     # DDP find_unused_parameters
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================
# DDP 设置 / DDP Setup
# ======================================================================

def setup_ddp(backend: str = "nccl") -> int:
    """初始化 DDP 进程组，返回 local_rank。
    Initialize DDP process group, return local_rank.

    如果不在 torchrun 环境下调用，返回 0 且不初始化 DDP。
    If not launched via torchrun, returns 0 without initializing DDP.
    """
    if "RANK" not in os.environ:
        return 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        logger.info(f"DDP initialized: rank={rank}, local_rank={local_rank}, world={world_size}")

    return local_rank


def cleanup_ddp():
    """销毁 DDP 进程组。/ Destroy DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """当前进程是否为主进程（rank 0）。/ Whether current process is main (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """获取总进程数。/ Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def wrap_parallel(model: nn.Module, local_rank: int = 0,
                  mode: str = "auto",
                  sync_bn: bool = True, find_unused: bool = False) -> nn.Module:
    """将模型包装为并行模式。/ Wrap model with parallel mode.

    Args:
        model: 已在 device 上的模型 / Model already on device.
        local_rank: 本地 GPU 编号 / Local GPU index.
        mode: 并行模式 / Parallel mode:
            "auto" — 自动选择：torchrun 环境用 DDP，多卡用 DP，单卡不包装
                     Auto: DDP if torchrun, DP if multi-GPU, none if single
            "ddp"  — 强制 DistributedDataParallel（需要 torchrun 启动）
            "dp"   — 强制 DataParallel（单进程多卡）
            "none" — 不包装
        sync_bn: DDP 时转换 BN 为 SyncBN / Convert BN to SyncBN in DDP.
        find_unused: DDP find_unused_parameters.

    yaml 配置 / yaml config:
        training:
          parallel: auto     # auto / ddp / dp / none
          sync_bn: true
          find_unused: false
    """
    if mode == "none":
        return model

    # DDP 模式 / DDP mode
    if mode == "ddp" or (mode == "auto" and dist.is_initialized()):
        if not dist.is_initialized():
            raise RuntimeError(
                "DDP requires torchrun launch. Use: "
                "torchrun --nproc_per_node=N train.py --config xxx.yaml"
            )
        if sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info("BatchNorm → SyncBatchNorm")
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)
        logger.info(f"Model wrapped with DDP (device={local_rank})")
        return model

    # DP 模式 / DataParallel mode
    n_gpu = torch.cuda.device_count()
    if mode == "dp" or (mode == "auto" and n_gpu > 1):
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel ({n_gpu} GPUs)")
        return model

    # 单卡 / Single GPU — no wrapping
    return model


def ddp_sampler(dataset, shuffle: bool = True):
    """为 DDP 创建 DistributedSampler。/ Create DistributedSampler for DDP.

    非 DDP 环境返回 None（DataLoader 用默认 shuffle）。
    Returns None in non-DDP (DataLoader uses default shuffle).
    """
    if not dist.is_initialized():
        return None
    from torch.utils.data.distributed import DistributedSampler
    return DistributedSampler(dataset, shuffle=shuffle)


# ======================================================================
# AMP 工具 / AMP Utilities
# ======================================================================

class AMPScaler:
    """AMP GradScaler 的简单封装，支持 enable/disable 切换。
    Simple wrapper around GradScaler with enable/disable toggle.

    用法 / Usage:
        scaler = AMPScaler(enabled=True)

        with scaler.autocast():
            loss = model(x)

        scaler.scale_and_step(loss, optimizer)
        scaler.update()
    """

    def __init__(self, enabled: bool = True, device_type: str = "cuda"):
        self.enabled = enabled and torch.cuda.is_available()
        self.device_type = device_type
        self._scaler = torch.amp.GradScaler(enabled=self.enabled)

    def autocast(self):
        """返回 autocast 上下文管理器。/ Return autocast context manager."""
        return torch.amp.autocast(device_type=self.device_type, enabled=self.enabled)

    def scale_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                       max_norm: float = 1.0, model: Optional[nn.Module] = None):
        """缩放 loss → backward → unscale → clip grad → step。
        Scale loss → backward → unscale → clip grad → step."""
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(optimizer)
        if model is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        self._scaler.step(optimizer)

    def update(self):
        """更新 scaler 的缩放因子。/ Update scaler's scale factor."""
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state):
        self._scaler.load_state_dict(state)
