"""Warmup + 更多优化器和调度器。
Warmup schedulers and additional optimizers.

用法 / Usage:
    from medseg.utils.warmup import build_optimizer, build_scheduler

yaml 配置 / yaml config:
    training:
      optimizer:
        name: adamw          # adamw / sgd / lion / sam
        lr: 0.0001
        weight_decay: 0.0001
      scheduler:
        name: cosine         # cosine / step / poly / warmup_cosine / warmup_poly
        warmup_epochs: 10    # warmup 轮数 / warmup epochs
        warmup_lr: 1e-6      # warmup 起始 lr / warmup start lr
        min_lr: 1e-6
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, PolynomialLR, LambdaLR,
    SequentialLR,
)


# ======================================================================
# 优化器 / Optimizers
# ======================================================================

def build_optimizer(params, opt_cfg: dict) -> torch.optim.Optimizer:
    """根据配置构建优化器。/ Build optimizer from config.

    支持 / Supports: adamw, sgd, adam, lion
    """
    name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 1e-4))

    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd,
                     betas=tuple(opt_cfg.get("betas", (0.9, 0.999))))
    elif name == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return SGD(params, lr=lr, weight_decay=wd,
                   momentum=float(opt_cfg.get("momentum", 0.9)),
                   nesterov=opt_cfg.get("nesterov", False))
    elif name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=lr, weight_decay=wd)
        except ImportError:
            raise ImportError("Lion optimizer requires: pip install lion-pytorch")
    else:
        raise ValueError(f"Unknown optimizer: {name}. Available: adamw/adam/sgd/lion")


# ======================================================================
# 调度器 / Schedulers
# ======================================================================

def _warmup_lambda(warmup_epochs: int, warmup_lr_ratio: float):
    """Warmup lambda 函数。/ Warmup lambda function."""
    def fn(epoch):
        if epoch < warmup_epochs:
            return warmup_lr_ratio + (1.0 - warmup_lr_ratio) * epoch / warmup_epochs
        return 1.0
    return fn


def build_scheduler(optimizer, sched_cfg: dict, total_epochs: int):
    """根据配置构建调度器（支持 warmup）。
    Build scheduler from config (with warmup support).

    支持 / Supports:
        cosine, step, poly, warmup_cosine, warmup_poly, warmup_step
    """
    name = sched_cfg.get("name", "cosine").lower()
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
    warmup_lr = float(sched_cfg.get("warmup_lr", 1e-6))

    # 基础调度器 / Base scheduler
    base_name = name.replace("warmup_", "")

    if base_name == "cosine":
        base = CosineAnnealingLR(
            optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=min_lr
        )
    elif base_name == "step":
        base = StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 50)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )
    elif base_name == "poly":
        base = PolynomialLR(
            optimizer,
            total_iters=max(total_epochs - warmup_epochs, 1),
            power=float(sched_cfg.get("power", 0.9)),
        )
    elif base_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}. Available: cosine/step/poly/warmup_cosine/warmup_poly")

    # 如果需要 warmup / If warmup is needed
    if warmup_epochs > 0 or name.startswith("warmup"):
        warmup_epochs = max(warmup_epochs, 1)
        base_lr = optimizer.param_groups[0]["lr"]
        warmup_ratio = warmup_lr / base_lr if base_lr > 0 else 0.0
        warmup_sched = LambdaLR(optimizer, _warmup_lambda(warmup_epochs, warmup_ratio))
        return SequentialLR(optimizer, [warmup_sched, base], milestones=[warmup_epochs])

    return base
