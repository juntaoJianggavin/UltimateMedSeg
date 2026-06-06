"""统一训练日志：支持 TensorBoard / WandB / 控制台。
Unified training logger: supports TensorBoard / WandB / console.

用法 / Usage:
    from medseg.utils.logger import TrainLogger

    logger = TrainLogger(cfg, output_dir="./output")
    logger.log_scalar("train/loss", 0.5, step=100)
    logger.log_scalars("val", {"dice": 0.8, "iou": 0.7}, step=100)
    logger.log_image("pred", image_tensor, step=100)
    logger.close()

yaml 配置 / yaml config:
    training:
      logger: tensorboard   # tensorboard / wandb / both / none
      wandb_project: medseg  # WandB 项目名（仅 wandb/both 时需要）
      wandb_entity: null     # WandB 实体名（可选）
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

import numpy as np

_log = logging.getLogger(__name__)


class TrainLogger:
    """统一训练日志接口。/ Unified training logger interface.

    自动根据 yaml 配置选择 backend（TensorBoard / WandB / 两者都用 / 都不用）。
    Auto-selects backend based on yaml config.

    Args:
        cfg: 完整的 yaml 配置 dict / Full yaml config dict.
        output_dir: 输出目录（TensorBoard 日志写这里）/ Output dir for TB logs.
        experiment_name: 实验名（WandB run name）/ Experiment name for WandB.
    """

    def __init__(
        self,
        cfg: dict,
        output_dir: str = "./output",
        experiment_name: str = "medseg",
    ):
        train_cfg = cfg.get("training", {})
        backend = train_cfg.get("logger", "tensorboard")
        self._tb_writer = None
        self._wandb_run = None

        # TensorBoard
        if backend in ("tensorboard", "both"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(output_dir, "tb_logs")
                os.makedirs(tb_dir, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=tb_dir)
                _log.info(f"TensorBoard logger → {tb_dir}")
            except ImportError:
                _log.warning("tensorboard not installed; pip install tensorboard")

        # WandB
        if backend in ("wandb", "both"):
            try:
                import wandb
                project = train_cfg.get("wandb_project", "medseg")
                entity = train_cfg.get("wandb_entity", None)
                self._wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=experiment_name,
                    config=cfg,
                    reinit=True,
                )
                _log.info(f"WandB logger → project={project}")
            except ImportError:
                _log.warning("wandb not installed; pip install wandb")

        if self._tb_writer is None and self._wandb_run is None and backend != "none":
            _log.warning(f"No logger backend available (requested: {backend})")

    # ------------------------------------------------------------------
    # 标量 / Scalars
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int):
        """记录单个标量。/ Log a single scalar."""
        if self._tb_writer:
            self._tb_writer.add_scalar(tag, value, step)
        if self._wandb_run:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, prefix: str, values: Dict[str, float], step: int):
        """记录一组标量。/ Log a group of scalars."""
        for k, v in values.items():
            self.log_scalar(f"{prefix}/{k}", v, step)

    # ------------------------------------------------------------------
    # 图像 / Images
    # ------------------------------------------------------------------

    def log_image(self, tag: str, image, step: int):
        """记录图像（支持 Tensor / ndarray）。/ Log an image."""
        if self._tb_writer:
            import torch
            if isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image = image[None, :, :]  # HW → CHW
                elif image.ndim == 3 and image.shape[2] in (1, 3):
                    image = image.transpose(2, 0, 1)  # HWC → CHW
                image = torch.from_numpy(image).float()
            self._tb_writer.add_image(tag, image, step)
        if self._wandb_run:
            import wandb
            if hasattr(image, 'cpu'):
                image = image.cpu().numpy()
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = image.transpose(1, 2, 0)
            wandb.log({tag: wandb.Image(image)}, step=step)

    # ------------------------------------------------------------------
    # 模型 / Model
    # ------------------------------------------------------------------

    def log_model_graph(self, model, input_shape=(1, 3, 224, 224)):
        """记录模型计算图（仅 TensorBoard）。/ Log model graph (TB only)."""
        if self._tb_writer:
            import torch
            try:
                dummy = torch.randn(*input_shape)
                if next(model.parameters()).is_cuda:
                    dummy = dummy.cuda()
                self._tb_writer.add_graph(model, dummy)
            except Exception as e:
                _log.warning(f"Failed to log model graph: {e}")

    # ------------------------------------------------------------------
    # 超参 / Hyperparameters
    # ------------------------------------------------------------------

    def log_hyperparams(self, hparams: dict, metrics: Optional[dict] = None):
        """记录超参数。/ Log hyperparameters."""
        if self._tb_writer and metrics:
            self._tb_writer.add_hparams(hparams, metrics)
        if self._wandb_run:
            import wandb
            wandb.config.update(hparams, allow_val_change=True)

    # ------------------------------------------------------------------
    # 学习率 / Learning rate
    # ------------------------------------------------------------------

    def log_lr(self, optimizer, step: int):
        """记录当前学习率。/ Log current learning rate."""
        for i, pg in enumerate(optimizer.param_groups):
            self.log_scalar(f"lr/group{i}", pg["lr"], step)

    # ------------------------------------------------------------------
    # 关闭 / Close
    # ------------------------------------------------------------------

    def close(self):
        """关闭所有 logger。/ Close all loggers."""
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            import wandb
            wandb.finish()
