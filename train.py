"""Training script for modular medical image segmentation framework.

(Self-contained: no single canonical GitHub source.)
"""

import os
import sys
import argparse
import logging
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, PolynomialLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medseg.model_builder import build_model
from medseg.registry import LOSS_REGISTRY
from medseg.datasets import SynapseDataset, GenericDataset, get_train_transforms, get_val_transforms
from medseg.utils.augmentation import build_transforms
from medseg.utils.metrics import compute_metrics

# Import all modules to trigger registration
import medseg.models.encoders
import medseg.models.decoders
import medseg.models.skip_connections
import medseg.models.bottlenecks
import medseg.losses
import medseg.datasets.advanced_aug  # noqa: trigger AUGMENTATION_REGISTRY registration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_dataset(data_cfg, split='train', cfg=None):
    """Build dataset from config.

    Uses build_transforms() for YAML-configurable augmentation pipeline when
    the full config (cfg) is provided. Falls back to get_train_transforms()
    for backwards compatibility.
    """
    dataset_type = data_cfg.get('type', 'synapse')
    img_size = data_cfg.get('img_size', 224)

    # Use YAML-configurable augmentation pipeline if full config available
    if cfg is not None and cfg.get('training', {}).get('augmentation') == 'pipeline':
        transform = build_transforms(cfg, split=split)
    else:
        transform = get_train_transforms(img_size) if split == 'train' else get_val_transforms(img_size)

    if dataset_type == 'synapse':
        return SynapseDataset(
            root_dir=data_cfg[f'{split}_dir'],
            split=split,
            list_file=data_cfg.get(f'{split}_list', None),
            transform=transform,
            img_size=img_size,
        )
    elif dataset_type in ('image_mask', 'binary', 'generic'):
        return GenericDataset(
            root_dir=data_cfg.get(f'{split}_dir', data_cfg.get('root_dir')),
            split=split,
            transform=transform,
            img_size=img_size,
            img_suffix=data_cfg.get('img_suffix', '.png'),
            mask_suffix=data_cfg.get('mask_suffix', '.png'),
            train_ratio=data_cfg.get('train_ratio', 0.7),
            val_ratio=data_cfg.get('val_ratio', 0.15),
            random_state=data_cfg.get('random_state', 42),
            n_splits=data_cfg.get('n_splits', 0),
            fold_idx=data_cfg.get('fold_idx', 0),
            kfold_mode=split if split in ('train', 'val') else 'val',
            file_list=data_cfg.get(f'{split}_list', None),
        )
    elif dataset_type in ('synapse', 'acdc'):
        # ACDC uses the same dataset class as Synapse
        return SynapseDataset(
            root_dir=data_cfg[f'{split}_dir'],
            split=split,
            list_file=data_cfg.get(f'{split}_list', None),
            transform=transform,
            img_size=img_size,
        )
    elif dataset_type in ('qata_covid19', 'mosmed_plus'):
        # QaTa-COV19 / MosMedData+ (LViT enriched): 带 per-image 文本标注
        # QaTa-COV19 / MosMedData+ (LViT enriched): with per-image text annotations
        from medseg.datasets import QaTaCOV19Dataset, MosMedPlusDataset
        ds_cls = QaTaCOV19Dataset if dataset_type == 'qata_covid19' else MosMedPlusDataset
        return ds_cls(
            data_root=data_cfg.get('data_root', data_cfg.get('root_dir')),
            split=split,
            img_size=img_size,
            tokenizer_name=data_cfg.get('tokenizer_name'),
            text_max_length=data_cfg.get('text_max_length', 24),
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def build_optimizer(params, opt_cfg):
    """Build optimizer from config."""
    name = opt_cfg.get('name', 'adamw')
    lr = opt_cfg.get('lr', 1e-4)
    weight_decay = opt_cfg.get('weight_decay', 1e-4)
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=opt_cfg.get('momentum', 0.9))
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, sched_cfg, total_epochs):
    """Build LR scheduler from config."""
    name = sched_cfg.get('name', 'cosine')
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=sched_cfg.get('min_lr', 1e-6))
    elif name == 'step':
        return StepLR(optimizer, step_size=sched_cfg.get('step_size', 50), gamma=sched_cfg.get('gamma', 0.1))
    elif name == 'poly':
        return PolynomialLR(optimizer, total_iters=total_epochs, power=sched_cfg.get('power', 0.9))
    else:
        return None


def build_loss(loss_cfg):
    """Build loss function from config."""
    name = loss_cfg.get('name', 'compound')
    params = loss_cfg.get('params', {})
    loss_cls = LOSS_REGISTRY.get(name)
    return loss_cls(**params)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch.

    Works transparently for both single models and ``EnsembleModel`` since
    the latter exposes a flat ``forward`` that returns averaged logits;
    the criterion is applied once on the averaged output.
    """
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 50 == 0:
            logger.info(f"Epoch [{epoch}] Step [{i+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, num_classes, device):
    """Validate the model."""
    model.eval()
    all_dice = []
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].numpy()

        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        preds = outputs.argmax(dim=1).cpu().numpy()

        for pred, target in zip(preds, labels):
            metrics = compute_metrics(pred, target, num_classes)
            dice_vals = list(metrics['dice'].values())
            if dice_vals:
                all_dice.append(np.mean(dice_vals))

    return np.mean(all_dice) if all_dice else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    # P0: AMP + DDP
    parser.add_argument('--amp', action='store_true', help='启用混合精度 / Enable AMP')
    parser.add_argument(
        '--override', nargs='+', metavar='KEY=VALUE',
        help='覆盖配置 / Override config values (dot-separated keys). '
             'Example: --override model.architecture=unet training.epochs=100'
    )
    args = parser.parse_args()

    # P1: 配置继承 / Config inheritance (_base_ support)
    from medseg.utils.config import load_config
    cfg = load_config(args.config)

    # 应用 CLI 覆盖 / Apply CLI overrides
    if args.override:
        for item in args.override:
            if '=' not in item:
                logger.warning(f"跳过无效的覆盖 / Skipping invalid override (no '='): {item}")
                continue
            key, value = item.split('=', 1)
            # 尝试类型转换 / Try type conversion
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # 保持为字符串 / Keep as string
            # 按点号路径设置 / Set by dot-separated path
            parts = key.split('.')
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value
            logger.info(f"覆盖 / Override: {key} = {value}")

    # 可复现性设置：优先读 yaml 中的 training.random_state，其次用 CLI --seed
    # Reproducibility: prefer yaml training.random_state, fallback to CLI --seed
    from medseg.utils.reproducibility import set_seed, worker_init_fn, get_generator
    train_cfg = cfg.get('training', {})
    seed = train_cfg.get('random_state', args.seed)
    deterministic = train_cfg.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)

    os.makedirs(args.output_dir, exist_ok=True)

    # P0: DDP 初始化 / DDP setup
    from medseg.utils.amp_ddp import setup_ddp, wrap_parallel, ddp_sampler, AMPScaler, is_main_process
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    use_amp = args.amp or train_cfg.get('amp', False)

    # P0: AMP scaler
    scaler = AMPScaler(enabled=use_amp)

    # P0: Logger（只在主进程创建）/ Logger (main process only)
    from medseg.utils.logger import TrainLogger
    tb_logger = TrainLogger(cfg, args.output_dir) if is_main_process() else None

    # Build model
    model = build_model(cfg)
    model = model.to(device)

    # 并行包装：auto 自动选 DDP/DP/单卡 / Parallel wrap: auto selects DDP/DP/single
    parallel_mode = train_cfg.get('parallel', 'auto')
    model = wrap_parallel(model, local_rank, mode=parallel_mode,
                          sync_bn=train_cfg.get('sync_bn', True),
                          find_unused=train_cfg.get('find_unused', False))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    if hasattr(model, 'models') and isinstance(model.models, torch.nn.ModuleList):
        # EnsembleModel: log per-member size as well
        per = [
            sum(p.numel() for p in m.parameters()) / 1e6
            for m in model.models
        ]
        logger.info(
            f"EnsembleModel built. Members={len(model.models)} "
            f"per-member-params={['%.2fM' % p for p in per]} "
            f"total={n_params:.2f}M (trainable={n_trainable:.2f}M)"
        )
    else:
        logger.info(
            f"Model built. Parameters: {n_params:.2f}M "
            f"(trainable: {n_trainable:.2f}M)"
        )

    # Build datasets
    data_cfg = cfg.get('data', {})
    train_dataset = build_dataset(data_cfg, 'train', cfg=cfg)
    # DDP sampler（DDP 时不用 shuffle，由 sampler 控制）
    train_sampler = ddp_sampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get('batch_size', 8),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=get_generator(seed),
    )
    logger.info(f"Train dataset: {len(train_dataset)} samples")

    val_dataset = None
    val_loader = None
    if 'val_dir' in data_cfg or 'test_dir' in data_cfg:
        val_key = 'val' if 'val_dir' in data_cfg else 'test'
        val_dataset = build_dataset(data_cfg, val_key, cfg=cfg)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get('training', {}).get('batch_size', 8),
            shuffle=False,
            num_workers=cfg.get('training', {}).get('num_workers', 4),
            pin_memory=True,
        )
        logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Build loss + P2: warmup 优化器/调度器 / Build loss + warmup optimizer/scheduler
    criterion = build_loss(train_cfg.get('loss', {'name': 'compound'}))
    from medseg.utils.warmup import build_optimizer as build_opt_v2, build_scheduler as build_sched_v2
    optimizer = build_opt_v2(model.parameters(), train_cfg.get('optimizer', {}))
    total_epochs = train_cfg.get('epochs', 150)
    scheduler = build_sched_v2(optimizer, train_cfg.get('scheduler', {}), total_epochs)

    # Resume
    start_epoch = 0
    best_dice = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_dice = ckpt.get('best_dice', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

    # Training loop (with AMP + DDP + Logger)
    global_step = 0
    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        # DDP: 设置 epoch 让 sampler shuffle 不同 / Set epoch for sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 训练一轮（AMP 包裹）/ Train one epoch (AMP wrapped)
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with scaler.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale_and_step(loss, optimizer, max_norm=1.0, model=model)
            scaler.update()

            total_loss += loss.item()
            global_step += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Epoch [{epoch}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss = total_loss / max(len(train_loader), 1)

        if scheduler:
            scheduler.step()

        # P0: Logger 记录 / Log metrics
        if tb_logger:
            tb_logger.log_scalar("train/loss", train_loss, epoch)
            tb_logger.log_lr(optimizer, epoch)

        log_msg = f"Epoch [{epoch}/{total_epochs}] Loss: {train_loss:.4f} LR: {optimizer.param_groups[0]['lr']:.6f} Time: {time.time()-t0:.1f}s"

        # Validation
        if val_loader is not None and (epoch + 1) % train_cfg.get('val_interval', 10) == 0:
            eval_model = model.module if hasattr(model, 'module') else model
            dice = validate(eval_model, val_loader, cfg['model']['num_classes'], device)
            log_msg += f" Val_Dice: {dice:.4f}"
            if dice > best_dice:
                best_dice = dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                }, os.path.join(args.output_dir, 'best_model.pth'))
                logger.info(f"New best model saved! Dice: {best_dice:.4f}")

        logger.info(log_msg)

        # Save latest checkpoint
        if (epoch + 1) % train_cfg.get('save_interval', 50) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))

    logger.info(f"Training complete. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
