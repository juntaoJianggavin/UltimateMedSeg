"""Semi-supervised training script for medical image segmentation.

Supports multiple semi-supervised methods:
  - Mean Teacher (Tarvainen & Valpola, NeurIPS 2017)
  - Cross Pseudo Supervision / CPS (Chen et al., CVPR 2021)
  - Cross Consistency Training / CCT (Ouali et al., BMVC 2020)
  - UniMatch (Yang et al., CVPR 2023)

Usage:
    python semi_train.py --config configs/training_paradigms/semi_supervision/mean_teacher.yaml --output_dir output/semi_mt

Reference implementation: https://github.com/HiLab-git/SSL4MIS
"""

import os
import sys
import argparse
import logging
import time
import itertools
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
from medseg.datasets import (
    SynapseDataset, GenericDataset, UnlabeledDataset,
    get_train_transforms, get_val_transforms,
)
from medseg.training.semi import build_semi_method
from medseg.utils.metrics import compute_metrics

# Import all modules to trigger registration
import medseg.models.encoders
import medseg.models.decoders
import medseg.models.skip_connections
import medseg.models.bottlenecks
import medseg.losses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _resolve_split_dir(data_cfg, split):
    """Resolve the directory for a split, allowing common aliases.

    For split='train': accepts train_dir / labeled_dir.
    For split='val'/'test': accepts <split>_dir.
    """
    if split == 'train':
        for key in ('train_dir', 'labeled_dir'):
            if key in data_cfg:
                return data_cfg[key], data_cfg.get('train_list', data_cfg.get('labeled_list'))
        raise KeyError("data.train_dir (or data.labeled_dir) must be specified")
    key = f'{split}_dir'
    if key in data_cfg:
        return data_cfg[key], data_cfg.get(f'{split}_list')
    raise KeyError(f"data.{key} must be specified")


def build_labeled_dataset(data_cfg, split='train'):
    """Build labeled dataset from config (same as train.py).

    Accepts `data.labeled_dir` / `data.labeled_list` as aliases for
    `data.train_dir` / `data.train_list` so that semi configs that use
    the 'labeled_dir' wording also work.
    """
    dataset_type = data_cfg.get('type', 'synapse')
    img_size = data_cfg.get('img_size', 224)
    transform = get_train_transforms(img_size) if split == 'train' else get_val_transforms(img_size)

    if dataset_type in ('synapse', 'acdc'):
        root_dir, list_file = _resolve_split_dir(data_cfg, split)
        return SynapseDataset(
            root_dir=root_dir,
            split=split,
            list_file=list_file,
            transform=transform,
            img_size=img_size,
        )
    elif dataset_type in ('image_mask', 'binary'):
        return GenericDataset(
            root_dir=data_cfg.get('root_dir'),
            split=split,
            transform=transform,
            img_size=img_size,
            **{k: v for k, v in data_cfg.items()
               if k in ('train_ratio', 'val_ratio', 'random_state',
                         'n_splits', 'fold_idx', 'kfold_mode', 'file_list',
                         'img_suffix', 'mask_suffix')},
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def build_unlabeled_dataset(semi_cfg, img_size=224, data_cfg=None):
    """Build unlabeled dataset from semi config.

    Resolution order for the unlabeled image directory:
      1. semi.unlabeled_data.root / .root_dir
      2. data.unlabeled_dir  (preferred form in current configs)
    """
    ul_cfg = semi_cfg.get('unlabeled_data', {}) or {}
    root = ul_cfg.get('root', ul_cfg.get('root_dir', None))
    if root is None and data_cfg is not None:
        root = data_cfg.get('unlabeled_dir')
    if root is None:
        raise ValueError(
            "Unlabeled image dir must be set via either "
            "semi.unlabeled_data.root or data.unlabeled_dir"
        )

    transform = get_train_transforms(img_size, augment_level='light')
    return UnlabeledDataset(
        root_dir=root,
        transform=transform,
        img_size=img_size,
        img_suffix=ul_cfg.get('img_suffix', None),
        use_subdir=ul_cfg.get('use_subdir', False),
    )


# ---------------------------------------------------------------------------
# Optimizer / Scheduler / Loss builders
# ---------------------------------------------------------------------------

def build_optimizer(params, opt_cfg):
    name = opt_cfg.get('name', 'adamw')
    lr = opt_cfg.get('lr', 1e-4)
    weight_decay = opt_cfg.get('weight_decay', 1e-4)
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return SGD(params, lr=lr, weight_decay=weight_decay,
                   momentum=opt_cfg.get('momentum', 0.9))
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, sched_cfg, total_epochs):
    name = sched_cfg.get('name', 'cosine')
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=total_epochs,
                                 eta_min=sched_cfg.get('min_lr', 1e-6))
    elif name == 'step':
        return StepLR(optimizer, step_size=sched_cfg.get('step_size', 50),
                      gamma=sched_cfg.get('gamma', 0.1))
    elif name == 'poly':
        return PolynomialLR(optimizer, total_iters=total_epochs,
                            power=sched_cfg.get('power', 0.9))
    return None


def build_loss(loss_cfg):
    name = loss_cfg.get('name', 'compound')
    params = loss_cfg.get('params', {})
    loss_cls = LOSS_REGISTRY.get(name)
    return loss_cls(**params)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Semi-supervised segmentation training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--output_dir', type=str, default='./output_semi', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 可复现性设置 / Reproducibility setup
    from medseg.utils.reproducibility import set_seed, worker_init_fn, get_generator
    train_cfg_r = cfg.get('training', {})
    seed = train_cfg_r.get('random_state', args.seed)
    deterministic = train_cfg_r.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Build model ----
    model_cfg = cfg.get('model', cfg)
    model = build_model(cfg)
    model = model.to(device)
    logger.info(f"Model built. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ---- Build semi-supervised method ----
    semi_cfg = cfg.get('semi', {})
    img_size = cfg.get('data', {}).get('img_size', model_cfg.get('img_size', 224))
    semi_method = build_semi_method(semi_cfg, model, device, img_size=img_size)
    logger.info(f"Semi-supervised method: {semi_cfg.get('method', 'mean_teacher')}")

    # ---- Build datasets ----
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('training', {})

    labeled_dataset = build_labeled_dataset(data_cfg, 'train')
    unlabeled_dataset = build_unlabeled_dataset(semi_cfg, img_size, data_cfg=data_cfg)

    labeled_bs = train_cfg.get('labeled_batch_size', train_cfg.get('batch_size', 8))
    unlabeled_bs = train_cfg.get('unlabeled_batch_size', train_cfg.get('batch_size', 8))
    num_workers = train_cfg.get('num_workers', 4)

    labeled_loader = DataLoader(
        labeled_dataset, batch_size=labeled_bs, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=unlabeled_bs, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    logger.info(f"Labeled: {len(labeled_dataset)} samples (bs={labeled_bs})")
    logger.info(f"Unlabeled: {len(unlabeled_dataset)} samples (bs={unlabeled_bs})")

    # Validation loader
    val_loader = None
    if 'val_dir' in data_cfg or 'test_dir' in data_cfg:
        val_key = 'val' if 'val_dir' in data_cfg else 'test'
        val_dataset = build_labeled_dataset(data_cfg, val_key)
        val_loader = DataLoader(
            val_dataset, batch_size=train_cfg.get('batch_size', 8),
            shuffle=False, num_workers=num_workers, pin_memory=True)
        logger.info(f"Val: {len(val_dataset)} samples")

    # ---- Build loss, optimizer, scheduler ----
    criterion = build_loss(train_cfg.get('loss', {'name': 'compound'}))
    total_epochs = train_cfg.get('epochs', 150)

    # Combine model params + semi method extra params
    all_params = list(model.parameters()) + semi_method.extra_params()
    optimizer = build_optimizer(all_params, train_cfg.get('optimizer', {}))
    scheduler = build_scheduler(optimizer, train_cfg.get('scheduler', {}), total_epochs)

    # Build extra optimizers (e.g. SASSNet discriminator)
    extra_opt_cfg = train_cfg.get('optimizer', {})
    extra_lr = train_cfg.get('D_lr', extra_opt_cfg.get('lr', 1e-4))
    extra_optimizers = semi_method.extra_optimizers(lr=extra_lr)

    # ---- Resume ----
    start_epoch = 0
    best_dice = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_dice = ckpt.get('best_dice', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

    # ---- Training loop ----
    num_classes = model_cfg.get('num_classes', 2)
    val_interval = train_cfg.get('val_interval', 10)
    save_interval = train_cfg.get('save_interval', 50)

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()

        epoch_loss = 0.0
        epoch_sup = 0.0
        epoch_unsup = 0.0
        n_steps = 0

        # Iterate labeled + unlabeled (cycle unlabeled if shorter)
        unlabeled_iter = itertools.cycle(unlabeled_loader)
        for labeled_batch in labeled_loader:
            unlabeled_batch = next(unlabeled_iter)

            loss_dict = semi_method.train_step(
                labeled_batch, unlabeled_batch,
                criterion, optimizer,
                epoch, total_epochs)

            semi_method.update(epoch)

            # Step extra optimizers (e.g. SASSNet discriminator).
            # Methods accumulate gradients into these params inside train_step
            # via their own .backward() calls, so the correct order here is
            # step (apply the just-accumulated gradients) THEN zero_grad
            # (clear them ahead of the next iteration). The previous order
            # zeroed gradients *before* stepping, which silently dropped
            # every D update (SASSNet's adversarial signal collapsed to 0).
            for extra_opt, _ in extra_optimizers:
                extra_opt.step()
                extra_opt.zero_grad()

            epoch_loss += loss_dict['loss']
            epoch_sup += loss_dict['sup_loss']
            epoch_unsup += loss_dict['unsup_loss']
            n_steps += 1

            if n_steps % 50 == 0:
                logger.info(
                    f"Epoch [{epoch}] Step [{n_steps}/{len(labeled_loader)}] "
                    f"Loss: {loss_dict['loss']:.4f} "
                    f"Sup: {loss_dict['sup_loss']:.4f} "
                    f"Unsup: {loss_dict['unsup_loss']:.4f} "
                    f"W: {loss_dict.get('w', 0):.3f}")

        if scheduler:
            scheduler.step()

        avg_loss = epoch_loss / max(n_steps, 1)
        avg_sup = epoch_sup / max(n_steps, 1)
        avg_unsup = epoch_unsup / max(n_steps, 1)
        lr = optimizer.param_groups[0]['lr']

        log_msg = (f"Epoch [{epoch}/{total_epochs}] "
                   f"Loss: {avg_loss:.4f} Sup: {avg_sup:.4f} Unsup: {avg_unsup:.4f} "
                   f"LR: {lr:.6f} Time: {time.time()-t0:.1f}s")

        # ---- Validation ----
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            eval_model = semi_method.get_eval_model()
            dice = validate(eval_model, val_loader, num_classes, device)
            log_msg += f" Val_Dice: {dice:.4f}"
            if dice > best_dice:
                best_dice = dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'semi_method': semi_cfg.get('method', 'mean_teacher'),
                }, os.path.join(args.output_dir, 'best_model.pth'))
                logger.info(f"New best model saved! Dice: {best_dice:.4f}")

        logger.info(log_msg)

        # ---- Save checkpoint ----
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'semi_method': semi_cfg.get('method', 'mean_teacher'),
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))

    logger.info(f"Training complete. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
