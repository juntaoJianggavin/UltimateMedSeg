"""Domain Adaptation training script for medical image segmentation.

Supports multiple domain adaptation methods from ADA4MIA benchmark:
  - AdvEnt: Adversarial Entropy Minimization (CVPR 2019)
  - Tent: Test-Time Adaptation (ICLR 2021)
  - DPL: Denoised Pseudo-Labeling (MICCAI 2021)
  - FSM: Fourier Style Mining (MIA 2022)
  - CBMT: Class-Balanced Mean Teacher (MICCAI 2023)
  - UGTST: Uncertainty-guided Tiered Self-training (MICCAI 2024)
  - STDR: Dual-Reference Source-Free ADA (TMI 2024)

Usage:
    python train_domain_adaptation.py --config configs/training_paradigms/domain_adaptation/advent.yaml --output_dir output/da_advent

Reference implementation: https://github.com/whq-xxh/ADA4MIA
"""

import os
import sys
import copy
import argparse
import logging
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from medseg.datasets.domain_adaptation_datasets import (
    DomainAdaptationDataset, SourceFreeDataset,
)
from medseg.utils.metrics import compute_metrics

# Import all modules to trigger registration
import medseg.models.encoders
import medseg.models.decoders
import medseg.models.skip_connections
import medseg.models.bottlenecks
import medseg.losses
import medseg.training.domain_adaptation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _pick_root(sub_cfg, *keys):
    """Return the first present value across the candidate keys, else None."""
    for k in keys:
        if k in sub_cfg and sub_cfg[k] is not None:
            return sub_cfg[k]
    return None


def build_source_dataset(data_cfg, img_size=224):
    """Build source domain dataset (labeled).

    Accepts both the synapse-style (`train_dir`/`train_list`) and the
    DA-style (`image_dir`/`mask_dir`) field naming.
    """
    source_cfg = data_cfg.get('source', data_cfg)
    dataset_type = source_cfg.get('type', data_cfg.get('type', 'synapse'))
    transform = get_train_transforms(img_size)

    if dataset_type in ('synapse', 'acdc'):
        root = _pick_root(source_cfg, 'train_dir', 'image_dir', 'root_dir', 'root')
        if root is None:
            raise KeyError(
                "data.source needs one of: train_dir / image_dir / root_dir / root"
            )
        return SynapseDataset(
            root_dir=root,
            split='train',
            list_file=source_cfg.get('train_list'),
            transform=transform,
            img_size=img_size,
        )
    elif dataset_type in ('image_mask', 'binary'):
        return GenericDataset(
            root_dir=_pick_root(source_cfg, 'root_dir', 'image_dir', 'train_dir'),
            split='train',
            transform=transform,
            img_size=img_size,
        )
    else:
        raise ValueError(f"Unknown source dataset type: {dataset_type}")


def build_target_dataset(data_cfg, img_size=224):
    """Build target domain dataset (unlabeled or partially labeled).

    Accepts ``data.target.{root, root_dir, image_dir}`` interchangeably.
    """
    target_cfg = data_cfg.get('target', {})
    root = _pick_root(target_cfg, 'root', 'root_dir', 'image_dir', 'train_dir')
    if root is None:
        raise ValueError(
            "data.target needs one of: root / root_dir / image_dir / train_dir"
        )

    transform = get_train_transforms(img_size, augment_level='light')
    return UnlabeledDataset(
        root_dir=root,
        transform=transform,
        img_size=img_size,
        img_suffix=target_cfg.get('img_suffix', None),
        use_subdir=target_cfg.get('use_subdir', False),
    )


def build_val_dataset(data_cfg, img_size=224):
    """Build validation dataset (target domain with labels).

    Resolution order: ``data.val`` → ``data.validation`` → ``data.test`` →
    top-level. Accepts ``image_dir``/``mask_dir`` or ``val_dir``/``test_dir``.
    """
    for key in ('val', 'validation', 'test'):
        if key in data_cfg and isinstance(data_cfg[key], dict):
            val_cfg = data_cfg[key]
            split = 'val' if key in ('val', 'validation') else 'test'
            break
    else:
        val_cfg = data_cfg
        split = 'val' if 'val_dir' in data_cfg else 'test'

    dataset_type = val_cfg.get('type', data_cfg.get('type', 'synapse'))
    transform = get_val_transforms(img_size)

    if dataset_type in ('synapse', 'acdc'):
        root = _pick_root(val_cfg, f'{split}_dir', 'test_dir', 'val_dir',
                          'image_dir', 'root_dir', 'root')
        if root is None:
            return None
        return SynapseDataset(
            root_dir=root,
            split=split,
            list_file=val_cfg.get(f'{split}_list'),
            transform=transform,
            img_size=img_size,
        )
    elif dataset_type in ('image_mask', 'binary'):
        return GenericDataset(
            root_dir=_pick_root(val_cfg, 'root_dir', 'image_dir', 'val_dir', 'test_dir'),
            split=split,
            transform=transform,
            img_size=img_size,
        )
    else:
        raise ValueError(f"Unknown val dataset type: {dataset_type}")


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
# EMA Model
# ---------------------------------------------------------------------------

def create_ema_model(model):
    """Create exponential moving average model."""
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return ema_model


def update_ema(model, ema_model, alpha=0.999):
    """Update EMA model parameters."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, dataloader, num_classes, device):
    """Run validation and return metrics."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label']
        preds = model(images)
        preds = preds.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_preds, all_labels, num_classes)
    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Training loops for different DA methods
# ---------------------------------------------------------------------------

import inspect


def _filter_kwargs_for(fn, ctx):
    """Return the subset of ``ctx`` keys that ``fn`` will accept.

    If ``fn`` has a **kwargs catch-all, all keys are passed through; otherwise
    only the named parameters are kept. This lets us call DA losses that have
    wildly different forward() signatures from a single training loop.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ctx
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return ctx
    return {k: v for k, v in ctx.items() if k in params}


def _call_da_loss(da_loss_fn, **ctx):
    """Call a DA loss with whatever subset of ``ctx`` its forward accepts."""
    kept = _filter_kwargs_for(da_loss_fn.forward, ctx)
    return da_loss_fn(**kept)


def train_advent(model, source_loader, target_loader, da_loss_fn,
                 supervised_loss_fn, optimizer, device, epoch, total_epochs):
    """Source+target DA training step (AdvEnt-style and the 14 others that
    follow the same fan-out)."""
    model.train()
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    total_loss = 0.0
    num_batches = min(len(source_loader), len(target_loader))

    for i in range(num_batches):
        try:
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_batch = next(source_iter)

        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)

        source_images = source_batch['image'].to(device)
        source_labels = source_batch['label'].to(device)
        target_images = target_batch['image'].to(device)

        # Forward
        source_pred = model(source_images)
        target_pred = model(target_images)

        # Supervised loss on source
        sup_loss = supervised_loss_fn(source_pred, source_labels)

        # Build the full DA context. Different DA losses ask for different
        # subsets — some want (source_pred, target_pred, labeled_loss), some
        # want (predictions, teacher_pred, ...), some want (pred1, pred2),
        # and several need source_labels or images. We pass the union and
        # let _call_da_loss keep only what the loss actually accepts.
        ctx = {
            'source_pred': source_pred,
            'target_pred': target_pred,
            'source_labels': source_labels,
            'labeled_loss': sup_loss,
            'source_images': source_images,
            'target_images': target_images,
            'predictions': source_pred,        # alias used by ~6 DA losses
            'teacher_pred': target_pred,       # alias used by ~4 DA losses
            'pred1': source_pred,              # alias used by CoNMix
            'pred2': target_pred,
            'augmented_pred': target_pred,     # FPL+/SSiT placeholder
        }
        da_loss = _call_da_loss(da_loss_fn, **ctx)

        optimizer.zero_grad()
        da_loss.backward()
        optimizer.step()

        total_loss += da_loss.item()

    return total_loss / max(num_batches, 1)


def train_source_free(model, ema_model, target_loader, da_loss_fn,
                      optimizer, device, epoch, total_epochs, method='dpl'):
    """Source-free DA training (Tent / DPL / CBMT / SHOT / Adamss / SF-TTA …)."""
    model.train()
    total_loss = 0.0

    for batch in target_loader:
        target_images = batch['image'].to(device)

        # Student prediction
        target_pred = model(target_images)

        # Teacher (EMA) prediction
        with torch.no_grad():
            ema_pred = ema_model(target_images) if ema_model is not None else None

        ctx = {
            'target_pred': target_pred,
            'target_images': target_images,
            'predictions': target_pred,
            'ema_pred': ema_pred,
            'teacher_pred': ema_pred,
            'student_pred': target_pred,
        }
        loss = _call_da_loss(da_loss_fn, **ctx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA (only if we actually have a teacher)
        if ema_model is not None:
            update_ema(model, ema_model)

        total_loss += loss.item()

        # Update epoch for rampup
        if hasattr(da_loss_fn, 'update_epoch'):
            da_loss_fn.update_epoch(epoch)

    return total_loss / max(len(target_loader), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Domain Adaptation Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--output_dir', type=str, default='output/domain_adaptation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 可复现性设置 / Reproducibility setup
    from medseg.utils.reproducibility import set_seed, worker_init_fn, get_generator
    train_cfg_r = cfg.get('training', {})
    seed = train_cfg_r.get('random_state', 42)
    deterministic = train_cfg_r.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Build model
    model_cfg = cfg['model']
    num_classes = model_cfg.get('num_classes', 4)
    img_size = model_cfg.get('img_size', 224)
    model = build_model(model_cfg).to(device)
    logger.info(f"Model built: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # Build datasets
    data_cfg = cfg['data']
    train_cfg = cfg['training']
    da_cfg = cfg.get('domain_adaptation', {})
    da_method = da_cfg.get('method', 'advent')
    da_params = da_cfg.get('params', {})

    # Determine if source-free. Methods listed here skip building a source
    # loader and run the target-only training path with an EMA teacher.
    # Explicit override: set ``domain_adaptation.source_free: true|false`` in
    # the yaml to force the dispatch regardless of method name.
    source_free_methods = {
        'tent', 'dpl', 'class_balanced_mt', 'uncertainty_self_training',
        'dual_reference',
        # Methods that semantically are source-free (only see target during
        # adaptation, optionally with a pretrained source-trained checkpoint):
        'shot_loss', 'adamss_loss', 'sf_tta_loss', 'fpl_plus_loss',
    }
    is_source_free = da_cfg.get('source_free', da_method in source_free_methods)

    if not is_source_free:
        source_dataset = build_source_dataset(data_cfg, img_size)
        source_loader = DataLoader(
            source_dataset,
            batch_size=train_cfg.get('batch_size', 8),
            shuffle=True,
            num_workers=train_cfg.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
        )
        logger.info(f"Source dataset: {len(source_dataset)} samples")

    target_dataset = build_target_dataset(data_cfg, img_size)
    target_loader = DataLoader(
        target_dataset,
        batch_size=train_cfg.get('batch_size', 8),
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Target dataset: {len(target_dataset)} samples")

    # Build DA loss
    da_loss_fn = LOSS_REGISTRY.get(da_method)(**da_params)
    logger.info(f"DA method: {da_method}")

    # Build supervised loss (for non source-free methods)
    if not is_source_free:
        sup_loss_cfg = train_cfg.get('loss', {'name': 'compound', 'params': {}})
        supervised_loss_fn = build_loss(sup_loss_cfg)

    # Load source-pretrained checkpoint. Source-free methods REQUIRE this:
    # without it the EMA teacher would start from random weights and adapt
    # toward noise. We refuse to silently degrade and raise instead.
    pretrained = data_cfg.get('pretrained_model') or cfg.get('pretrained_model')
    if pretrained:
        if not os.path.exists(pretrained):
            raise FileNotFoundError(
                f"pretrained_model not found: {pretrained}"
            )
        ckpt = torch.load(pretrained, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        total = sum(1 for _ in model.parameters())
        if total and len(missing) > 0.5 * total:
            raise RuntimeError(
                f"pretrained_model at {pretrained} matched only a minority of "
                f"params (missing={len(missing)}/{total}). Refusing to start "
                f"from a near-random init. Check that the checkpoint matches "
                f"the model config."
            )
        logger.info(f"Loaded pretrained weights from {pretrained} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})")
    elif is_source_free:
        raise RuntimeError(
            f"Source-free DA method '{da_method}' requires "
            f"data.pretrained_model in the yaml. Refusing to adapt a teacher "
            f"that starts from random weights."
        )

    # EMA model (for source-free methods) — built AFTER loading the
    # pretrained weights so the teacher inherits them.
    ema_model = None
    if is_source_free:
        ema_model = create_ema_model(model)

    # Optional validation loader
    val_dataset = build_val_dataset(data_cfg, img_size)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.get('batch_size', 8),
            shuffle=False,
            num_workers=train_cfg.get('num_workers', 4),
            pin_memory=True,
        )
        logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Optimizer and scheduler
    total_epochs = train_cfg.get('epochs', 100)
    optimizer = build_optimizer(model.parameters(), train_cfg.get('optimizer', {}))
    scheduler = build_scheduler(optimizer, train_cfg.get('scheduler', {}), total_epochs)

    # Training loop
    logger.info(f"Starting domain adaptation training: {da_method}")
    logger.info(f"  Source-free: {is_source_free}")
    logger.info(f"  Epochs: {total_epochs}")

    best_metric = 0.0
    val_interval = train_cfg.get('val_interval', 10)

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        if is_source_free:
            avg_loss = train_source_free(
                model, ema_model, target_loader, da_loss_fn,
                optimizer, device, epoch, total_epochs, method=da_method
            )
        else:
            avg_loss = train_advent(
                model, source_loader, target_loader, da_loss_fn,
                supervised_loss_fn, optimizer, device, epoch, total_epochs
            )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        log_line = (
            f"Epoch {epoch}/{total_epochs} | Loss: {avg_loss:.4f} "
            f"| Time: {elapsed:.1f}s"
        )

        # Validation
        if val_loader is not None and epoch % val_interval == 0:
            eval_model = ema_model if (is_source_free and ema_model is not None) else model
            metrics = validate(eval_model, val_loader, num_classes, device)
            dice_vals = list(metrics.get('dice', {}).values())
            mean_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
            log_line += f" | Val_Dice: {mean_dice:.4f}"
            if mean_dice > best_metric:
                best_metric = mean_dice
                best_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_metric,
                    'da_method': da_method,
                }, best_path)
                log_line += " (new best)"

        logger.info(log_line)

        # Save checkpoint
        if epoch % train_cfg.get('save_interval', 50) == 0:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved: {final_path}. "
                f"Best Val_Dice: {best_metric:.4f}")


if __name__ == '__main__':
    main()
