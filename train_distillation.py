"""Training script for UNet knowledge distillation.

Supports:
- Logit-based distillation (Hinton style)
- Feature-based distillation
- Attention-based distillation
- Multi-scale distillation
- Hint-based distillation (FitNets)

Usage:
    python train_distillation.py \
        --teacher_config configs/training_paradigms/distillation/teacher_large.yaml \
        --student_config configs/training_paradigms/distillation/aicsd.yaml \
        --distillation_type logit \
        --temperature 4.0 \
        --alpha 0.5

Reference implementation: https://arxiv.org/abs/1503.02531
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inspect
from medseg.model_builder import build_model
from medseg.training.distillation import UNetDistillationLoss, HintDistillationLoss, AttentionMimicryLoss
from medseg.training.distillation.feature_extractor import FeatureExtractor
from medseg.datasets import SynapseDataset, GenericDataset, get_train_transforms, get_val_transforms
from medseg.utils.metrics import compute_metrics
from medseg.registry import LOSS_REGISTRY

# Import all modules
import medseg.models.encoders
import medseg.models.decoders
import medseg.losses
import medseg.training.distillation  # noqa: F401  triggers KD loss registrations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resolve_split_dir(data_cfg, split):
    """Pick the directory for a split, supporting the three styles configs
    in this repo use:
      1. flat: ``data.train_dir``, ``data.val_dir``, ``data.test_dir``
      2. DA-style nested: ``data.{train,val,test,source}.image_dir``
      3. root only: ``data.root_dir`` (GenericDataset auto-splits)
    Returns ``(root_dir_or_None, list_file_or_None)``.
    """
    flat_key = f'{split}_dir'
    if flat_key in data_cfg and data_cfg[flat_key]:
        return data_cfg[flat_key], data_cfg.get(f'{split}_list')

    nested_keys = ['val', 'validation'] if split == 'val' else (
        ['test'] if split == 'test' else ['train', 'source']
    )
    for nk in nested_keys:
        sub = data_cfg.get(nk)
        if isinstance(sub, dict):
            for cand in ('train_dir', 'image_dir', 'root_dir', 'root'):
                if cand in sub and sub[cand]:
                    return sub[cand], sub.get(f'{split}_list')

    return data_cfg.get('root_dir'), data_cfg.get(f'{split}_list')


def build_dataset(data_cfg, split='train'):
    """Build dataset from config.

    Accepts both the synapse-style (``train_dir``/``val_dir``/``test_dir``)
    and the DA-style nested (``data.{train,val,test}.image_dir``) field
    naming so distillation configs originally laid out for DA can be
    consumed without rewriting.
    """
    dataset_type = data_cfg.get('type', 'synapse')
    img_size = data_cfg.get('img_size', 224)
    transform = get_train_transforms(img_size) if split == 'train' else get_val_transforms(img_size)
    root_dir, list_file = _resolve_split_dir(data_cfg, split)

    if dataset_type in ('synapse', 'acdc'):
        if root_dir is None:
            raise KeyError(
                f"data is missing a directory for split={split}. Looked for "
                f"data.{split}_dir, data.{split}.image_dir, data.root_dir."
            )
        return SynapseDataset(
            root_dir=root_dir,
            split=split,
            list_file=list_file,
            transform=transform,
            img_size=img_size,
        )
    elif dataset_type in ('image_mask', 'binary'):
        return GenericDataset(
            root_dir=root_dir or data_cfg.get('root_dir'),
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
            file_list=list_file,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def _filter_kwargs_for(fn, ctx):
    """Subset ctx to keys that fn accepts (full passthrough if **kwargs)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return ctx
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return ctx
    return {k: v for k, v in ctx.items() if k in params}


def _load_teacher_ckpt(model, path, device):
    """Load teacher weights, tolerating ``model_state_dict``/raw-dict forms.

    Strict: per project policy a missing or unreadable checkpoint is a hard
    error. KD against a random teacher is not a meaningful experiment, so
    we refuse to silently fall back.
    """
    if not path:
        raise RuntimeError(
            "Teacher checkpoint path is required. Pass --teacher_ckpt or set "
            "teacher_cfg.pretrained_model / student_cfg.distillation.teacher_ckpt."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing and len(missing) > 0.5 * sum(1 for _ in model.parameters()):
        raise RuntimeError(
            f"Teacher checkpoint at {path} only matched a minority of params "
            f"(missing={len(missing)}). Refusing to train against a near-random "
            f"teacher. Make sure teacher_config matches the checkpoint's model."
        )
    logger.info(
        f"Teacher weights loaded from {path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return True


def train_one_epoch(
    teacher_model,
    student_model,
    distillation_loss,
    base_loss,
    distill_weight,
    dataloader,
    optimizer,
    device,
    epoch,
    teacher_extractor=None,
    student_extractor=None,
):
    """Train student model with teacher guidance for one epoch.

    The KD loss is called with the union of plausible kwargs (logits,
    features, labels) and inspect-filtered down to the ones its forward
    actually accepts. Whatever the KD loss returns is added to the
    supervised base loss with weight ``distill_weight``.
    """
    teacher_model.eval()
    student_model.train()

    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Teacher: no grad, optionally also capture features via hook.
        with torch.no_grad():
            if teacher_extractor is not None:
                teacher_output, teacher_feats = teacher_extractor(images)
            else:
                teacher_output = teacher_model(images)
                teacher_feats = {}

        # Student: with grad, optionally capture features.
        if student_extractor is not None:
            student_output, student_feats = student_extractor(images)
        else:
            student_output = student_model(images)
            student_feats = {}

        # Convert feature dicts → ordered lists (most KD losses expect lists).
        s_feat_list = [v for _, v in sorted(student_feats.items())]
        t_feat_list = [v for _, v in sorted(teacher_feats.items())]

        # Supervised loss on student.
        task_loss = base_loss(student_output, labels)

        # KD loss with flexible kwargs.
        ctx = {
            'student_output': student_output, 'teacher_output': teacher_output,
            'student_logits': student_output, 'teacher_logits': teacher_output,
            'student_pred': student_output, 'teacher_pred': teacher_output,
            'student': student_output, 'teacher': teacher_output,
            'student_features': student_feats, 'teacher_features': teacher_feats,
            'feat_S': s_feat_list, 'feat_T': t_feat_list,
            's_feats': s_feat_list, 't_feats': t_feat_list,
            'features_student': s_feat_list, 'features_teacher': t_feat_list,
            'target': labels, 'labels': labels,
        }
        kept = _filter_kwargs_for(distillation_loss.forward, ctx)
        kd_loss = distillation_loss(**kept)

        loss = task_loss + distill_weight * kd_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            logger.info(
                f"Epoch [{epoch}] Step [{i+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (task={task_loss.item():.4f}, "
                f"kd={float(kd_loss):.4f})"
            )

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
    parser = argparse.ArgumentParser(description='Train UNet with knowledge distillation')
    parser.add_argument('--teacher_config', type=str, required=True, help='Teacher model config')
    parser.add_argument('--student_config', type=str, required=True, help='Student model config')
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='Path to teacher checkpoint. Overrides teacher_cfg.pretrained_model. '
                             'Without a trained teacher KD is meaningless.')
    parser.add_argument('--distillation_type', type=str, default=None,
                        help='Built-in KD type (logit/feature/attention/multi_scale/hint). '
                             'Overrides student_cfg.distillation.method.')
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight on the KD loss when added to the supervised base loss.')
    parser.add_argument('--output_dir', type=str, default='./output_distillation')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.teacher_config, 'r') as f:
        teacher_cfg = yaml.safe_load(f)
    with open(args.student_config, 'r') as f:
        student_cfg = yaml.safe_load(f)

    # 可复现性设置 / Reproducibility setup
    from medseg.utils.reproducibility import set_seed, worker_init_fn, get_generator
    train_cfg_r = student_cfg.get('training', {})
    seed = train_cfg_r.get('random_state', args.seed)
    deterministic = train_cfg_r.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Build teacher (and load weights — KD without trained teacher is noise) ----
    teacher_model = build_model(teacher_cfg).to(device)
    teacher_ckpt = (args.teacher_ckpt or
                    teacher_cfg.get('pretrained_model') or
                    student_cfg.get('distillation', {}).get('teacher_ckpt'))
    # Strict: _load_teacher_ckpt raises if the path is missing/unreadable/
    # only partially matches. KD against a random teacher is meaningless;
    # the trainer refuses to silently degrade.
    _load_teacher_ckpt(teacher_model, teacher_ckpt, device)
    teacher_model.eval()
    logger.info(f"Teacher: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M params")

    # ---- Build student ----
    student_model = build_model(student_cfg).to(device)
    logger.info(f"Student: {sum(p.numel() for p in student_model.parameters())/1e6:.2f}M params")

    # ---- Resolve KD method ----
    distill_cfg = student_cfg.get('distillation', {})
    method = args.distillation_type or distill_cfg.get('method', 'logit')
    distill_params = dict(distill_cfg.get('params', {}) or {})
    # CLI temp/alpha defaults if user didn't put them in the yaml.
    distill_params.setdefault('temperature', args.temperature)
    distill_weight = float(distill_cfg.get('weight', args.alpha))

    if method in ('logit', 'feature', 'attention', 'multi_scale'):
        distillation_loss = UNetDistillationLoss(
            temperature=distill_params.get('temperature', args.temperature),
            alpha=1.0,                # weighting is handled in train_one_epoch
            distillation_type=method,
        )
    elif method == 'hint':
        distillation_loss = HintDistillationLoss(
            temperature=distill_params.get('temperature', args.temperature),
            alpha=1.0,
        )
    elif method == 'attention_mimicry':
        distillation_loss = AttentionMimicryLoss(alpha=1.0)
    elif method in LOSS_REGISTRY:
        distillation_loss = LOSS_REGISTRY.get(method)(**distill_params)
    else:
        raise ValueError(
            f"Unknown distillation method: {method}. Set "
            f"student_cfg.distillation.method to a LOSS_REGISTRY key, or "
            f"pass --distillation_type logit/feature/attention/multi_scale/hint."
        )
    distillation_loss = distillation_loss.to(device)
    logger.info(f"KD method: {method}, weight: {distill_weight}")

    # ---- Set up feature extractors if the KD loss needs intermediate features ----
    feature_layers = distill_cfg.get('feature_layers') or distill_params.get('feature_layers')
    teacher_extractor = student_extractor = None
    if feature_layers:
        teacher_extractor = FeatureExtractor(teacher_model, feature_layers)
        student_extractor = FeatureExtractor(student_model, feature_layers)
        logger.info(f"Hooked feature layers: {feature_layers}")

    # ---- Supervised base loss for the student ----
    from medseg.losses.compound_loss import CompoundLoss
    base_loss_cfg = student_cfg.get('training', {}).get('loss', {'name': 'compound'})
    if base_loss_cfg.get('name', 'compound') in LOSS_REGISTRY:
        base_loss = LOSS_REGISTRY.get(base_loss_cfg['name'])(**(base_loss_cfg.get('params') or {}))
    else:
        base_loss = CompoundLoss()
    base_loss = base_loss.to(device)
    
    # Build datasets
    data_cfg = student_cfg.get('data', {})
    train_dataset = build_dataset(data_cfg, 'train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=student_cfg.get('training', {}).get('batch_size', 8),
        shuffle=True,
        num_workers=student_cfg.get('training', {}).get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    
    val_loader = None
    if 'val_dir' in data_cfg or 'test_dir' in data_cfg:
        val_key = 'val' if 'val_dir' in data_cfg else 'test'
        val_dataset = build_dataset(data_cfg, val_key)
        val_loader = DataLoader(
            val_dataset,
            batch_size=student_cfg.get('training', {}).get('batch_size', 8),
            shuffle=False,
            num_workers=student_cfg.get('training', {}).get('num_workers', 4),
            pin_memory=True,
        )
        logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Build optimizer (only for student)
    training_cfg = student_cfg.get('training', {})
    opt_cfg = training_cfg.get('optimizer', {'name': 'adamw', 'lr': 1e-4})
    optimizer = AdamW(
        student_model.parameters(),
        lr=opt_cfg.get('lr', 1e-4),
        weight_decay=opt_cfg.get('weight_decay', 1e-4)
    )
    
    # Build scheduler
    total_epochs = training_cfg.get('epochs', 200)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    
    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        student_model.load_state_dict(checkpoint['student_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info(f"Starting distillation training for {total_epochs} epochs")
    
    for epoch in range(start_epoch, total_epochs):
        # Train
        train_loss = train_one_epoch(
            teacher_model,
            student_model,
            distillation_loss,
            base_loss,
            distill_weight,
            train_loader,
            optimizer,
            device,
            epoch,
            teacher_extractor=teacher_extractor,
            student_extractor=student_extractor,
        )
        
        # Update LR
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch}/{total_epochs}] Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        
        # Validate
        if val_loader is not None and (epoch + 1) % training_cfg.get('val_interval', 10) == 0:
            num_classes = student_cfg.get('model', {}).get('num_classes', 9)
            val_dice = validate(student_model, val_loader, num_classes, device)
            logger.info(f"Epoch [{epoch}/{total_epochs}] Val Dice: {val_dice:.4f}")
            
            # Save best student model
            if val_dice > best_dice:
                best_dice = val_dice
                save_path = os.path.join(args.output_dir, 'best_student.pth')
                torch.save({
                    'epoch': epoch,
                    'student_state_dict': student_model.state_dict(),
                    'teacher_state_dict': teacher_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'distillation_config': {
                        'type': args.distillation_type,
                        'temperature': args.temperature,
                        'alpha': args.alpha,
                    }
                }, save_path)
                logger.info(f"Saved best student model (Dice: {best_dice:.4f})")
    
    logger.info(f"Distillation training completed. Best Student Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
