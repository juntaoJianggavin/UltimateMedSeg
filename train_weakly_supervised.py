"""Training script for weakly supervised segmentation.

Supports:
- Image-level classification labels only
- Bounding box annotations only
- Mixed weak + strong annotations

Usage:
    # Box-supervised training
    python train_weakly_supervised.py \
        --config configs/training_paradigms/weak_supervision/box_supervised.yaml \
        --supervision_type box
    
    # CAM-based training
    python train_weakly_supervised.py \
        --config configs/training_paradigms/weak_supervision/cam.yaml \
        --supervision_type cam
    
    # Multi-instance learning
    python train_weakly_supervised.py \
        --config configs/training_paradigms/weak_supervision/mil.yaml \
        --supervision_type mil

(Self-contained: no single canonical GitHub source.)
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

from medseg.model_builder import build_model
from medseg.training.weakly_supervised import (
    BoxSupervisedLoss,
    CAMLoss,
    MILLoss,
    EMPseudoLabelLoss,
    CAMGenerator
)
from medseg.registry import LOSS_REGISTRY
from medseg.utils.metrics import compute_metrics

# Import all modules
import medseg.models.encoders
import medseg.models.decoders
import medseg.losses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeaklySupervisedDataset:
    """Dataset wrapper for weakly supervised learning.
    
    Supports:
    - Images with full pixel annotations
    - Images with only bounding boxes
    - Images with only image-level labels
    """
    
    def __init__(self, dataset, supervision_type='box'):
        self.dataset = dataset
        self.supervision_type = supervision_type
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Add weak supervision annotations
        if self.supervision_type == 'box':
            # Generate or load boxes
            item['boxes'] = self._get_or_generate_boxes(item)
            item['image_labels'] = self._get_image_labels(item)
        elif self.supervision_type == 'image_label':
            item['image_labels'] = self._get_image_labels(item)
        
        return item
    
    def _get_or_generate_boxes(self, item):
        """Get bounding boxes from annotation or generate from label."""
        if 'boxes' in item:
            return item['boxes']
        
        # Generate pseudo-boxes from label statistics
        label = item['label']
        boxes = []
        
        for cls in range(label.max() + 1):
            if cls == 0:
                continue
            mask = (label == cls)
            if mask.sum() > 0:
                # Get bounding box
                y_indices, x_indices = torch.where(mask)
                if len(y_indices) > 0:
                    x1, y1 = x_indices.min(), y_indices.min()
                    x2, y2 = x_indices.max(), y_indices.max()
                    boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
        
        return torch.tensor(boxes) if boxes else torch.empty(0, 4)
    
    def _get_image_labels(self, item):
        """Get image-level classification labels."""
        if 'image_labels' in item:
            return item['image_labels']
        
        # Generate from segmentation mask
        label = item['label']
        num_classes = label.max() + 1
        image_labels = torch.zeros(num_classes)
        
        for cls in range(num_classes):
            if (label == cls).sum() > 0:
                image_labels[cls] = 1.0
        
        return image_labels


def build_loss(supervision_type, cfg):
    """Build weakly supervised loss function.

    Resolution order:
      1. If ``cfg['name']`` names a key in LOSS_REGISTRY, build via registry
         (this is the path that exposes the 21 weak-supervision yaml methods).
      2. Otherwise dispatch to one of the four hard-coded loss classes by
         ``supervision_type``.

    The previous implementation passed top-level defaults as named kwargs
    AND spread ``cfg['params']`` on top of them, which raises
    ``TypeError: multiple values for keyword 'box_penalty'`` whenever the
    yaml put ``box_penalty`` under ``params``. The fix merges them first
    (params win) and only then calls the constructor.
    """
    params = dict(cfg.get('params', {}) or {})

    name = cfg.get('name')
    if name and name in LOSS_REGISTRY:
        loss_cls = LOSS_REGISTRY.get(name)
        return loss_cls(**params)

    def _merged(defaults):
        out = dict(defaults)
        out.update(params)  # params from yaml override top-level defaults
        return out

    if supervision_type == 'box':
        defaults = {'box_penalty': cfg.get('box_penalty', 0.1)}
        return BoxSupervisedLoss(**_merged(defaults))
    elif supervision_type == 'cam':
        defaults = {'cam_threshold': cfg.get('cam_threshold', 0.5)}
        return CAMLoss(**_merged(defaults))
    elif supervision_type == 'mil':
        defaults = {'patch_size': cfg.get('patch_size', 32)}
        return MILLoss(**_merged(defaults))
    elif supervision_type == 'em':
        defaults = {'num_iterations': cfg.get('num_iterations', 5)}
        return EMPseudoLabelLoss(**_merged(defaults))
    elif supervision_type == 'image_label':
        # Image-level multi-label CE — MIL with patch_size=image-size effectively.
        defaults = {'patch_size': cfg.get('patch_size', 32)}
        return MILLoss(**_merged(defaults))
    else:
        raise ValueError(
            f"Unknown supervision type: {supervision_type}. "
            f"Either set --supervision_type ∈ {{box, cam, mil, em, image_label}} "
            f"or set training.loss.name to a registered LOSS_REGISTRY key."
        )


def train_one_epoch_weakly(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    supervision_type='box',
    cam_generator=None
):
    """Train with weak supervision for one epoch."""
    model.train()
    total_loss = 0.0
    
    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(images)
        
        # Compute weakly supervised loss
        if supervision_type == 'box':
            boxes = batch.get('boxes', None)
            image_labels = batch.get('image_labels', None)
            target = batch.get('label', None)
            
            if boxes is not None:
                boxes = boxes.to(device)
            if image_labels is not None:
                image_labels = image_labels.to(device)
            if target is not None:
                target = target.to(device)
            
            loss = criterion(predictions, boxes, image_labels, target)
            
        elif supervision_type == 'cam':
            if cam_generator is not None:
                cams = cam_generator.generate_batch_cam(
                    images,
                    batch.get('image_labels', None)
                )
                image_labels = batch['image_labels'].to(device)
                target = batch.get('label', None)
                if target is not None:
                    target = target.to(device)
                
                loss = criterion(predictions, cams.to(device), image_labels, target)
            else:
                loss = criterion(predictions, predictions, batch['image_labels'].to(device))
        
        elif supervision_type == 'mil':
            image_labels = batch['image_labels'].to(device)
            loss = criterion(predictions, image_labels)
        
        elif supervision_type == 'em':
            weak_labels = batch.get('boxes', batch.get('image_labels', None))
            target = batch.get('label', None)
            if target is not None:
                target = target.to(device)

            loss = criterion(predictions, weak_labels, target)

        else:
            # Generic dispatch for losses chosen via training.loss.name from
            # LOSS_REGISTRY. We hand the loss every plausible field from the
            # batch as a kwarg; the loss's forward() is expected to ignore
            # what it does not need via **kwargs.
            ctx = {}
            for k, v in batch.items():
                if k == 'image':
                    continue
                if isinstance(v, torch.Tensor):
                    ctx[k] = v.to(device)
                else:
                    ctx[k] = v
            target = ctx.pop('label', None)
            try:
                loss = criterion(predictions, target, **ctx) if target is not None \
                       else criterion(predictions, **ctx)
            except TypeError as e:
                raise TypeError(
                    f"Loss {type(criterion).__name__} forward() rejected the "
                    f"batch fields {list(ctx.keys())}. Either add **kwargs to "
                    f"its forward() or extend WeaklySupervisedDataset to emit "
                    f"the field names it expects. Original error: {e}"
                )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i + 1) % 50 == 0:
            logger.info(
                f"Epoch [{epoch}] Step [{i+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_weakly(model, dataloader, num_classes, device):
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
    parser = argparse.ArgumentParser(description='Train with weak supervision')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--supervision_type', type=str, default='box',
                        help='Type of weak supervision. Built-in handlers: '
                             'box / cam / mil / em / image_label. For any '
                             'other value, the training loop calls '
                             'training.loss (built from LOSS_REGISTRY) with '
                             'predictions + every tensor in the batch as '
                             'kwargs; the loss must accept **kwargs.')
    parser.add_argument('--output_dir', type=str, default='./output_weakly_supervised',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Build model
    model = build_model(cfg)
    model = model.to(device)
    logger.info(f"Model built. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Build dataset
    from train import build_dataset
    data_cfg = cfg.get('data', {})
    train_dataset = build_dataset(data_cfg, 'train')
    train_dataset = WeaklySupervisedDataset(train_dataset, args.supervision_type)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('training', {}).get('batch_size', 8),
        shuffle=True,
        num_workers=cfg.get('training', {}).get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Train dataset: {len(train_dataset)} samples ({args.supervision_type} supervision)")
    
    val_loader = None
    if 'val_dir' in data_cfg or 'test_dir' in data_cfg:
        val_key = 'val' if 'val_dir' in data_cfg else 'test'
        val_dataset = build_dataset(data_cfg, val_key)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get('training', {}).get('batch_size', 8),
            shuffle=False,
            num_workers=cfg.get('training', {}).get('num_workers', 4),
            pin_memory=True,
        )
        logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Build loss
    loss_cfg = cfg.get('training', {}).get('loss', {})
    criterion = build_loss(args.supervision_type, loss_cfg)
    criterion = criterion.to(device)
    logger.info(f"Loss: {args.supervision_type}")
    
    # Build optimizer
    training_cfg = cfg.get('training', {})
    opt_cfg = training_cfg.get('optimizer', {'name': 'adamw', 'lr': 1e-4})
    optimizer = AdamW(
        model.parameters(),
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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Build CAM generator if needed
    cam_generator = None
    if args.supervision_type == 'cam':
        cam_generator = CAMGenerator(model, target_layer='encoder.layer4')
    
    # Training loop
    logger.info(f"Starting weakly supervised training ({args.supervision_type}) for {total_epochs} epochs")
    
    for epoch in range(start_epoch, total_epochs):
        # Train
        train_loss = train_one_epoch_weakly(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.supervision_type,
            cam_generator
        )
        
        # Update LR
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch}/{total_epochs}] Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        
        # Validate
        if val_loader is not None and (epoch + 1) % training_cfg.get('val_interval', 10) == 0:
            num_classes = cfg.get('model', {}).get('num_classes', 9)
            val_dice = validate_weakly(model, val_loader, num_classes, device)
            logger.info(f"Epoch [{epoch}/{total_epochs}] Val Dice: {val_dice:.4f}")
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                save_path = os.path.join(args.output_dir, 'best_model_weakly.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'supervision_type': args.supervision_type,
                }, save_path)
                logger.info(f"Saved best model (Dice: {best_dice:.4f})")
    
    logger.info(f"Weakly supervised training completed. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
