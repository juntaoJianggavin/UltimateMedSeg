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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medseg.model_builder import build_model
from medseg.training.weakly_supervised import (
    BoxSupervisedLoss,
    CAMLoss,
    MILLoss,
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

    Loads weak annotations from JSON files specified in data_cfg:
    - annotation_file: boxes.json, points.json, scribbles.json, etc.
    - label_file: image_labels.json

    JSON annotations use normalized coordinates (0-1) which are converted
    to pixel coordinates using the image tensor dimensions. Falls back to
    generating annotations from segmentation masks when JSON is unavailable.
    """

    def __init__(self, dataset, supervision_type='box', num_classes=None,
                 data_cfg=None):
        self.dataset = dataset
        self.supervision_type = supervision_type
        self.num_classes = num_classes
        self.data_cfg = data_cfg or {}
        self._annotations = {}
        self._load_annotations()

    def _load_annotations(self):
        """Load JSON annotation files indexed by image filename."""
        import json as _json
        for key in ('annotation_file', 'label_file'):
            fpath = self.data_cfg.get(key)
            if fpath and os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    entries = _json.load(f)
                for entry in entries:
                    img_name = entry.get('image', '')
                    if img_name not in self._annotations:
                        self._annotations[img_name] = {}
                    self._annotations[img_name].update(entry)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        case_name = item.get('case_name', '')
        ann = self._annotations.get(case_name, {})

        if 'boxes' in ann:
            boxes, box_classes = self._parse_boxes(ann['boxes'], item)
            item['boxes'] = boxes
            if box_classes is not None:
                item['box_classes'] = box_classes
        elif self.supervision_type == 'box':
            item['boxes'] = self._get_or_generate_boxes(item)

        if 'image_labels' in ann:
            item['image_labels'] = self._parse_image_labels(ann['image_labels'])
        else:
            item['image_labels'] = self._get_image_labels(item)

        if 'scribbles' in ann:
            scribbles, scribble_classes = self._parse_scribbles(
                ann['scribbles'], item)
            item['scribbles'] = scribbles
            item['scribble_classes'] = scribble_classes

        if 'points' in ann:
            points, point_classes = self._parse_points(ann['points'], item)
            item['points'] = points
            item['point_classes'] = point_classes

        return item

    def _parse_boxes(self, boxes_data, item):
        """Parse boxes from JSON. Handles class-annotated and plain formats.

        Returns (boxes_tensor, box_classes_tensor_or_None).
        """
        _, H, W = item['image'].shape
        boxes = []
        classes = []
        has_class = False
        for b in boxes_data:
            if isinstance(b, dict):
                box = b['box']
                cls = b.get('class', 1)
                has_class = True
            else:
                box = b
                cls = 1
            boxes.append([box[0] * W, box[1] * H, box[2] * W, box[3] * H])
            classes.append(cls)
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 4)
        classes_t = torch.tensor(classes, dtype=torch.long) if has_class and classes else None
        return boxes_t, classes_t

    def _parse_image_labels(self, labels_data):
        """Parse image-level labels from JSON. Returns multi-hot tensor."""
        num_classes = self.num_classes if self.num_classes is not None else (
            max(labels_data) + 1 if labels_data else 2)
        image_labels = torch.zeros(num_classes)
        for cls in labels_data:
            if 0 <= cls < num_classes:
                image_labels[cls] = 1.0
        return image_labels

    def _parse_scribbles(self, scribbles_data, item):
        """Parse scribbles from JSON. Returns (coords_tensor, classes_tensor)."""
        _, H, W = item['image'].shape
        coords = []
        classes = []
        for s in scribbles_data:
            if isinstance(s, dict):
                pts = s['scribble']
                cls = s.get('class', 0)
            else:
                pts = s
                cls = 0
            for pt in pts:
                coords.append([int(pt[0] * W), int(pt[1] * H)])
                classes.append(cls)
        if coords:
            return (torch.tensor(coords, dtype=torch.long),
                    torch.tensor(classes, dtype=torch.long))
        return torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long)

    def _parse_points(self, points_data, item):
        """Parse points from JSON. Returns (points_tensor, classes_tensor)."""
        _, H, W = item['image'].shape
        pts = []
        classes = []
        for p in points_data:
            if isinstance(p, dict):
                pt = p['point']
                cls = p.get('class', 0)
            else:
                pt = p
                cls = 0
            pts.append([int(pt[0] * W), int(pt[1] * H)])
            classes.append(cls)
        if pts:
            return (torch.tensor(pts, dtype=torch.long),
                    torch.tensor(classes, dtype=torch.long))
        return torch.empty(0, 2, dtype=torch.long), torch.empty(0, dtype=torch.long)

    def _get_or_generate_boxes(self, item):
        """Generate pseudo-boxes from label statistics (fallback)."""
        if 'boxes' in item:
            return item['boxes']
        label = item['label']
        boxes = []
        for cls in range(label.max() + 1):
            if cls == 0:
                continue
            mask = (label == cls)
            if mask.sum() > 0:
                y_indices, x_indices = torch.where(mask)
                if len(y_indices) > 0:
                    x1, y1 = x_indices.min(), y_indices.min()
                    x2, y2 = x_indices.max(), y_indices.max()
                    boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
        return torch.tensor(boxes) if boxes else torch.empty(0, 4)

    def _get_image_labels(self, item):
        """Generate image-level labels from mask (fallback)."""
        if 'image_labels' in item:
            return item['image_labels']
        label = item['label']
        num_classes = self.num_classes if self.num_classes is not None else label.max() + 1
        image_labels = torch.zeros(num_classes)
        for cls in range(num_classes):
            if (label == cls).sum() > 0:
                image_labels[cls] = 1.0
        return image_labels


def weak_collate_fn(batch):
    """Collate function that handles variable-length weak annotations.

    Keeps boxes/points/scribbles as lists of per-sample tensors so that
    loss functions expecting list format (e.g. BoxSupervisedLoss) work correctly.
    """
    elem = batch[0]
    result = {}
    for key in elem:
        if key in ('boxes', 'points', 'scribbles', 'box_classes', 'point_classes', 'scribble_classes'):
            # Keep as list of per-sample tensors
            result[key] = [b[key] for b in batch if key in b]
        elif key == 'case_name':
            result[key] = [b[key] for b in batch]
        else:
            vals = [b[key] for b in batch if key in b]
            if vals and isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals, 0)
            else:
                result[key] = vals
    return result


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
        defaults = {
            'mask_type': cfg.get('mask_type', 'ellipse'),
            'outside_penalty': cfg.get('outside_penalty', 0.1),
        }
        return BoxSupervisedLoss(**_merged(defaults))
    elif supervision_type == 'cam':
        defaults = {'cam_threshold': cfg.get('cam_threshold', 0.5)}
        return CAMLoss(**_merged(defaults))
    elif supervision_type == 'mil':
        defaults = {'patch_size': cfg.get('patch_size', 32)}
        return MILLoss(**_merged(defaults))
    elif supervision_type == 'image_label':
        # Image-level multi-label CE — MIL with patch_size=image-size effectively.
        defaults = {'patch_size': cfg.get('patch_size', 32)}
        return MILLoss(**_merged(defaults))
    else:
        raise ValueError(
            f"Unknown supervision type: {supervision_type}. "
            f"Either set --supervision_type ∈ {{box, cam, mil, image_label}} "
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
    cam_generator=None,
    cfg=None
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
        if supervision_type in ('box', 'box_supervised', 'boxinst'):
            boxes = batch.get('boxes', None)
            box_classes = batch.get('box_classes', None)
            image_labels = batch.get('image_labels', None)
            target = batch.get('label', None)
            
            if image_labels is not None:
                image_labels = image_labels.to(device)
            if target is not None:
                target = target.to(device)
            
            # Dispatch based on actual loss type
            loss_name = cfg.get('training', {}).get('loss', {}).get('name', '')
            if loss_name == 'boxinst':
                # BoxInstLoss: (predictions, images=, boxes=)
                loss = criterion(predictions, images=images, boxes=boxes)
            else:
                # BoxSupervisedLoss: keyword arguments
                loss = criterion(
                    predictions,
                    boxes=boxes,
                    box_classes=box_classes,
                    image_labels=image_labels,
                    target=target,
                )
            
        elif supervision_type == 'cam':
            if cam_generator is not None:
                image_labels_raw = batch.get('image_labels', None)
                if image_labels_raw is not None:
                    # Convert multi-hot (B, num_classes) float to primary class index (B,) long
                    primary_classes = image_labels_raw.argmax(dim=1).long().to(device)
                else:
                    primary_classes = torch.zeros(images.shape[0], dtype=torch.long, device=device)
                cams = cam_generator.generate_batch_cam(
                    images, primary_classes
                )
                # generate_batch_cam returns (B, H_cam, W_cam) at feature-map resolution;
                # CAMLoss expects (B, C, H, W) matching predictions spatial size.
                num_cls = batch['image_labels'].shape[1]
                cams = cams.unsqueeze(1).expand(-1, num_cls, -1, -1)
                cams = F.interpolate(
                    cams, size=predictions.shape[-2:],
                    mode='bilinear', align_corners=False,
                ).to(device)
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

        else:
            # Generic dispatch for losses chosen via training.loss.name from
            # LOSS_REGISTRY. Different losses have different forward signatures,
            # so we handle each explicitly.
            loss_name = cfg.get('training', {}).get('loss', {}).get('name', '')
            ctx = {}
            for k, v in batch.items():
                if k == 'case_name':
                    continue
                if isinstance(v, torch.Tensor):
                    ctx[k] = v.to(device)

            try:
                if loss_name == 'dupl':
                    seg_a = predictions
                    seg_b = predictions
                    cam_a = ctx.pop('cam_a', predictions)
                    cam_b = ctx.pop('cam_b', predictions)
                    image_labels = ctx.pop('image_labels', None)
                    loss = criterion(seg_a, seg_b, cam_a, cam_b, image_labels)
                elif loss_name == 'mars':
                    cam_orig = ctx.pop('cam_orig', predictions)
                    cam_react = ctx.pop('cam_react', predictions)
                    erase_mask = ctx.pop('erase_mask', torch.zeros_like(predictions[:, :1]))
                    image_labels = ctx.pop('image_labels', None)
                    loss = criterion(cam_orig, cam_react, erase_mask, image_labels)
                elif loss_name == 'lpcam':
                    features = ctx.pop('features', predictions)
                    image_labels = ctx.pop('image_labels', None)
                    loss = criterion(features, image_labels)
                elif loss_name == 'tree_energy':
                    loss = criterion(predictions, batch['image'].to(device))
                elif loss_name == 'cam_loss':
                    image_labels = ctx.pop('image_labels', None)
                    target = ctx.pop('label', None)
                    if cam_generator is not None:
                        cams = cam_generator.generate_batch_cam(images, image_labels)
                    else:
                        cams = predictions
                    loss = criterion(predictions, cams, image_labels, target)
                elif loss_name == 'mil_loss':
                    image_labels = ctx.pop('image_labels', None)
                    loss = criterion(predictions, image_labels)
                elif loss_name == 'scribble_sup':
                    scribbles_list = batch.get('scribbles', None)
                    scribble_classes_list = batch.get('scribble_classes', None)
                    images_t = batch['image'].to(device)
                    if scribbles_list is not None and isinstance(scribbles_list, list):
                        # Convert scribble coordinates to a mask
                        B_s, _, H_s, W_s = predictions.shape
                        scribble_mask = torch.full((B_s, H_s, W_s), -1, dtype=torch.long, device=device)
                        for b_idx, scrib in enumerate(scribbles_list):
                            if isinstance(scrib, torch.Tensor) and scrib.numel() > 0 and scrib.dim() == 2:
                                x_coords = scrib[:, 0].clamp(0, W_s - 1).long().to(device)
                                y_coords = scrib[:, 1].clamp(0, H_s - 1).long().to(device)
                                if scribble_classes_list is not None and b_idx < len(scribble_classes_list):
                                    cls = scribble_classes_list[b_idx].long().to(device)
                                    scribble_mask[b_idx, y_coords, x_coords] = cls
                                else:
                                    scribble_mask[b_idx, y_coords, x_coords] = 0
                    else:
                        scribble_mask = torch.zeros(predictions.shape[0], predictions.shape[2], predictions.shape[3], dtype=torch.long, device=device)
                    loss = criterion(predictions, scribble_mask, images_t)
                elif loss_name == 'point_supervised':
                    points_list = batch.get('points', None)
                    point_classes_list = batch.get('point_classes', None)
                    images_t = batch['image'].to(device)
                    if points_list is not None and isinstance(points_list, list) and len(points_list) >= 1:
                        # Convert point coordinates to a mask
                        B_p, _, H_p, W_p = predictions.shape
                        point_mask = torch.full((B_p, H_p, W_p), -1, dtype=torch.long, device=device)
                        for b_idx, pts in enumerate(points_list):
                            if isinstance(pts, torch.Tensor) and pts.numel() > 0 and pts.dim() == 2:
                                x_coords = (pts[:, 0].clamp(0, W_p - 1).long())
                                y_coords = (pts[:, 1].clamp(0, H_p - 1).long())
                                if point_classes_list is not None and isinstance(point_classes_list, list) and b_idx < len(point_classes_list):
                                    cls = point_classes_list[b_idx].long().to(device)
                                else:
                                    cls = torch.zeros(pts.shape[0], dtype=torch.long, device=device)
                                point_mask[b_idx, y_coords, x_coords] = cls
                    else:
                        point_mask = torch.zeros(predictions.shape[0], predictions.shape[2], predictions.shape[3], dtype=torch.long, device=device)
                    loss = criterion(predictions, point_mask, images_t)
                elif loss_name in ('more', 'psdpm', 'recam', 'semples', 'toco'):
                    image_labels = ctx.pop('image_labels', None)
                    if cam_generator is not None:
                        features, cam_logits = cam_generator.extract_features_and_cam(
                            batch['image'].to(device), image_labels, upsample=False
                        )
                    else:
                        features = predictions
                        cam_logits = predictions

                    if loss_name == 'more':
                        B_f, D_f, H_f, W_f = features.shape
                        N = H_f * W_f
                        cam_flat = cam_logits.view(B_f, cam_logits.shape[1], -1)
                        cam_sum = cam_flat.sum(dim=2, keepdim=True).clamp_min(1e-6)
                        class_attn = cam_flat / cam_sum
                        patch_tokens = features.permute(0, 2, 3, 1).reshape(B_f, N, D_f)
                        loss = criterion(class_attn, patch_tokens, image_labels, cam_logits=cam_logits)
                    elif loss_name == 'psdpm':
                        pred_interp = F.interpolate(predictions, size=cam_logits.shape[-2:], mode='bilinear', align_corners=False)
                        bg_channel = torch.zeros_like(pred_interp[:, :1])
                        pred_with_bg = torch.cat([bg_channel, pred_interp], dim=1)
                        loss = criterion(pred_with_bg, features, cam_logits, image_labels)
                    elif loss_name == 'recam':
                        loss = criterion(cam_logits, features, image_labels)
                    elif loss_name == 'semples':
                        B_f, D_f, H_f, W_f = features.shape
                        C_lab = image_labels.shape[1]
                        cam_norm = torch.sigmoid(cam_logits)
                        class_image_emb = torch.einsum('bchw,bdhw->bcd', cam_norm, features)
                        denom = cam_norm.sum(dim=(2, 3)).unsqueeze(-1).clamp_min(1e-6)
                        class_image_emb = class_image_emb / denom
                        cls_prompts = class_image_emb.mean(dim=0)
                        cam_learn = cam_logits
                        cam_teacher = cam_logits.detach()
                        bg_prompts = torch.randn_like(cls_prompts)
                        loss = criterion(class_image_emb, cls_prompts, cam_learn, cam_teacher, image_labels, bg_prompts=bg_prompts)
                    elif loss_name == 'toco':
                        B_f, D_f, H_f, W_f = features.shape
                        N = H_f * W_f
                        C_tok = cam_logits.shape[1]
                        patch_tokens = features.permute(0, 2, 3, 1).reshape(B_f, N, D_f)
                        cam_tokens = cam_logits.view(B_f, C_tok, N)
                        cls_token_full = features.mean(dim=(2, 3))
                        cam_mask = (cam_logits > 0.5).float()
                        cam_mask_exp = cam_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
                        cls_token_masked = (features * cam_mask_exp).mean(dim=(2, 3))
                        loss = criterion(patch_tokens, cam_tokens, cls_token_full, cls_token_masked, image_labels)
                elif loss_name in ('advcam_loss', 'mctformer_loss'):
                    # These losses all take (predictions, image_labels, ...)
                    image_labels = ctx.pop('image_labels', None)
                    loss = criterion(predictions, image_labels)
                elif loss_name == 'puzzle_cam':
                    # PuzzleCAMLoss: (features_full, features_tiled_merged,
                    #                 predictions_full, predictions_tiled, image_labels)
                    image_labels = ctx.pop('image_labels', None)
                    cls_preds = F.adaptive_avg_pool2d(predictions, 1).flatten(1)
                    loss = criterion(predictions, predictions, cls_preds, cls_preds, image_labels)
                elif loss_name == 'seam_loss':
                    # SEAMLoss: (cam1_raw, cam_rv1_raw, cam2_raw, cam_rv2_raw, image_labels)
                    # SEAM expects C+1 channels (bg + classes)
                    image_labels = ctx.pop('image_labels', None)
                    bg = torch.zeros_like(predictions[:, :1])
                    cams_with_bg = torch.cat([bg, predictions], dim=1)
                    loss = criterion(cams_with_bg, cams_with_bg, cams_with_bg, cams_with_bg, image_labels)
                else:
                    target = ctx.pop('label', None)
                    loss = criterion(predictions, target, **ctx) if target is not None else criterion(predictions, **ctx)
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
    has_labels = True

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch.get('label', None)
        if labels is None:
            has_labels = False
            continue
        labels = labels.numpy()

        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        preds = outputs.argmax(dim=1).cpu().numpy()

        for pred, target in zip(preds, labels):
            metrics = compute_metrics(pred, target, num_classes)
            dice_vals = list(metrics['dice'].values())
            if dice_vals:
                all_dice.append(np.mean(dice_vals))

    if not has_labels:
        return 0.0
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
    with open(args.config, 'r', encoding='utf-8') as f:
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
    num_classes = cfg.get('model', {}).get('num_classes', 9)
    train_dataset = WeaklySupervisedDataset(
        train_dataset, args.supervision_type, num_classes, data_cfg)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('training', {}).get('batch_size', 8),
        shuffle=True,
        num_workers=cfg.get('training', {}).get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=weak_collate_fn,
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
            collate_fn=weak_collate_fn,
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
        lr=float(opt_cfg.get('lr', 1e-4)),
        weight_decay=float(opt_cfg.get('weight_decay', 1e-4))
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
    needs_cam_gen = args.supervision_type == 'cam'
    loss_name_cfg = cfg.get('training', {}).get('loss', {}).get('name', '')
    if loss_name_cfg in ('more', 'psdpm', 'recam', 'semples', 'toco'):
        needs_cam_gen = True
    if needs_cam_gen:
        possible_layers = ['encoder.model.layer4', 'encoder.layer4', 'model.layer4', 'layer4']
        target_layer = None
        for layer in possible_layers:
            if any(layer == n for n, _ in model.named_modules()):
                target_layer = layer
                break
        if target_layer is None:
            for name, _ in model.named_modules():
                if 'layer4' in name or 'stage4' in name:
                    target_layer = name
        if target_layer is None:
            raise ValueError("Could not find a suitable target_layer for CAM.")
        cam_generator = CAMGenerator(model, target_layer=target_layer)
        logger.info(f"CAMGenerator using target_layer: {target_layer}")
    
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
            cam_generator,
            cfg
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
