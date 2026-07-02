"""Training script for text-guided segmentation models.

Supports:
- TextPromptUNet (cross-attention + feature modulation)
- SemanticGuidedUNet (class embeddings + multi-scale attention)
- CLIP text-guided models

Usage:
    python train_text_guided.py --config configs/training_paradigms/text_guided/synapse_clip.yaml
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medseg.model_builder import build_model
from medseg.text_guided import TextPromptUNet, SemanticGuidedUNet
from medseg.registry import LOSS_REGISTRY
from medseg.datasets import SynapseDataset, GenericDataset, get_train_transforms, get_val_transforms
from medseg.utils.metrics import compute_metrics

# Import all modules to trigger registration
import medseg.models.encoders
import medseg.models.decoders
import medseg.models.bottlenecks
import medseg.losses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextGuidedDatasetWrapper:
    """Wrapper to add text/class information to datasets.

    Adds class names and text prompts to each batch for text-guided models.
    When text_tokenizer and text_encoder are provided (for LViT etc.), encodes
    per-image captions into BERT embeddings.
    """

    def __init__(self, dataset, class_names, prompt_mode='learnable',
                 text_tokenizer=None, text_encoder=None, text_max_length=10,
                 text_embed_dim=768, device='cpu'):
        self.dataset = dataset
        self.class_names = class_names
        self.prompt_mode = prompt_mode
        self.num_classes = len(class_names)
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        self.text_max_length = text_max_length
        self.text_embed_dim = text_embed_dim
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Add text information
        item['class_names'] = self.class_names
        item['num_classes'] = self.num_classes

        # Encode per-image text caption into BERT embeddings (for LViT etc.)
        if self.text_tokenizer is not None and self.text_encoder is not None:
            caption = item.get('text', '')
            if not caption:
                # Fallback to class name as caption
                caption = self.class_names[0] if self.class_names else 'foreground'
            inputs = self.text_tokenizer(
                caption, return_tensors='pt', padding='max_length',
                max_length=self.text_max_length, truncation=True
            )
            # Move tokenizer inputs to the encoder's device
            encoder_device = next(self.text_encoder.parameters()).device
            inputs = {k: v.to(encoder_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                # Use last hidden state: (1, L, D) -> (L, D)
                embeddings = outputs.last_hidden_state.squeeze(0)  # (L, D)
            item['text'] = embeddings

        return item


def build_text_guided_model(cfg, device):
    """Build text-guided segmentation model.

    Supports both standalone text-guided models and hybrid approaches.
    """
    model_cfg = cfg.get('model', {})
    text_cfg = model_cfg.get('text_guided', {})

    # Get encoder channels from the encoder config (must match actual encoder output)
    enc_cfg = model_cfg.get('encoder', {})
    encoder_channels = enc_cfg.get('params', {}).get(
        'out_channels', text_cfg.get('encoder_channels')
    )
    if encoder_channels is None:
        raise ValueError(
            "Cannot determine encoder_channels. Set model.encoder.params.out_channels "
            "(e.g. [128, 256, 512, 1024]) or model.text_guided.encoder_channels."
        )
    class_names = text_cfg.get('class_names', ['background', 'foreground'])
    prompt_mode = text_cfg.get('prompt_mode', 'learnable')
    embed_dim = text_cfg.get('embed_dim', 512)
    num_classes = len(class_names)

    # Build text-guided model
    model_type = text_cfg.get('model_type', 'TextPromptUNet')

    if model_type == 'TextPromptUNet':
        model = TextPromptUNet(
            num_classes=num_classes,
            class_names=class_names,
            encoder_channels=encoder_channels,
            prompt_mode=prompt_mode,
            embed_dim=embed_dim,
        )
    elif model_type == 'SemanticGuidedUNet':
        model = SemanticGuidedUNet(
            num_classes=num_classes,
            encoder_channels=encoder_channels,
            embed_dim=embed_dim,
        )
    else:
        raise ValueError(f"Unknown text-guided model type: {model_type}")

    model = model.to(device)
    logger.info(f"Text-guided model built: {model_type}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    logger.info(f"Class names: {class_names}")
    logger.info(f"Prompt mode: {prompt_mode}")

    return model


def build_encoder(cfg, device):
    """Build standalone encoder for text-guided models."""
    model_cfg = cfg.get('model', {})
    text_cfg = model_cfg.get('text_guided', {})

    # Build encoder using model_builder
    encoder_cfg = {
        'model': {
            'encoder': model_cfg.get('encoder', {'name': 'timm_resnet50', 'pretrained': True}),
            'bottleneck': {'name': 'none'},
            'decoder': {'name': 'none'},  # We only need encoder
            'num_classes': 1,
        }
    }

    from medseg.registry import ENCODER_REGISTRY
    enc_cfg = encoder_cfg['model']['encoder']
    enc_cls = ENCODER_REGISTRY.get(enc_cfg['name'])
    encoder = enc_cls(
        pretrained=enc_cfg.get('pretrained', False),
        in_channels=enc_cfg.get('in_channels', 3),
        img_size=cfg.get('data', {}).get('img_size', 224),
        **enc_cfg.get('params', {}),
    )
    encoder = encoder.to(device)
    encoder.eval()  # Encoder is typically frozen or pre-extracted

    logger.info(f"Encoder built: {enc_cfg['name']}")
    return encoder


@torch.no_grad()
def extract_features(encoder, images, device):
    """Extract multi-scale features from encoder.

    Returns list of feature tensors at different resolutions.
    """
    encoder.eval()
    images = images.to(device)
    features = encoder(images)

    # Ensure features is a list
    if not isinstance(features, (list, tuple)):
        features = [features]

    return features


def train_one_epoch_text_guided(
    model, encoder, dataloader, criterion, optimizer, device, epoch,
    use_cached_encoder=False, is_text_unet=False
):
    """Train text-guided model for one epoch.

    Args:
        model: Text-guided decoder model (or end-to-end text_unet model)
        encoder: Feature encoder (None for text_unet models)
        dataloader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epoch: Current epoch
        use_cached_encoder: Whether to cache encoder features
        is_text_unet: True for languide/lvit/etc that take (image, text)
    """
    model.train()
    if encoder is not None and not use_cached_encoder:
        encoder.train()

    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        if is_text_unet:
            # Text-guided end-to-end models: forward(image, text)
            text = batch.get('text', None)
            if text is not None:
                if isinstance(text, torch.Tensor):
                    text = text.to(device)
                outputs = model(images, text=text)
            else:
                # Model will use _default_text_prompts or raise
                outputs = model(images)
        else:
            # TextPromptUNet path: extract encoder features
            if encoder is not None:
                if use_cached_encoder:
                    encoder.eval()
                encoder_features = extract_features(encoder, images, device)
            else:
                encoder_features = None
            outputs = model(encoder_features)

        # Upsample predictions to match target resolution
        if outputs.shape[-2:] != labels.shape[-2:]:
            outputs = F.interpolate(outputs, size=labels.shape[-2:],
                                    mode='bilinear', align_corners=False)

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
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
def validate_text_guided(model, encoder, dataloader, num_classes, device,
                         is_text_unet=False):
    """Validate text-guided model."""
    model.eval()
    if encoder is not None:
        encoder.eval()

    all_dice = []

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].numpy()

        if is_text_unet:
            text = batch.get('text', None)
            if text is not None and isinstance(text, torch.Tensor):
                text = text.to(device)
            outputs = model(images, text=text)
        else:
            if encoder is not None:
                encoder_features = extract_features(encoder, images, device)
            else:
                encoder_features = None
            outputs = model(encoder_features)

        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        # Upsample predictions to match target resolution
        if outputs.shape[-2:] != images.shape[-2:]:
            outputs = F.interpolate(outputs, size=images.shape[-2:],
                                    mode='bilinear', align_corners=False)

        preds = outputs.argmax(dim=1).cpu().numpy()

        # Compute metrics
        for pred, target in zip(preds, labels):
            metrics = compute_metrics(pred, target, num_classes)
            dice_vals = list(metrics['dice'].values())
            if dice_vals:
                all_dice.append(np.mean(dice_vals))

    return np.mean(all_dice) if all_dice else 0.0


def main():
    parser = argparse.ArgumentParser(description='Train text-guided segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--output_dir', type=str, default='./output_text_guided', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder during training')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    # Build text-guided model — route through the correct builder
    model_cfg = cfg.get('model', {})
    arch_name = model_cfg.get('architecture', '').lower()

    # Check if this is a text+UNet architecture (languide, lvit, tganet, cris, etc.)
    from medseg.models.networks import _TEXT_UNET_ARCHS, build_special_arch

    if arch_name in _TEXT_UNET_ARCHS:
        # Text-guided end-to-end models with their own encoders
        model = build_special_arch(arch_name, cfg)
        model = model.to(device)
        encoder = None  # These models have built-in encoders
        use_external_encoder = False
        # Extract class_names from arch_params or text_guided for these models
        arch_params = model_cfg.get('arch_params', {})
        text_cfg = model_cfg.get('text_guided', {})
        class_names = text_cfg.get('class_names', arch_params.get('class_names', ['background', 'foreground']))
        prompt_mode = 'learnable'
        is_text_unet = True
        logger.info(f"Built text-guided architecture: {arch_name}")
    elif 'text_guided' in model_cfg:
        # TextPromptUNet / SemanticGuidedUNet path
        model = build_text_guided_model(cfg, device)
        text_cfg = model_cfg.get('text_guided', {})
        use_external_encoder = text_cfg.get('use_external_encoder', True)
        encoder = None
        if use_external_encoder:
            encoder = build_encoder(cfg, device)
            if args.freeze_encoder:
                for param in encoder.parameters():
                    param.requires_grad = False
                logger.info("Encoder frozen")
        class_names = text_cfg.get('class_names', ['background', 'foreground'])
        prompt_mode = text_cfg.get('prompt_mode', 'learnable')
        is_text_unet = False
    else:
        raise ValueError(
            f"Cannot build text-guided model: no 'architecture' in _TEXT_UNET_ARCHS "
            f"and no 'text_guided' section. arch={arch_name!r}"
        )

    # Build datasets
    data_cfg = cfg.get('data', {})

    # Import dataset builder from train.py
    from train import build_dataset

    # For text_unet models that need per-image text (LViT), build BERT encoder first
    arch_params = model_cfg.get('arch_params', {})
    _bert_tokenizer = None
    _bert_encoder = None
    _text_max_len = data_cfg.get('text_max_length', arch_params.get('text_len', 10))
    _text_embed_dim = arch_params.get('text_embed_dim', 768)

    if is_text_unet and arch_name == 'lvit':
        from transformers import AutoTokenizer, AutoModel
        tokenizer_name = data_cfg.get('tokenizer_name', 'bert-base-uncased')
        logger.info(f"Loading BERT tokenizer: {tokenizer_name}")
        _bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Loading BERT text encoder for embeddings (dim={_text_embed_dim})")
        _bert_encoder = AutoModel.from_pretrained(tokenizer_name)
        _bert_encoder.eval()
        _bert_encoder = _bert_encoder.to(device)
        for p in _bert_encoder.parameters():
            p.requires_grad = False

    train_dataset = build_dataset(data_cfg, 'train')
    train_dataset = TextGuidedDatasetWrapper(
        train_dataset, class_names, prompt_mode,
        text_tokenizer=_bert_tokenizer, text_encoder=_bert_encoder,
        text_max_length=_text_max_len, text_embed_dim=_text_embed_dim,
        device=str(device)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('training', {}).get('batch_size', 8),
        shuffle=True,
        num_workers=cfg.get('training', {}).get('num_workers', 4),
        pin_memory=not (_bert_encoder is not None),  # Disable pin when BERT embeddings already on GPU
        drop_last=True,
    )
    logger.info(f"Train dataset: {len(train_dataset)} samples")

    val_loader = None
    if 'val_dir' in data_cfg or 'test_dir' in data_cfg:
        val_key = 'val' if 'val_dir' in data_cfg else 'test'
        val_dataset = build_dataset(data_cfg, val_key)
        val_dataset = TextGuidedDatasetWrapper(
            val_dataset, class_names, prompt_mode,
            text_tokenizer=_bert_tokenizer, text_encoder=_bert_encoder,
            text_max_length=_text_max_len, text_embed_dim=_text_embed_dim,
            device=str(device)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get('training', {}).get('batch_size', 8),
            shuffle=False,
            num_workers=cfg.get('training', {}).get('num_workers', 4),
            pin_memory=not (_bert_encoder is not None),  # Disable pin when BERT embeddings already on GPU
        )
        logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Build optimizer
    training_cfg = cfg.get('training', {})
    opt_cfg = training_cfg.get('optimizer', {'name': 'adamw', 'lr': 1e-4})

    # Only optimize model parameters (not encoder if frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if encoder is not None and not args.freeze_encoder:
        trainable_params.extend([p for p in encoder.parameters() if p.requires_grad])

    if not trainable_params:
        raise ValueError(
            f"No trainable parameters found for {arch_name!r}. "
            f"If this is a frozen MLLM (e.g. MediSee), it cannot be trained "
            f"end-to-end; use the inference pipeline instead."
        )

    optimizer = AdamW(
        trainable_params,
        lr=float(opt_cfg.get('lr', 1e-4)),
        weight_decay=float(opt_cfg.get('weight_decay', 1e-4))
    )

    # Build scheduler
    total_epochs = training_cfg.get('epochs', 200)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # Build loss
    loss_cfg = training_cfg.get('loss', {'name': 'compound'})
    criterion = LOSS_REGISTRY.get(loss_cfg['name'])(**loss_cfg.get('params', {}))

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if encoder is not None and 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info(f"Starting training for {total_epochs} epochs")
    logger.info(f"Optimizer: {opt_cfg.get('name', 'adamw')}, LR: {opt_cfg.get('lr', 1e-4)}")

    for epoch in range(start_epoch, total_epochs):
        # Train
        train_loss = train_one_epoch_text_guided(
            model, encoder, train_loader, criterion, optimizer, device, epoch,
            use_cached_encoder=args.freeze_encoder,
            is_text_unet=is_text_unet
        )

        # Update LR
        if scheduler is not None:
            scheduler.step()

        logger.info(f"Epoch [{epoch}/{total_epochs}] Train Loss: {train_loss:.4f}")

        # Validate
        if val_loader is not None and (epoch + 1) % training_cfg.get('val_interval', 10) == 0:
            val_dice = validate_text_guided(
                model, encoder, val_loader, len(class_names), device,
                is_text_unet=is_text_unet
            )
            logger.info(f"Epoch [{epoch}/{total_epochs}] Val Dice: {val_dice:.4f}")

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                save_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'encoder_state_dict': encoder.state_dict() if encoder is not None else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'class_names': class_names,
                }, save_path)
                logger.info(f"Saved best model (Dice: {best_dice:.4f})")

        # Save checkpoint
        if (epoch + 1) % training_cfg.get('save_interval', 50) == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': encoder.state_dict() if encoder is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, save_path)

    logger.info(f"Training completed. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
