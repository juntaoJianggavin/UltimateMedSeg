"""Testing/inference script for segmentation models.

Standard usage (single model)::

    python test.py --config cfg.yaml --checkpoint best.pth

Multi-checkpoint logit-averaging ensemble (inference-time)::

    python test.py --config cfg.yaml \\
        --checkpoint best_a.pth best_b.pth best_c.pth \\
        --ensemble-weights 0.5 0.3 0.2 \\
        --ensemble-average logit

If your config defines ``model.type: ensemble`` with multiple ``members``,
``--checkpoint`` may either be **one path per member** (loaded as
``model_state_dict``) or **one path** holding the full ensemble state.

Test-Time Augmentation::

    python test.py --config cfg.yaml --checkpoint best.pth \\
        --tta \\
        --tta-augs identity rot90 rot180 rot270 hflip vflip \\
                    brightness_up brightness_down \\
        --tta-merge mean

TTA can be combined with ensemble; ensemble runs first, then TTA wraps the
ensemble.
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medseg.model_builder import build_model
from medseg.datasets import SynapseDataset, GenericDataset, get_val_transforms
from medseg.utils.metrics import compute_metrics
from medseg.inference import (
    EnsembleModel,
    TTAWrapper,
    AVAILABLE_TTAS,
    load_ensemble_from_checkpoints,
)

import medseg.models.encoders
import medseg.models.decoders
import medseg.models.skip_connections
import medseg.models.bottlenecks
import medseg.losses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_test_dataset(data_cfg):
    """Build test dataset."""
    dataset_type = data_cfg.get('type', 'synapse')
    img_size = data_cfg.get('img_size', 224)
    transform = get_val_transforms(img_size)

    if dataset_type == 'synapse':
        return SynapseDataset(
            root_dir=data_cfg['test_dir'],
            split='test',
            list_file=data_cfg.get('test_list', None),
            transform=transform,
            img_size=img_size,
        )
    elif dataset_type in ('image_mask', 'binary'):
        return GenericDataset(
            root_dir=data_cfg.get('test_dir', data_cfg.get('root_dir')),
            split='test',
            transform=transform,
            img_size=img_size,
            img_suffix=data_cfg.get('img_suffix', '.png'),
            mask_suffix=data_cfg.get('mask_suffix', '.png'),
            file_list=data_cfg.get('test_list', None),
        )
    elif dataset_type in ('synapse', 'acdc'):
        return SynapseDataset(
            root_dir=data_cfg['test_dir'],
            split='test',
            list_file=data_cfg.get('test_list', None),
            transform=transform,
            img_size=img_size,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# ---------------------------------------------------------------------------
def _load_single_checkpoint(model: torch.nn.Module, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    model.load_state_dict(state)
    return model


def build_inference_model(cfg, args):
    """Build the model used for inference, optionally wrapping with
    EnsembleModel + TTAWrapper according to CLI args.

    Logic:
    1. If config declares ``model.type == ensemble`` and only one
       ``--checkpoint`` is given, treat that ckpt as the full state of the
       built ensemble.
    2. If config declares a single model and N checkpoints are given,
       N copies are built from the same config and loaded individually.
    3. Otherwise the standard single-model flow is taken.
    """
    model_cfg = cfg.get('model', {})
    is_cfg_ensemble = model_cfg.get('type', None) in ('ensemble', 'logit_ensemble')
    n_ckpts = len(args.checkpoint)

    if not is_cfg_ensemble and n_ckpts > 1:
        # Replicate config N times to form an inference-time ensemble
        member_cfgs = [model_cfg] * n_ckpts
        weights = args.ensemble_weights if args.ensemble_weights else None
        model = load_ensemble_from_checkpoints(
            member_cfgs=member_cfgs,
            checkpoints=args.checkpoint,
            build_one_fn=build_model,
            weights=weights,
            average=args.ensemble_average,
            map_location=args.device,
            strict=not args.no_strict,
        )
        logger.info(
            f"Built inference-time ensemble of {n_ckpts} models from "
            f"replicated config; average='{args.ensemble_average}'."
        )
    else:
        model = build_model(cfg)
        if is_cfg_ensemble and n_ckpts == len(model.models):
            # Per-member checkpoints
            for sub, ckpt in zip(model.models, args.checkpoint):
                _load_single_checkpoint(sub, ckpt, args.device)
                logger.info(f"Loaded ensemble member from {ckpt}")
            if args.ensemble_weights:
                w = torch.tensor(args.ensemble_weights, dtype=torch.float32)
                w = w / w.sum().clamp(min=1e-8)
                model.weights.copy_(w)
            model.freeze_sub_models()
        elif n_ckpts >= 1:
            _load_single_checkpoint(model, args.checkpoint[0], args.device)
            logger.info(f"Model loaded from {args.checkpoint[0]}")

    if args.tta:
        model = TTAWrapper(
            model,
            augmentations=args.tta_augs,
            merge=args.tta_merge,
            brightness_delta=args.tta_brightness_delta,
            contrast_delta=args.tta_contrast_delta,
            gamma_delta=args.tta_gamma_delta,
            ignore_unknown=True,
        )
        logger.info(
            f"TTA enabled: augs={model.augmentations} merge='{model.merge}'"
        )

    return model.to(args.device)


# ---------------------------------------------------------------------------
@torch.no_grad()
def test(model, dataloader, num_classes, device, save_dir=None):
    """Test model and compute metrics."""
    model.eval()
    all_metrics = {'dice': {}, 'iou': {}, 'hd95': {}}
    case_results = []

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].numpy()
        case_names = batch['case_name']

        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        preds = outputs.argmax(dim=1).cpu().numpy()

        for pred, target, case_name in zip(preds, labels, case_names):
            metrics = compute_metrics(pred, target, num_classes)

            result = {'case_name': case_name}
            for metric_name in ['dice', 'iou', 'hd95']:
                vals = metrics[metric_name]
                for c, v in vals.items():
                    if c not in all_metrics[metric_name]:
                        all_metrics[metric_name][c] = []
                    all_metrics[metric_name][c].append(v)
                result[metric_name] = {c: v for c, v in vals.items()}
            case_results.append(result)

            # Optionally save predictions
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, f'{case_name}_pred.npy'), pred)

    # Summary
    logger.info("=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    for metric_name in ['dice', 'iou', 'hd95']:
        logger.info(f"\n{metric_name.upper()}:")
        total_vals = []
        for c in sorted(all_metrics[metric_name].keys()):
            vals = all_metrics[metric_name][c]
            valid_vals = [v for v in vals if not np.isnan(v)]
            mean_val = np.mean(valid_vals) if valid_vals else float('nan')
            logger.info(f"  Class {c}: {mean_val:.4f}")
            total_vals.extend(valid_vals)
        if total_vals:
            logger.info(f"  Mean: {np.mean(total_vals):.4f}")
    logger.info("=" * 60)

    return case_results


def main():
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument(
        '--checkpoint', type=str, nargs='+', required=True,
        help='One or more checkpoint paths. Multiple paths trigger '
             'inference-time ensemble (logit averaging).',
    )
    parser.add_argument('--output_dir', type=str, default='./test_output')
    parser.add_argument('--save_pred', action='store_true', help='Save predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-strict', action='store_true',
                        help='Disable strict load_state_dict for ensemble members.')

    # Ensemble (inference-time) options
    ens = parser.add_argument_group('ensemble (inference-time)')
    ens.add_argument('--ensemble-weights', type=float, nargs='*', default=None,
                     help='Per-checkpoint weights; default: equal weights.')
    ens.add_argument('--ensemble-average', type=str, default='logit',
                     choices=['logit', 'softmax', 'sigmoid'],
                     help='Logit averaging mode.')

    # TTA options
    tta_grp = parser.add_argument_group('test-time augmentation')
    tta_grp.add_argument('--tta', action='store_true', help='Enable TTA.')
    tta_grp.add_argument(
        '--tta-augs', nargs='*', default=None,
        help=f'Augmentation names. Available: {list(AVAILABLE_TTAS)}. '
             f'Default: identity + 3×rot90 + hflip + vflip + brightness ±.',
    )
    tta_grp.add_argument('--tta-merge', type=str, default='mean',
                         choices=['mean', 'gmean', 'max', 'median'])
    tta_grp.add_argument('--tta-brightness-delta', type=float, default=0.05)
    tta_grp.add_argument('--tta-contrast-delta', type=float, default=0.10)
    tta_grp.add_argument('--tta-gamma-delta', type=float, default=0.10)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    model = build_inference_model(cfg, args)

    # Build test dataset
    data_cfg = cfg.get('data', {})
    test_dataset = build_test_dataset(data_cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.get('training', {}).get('batch_size', 1),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    logger.info(f"Test dataset: {len(test_dataset)} samples")

    save_dir = os.path.join(args.output_dir, 'predictions') if args.save_pred else None
    test(model, test_loader, cfg['model']['num_classes'], args.device, save_dir)


if __name__ == '__main__':
    main()
