#!/usr/bin/env python3
"""Test all configs: create dummy data, build model, forward, compute loss.

Usage:
    cd segmentation_tool/
    python scripts/test_all_configs.py [--category semi|da|distillation|weak|foundation|text_guided]
    python scripts/test_all_configs.py --all

This script:
  1. Creates a minimal dummy dataset (4 images × 3 splits).
  2. For each yaml config in the chosen category:
     a. Patches data paths to point at the dummy data.
     b. Builds the model via the project's build_model / build_special_arch.
     c. Runs one forward pass on a random batch.
     d. Computes loss and calls .backward().
     e. Reports OK / FAIL with the error message.

No GPU required — runs on CPU with tiny tensors.
"""

import argparse
import os
import sys
import glob
import traceback
import yaml
import numpy as np
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def create_dummy_data():
    """Create minimal dummy datasets that all configs can point to."""
    base = ROOT / "data" / "_test_dummy"
    for split in ("train", "val", "test"):
        img_dir = base / split / "images"
        mask_dir = base / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            img_path = img_dir / f"img_{i:04d}.png"
            mask_path = mask_dir / f"img_{i:04d}.png"
            if not img_path.exists():
                from PIL import Image
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(img).save(str(img_path))
                mask = np.random.randint(0, 4, (64, 64), dtype=np.uint8)
                Image.fromarray(mask).save(str(mask_path))

    # Symlink common aliases
    for alias in ("YourDataset", "test_dummy", "source", "target", "target_val"):
        link = ROOT / "data" / alias
        if not link.exists():
            try:
                link.symlink_to(base)
            except OSError:
                pass

    print(f"[OK] Dummy data at {base} (4 images × 3 splits, 64×64)")
    return str(base)


def patch_data_paths(cfg, dummy_root):
    """Recursively replace all data directory paths with the dummy root."""
    if isinstance(cfg, dict):
        for k, v in list(cfg.items()):
            if isinstance(v, str) and ("./data/" in v or "data/" in v):
                # Replace with dummy path
                if "image" in k or "img" in k:
                    cfg[k] = os.path.join(dummy_root, "train", "images")
                elif "mask" in k:
                    cfg[k] = os.path.join(dummy_root, "train", "masks")
                elif "train" in k:
                    cfg[k] = os.path.join(dummy_root, "train")
                elif "val" in k:
                    cfg[k] = os.path.join(dummy_root, "val")
                elif "test" in k:
                    cfg[k] = os.path.join(dummy_root, "test")
                elif "root" in k or "dir" in k:
                    cfg[k] = dummy_root
                else:
                    cfg[k] = dummy_root
            elif isinstance(v, (dict, list)):
                patch_data_paths(v, dummy_root)
    elif isinstance(cfg, list):
        for item in cfg:
            patch_data_paths(item, dummy_root)


def test_foundation_config(yaml_path, dummy_root):
    """Test a foundation / architecture config: build model + forward + loss."""
    import torch
    import warnings
    warnings.filterwarnings('ignore')

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return "SKIP", "empty yaml"

    patch_data_paths(cfg, dummy_root)

    model_cfg = cfg.get("model", cfg)
    num_classes = model_cfg.get("num_classes", 2)
    img_size = model_cfg.get("img_size", 64)
    if img_size == "native" or img_size is None:
        img_size = 64

    # Build model
    from medseg.model_builder import build_model
    model = build_model(cfg)
    model.eval()

    # Forward pass
    B = 2
    x = torch.randn(B, 3, img_size, img_size)

    # Check if text-guided
    is_text_guided = getattr(model, 'is_text_guided', False)
    if is_text_guided:
        # Try forward without text first (some models support it)
        try:
            out = model(x)
        except (ValueError, TypeError):
            # Needs text — create dummy text
            out = model(x, text=None)
    else:
        out = model(x)

    if isinstance(out, (list, tuple)):
        out = out[0]

    # Compute loss
    from medseg.registry import LOSS_REGISTRY
    loss_cfg = cfg.get("training", {}).get("loss", {"name": "compound"})
    loss_name = loss_cfg.get("name", "compound")
    loss_params = loss_cfg.get("params", {}) or {}
    loss_fn = LOSS_REGISTRY.get(loss_name)(**loss_params)

    target = torch.randint(0, max(num_classes, 2), (B, img_size, img_size))

    # Resize output if needed
    if out.shape[2:] != target.shape[1:]:
        import torch.nn.functional as F
        out = F.interpolate(out, size=target.shape[1:], mode='bilinear', align_corners=False)

    loss = loss_fn(out, target)
    loss.backward()

    return "OK", f"loss={float(loss):.4f}, out_shape={tuple(out.shape)}"


def test_semi_config(yaml_path, dummy_root):
    """Test a semi config: build model + build_semi_method + one train_step."""
    import torch
    import warnings
    warnings.filterwarnings('ignore')

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        return "SKIP", "empty yaml"

    patch_data_paths(cfg, dummy_root)

    model_cfg = cfg.get("model", cfg)
    num_classes = model_cfg.get("num_classes", 2)
    img_size = model_cfg.get("img_size", 64)
    if img_size == "native" or img_size is None:
        img_size = 64

    # Force small size for speed
    if img_size > 128:
        img_size = 64

    from medseg.model_builder import build_model
    model = build_model(cfg)
    model.train()
    device = torch.device('cpu')

    # Build semi method
    from medseg.training.semi import build_semi_method
    semi_cfg = cfg.get("semi", {})
    semi_method = build_semi_method(semi_cfg, model, device, img_size=img_size)

    # Build loss
    from medseg.registry import LOSS_REGISTRY
    train_cfg = cfg.get("training", {})
    loss_cfg = train_cfg.get("loss", {"name": "compound"})
    loss_name = loss_cfg.get("name", "compound")
    loss_params = loss_cfg.get("params", {}) or {}
    criterion = LOSS_REGISTRY.get(loss_name)(**loss_params)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Fake batch
    B = 2
    labeled_batch = {
        'image': torch.randn(B, 3, img_size, img_size),
        'label': torch.randint(0, num_classes, (B, img_size, img_size)),
    }
    unlabeled_batch = {
        'image': torch.randn(B, 3, img_size, img_size),
    }

    # One train step
    loss_dict = semi_method.train_step(
        labeled_batch, unlabeled_batch,
        criterion, optimizer, epoch=0, total_epochs=10
    )

    return "OK", f"loss={loss_dict['loss']:.4f}, sup={loss_dict['sup_loss']:.4f}"


def test_config(yaml_path, dummy_root, category):
    """Route to the appropriate test function."""
    try:
        if category in ("semi",):
            return test_semi_config(yaml_path, dummy_root)
        else:
            return test_foundation_config(yaml_path, dummy_root)
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {str(e)[:200]}"


def get_configs(category):
    """Get list of yaml configs for a category."""
    dirs = {
        "semi": "configs/training_paradigms/semi_supervision",
        "da": "configs/training_paradigms/domain_adaptation",
        "distillation": "configs/training_paradigms/distillation",
        "weak": "configs/training_paradigms/weak_supervision",
        "foundation": "configs/architectures/foundation",
        "text_guided": "configs/training_paradigms/text_guided",
    }
    if category == "all":
        paths = []
        for d in dirs.values():
            paths.extend(sorted(glob.glob(f"{d}/**/*.yaml", recursive=True)))
        return paths
    d = dirs.get(category)
    if d is None:
        print(f"Unknown category: {category}. Choose from: {list(dirs.keys())} or 'all'")
        sys.exit(1)
    return sorted(glob.glob(f"{d}/**/*.yaml", recursive=True))


def main():
    parser = argparse.ArgumentParser(description="Test all configs produce a loss")
    parser.add_argument("--category", type=str, default="foundation",
                        help="Category: semi|da|distillation|weak|foundation|text_guided|all")
    parser.add_argument("--all", action="store_true", help="Test all categories")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.category = "all"

    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')

    # Create dummy data
    dummy_root = create_dummy_data()

    # Import medseg to trigger registrations
    import medseg.models.encoders
    import medseg.models.decoders
    import medseg.models.skip_connections
    import medseg.models.bottlenecks
    import medseg.losses

    configs = get_configs(args.category)
    print(f"\nTesting {len(configs)} configs (category={args.category})...\n")

    results = {"OK": [], "FAIL": [], "SKIP": []}

    for i, path in enumerate(configs, 1):
        rel = os.path.relpath(path)
        cat = "semi" if "/semi/" in path else (
            "da" if "/domain_adaptation/" in path else (
            "distillation" if "/distillation/" in path else (
            "weak" if "/weak_supervision/" in path else "foundation")))

        status, msg = test_config(path, dummy_root, cat)
        results[status].append((rel, msg))

        icon = {"OK": "✓", "FAIL": "✗", "SKIP": "—"}[status]
        line = f"  [{icon}] {rel}"
        if status == "FAIL" or args.verbose:
            line += f"  ({msg})"
        print(line)

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results['OK'])} OK, {len(results['FAIL'])} FAIL, {len(results['SKIP'])} SKIP")
    print(f"{'='*60}")

    if results['FAIL']:
        print("\nFAILED configs:")
        for path, msg in results['FAIL']:
            print(f"  {path}")
            print(f"    → {msg}")

    return 0 if not results['FAIL'] else 1


if __name__ == "__main__":
    sys.exit(main())
