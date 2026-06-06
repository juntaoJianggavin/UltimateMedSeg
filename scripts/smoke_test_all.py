#!/usr/bin/env python3
"""Smoke-test ALL yaml configs: build model, forward pass, compute loss.

Usage:
    cd UltimateMedSeg-main/
    python scripts/smoke_test_all.py

Tests every YAML under configs/ — architectures + training_paradigms.
"""

import argparse
import glob
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

IMG_SIZE = 64  # tiny for speed
BATCH = 2


def create_dummy_data():
    """Create minimal dummy datasets."""
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
                import numpy as np
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(img).save(str(img_path))
                mask = np.random.randint(0, 4, (64, 64), dtype=np.uint8)
                Image.fromarray(mask).save(str(mask_path))
    # Symlinks
    for alias in ("YourDataset", "test_dummy", "source", "target", "target_val"):
        link = ROOT / "data" / alias
        if not link.exists():
            try:
                link.symlink_to(base, target_is_directory=True)
            except OSError:
                pass
    return str(base)


def patch_yaml(cfg, dummy_root):
    """Recursively replace data paths with dummy root."""
    if isinstance(cfg, dict):
        for k, v in list(cfg.items()):
            if isinstance(v, str) and ("data/" in v or "./data" in v):
                if "image" in k or "img" in k:
                    cfg[k] = os.path.join(dummy_root, "train", "images")
                elif "mask" in k:
                    cfg[k] = os.path.join(dummy_root, "train", "masks")
                elif "val" in k:
                    cfg[k] = os.path.join(dummy_root, "val")
                elif "test" in k:
                    cfg[k] = os.path.join(dummy_root, "test")
                elif "train" in k:
                    cfg[k] = os.path.join(dummy_root, "train")
                else:
                    cfg[k] = dummy_root
            elif isinstance(v, (dict, list)):
                patch_yaml(v, dummy_root)
    elif isinstance(cfg, list):
        for item in cfg:
            patch_yaml(item, dummy_root)


def force_small_img(cfg, yaml_path=""):
    """Force img_size to IMG_SIZE for speed, disable pretrained.

    Some architectures require larger images due to window/padding constraints.
    Uses ARCH_MIN_IMG_SIZE to determine the appropriate size.
    """
    # Architectures that need larger images (window size, padding constraints)
    ARCH_MIN_IMG_SIZE = {
        "swinunet": 112,   # Swin window_size=7, needs 7*16=112 minimum
        "swin": 112,
        "lawin": 256,      # Large padding exceeds feature map at smaller sizes
        "maxvit": 224,     # Window size 7, needs 7*32=224
        "coatnet": 128,
        "coat": 128,
        "dinov3": 96,      # patch_size=16, needs at least 6 patches
        "dinov2": 96,
        "dino": 96,
        "eva": 96,
        "vit_": 96,
        "cfanet": 256,     # Res2Net spatial hierarchy assumed
        "ege_unet": 128,   # GHPA needs larger features
        "malunet": 128,    # DGA needs larger features
        "mtunet": 224,     # Transformer sequence needs large spatial
    }
    model_cfg = cfg.get("model", cfg)
    if isinstance(model_cfg, dict):
        # Determine required img_size based on architecture/encoder/decoder names
        arch = str(model_cfg.get("architecture", ""))
        encoder_name = model_cfg.get("encoder", {}).get("name", "")
        decoder_name = model_cfg.get("decoder", {}).get("name", "")
        yaml_lower = yaml_path.lower()

        min_size = IMG_SIZE
        for key, req_size in ARCH_MIN_IMG_SIZE.items():
            if key in arch.lower() or key in encoder_name.lower() or \
               key in decoder_name.lower() or key in yaml_lower:
                min_size = max(min_size, req_size)

        old = model_cfg.get("img_size", None)
        if old != "native":
            model_cfg["img_size"] = min_size
        # Disable pretrained to avoid downloads
        enc = model_cfg.get("encoder", {})
        if isinstance(enc, dict):
            enc["pretrained"] = False


def test_one_config(yaml_path, dummy_root):
    """Build model, forward, compute loss. Returns (status, message)."""
    import yaml
    import torch
    import torch.nn.functional as F

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return "SKIP", "empty yaml"

    patch_yaml(cfg, dummy_root)
    force_small_img(cfg, yaml_path)

    model_cfg = cfg.get("model", cfg)
    num_classes = model_cfg.get("num_classes", 2)

    # Build model
    from medseg.model_builder import build_model, IncompatibleEncoderError
    try:
        model = build_model(cfg)
    except IncompatibleEncoderError as e:
        return "SKIP", str(e)[:200]
    model.eval()

    # Forward (use actual img_size from config, which may differ from IMG_SIZE)
    actual_img_size = model_cfg.get("img_size", IMG_SIZE)
    if isinstance(actual_img_size, str):
        actual_img_size = IMG_SIZE
    x = torch.randn(BATCH, 3, actual_img_size, actual_img_size)
    is_text = getattr(model, "is_text_guided", False)

    if is_text:
        try:
            out = model(x)
        except (ValueError, TypeError):
            out = model(x, text=None)
    else:
        out = model(x)

    if isinstance(out, (list, tuple)):
        out = out[0]

    # Loss
    from medseg.registry import LOSS_REGISTRY
    loss_cfg = cfg.get("training", {}).get("loss", {"name": "compound"})
    loss_name = loss_cfg.get("name", "compound")
    loss_params = loss_cfg.get("params", {}) or {}
    loss_fn = LOSS_REGISTRY.get(loss_name)(**loss_params)

    target = torch.randint(0, max(num_classes, 2), (BATCH, actual_img_size, actual_img_size))
    if out.shape[2:] != target.shape[1:]:
        out = F.interpolate(out, size=target.shape[1:], mode="bilinear",
                            align_corners=False)
    loss = loss_fn(out, target)
    loss.backward()

    return "OK", f"loss={float(loss):.4f} shape={tuple(out.shape)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-f", type=str, default=None,
                        help="Only test yamls whose path contains this substring")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    import medseg.models.encoders   # noqa
    import medseg.models.decoders   # noqa
    import medseg.models.skip_connections  # noqa
    import medseg.models.bottlenecks  # noqa
    import medseg.losses  # noqa

    dummy_root = create_dummy_data()
    all_yamls = sorted(glob.glob("configs/**/*.yaml", recursive=True))
    if args.filter:
        all_yamls = [p for p in all_yamls if args.filter in p]

    print(f"Smoke-testing {len(all_yamls)} YAML configs (img_size={IMG_SIZE})...\n")

    # Also log to file
    log_path = ROOT / "smoke_results.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    ok, fail, skip = [], [], []
    t0 = time.time()

    for i, path in enumerate(all_yamls, 1):
        try:
            status, msg = test_one_config(path, dummy_root)
        except Exception as e:
            status, msg = "FAIL", f"{type(e).__name__}: {str(e)[:200]}"

        if status == "OK":
            ok.append((path, msg))
        elif status == "SKIP":
            skip.append((path, msg))
        else:
            fail.append((path, msg))

        icon = {"OK": "+", "FAIL": "X", "SKIP": "-"}[status]
        line = f"  [{icon}] {i}/{len(all_yamls)} {path}"
        if status == "FAIL" or args.verbose:
            line += f"  ({msg})"
        log(line)

    elapsed = time.time() - t0
    log(f"\n{'='*70}")
    log(f"RESULTS ({elapsed:.1f}s): {len(ok)} OK, {len(fail)} FAIL, {len(skip)} SKIP")
    log(f"{'='*70}")

    if fail:
        log(f"\n{len(fail)} FAILED configs:")
        for path, msg in fail:
            log(f"  {path}")
            log(f"    -> {msg}")

    log(f"\nFull log saved to: {log_path}")
    log_file.close()
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
