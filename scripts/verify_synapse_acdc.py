#!/usr/bin/env python3
"""Verify TransUNet-style Synapse / ACDC data before training.

Usage::

    python scripts/verify_synapse_acdc.py \\
        --train-dir ~/datasets/project_TransUNet/data/Synapse/train_npz

    python scripts/verify_synapse_acdc.py \\
        --train-dir ./data/ACDC/train_npz \\
        --val-dir ./data/ACDC/val_npz \\
        --test-dir ./data/ACDC/test_vol
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medseg.datasets import SynapseDataset, get_train_transforms, get_val_transforms  # noqa: E402
from medseg.datasets.synapse_dataset import verify_dataset_root, VolumeLayout  # noqa: E402


def _check_loader(root: str, split: str, img_size: int) -> None:
    summary = verify_dataset_root(root, split=split, sample_limit=3)
    if split != "train" and VolumeLayout.VOLUME_3D.value in summary["layouts"]:
        print(
            f"  loader skip [{split}]: 3D volume dir — use test.py for evaluation; "
            "train.py val expects 2D val_npz slices."
        )
        return

    transform = get_train_transforms(img_size) if split == "train" else get_val_transforms(img_size)
    ds = SynapseDataset(root, split=split, transform=transform, img_size=img_size)
    sample = ds[0]
    image = sample["image"]
    label = sample["label"]
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image tensor (3,H,W), got {tuple(image.shape)}")
    if label.dim() != 2:
        raise ValueError(f"Expected label tensor (H,W), got {tuple(label.shape)}")
    print(
        f"  loader OK [{split}]: len={len(ds)}, "
        f"image={tuple(image.shape)}, label={tuple(label.shape)}, case={sample['case_name']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Synapse/ACDC TransUNet data layout")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    dirs = [
        ("train", args.train_dir),
        ("val", args.val_dir),
        ("test", args.test_dir),
    ]
    if not any(p for _, p in dirs):
        parser.error("provide at least one of --train-dir, --val-dir, --test-dir")

    ok = True
    for split, path in dirs:
        if not path:
            continue
        print(f"\n=== {split}: {path} ===")
        try:
            summary = verify_dataset_root(path, split=split)
            print(f"  files: {summary['num_files']}, samples: {summary['num_samples']}")
            print(f"  layouts: {summary['layouts']}")
            for s in summary["samples"]:
                print(
                    f"    - {Path(s['path']).name}: "
                    f"layout={s['layout']} image={s['image_shape']} "
                    f"slices={s['num_slices']}"
                )
            loader_split = "train" if split == "train" else "val"
            _check_loader(path, loader_split, args.img_size)
        except Exception as exc:
            ok = False
            print(f"  FAIL: {type(exc).__name__}: {exc}")

    print("\n" + ("All checks passed." if ok else "Some checks failed."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
