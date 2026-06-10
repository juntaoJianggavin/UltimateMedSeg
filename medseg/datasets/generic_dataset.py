"""Image-Mask segmentation dataset with train/val/test split and K-fold cross-validation support.

Supports multi-class segmentation with image/mask pairs. Mask pixel values are
used directly as class labels (0 = background, 1..N = foreground classes).

Supports two operational modes:
1. **Split mode (train/val/test)**: Either specify explicit directories for each split,
   or provide a single root directory to auto-split by ratio with a configurable random_state.
2. **K-fold mode**: Provide a single root directory and fold parameters (n_splits, fold_idx)
   to perform K-fold cross-validation.

Expected directory layout (per split or root):
    <dir>/
        images/
            case_001.png
            case_002.png
            ...
        masks/
            case_001.png
            case_002.png
            ...

Masks are single-channel images where pixel values represent class indices
(0 = background, 1..N = foreground classes).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, List, Tuple


class GenericDataset(Dataset):
    """Image-mask segmentation dataset for arbitrary number of classes.

    Supports binary (1 foreground class, num_classes=2 including background),
    multi-class (e.g. num_classes=4 / 9 / N+1), with both auto-split and
    K-fold modes. Input format is unchanged from the previous `BinaryDataset`:
    images/ and masks/ sub-folders. Mask pixel values are used directly as
    class indices — NO binarization.

    Args:
        root_dir: Root directory containing images/ and masks/ sub-folders.
            Used when explicit image_dir/mask_dir are not provided OR for auto-split/k-fold.
        image_dir: Explicit path to images folder (overrides root_dir/images).
        mask_dir: Explicit path to masks folder (overrides root_dir/masks).
        split: One of 'train', 'val', 'test'. Only used in split mode.
        transform: Optional transform (dict-based: {'image': ..., 'label': ...}).
        img_suffix: Image file extension (default '.png').
        mask_suffix: Mask file extension (default '.png').
        img_size: Target image size (int or tuple).

        --- Split mode (auto) ---
        train_ratio: Ratio for training set when auto-splitting (default 0.7).
        val_ratio: Ratio for validation set when auto-splitting (default 0.15).
            test_ratio is inferred as 1 - train_ratio - val_ratio.
        random_state: Random seed for reproducible auto-split (default 42).

        --- K-fold mode ---
        n_splits: Number of folds for K-fold cross-validation (default 5).
            Set to a value > 1 to enable K-fold mode.
        fold_idx: Current fold index (0-based). Required when n_splits > 1.
        kfold_mode: One of 'train', 'val'. In fold mode there is no test set;
            the fold is used as val and the rest as train.

        --- Explicit file list ---
        file_list: Optional path to a text file listing sample base-names (one per line).
            When provided, only these samples are used (no auto-split/k-fold).
    """

    def __init__(
        self,
        root_dir: str = None,
        image_dir: str = None,
        mask_dir: str = None,
        split: str = 'train',
        transform=None,
        img_suffix: str = '.png',
        mask_suffix: str = '.png',
        img_size: int = 224,
        # Split-mode params
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_state: int = 42,
        # K-fold params
        n_splits: int = 0,
        fold_idx: int = 0,
        kfold_mode: str = 'train',
        # Explicit file list
        file_list: str = None,
    ):
        super().__init__()
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        # Resolve directories
        if image_dir is not None and mask_dir is not None:
            self._image_dir = image_dir
            self._mask_dir = mask_dir
        elif root_dir is not None:
            self._image_dir = os.path.join(root_dir, 'images')
            self._mask_dir = os.path.join(root_dir, 'masks')
        else:
            raise ValueError("Must provide either root_dir or both image_dir and mask_dir.")

        # Collect all valid sample base-names
        all_bases = self._collect_bases(img_suffix, mask_suffix)

        # Determine which samples belong to this instance
        if file_list is not None and os.path.exists(file_list):
            # Explicit file list mode
            with open(file_list, 'r') as f:
                listed = set(line.strip() for line in f if line.strip())
            bases = [b for b in all_bases if b in listed]
        elif n_splits > 1:
            # K-fold cross-validation mode
            bases = self._kfold_select(all_bases, n_splits, fold_idx, kfold_mode, random_state)
        else:
            # Auto train/val/test split mode
            bases = self._split_select(all_bases, split, train_ratio, val_ratio, random_state)

        # Build final sample list as (img_filename, mask_filename)
        self.samples: List[Tuple[str, str]] = [
            (b + img_suffix, b + mask_suffix) for b in bases
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_bases(self, img_suffix: str, mask_suffix: str) -> List[str]:
        """Return sorted list of base-names that have both image and mask."""
        img_files = {f.replace(img_suffix, '')
                     for f in os.listdir(self._image_dir)
                     if f.endswith(img_suffix)}
        mask_files = {f.replace(mask_suffix, '')
                      for f in os.listdir(self._mask_dir)
                      if f.endswith(mask_suffix)}
        return sorted(img_files & mask_files)

    @staticmethod
    def _split_select(bases: List[str], split: str,
                      train_ratio: float, val_ratio: float,
                      random_state: int) -> List[str]:
        """Auto-split bases into train / val / test by ratio."""
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(len(bases))
        n = len(bases)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == 'train':
            sel = indices[:n_train]
        elif split == 'val':
            sel = indices[n_train:n_train + n_val]
        elif split == 'test':
            sel = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split '{split}', expected train/val/test.")
        return [bases[i] for i in sorted(sel)]

    @staticmethod
    def _kfold_select(bases: List[str], n_splits: int, fold_idx: int,
                      mode: str, random_state: int) -> List[str]:
        """Select samples for a given K-fold split."""
        assert 0 <= fold_idx < n_splits, f"fold_idx={fold_idx} out of range [0, {n_splits})"
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(len(bases))
        fold_sizes = np.full(n_splits, len(bases) // n_splits, dtype=int)
        fold_sizes[:len(bases) % n_splits] += 1
        folds = np.split(indices, np.cumsum(fold_sizes)[:-1])

        val_indices = set(folds[fold_idx].tolist())
        if mode == 'val':
            sel = [i for i in range(len(bases)) if i in val_indices]
        elif mode == 'train':
            sel = [i for i in range(len(bases)) if i not in val_indices]
        else:
            raise ValueError(f"Unknown kfold_mode '{mode}', expected train/val.")
        return [bases[i] for i in sorted(sel)]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, mask_file = self.samples[idx]

        # Load image (RGB)
        image = Image.open(os.path.join(self._image_dir, img_file)).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0

        # Load mask (multi-class: pixel values = class indices)
        mask = Image.open(os.path.join(self._mask_dir, mask_file))
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.max() > 1:
            mask = (mask > 0).astype(np.int64)

        # Apply transforms (contract: image HWC float, label HW int64)
        if self.transform is not None:
            sample = self.transform({"image": image, "label": mask})
            image, mask = sample["image"], sample["label"]

        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.int64)
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        else:
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()

        return {"image": image, "label": mask, "case_name": img_file}
