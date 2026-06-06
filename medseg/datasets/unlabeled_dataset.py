"""Unlabeled dataset for semi-supervised segmentation.

Loads images only (no masks) from a directory.  Used as the unlabeled
data source in semi-supervised training pipelines.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class UnlabeledDataset(Dataset):
    """Dataset that loads images only (no masks).

    Args:
        root_dir: Directory containing image files directly, or a
            sub-directory ``images/`` when *use_subdir* is True.
        transform: Optional dict-based transform (only ``image`` key used).
        img_suffix: Image file extension filter (default ``'.png'``).
            Set to ``None`` to accept all common image formats.
        img_size: Target image size (int or tuple).
        use_subdir: If True, look for ``root_dir/images/``.
    """

    COMMON_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, root_dir: str, transform=None,
                 img_suffix: str = None, img_size: int = 224,
                 use_subdir: bool = False):
        super().__init__()
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        img_dir = os.path.join(root_dir, 'images') if use_subdir else root_dir
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.img_dir = img_dir
        self.samples = sorted([
            f for f in os.listdir(img_dir)
            if self._valid_ext(f, img_suffix)
        ])

    @staticmethod
    def _valid_ext(filename: str, suffix: str = None) -> bool:
        if suffix is not None:
            return filename.endswith(suffix)
        ext = os.path.splitext(filename)[1].lower()
        return ext in UnlabeledDataset.COMMON_EXTS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file = self.samples[idx]
        image = Image.open(os.path.join(self.img_dir, img_file)).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0

        if self.transform is not None:
            # Create dummy label for transform compatibility
            dummy = np.zeros(image.shape[:2], dtype=np.int64)
            sample = self.transform({"image": image, "label": dummy})
            image = sample["image"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return {"image": image, "case_name": img_file}
