"""Semi-supervised segmentation datasets.

Supports various semi-supervised training paradigms:
- Mean Teacher
- Cross Pseudo Supervision (CPS)
- Cross Consistency Training (CCT)
- UniMatch
- FixMatch
- etc.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, Tuple, Dict, Any


class SemiSupervisedDataset(Dataset):
    """Semi-supervised dataset with labeled and unlabeled samples.
    
    Returns both labeled and unlabeled data for semi-supervised training.
    
    Args:
        labeled_dir: Directory with labeled data (images + masks)
        unlabeled_dir: Directory with unlabeled data (images only)
        labeled_transform: Transform for labeled data
        unlabeled_transform: Transform for unlabeled data
        num_classes: Number of segmentation classes
        img_size: Target image size
    """
    
    def __init__(
        self,
        labeled_dir: str,
        unlabeled_dir: str,
        labeled_transform=None,
        unlabeled_transform=None,
        num_classes: int = 5,
        img_size: int = 224,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.labeled_transform = labeled_transform
        self.unlabeled_transform = unlabeled_transform
        
        # Load labeled samples
        self.labeled_dir = labeled_dir
        self.labeled_images = self._load_images(os.path.join(labeled_dir, 'images'))
        self.labeled_masks = self._load_images(os.path.join(labeled_dir, 'masks'))
        
        # Load unlabeled samples
        self.unlabeled_dir = unlabeled_dir
        self.unlabeled_images = self._load_images(os.path.join(unlabeled_dir, 'images'))
        
        assert len(self.labeled_images) == len(self.labeled_masks), \
            "Number of labeled images and masks must match"
    
    def _load_images(self, directory: str):
        """Load image file paths from directory."""
        if not os.path.isdir(directory):
            return []
        
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.npy', '.npz'}
        files = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in exts
        ])
        return files
    
    def __len__(self):
        return max(len(self.labeled_images), len(self.unlabeled_images))
    
    def __getitem__(self, idx):
        # Get labeled sample
        labeled_data = self._get_labeled(idx % len(self.labeled_images))
        
        # Get unlabeled sample
        unlabeled_data = self._get_unlabeled(idx % len(self.unlabeled_images))
        
        return {
            'labeled_image': labeled_data['image'],
            'labeled_mask': labeled_data['mask'],
            'unlabeled_image': unlabeled_data['image'],
            'case_name': labeled_data.get('case_name', unlabeled_data.get('case_name', ''))
        }
    
    def _get_labeled(self, idx):
        """Get labeled sample."""
        img_path = self.labeled_images[idx]
        mask_path = self.labeled_masks[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Load mask
        if mask_path.endswith('.npy') or mask_path.endswith('.npz'):
            mask = np.load(mask_path)
            if isinstance(mask, np.ndarray):
                if mask.ndim == 3:
                    mask = mask.squeeze()
            else:
                mask = np.zeros(image.shape[:2], dtype=np.int64)
        else:
            mask = Image.open(mask_path)
            mask = mask.resize(self.img_size, Image.NEAREST)
            mask = np.array(mask, dtype=np.int64)
        
        # Apply transform
        if self.labeled_transform is not None:
            sample = self.labeled_transform({"image": image, "label": mask})
            image = sample["image"]
            mask = sample["label"]
        else:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'case_name': os.path.basename(img_path)
        }
    
    def _get_unlabeled(self, idx):
        """Get unlabeled sample."""
        img_path = self.unlabeled_images[idx]
        image = self._load_image(img_path)
        
        # Apply transform
        if self.unlabeled_transform is not None:
            dummy_mask = np.zeros(image.shape[:2], dtype=np.int64)
            sample = self.unlabeled_transform({"image": image, "label": dummy_mask})
            image = sample["image"]
        else:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return {
            'image': image,
            'case_name': os.path.basename(img_path)
        }
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image from file."""
        if path.endswith('.npy') or path.endswith('.npz'):
            data = np.load(path)
            if isinstance(data, np.ndarray):
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data.squeeze(0)
                if data.ndim == 2:
                    data = np.stack([data] * 3, axis=-1)
            return data.astype(np.float32) / 255.0
        else:
            image = Image.open(path).convert('RGB')
            image = image.resize(self.img_size, Image.BILINEAR)
            return np.array(image, dtype=np.float32) / 255.0


class PairedLabeledUnlabeledDataset(Dataset):
    """Dataset for methods that need paired labeled/unlabeled data.
    
    Used by:
    - Mean Teacher (student sees labeled, teacher sees unlabeled)
    - CPS (two models exchange pseudo-labels)
    - CCT (cross consistency training)
    
    Args:
        labeled_dataset: Labeled dataset
        unlabeled_dataset: Unlabeled dataset
        paired: If True, return paired samples; if False, return independent samples
    """
    
    def __init__(
        self,
        labeled_dataset: Dataset,
        unlabeled_dataset: Dataset,
        paired: bool = False,
    ):
        super().__init__()
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.paired = paired
    
    def __len__(self):
        return max(len(self.labeled_dataset), len(self.unlabeled_dataset))
    
    def __getitem__(self, idx):
        # Get labeled sample
        labeled_idx = idx % len(self.labeled_dataset)
        labeled_data = self.labeled_dataset[labeled_idx]
        
        # Get unlabeled sample
        if self.paired:
            unlabeled_idx = idx % len(self.unlabeled_dataset)
        else:
            import random
            unlabeled_idx = random.randint(0, len(self.unlabeled_dataset) - 1)
        
        unlabeled_data = self.unlabeled_dataset[unlabeled_idx]
        
        return {
            'labeled': labeled_data,
            'unlabeled': unlabeled_data,
            'idx': idx
        }


def build_semi_dataloader(
    labeled_dir: str,
    unlabeled_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 224,
    paired: bool = False,
    **kwargs
) -> DataLoader:
    """Build semi-supervised dataloader.
    
    Args:
        labeled_dir: Directory with labeled data
        unlabeled_dir: Directory with unlabeled data
        batch_size: Batch size
        num_workers: Number of workers
        img_size: Image size
        paired: Whether to use paired dataset
        **kwargs: Additional arguments for transforms
    
    Returns:
        DataLoader for semi-supervised training
    """
    from .transforms import get_train_transforms
    
    # Build transforms
    labeled_transform = get_train_transforms(img_size=img_size, **kwargs)
    unlabeled_transform = get_train_transforms(img_size=img_size, **kwargs)
    
    # Build dataset
    if paired:
        labeled_ds = SemiSupervisedDataset(
            labeled_dir=labeled_dir,
            unlabeled_dir=labeled_dir,  # Dummy, not used
            labeled_transform=labeled_transform,
            unlabeled_transform=None,
            img_size=img_size,
            num_classes=kwargs.get('num_classes', 5)
        )
        unlabeled_ds = SemiSupervisedDataset(
            labeled_dir=unlabeled_dir,  # Dummy
            unlabeled_dir=unlabeled_dir,
            labeled_transform=None,
            unlabeled_transform=unlabeled_transform,
            img_size=img_size,
            num_classes=kwargs.get('num_classes', 5)
        )
        dataset = PairedLabeledUnlabeledDataset(
            labeled_ds, unlabeled_ds, paired=True
        )
    else:
        dataset = SemiSupervisedDataset(
            labeled_dir=labeled_dir,
            unlabeled_dir=unlabeled_dir,
            labeled_transform=labeled_transform,
            unlabeled_transform=unlabeled_transform,
            img_size=img_size,
            num_classes=kwargs.get('num_classes', 5)
        )
    
    # Build dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader
