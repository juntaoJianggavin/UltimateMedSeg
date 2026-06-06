"""Domain adaptation datasets.

Supports:
- Traditional domain adaptation (source + target data)
- Source-free domain adaptation (pretrained model + target data only)
- Multi-target domain adaptation
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Optional, Dict, Any, List


class DomainAdaptationDataset(Dataset):
    """Dataset for traditional domain adaptation.
    
    Provides both source (labeled) and target (unlabeled) data.
    
    Args:
        source_dir: Source domain directory (with labels)
        target_dir: Target domain directory (no labels)
        source_transform: Transform for source data
        target_transform: Transform for target data
        img_size: Target image size
        num_classes: Number of classes
    """
    
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        source_transform=None,
        target_transform=None,
        img_size: int = 224,
        num_classes: int = 5,
    ):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.num_classes = num_classes
        self.source_transform = source_transform
        self.target_transform = target_transform
        
        # Load source data (with labels)
        self.source_images = self._load_images(
            os.path.join(source_dir, 'images')
        )
        self.source_masks = self._load_images(
            os.path.join(source_dir, 'masks')
        )
        
        # Load target data (no labels)
        self.target_images = self._load_images(
            os.path.join(target_dir, 'images')
        )
        
        assert len(self.source_images) == len(self.source_masks), \
            "Source images and masks count mismatch"
    
    def _load_images(self, directory: str) -> List[str]:
        """Load image paths."""
        if not os.path.isdir(directory):
            return []
        
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.npy', '.npz'}
        return sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in exts
        ])
    
    def __len__(self):
        return max(len(self.source_images), len(self.target_images))
    
    def __getitem__(self, idx):
        # Get source sample (labeled)
        source_data = self._get_source(idx % len(self.source_images))
        
        # Get target sample (unlabeled)
        target_data = self._get_target(idx % len(self.target_images))
        
        return {
            'source_image': source_data['image'],
            'source_mask': source_data['mask'],
            'target_image': target_data['image'],
            'case_name': source_data.get('case_name', target_data.get('case_name', ''))
        }
    
    def _get_source(self, idx):
        """Get source domain sample."""
        img_path = self.source_images[idx]
        mask_path = self.source_masks[idx]
        
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        
        if self.source_transform is not None:
            sample = self.source_transform({"image": image, "label": mask})
            image = sample["image"]
            mask = sample["label"]
        else:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
        
        return {'image': image, 'mask': mask, 'case_name': os.path.basename(img_path)}
    
    def _get_target(self, idx):
        """Get target domain sample."""
        img_path = self.target_images[idx]
        image = self._load_image(img_path)
        
        if self.target_transform is not None:
            dummy_mask = np.zeros(image.shape[:2], dtype=np.int64)
            sample = self.target_transform({"image": image, "label": dummy_mask})
            image = sample["image"]
        else:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return {'image': image, 'case_name': os.path.basename(img_path)}
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image."""
        if path.endswith('.npy') or path.endswith('.npz'):
            data = np.load(path)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            return data.astype(np.float32) / 255.0
        else:
            image = Image.open(path).convert('RGB')
            image = image.resize(self.img_size, Image.BILINEAR)
            return np.array(image, dtype=np.float32) / 255.0
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask."""
        if path.endswith('.npy') or path.endswith('.npz'):
            mask = np.load(path)
            if mask.ndim == 3:
                mask = mask.squeeze()
            return mask.astype(np.int64)
        else:
            mask = Image.open(path)
            mask = mask.resize(self.img_size, Image.NEAREST)
            return np.array(mask, dtype=np.int64)


class SourceTargetDataset(Dataset):
    """Dataset for methods requiring explicit source/target separation.
    
    Used by:
    - AdvEnt (adversarial domain adaptation)
    - DANN (Domain Adversarial Neural Network)
    - AdaptSeg
    
    Returns paired source and target samples for adversarial training.
    """
    
    def __init__(
        self,
        source_dataset: Dataset,
        target_dataset: Dataset,
    ):
        super().__init__()
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
    
    def __len__(self):
        return max(len(self.source_dataset), len(self.target_dataset))
    
    def __getitem__(self, idx):
        source_data = self.source_dataset[idx % len(self.source_dataset)]
        target_data = self.target_dataset[idx % len(self.target_dataset)]
        
        return {
            'source': source_data,
            'target': target_data,
            'idx': idx
        }


class SourceFreeDataset(Dataset):
    """Dataset for source-free domain adaptation.
    
    Only target domain data available.
    Used by:
    - Tent (test-time adaptation)
    - DPL (denoised pseudo-labeling)
    - FSM (fourier style mining)
    - CBMT (class-balanced mean teacher)
    - STDR (dual-reference)
    - UGTST (uncertainty-guided self-training)
    
    Args:
        target_dir: Target domain directory
        transform: Data transform
        img_size: Target image size
        pretrained_model_path: Path to pretrained source model (optional)
    """
    
    def __init__(
        self,
        target_dir: str,
        transform=None,
        img_size: int = 224,
        pretrained_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.target_dir = target_dir
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.pretrained_model_path = pretrained_model_path
        
        # Load target images
        self.images = self._load_images(
            os.path.join(target_dir, 'images')
        )
        
        # Load target masks if available (for evaluation)
        masks_dir = os.path.join(target_dir, 'masks')
        if os.path.isdir(masks_dir):
            self.masks = self._load_images(masks_dir)
        else:
            self.masks = None
    
    def _load_images(self, directory: str) -> List[str]:
        """Load image paths."""
        if not os.path.isdir(directory):
            return []
        
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.npy', '.npz'}
        return sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in exts
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = self._load_image(img_path)
        
        # Load mask if available
        mask = None
        if self.masks is not None and idx < len(self.masks):
            mask = self._load_mask(self.masks[idx])
        
        # Apply transform
        if self.transform is not None:
            if mask is None:
                dummy_mask = np.zeros(image.shape[:2], dtype=np.int64)
                sample = self.transform({"image": image, "label": dummy_mask})
                image = sample["image"]
            else:
                sample = self.transform({"image": image, "label": mask})
                image = sample["image"]
                mask = sample["label"]
        else:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            if mask is not None and isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'case_name': os.path.basename(img_path)
        }
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image."""
        if path.endswith('.npy') or path.endswith('.npz'):
            data = np.load(path)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            return data.astype(np.float32) / 255.0
        else:
            image = Image.open(path).convert('RGB')
            image = image.resize(self.img_size, Image.BILINEAR)
            return np.array(image, dtype=np.float32) / 255.0
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask."""
        if path.endswith('.npy') or path.endswith('.npz'):
            mask = np.load(path)
            if mask.ndim == 3:
                mask = mask.squeeze()
            return mask.astype(np.int64)
        else:
            mask = Image.open(path)
            mask = mask.resize(self.img_size, Image.NEAREST)
            return np.array(mask, dtype=np.int64)


def build_da_dataloader(
    mode: str = 'traditional',
    source_dir: str = None,
    target_dir: str = None,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 224,
    num_classes: int = 5,
    **kwargs
) -> DataLoader:
    """Build domain adaptation dataloader.
    
    Args:
        mode: 'traditional' (source+target) or 'source_free' (target only)
        source_dir: Source domain directory
        target_dir: Target domain directory
        batch_size: Batch size
        num_workers: Number of workers
        img_size: Image size
        num_classes: Number of classes
        **kwargs: Additional arguments
    
    Returns:
        DataLoader for domain adaptation
    """
    from .transforms import get_train_transforms
    
    transform = get_train_transforms(img_size=img_size, **kwargs)
    
    if mode == 'traditional':
        dataset = DomainAdaptationDataset(
            source_dir=source_dir,
            target_dir=target_dir,
            source_transform=transform,
            target_transform=transform,
            img_size=img_size,
            num_classes=num_classes,
        )
    elif mode == 'source_free':
        dataset = SourceFreeDataset(
            target_dir=target_dir,
            transform=transform,
            img_size=img_size,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader
