"""Weakly supervised segmentation datasets.

Supports:
- Box-supervised segmentation (bounding box annotations)
- Image-level classification labels only
- CAM-based weak supervision
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any


class WeaklySupervisedDataset(Dataset):
    """Base weakly supervised dataset.
    
    Supports multiple types of weak supervision:
    - Bounding boxes
    - Image-level labels
    - Points
    - Scribbles
    
    Args:
        image_dir: Directory containing images
        annotation_file: JSON file with weak annotations
        supervision_type: Type of weak supervision ('box', 'image_label', 'point', 'scribble')
        transform: Data transform
        img_size: Target image size
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        supervision_type: str = 'box',
        transform=None,
        img_size: int = 224,
        num_classes: int = 5,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.supervision_type = supervision_type
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.num_classes = num_classes
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Validate supervision type
        assert supervision_type in ['box', 'image_label', 'point', 'scribble'], \
            f"Unknown supervision type: {supervision_type}"
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        image = self._load_image(ann['image'])
        
        # Load weak annotations based on type
        if self.supervision_type == 'box':
            boxes = self._load_boxes(ann)
            image_labels = self._boxes_to_image_labels(boxes)
            return {
                'image': image,
                'boxes': boxes,
                'image_labels': image_labels,
                'case_name': ann.get('case_name', os.path.basename(ann['image']))
            }
        elif self.supervision_type == 'image_label':
            image_labels = self._load_image_labels(ann)
            return {
                'image': image,
                'image_labels': image_labels,
                'case_name': ann.get('case_name', os.path.basename(ann['image']))
            }
        elif self.supervision_type == 'point':
            points = self._load_points(ann)
            return {
                'image': image,
                'points': points,
                'case_name': ann.get('case_name', os.path.basename(ann['image']))
            }
        elif self.supervision_type == 'scribble':
            scribbles = self._load_scribbles(ann)
            return {
                'image': image,
                'scribbles': scribbles,
                'case_name': ann.get('case_name', os.path.basename(ann['image']))
            }
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load image."""
        full_path = os.path.join(self.image_dir, path)
        image = Image.open(full_path).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        
        if self.transform is not None:
            dummy_mask = np.zeros(image.shape[:2], dtype=np.int64)
            sample = self.transform({"image": image, "label": dummy_mask})
            image = sample["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image
    
    def _load_boxes(self, ann: dict) -> torch.Tensor:
        """Load bounding boxes."""
        boxes = ann.get('boxes', [])
        if len(boxes) == 0:
            return torch.empty(0, 4)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        # Scale boxes to image size
        boxes[:, 0] = boxes[:, 0] * self.img_size[0]  # x1
        boxes[:, 1] = boxes[:, 1] * self.img_size[1]  # y1
        boxes[:, 2] = boxes[:, 2] * self.img_size[0]  # x2
        boxes[:, 3] = boxes[:, 3] * self.img_size[1]  # y2
        
        return boxes
    
    def _boxes_to_image_labels(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes to image-level labels."""
        image_labels = torch.zeros(self.num_classes)
        if len(boxes) > 0:
            # Assume all classes present in boxes are labeled
            for i in range(len(boxes)):
                if 'class' in boxes[i]:
                    class_id = int(boxes[i]['class'])
                    image_labels[class_id] = 1.0
                else:
                    # If no class info, assume all classes present
                    image_labels[:] = 1.0
        
        return image_labels
    
    def _load_image_labels(self, ann: dict) -> torch.Tensor:
        """Load image-level labels."""
        labels = ann.get('image_labels', [])
        image_labels = torch.zeros(self.num_classes)
        
        if isinstance(labels, list):
            for label in labels:
                if isinstance(label, int):
                    image_labels[label] = 1.0
                elif isinstance(label, dict):
                    class_id = label.get('class', 0)
                    image_labels[class_id] = 1.0
        elif isinstance(labels, dict):
            for class_id, present in labels.items():
                if present:
                    image_labels[int(class_id)] = 1.0
        
        return image_labels
    
    def _load_points(self, ann: dict) -> torch.Tensor:
        """Load point annotations."""
        points = ann.get('points', [])
        if len(points) == 0:
            return torch.empty(0, 3)
        
        return torch.tensor(points, dtype=torch.float32)
    
    def _load_scribbles(self, ann: dict) -> torch.Tensor:
        """Load scribble annotations."""
        scribbles = ann.get('scribbles', [])
        if len(scribbles) == 0:
            return torch.empty(0, 2)
        
        return torch.tensor(scribbles, dtype=torch.float32)


class BoxSupervisedDataset(WeaklySupervisedDataset):
    """Dataset for box-supervised segmentation.
    
    Only bounding box annotations required.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['supervision_type'] = 'box'
        super().__init__(*args, **kwargs)


class ImageLabelDataset(WeaklySupervisedDataset):
    """Dataset for image-level label supervision.
    
    Only image-level classification labels required.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['supervision_type'] = 'image_label'
        super().__init__(*args, **kwargs)


class CAMDataset(Dataset):
    """Dataset for CAM-based weak supervision.
    
    Uses pre-computed Class Activation Maps.
    
    Args:
        image_dir: Directory with images
        cam_dir: Directory with CAM files
        label_file: File with image-level labels
        transform: Data transform
        img_size: Target image size
        num_classes: Number of classes
    """
    
    def __init__(
        self,
        image_dir: str,
        cam_dir: str,
        label_file: str,
        transform=None,
        img_size: int = 224,
        num_classes: int = 5,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.cam_dir = cam_dir
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.num_classes = num_classes
        
        # Load labels
        with open(label_file, 'r') as f:
            self.labels = json.load(f)
        
        # Get image list
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        # Load image
        image = self._load_image(img_file)
        
        # Load CAM
        cam_file = os.path.splitext(img_file)[0] + '.npy'
        cam_path = os.path.join(self.cam_dir, cam_file)
        cam = self._load_cam(cam_path)
        
        # Load image-level labels
        image_labels = self._get_image_labels(img_file)
        
        return {
            'image': image,
            'cam': cam,
            'image_labels': image_labels,
            'case_name': img_file
        }
    
    def _load_image(self, filename: str) -> torch.Tensor:
        """Load image."""
        image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        
        if self.transform is not None:
            dummy_mask = np.zeros(image.shape[:2], dtype=np.int64)
            sample = self.transform({"image": image, "label": dummy_mask})
            image = sample["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image
    
    def _load_cam(self, path: str) -> torch.Tensor:
        """Load CAM."""
        if os.path.exists(path):
            cam = np.load(path)
        else:
            # Generate dummy CAM
            cam = np.random.rand(self.num_classes, self.img_size[0], self.img_size[1]).astype(np.float32)
        
        if isinstance(cam, np.ndarray):
            cam = torch.from_numpy(cam).float()
        
        return cam
    
    def _get_image_labels(self, filename: str) -> torch.Tensor:
        """Get image-level labels."""
        image_labels = torch.zeros(self.num_classes)
        
        if filename in self.labels:
            labels = self.labels[filename]
            if isinstance(labels, list):
                for label in labels:
                    image_labels[label] = 1.0
            elif isinstance(labels, dict):
                for class_id, present in labels.items():
                    if present:
                        image_labels[int(class_id)] = 1.0
        
        return image_labels
