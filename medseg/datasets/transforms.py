"""Data augmentation transforms for medical image segmentation.

Layout contract (all transforms must preserve this):
  - image: numpy float32, (H, W) grayscale or (H, W, C) with C in {1, 3, 4}
  - label: numpy int64, (H, W) only — per-pixel class indices, never RGB/HWC

Only GenericDataset.__getitem__ converts image HWC -> torch CHW at the end.
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as torchF


def _is_hwc_image(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[-1] in (1, 3, 4)


def _spatial_hw(arr) -> tuple[int, int]:
    """Return spatial (H, W) for HWC/HW image or HW label."""
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        if _is_hwc_image(arr):
            return int(arr.shape[0]), int(arr.shape[1])
        if arr.shape[0] in (1, 3, 4):
            return int(arr.shape[1]), int(arr.shape[2])
    raise ValueError(f"Expected HW/HWC/CHW array, got shape {getattr(arr, 'shape', arr)}")


def _spatial_axes_hw(arr) -> tuple[int, int]:
    """Return (height_axis, width_axis) for flip/rot90."""
    if arr.ndim == 2 or _is_hwc_image(arr):
        return 0, 1
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        return 1, 2
    raise ValueError(f"Cannot infer spatial axes for shape {arr.shape}")


def _ensure_hw_label(label) -> np.ndarray:
    """Force segmentation mask to 2D HW int64."""
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    label = np.asarray(label)
    if label.ndim == 3:
        if label.shape[-1] == 1:
            label = label[..., 0]
        elif label.shape[0] == 1:
            label = label[0]
        else:
            label = label[..., 0]
    label = np.squeeze(label)
    if label.ndim != 2:
        raise ValueError(f"Segmentation label must be 2D HW, got shape {label.shape}")
    return label.astype(np.int64, copy=False)


def _ensure_hwc_image(image) -> np.ndarray:
    """Force image to HWC float32 (expand grayscale HW -> HW1)."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    elif image.ndim == 3:
        if image.shape[0] in (1, 3, 4) and not _is_hwc_image(image):
            image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Image must be HW or HWC, got shape {image.shape}")
    return np.ascontiguousarray(image)


def _image_to_chw(image: np.ndarray) -> torch.Tensor:
    hwc = _ensure_hwc_image(image)
    return torch.from_numpy(hwc.transpose(2, 0, 1)).float()


def _chw_to_hwc(image: torch.Tensor) -> np.ndarray:
    if image.dim() == 4:
        image = image.squeeze(0)
    if image.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(image.shape)}")
    return image.permute(1, 2, 0).contiguous().numpy()


def _label_to_grid_tensor(label: np.ndarray) -> torch.Tensor:
    hw = _ensure_hw_label(label)
    return torch.from_numpy(hw).float().unsqueeze(0).unsqueeze(0)


def _grid_tensor_to_label(label: torch.Tensor) -> np.ndarray:
    while label.dim() > 2:
        label = label.squeeze(0)
    return label.long().numpy()


def _sanitize_sample(sample: dict) -> dict:
    sample["image"] = _ensure_hwc_image(sample["image"])
    sample["label"] = _ensure_hw_label(sample["label"])
    return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        sample = _sanitize_sample(sample)
        for t in self.transforms:
            sample = t(sample)
        return _sanitize_sample(sample)


class RandomFlip:
    """Random horizontal and vertical flip."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        h_ax, w_ax = _spatial_axes_hw(image)
        lh_ax, lw_ax = _spatial_axes_hw(label)
        if random.random() < self.p:
            image = np.flip(image, axis=w_ax).copy()
            label = np.flip(label, axis=lw_ax).copy()
        if random.random() < self.p:
            image = np.flip(image, axis=h_ax).copy()
            label = np.flip(label, axis=lh_ax).copy()
        return {"image": image, "label": label}


class RandomRotate90:
    """Random 90/180/270 degree rotation."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            k = random.randint(1, 3)
            h_ax, w_ax = _spatial_axes_hw(sample["image"])
            lh_ax, lw_ax = _spatial_axes_hw(sample["label"])
            sample["image"] = np.rot90(sample["image"], k, axes=(h_ax, w_ax)).copy()
            sample["label"] = np.rot90(sample["label"], k, axes=(lh_ax, lw_ax)).copy()
        return sample


class RandomRotation:
    """Random rotation by arbitrary angle with bilinear/nearest interpolation."""

    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        angle = random.uniform(-self.degrees, self.degrees)
        img_t = _image_to_chw(sample["image"])
        lab_t = _label_to_grid_tensor(sample["label"])
        theta = math.radians(angle)
        cos_v, sin_v = math.cos(theta), math.sin(theta)
        rot = torch.tensor([[cos_v, -sin_v, 0], [sin_v, cos_v, 0]], dtype=torch.float32).unsqueeze(0)
        grid = torchF.affine_grid(rot, img_t.unsqueeze(0).shape, align_corners=False)
        img_t = torchF.grid_sample(
            img_t.unsqueeze(0), grid, mode="bilinear", padding_mode="reflection", align_corners=False
        ).squeeze(0)
        lab_t = torchF.grid_sample(
            lab_t, grid, mode="nearest", padding_mode="zeros", align_corners=False
        )
        return {"image": _chw_to_hwc(img_t), "label": _grid_tensor_to_label(lab_t)}


class RandomScale:
    """Random scaling (zoom in/out) with resize back to original size."""

    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        scale = random.uniform(*self.scale_range)
        img_t = _image_to_chw(sample["image"])
        lab_t = _label_to_grid_tensor(sample["label"])
        h, w = img_t.shape[-2], img_t.shape[-1]
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        img_t = torchF.interpolate(img_t.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)
        img_t = torchF.interpolate(img_t, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        lab_t = torchF.interpolate(lab_t, size=(new_h, new_w), mode="nearest")
        lab_t = torchF.interpolate(lab_t, size=(h, w), mode="nearest")
        return {"image": _chw_to_hwc(img_t), "label": _grid_tensor_to_label(lab_t)}


class RandomCrop:
    """Random crop to specified size."""

    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        h, w = _spatial_hw(image)
        th, tw = self.size
        top = random.randint(0, h - th) if h > th else 0
        left = random.randint(0, w - tw) if w > tw else 0
        if _is_hwc_image(image):
            image = image[top : top + th, left : left + tw, :]
        elif image.ndim == 2:
            image = image[top : top + th, left : left + tw]
        else:
            image = image[:, top : top + th, left : left + tw]
        label = label[top : top + th, left : left + tw]
        return {"image": image, "label": label}


class RandomElasticDeform:
    """Random elastic deformation for medical image augmentation."""

    def __init__(self, alpha=50, sigma=5, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def _gaussian_filter(self, x, sigma):
        ks = int(4 * sigma + 1)
        if ks % 2 == 0:
            ks += 1
        ax = torch.arange(-ks // 2 + 1.0, ks // 2 + 1.0)
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel = (gauss / gauss.sum()).view(1, 1, -1)
        return torchF.conv1d(x.view(1, 1, -1), kernel, padding=ks // 2).view(x.shape)

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        img_t = _image_to_chw(sample["image"])
        lab_t = _label_to_grid_tensor(sample["label"])
        _, h, w = img_t.shape
        dx = torch.rand(h, w) * 2 - 1
        dy = torch.rand(h, w) * 2 - 1
        for i in range(h):
            dx[i] = self._gaussian_filter(dx[i], self.sigma)
            dy[i] = self._gaussian_filter(dy[i], self.sigma)
        dx = dx * self.alpha / w
        dy = dy * self.alpha / h
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        grid = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)
        img_t = torchF.grid_sample(
            img_t.unsqueeze(0), grid, mode="bilinear", padding_mode="reflection", align_corners=False
        ).squeeze(0)
        lab_t = torchF.grid_sample(lab_t, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        return {"image": _chw_to_hwc(img_t), "label": _grid_tensor_to_label(lab_t)}


class GaussianNoise:
    """Add random Gaussian noise to the image (label unchanged)."""

    def __init__(self, mean=0.0, std=0.05, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample["image"]
            noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
            sample["image"] = image + noise
        return sample


class GaussianBlur:
    """Apply Gaussian blur to the image (label unchanged)."""

    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), p=0.2):
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        img_t = _image_to_chw(sample["image"])
        sigma = random.uniform(*self.sigma)
        ks = self.kernel_size
        ax = torch.arange(-ks // 2 + 1.0, ks // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, ks, ks).repeat(img_t.shape[0], 1, 1, 1)
        img_t = torchF.conv2d(img_t.unsqueeze(0), kernel, padding=ks // 2, groups=img_t.shape[0]).squeeze(0)
        return {"image": _chw_to_hwc(img_t), "label": sample["label"]}


class BrightnessContrastJitter:
    """Randomly adjust brightness and contrast (label unchanged)."""

    def __init__(self, brightness=0.2, contrast=0.2, p=0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample["image"]
            b = random.uniform(-self.brightness, self.brightness)
            c = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = image.mean()
            sample["image"] = (image - mean) * c + mean + b
        return sample


class GammaCorrection:
    """Random gamma correction for intensity augmentation."""

    def __init__(self, gamma_range=(0.7, 1.5), p=0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            image = sample["image"]
            mn, mx = image.min(), image.max()
            if mx - mn > 0:
                sample["image"] = ((image - mn) / (mx - mn)) ** gamma * (mx - mn) + mn
        return sample


class CutOut:
    """Randomly mask out rectangular regions in the image (label unchanged)."""

    def __init__(self, num_holes=1, max_h_size=32, max_w_size=32, p=0.3):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.p = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        image = sample["image"].copy()
        h, w = _spatial_hw(image)
        for _ in range(self.num_holes):
            rh = random.randint(1, self.max_h_size)
            rw = random.randint(1, self.max_w_size)
            y = random.randint(0, max(0, h - rh))
            x = random.randint(0, max(0, w - rw))
            if _is_hwc_image(image):
                image[y : y + rh, x : x + rw, :] = 0
            else:
                image[y : y + rh, x : x + rw] = 0
        return {"image": image, "label": sample["label"]}


class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, sample):
        img_t = _image_to_chw(sample["image"])
        lab_t = _label_to_grid_tensor(sample["label"])
        img_t = torchF.interpolate(img_t.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)
        lab_t = torchF.interpolate(lab_t, size=self.size, mode="nearest")
        return {"image": _chw_to_hwc(img_t), "label": _grid_tensor_to_label(lab_t)}


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = _ensure_hwc_image(sample["image"])
        if self.mean is not None and self.std is not None:
            mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, -1)
            std = np.array(self.std, dtype=np.float32).reshape(1, 1, -1)
            image = (image - mean) / (std + 1e-8)
        else:
            mn, mx = image.min(), image.max()
            if mx - mn > 0:
                image = (image - mn) / (mx - mn)
        sample["image"] = image
        return sample


def get_train_transforms(img_size=224, augment_level="standard"):
    """Get training transforms.

    Args:
        img_size: Target image size.
        augment_level: 'light', 'standard', or 'heavy'.
    """
    base = [Resize(img_size)]
    if augment_level == "light":
        base += [RandomFlip(p=0.5)]
    elif augment_level == "heavy":
        base += [
            RandomFlip(p=0.5),
            RandomRotate90(p=0.5),
            RandomRotation(degrees=15, p=0.3),
            RandomScale(scale_range=(0.8, 1.2), p=0.3),
            RandomElasticDeform(alpha=50, sigma=5, p=0.2),
            GaussianNoise(std=0.05, p=0.2),
            GaussianBlur(kernel_size=3, p=0.15),
            BrightnessContrastJitter(brightness=0.2, contrast=0.2, p=0.2),
            GammaCorrection(gamma_range=(0.7, 1.5), p=0.2),
            CutOut(num_holes=1, max_h_size=32, max_w_size=32, p=0.15),
        ]
    else:
        base += [
            RandomFlip(p=0.5),
            RandomRotate90(p=0.5),
            RandomRotation(degrees=10, p=0.3),
            RandomScale(scale_range=(0.85, 1.15), p=0.3),
            GaussianNoise(std=0.03, p=0.2),
            BrightnessContrastJitter(brightness=0.15, contrast=0.15, p=0.2),
        ]
    base.append(Normalize())
    return Compose(base)


def get_val_transforms(img_size=224):
    return Compose([
        Resize(img_size),
        Normalize(),
    ])
