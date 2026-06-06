"""Data augmentation transforms for medical image segmentation."""

import numpy as np
import torch
import torch.nn.functional as torchF
import random
import math


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomFlip:
    """Random horizontal and vertical flip."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < self.p:
            image = np.flip(image, axis=-1).copy()
            label = np.flip(label, axis=-1).copy()
        if random.random() < self.p:
            image = np.flip(image, axis=-2).copy()
            label = np.flip(label, axis=-2).copy()
        return {'image': image, 'label': label}


class RandomRotate90:
    """Random 90/180/270 degree rotation."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            k = random.randint(1, 3)
            sample['image'] = np.rot90(sample['image'], k, axes=(-2, -1)).copy()
            sample['label'] = np.rot90(sample['label'], k, axes=(-2, -1)).copy()
        return sample


class RandomRotation:
    """Random rotation by arbitrary angle with bilinear/nearest interpolation."""
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            image, label = sample['image'], sample['label']
            is_np = isinstance(image, np.ndarray)
            if is_np:
                img_t = torch.from_numpy(image).float()
                lab_t = torch.from_numpy(label).float()
            else:
                img_t, lab_t = image.float(), label.float()
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)
                lab_t = lab_t.unsqueeze(0)
            # Build rotation matrix
            theta = math.radians(angle)
            cos_v, sin_v = math.cos(theta), math.sin(theta)
            rot = torch.tensor([[cos_v, -sin_v, 0], [sin_v, cos_v, 0]], dtype=torch.float32)
            rot = rot.unsqueeze(0)
            grid = torchF.affine_grid(rot, img_t.unsqueeze(0).shape, align_corners=False)
            img_t = torchF.grid_sample(img_t.unsqueeze(0), grid, mode='bilinear', padding_mode='reflection', align_corners=False).squeeze(0)
            lab_t = torchF.grid_sample(lab_t.unsqueeze(0), grid, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0)
            if is_np:
                sample['image'] = img_t.numpy()
                sample['label'] = lab_t.squeeze(0).long().numpy()
            else:
                sample['image'] = img_t
                sample['label'] = lab_t.squeeze(0).long()
        return sample


class RandomScale:
    """Random scaling (zoom in/out) with resize back to original size."""
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            scale = random.uniform(*self.scale_range)
            image, label = sample['image'], sample['label']
            h, w = image.shape[-2], image.shape[-1]
            new_h, new_w = int(h * scale), int(w * scale)
            is_np = isinstance(image, np.ndarray)
            if is_np:
                img_t = torch.from_numpy(image).float()
                lab_t = torch.from_numpy(label).float()
            else:
                img_t, lab_t = image.float(), label.float()
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)
            lab_t_4d = lab_t.unsqueeze(0).unsqueeze(0) if lab_t.dim() == 2 else lab_t.unsqueeze(0)
            img_t = torchF.interpolate(img_t.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            img_t = torchF.interpolate(img_t, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
            lab_t = torchF.interpolate(lab_t_4d, size=(new_h, new_w), mode='nearest')
            lab_t = torchF.interpolate(lab_t, size=(h, w), mode='nearest').squeeze(0).squeeze(0)
            if is_np:
                sample['image'] = img_t.numpy()
                sample['label'] = lab_t.long().numpy()
            else:
                sample['image'] = img_t
                sample['label'] = lab_t.long()
        return sample


class RandomCrop:
    """Random crop to specified size."""
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[-2], image.shape[-1]
        th, tw = self.size
        if h > th:
            top = random.randint(0, h - th)
        else:
            top = 0
        if w > tw:
            left = random.randint(0, w - tw)
        else:
            left = 0
        image = image[..., top:top + th, left:left + tw]
        label = label[..., top:top + th, left:left + tw]
        return {'image': image, 'label': label}


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
        ax = torch.arange(-ks // 2 + 1., ks // 2 + 1.)
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel = (gauss / gauss.sum()).view(1, 1, -1)
        x = torchF.conv1d(x.view(1, 1, -1), kernel, padding=ks // 2).view(x.shape)
        return x

    def __call__(self, sample):
        if random.random() < self.p:
            image, label = sample['image'], sample['label']
            is_np = isinstance(image, np.ndarray)
            if is_np:
                img_t = torch.from_numpy(image).float()
                lab_t = torch.from_numpy(label).float()
            else:
                img_t, lab_t = image.float(), label.float()
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)
            h, w = img_t.shape[-2], img_t.shape[-1]
            # Random displacement fields
            dx = torch.rand(h, w) * 2 - 1
            dy = torch.rand(h, w) * 2 - 1
            # Smooth with gaussian
            for i in range(h):
                dx[i] = self._gaussian_filter(dx[i], self.sigma)
                dy[i] = self._gaussian_filter(dy[i], self.sigma)
            dx = dx * self.alpha / w
            dy = dy * self.alpha / h
            # Build sampling grid
            grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
            grid = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)
            img_t = torchF.grid_sample(img_t.unsqueeze(0), grid, mode='bilinear', padding_mode='reflection', align_corners=False).squeeze(0)
            lab_4d = lab_t.unsqueeze(0).unsqueeze(0) if lab_t.dim() == 2 else lab_t.unsqueeze(0)
            lab_t = torchF.grid_sample(lab_4d, grid, mode='nearest', padding_mode='zeros', align_corners=False).squeeze(0).squeeze(0)
            if is_np:
                sample['image'] = img_t.numpy()
                sample['label'] = lab_t.long().numpy()
            else:
                sample['image'] = img_t
                sample['label'] = lab_t.long()
        return sample


class GaussianNoise:
    """Add random Gaussian noise to the image (label unchanged)."""
    def __init__(self, mean=0.0, std=0.05, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            if isinstance(image, np.ndarray):
                noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
                sample['image'] = image + noise
            else:
                noise = torch.randn_like(image) * self.std + self.mean
                sample['image'] = image + noise
        return sample


class GaussianBlur:
    """Apply Gaussian blur to the image (label unchanged)."""
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), p=0.2):
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            sigma = random.uniform(*self.sigma)
            is_np = isinstance(image, np.ndarray)
            if is_np:
                img_t = torch.from_numpy(image).float()
            else:
                img_t = image.float()
            if img_t.dim() == 2:
                img_t = img_t.unsqueeze(0)
            ks = self.kernel_size
            ax = torch.arange(-ks // 2 + 1., ks // 2 + 1.)
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, ks, ks).repeat(img_t.shape[0], 1, 1, 1)
            img_t = torchF.conv2d(img_t.unsqueeze(0), kernel, padding=ks // 2, groups=img_t.shape[0]).squeeze(0)
            if is_np:
                sample['image'] = img_t.numpy()
            else:
                sample['image'] = img_t
        return sample


class BrightnessContrastJitter:
    """Randomly adjust brightness and contrast (label unchanged)."""
    def __init__(self, brightness=0.2, contrast=0.2, p=0.3):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image = sample['image']
            b = random.uniform(-self.brightness, self.brightness)
            c = random.uniform(1 - self.contrast, 1 + self.contrast)
            if isinstance(image, np.ndarray):
                mean = image.mean()
                image = (image - mean) * c + mean + b
            else:
                mean = image.mean()
                image = (image - mean) * c + mean + b
            sample['image'] = image
        return sample


class GammaCorrection:
    """Random gamma correction for intensity augmentation."""
    def __init__(self, gamma_range=(0.7, 1.5), p=0.3):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            image = sample['image']
            if isinstance(image, np.ndarray):
                mn, mx = image.min(), image.max()
                if mx - mn > 0:
                    image = ((image - mn) / (mx - mn)) ** gamma * (mx - mn) + mn
            else:
                mn, mx = image.min(), image.max()
                if mx - mn > 0:
                    image = ((image - mn) / (mx - mn)) ** gamma * (mx - mn) + mn
            sample['image'] = image
        return sample


class CutOut:
    """Randomly mask out rectangular regions (both image and label set to 0)."""
    def __init__(self, num_holes=1, max_h_size=32, max_w_size=32, p=0.3):
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, label = sample['image'], sample['label']
            h, w = image.shape[-2], image.shape[-1]
            for _ in range(self.num_holes):
                rh = random.randint(1, self.max_h_size)
                rw = random.randint(1, self.max_w_size)
                y = random.randint(0, max(0, h - rh))
                x = random.randint(0, max(0, w - rw))
                image[..., y:y + rh, x:x + rw] = 0
            sample['image'] = image
        return sample


class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        was_np = isinstance(image, np.ndarray)
        # Use torch for resizing
        if was_np:
            img_t = torch.from_numpy(image).float()
            lab_t = torch.from_numpy(label).float()
        else:
            img_t = image
            lab_t = label.float()

        if img_t.dim() == 2:
            # (H, W) -> (1, 1, H, W)
            img_t = img_t.unsqueeze(0).unsqueeze(0)
            lab_t = lab_t.unsqueeze(0).unsqueeze(0)
        elif img_t.dim() == 3:
            if was_np and img_t.shape[-1] in (1, 3):
                # HWC numpy -> CHW tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
                img_t = img_t.permute(2, 0, 1).unsqueeze(0)
            else:
                # Already CHW: (C, H, W) -> (1, C, H, W)
                img_t = img_t.unsqueeze(0)
            lab_t = lab_t.unsqueeze(0).unsqueeze(0)

        img_t = torch.nn.functional.interpolate(img_t, size=self.size, mode='bilinear', align_corners=False)
        lab_t = torch.nn.functional.interpolate(lab_t, size=self.size, mode='nearest')

        if was_np:
            img_out = img_t.squeeze(0)
            # Convert back to HWC if it was HWC
            if img_out.shape[0] in (1, 3):
                img_out = img_out.permute(1, 2, 0)
            image = img_out.numpy()
        else:
            image = img_t.squeeze(0)
        label = lab_t.squeeze(0).squeeze(0).long().numpy() if was_np else lab_t.squeeze(0).squeeze(0).long()
        return {'image': image, 'label': label}


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        if self.mean is not None and self.std is not None:
            mean = np.array(self.mean).reshape(-1, 1, 1) if isinstance(image, np.ndarray) else torch.tensor(self.mean).reshape(-1, 1, 1)
            std = np.array(self.std).reshape(-1, 1, 1) if isinstance(image, np.ndarray) else torch.tensor(self.std).reshape(-1, 1, 1)
            image = (image - mean) / (std + 1e-8)
        else:
            # Simple min-max normalization
            if isinstance(image, np.ndarray):
                mn, mx = image.min(), image.max()
                if mx - mn > 0:
                    image = (image - mn) / (mx - mn)
            else:
                mn, mx = image.min(), image.max()
                if mx - mn > 0:
                    image = (image - mn) / (mx - mn)
        sample['image'] = image
        return sample


def get_train_transforms(img_size=224, augment_level='standard'):
    """Get training transforms.

    Args:
        img_size: Target image size.
        augment_level: 'light', 'standard', or 'heavy'.
    """
    base = [Resize(img_size)]
    if augment_level == 'light':
        base += [
            RandomFlip(p=0.5),
        ]
    elif augment_level == 'heavy':
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
    else:  # standard
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
