"""Advanced data augmentation methods for medical image segmentation.

All transforms follow the dict interface: {"image": ndarray (H,W,C), "label": ndarray (H,W)}.
All transforms are registered to AUGMENTATION_REGISTRY for YAML configuration.

Includes sample-level transforms (CopyPaste, Mosaic) that need dataset access,
and pixel-level transforms (PhotometricDistortion, GridMask, etc.).

Reference papers:
- CopyPaste: "Simple Copy-Paste is a Strong Method for Self-Supervised Learning" (Ghiasi et al., CVPR 2021)
- Mosaic: "YOLOv4: Optimal Speed and Accuracy of Object Detection" (Bochkovskiy et al., 2020)
- GridMask: "GridMask Data Augmentation" (Chen et al., 2020)
"""

import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Callable

from medseg.registry import AUGMENTATION_REGISTRY


# ─── Helper functions ────────────────────────────────────────────────────────


def _to_numpy_hwc(image):
    """Convert image to numpy HWC format (float32, 0-1 range)."""
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0).numpy()
        else:
            image = image.numpy()
    return image.astype(np.float32)


def _img_to_chw_torch(image):
    """Convert HWC numpy/tensor to CHW 4D tensor (B=1, C, H, W)."""
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            t = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        else:
            t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    else:
        t = image.float()
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            if t.shape[0] in (1, 3):
                t = t.unsqueeze(0)  # already CHW
            else:
                t = t.permute(2, 0, 1).unsqueeze(0)
    return t


def _lab_to_4d_torch(label):
    """Convert HW label to 4D tensor (B=1, 1, H, W) for nearest interpolation."""
    if isinstance(label, np.ndarray):
        t = torch.from_numpy(label).float().unsqueeze(0).unsqueeze(0)
    else:
        t = label.float()
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            t = t.unsqueeze(0)
    return t


def _from_chw_to_hwc(tensor, was_np: bool):
    """Convert 4D CHW tensor back to HWC numpy (if was_np) or 3D CHW tensor."""
    t = tensor.squeeze(0)
    if was_np:
        if t.dim() == 3:
            return t.permute(1, 2, 0).numpy()
        return t.numpy()
    return t


def _from_4d_to_hw(tensor, was_np: bool):
    """Convert 4D label tensor back to HW numpy or 2D tensor."""
    t = tensor.squeeze(0).squeeze(0)
    if was_np:
        return t.long().numpy()
    return t.long()


def _mask_region_props(mask: np.ndarray, ignore_bg: int = 0) -> List[dict]:
    """Extract connected component properties from a mask."""
    props = []
    unique_labels = np.unique(mask)
    for lbl in unique_labels:
        if lbl == ignore_bg:
            continue
        region = (mask == lbl)
        ys, xs = np.where(region)
        if len(ys) == 0:
            continue
        props.append({
            "label": int(lbl),
            "bbox": (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1),
            "area": len(ys),
            "mask": region,
        })
    return props


# ─── CopyPaste ───────────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("copy_paste")
class CopyPaste:
    """Copy-Paste augmentation for segmentation.

    Pastes foreground objects (from mask regions) of a donor image onto the
    current image, with corresponding mask updates.

    Args:
        p: Probability of applying the augmentation.
        max_objects: Max number of objects to paste per sample.
        scale_range: Scale range for pasted objects (relative to original size).
        blend_ratio_range: Alpha blending range at boundary (0=hard paste).
        dataset: Dataset reference for sampling donors.
    """

    def __init__(self, p=0.5, max_objects=3, scale_range=(0.5, 1.5),
                 blend_ratio_range=(0.0, 0.0), dataset=None, **kwargs):
        self.p = p
        self.max_objects = max_objects
        self.scale_range = scale_range
        self.blend_ratio_range = blend_ratio_range
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p or self.dataset is None:
            return sample
        if len(self.dataset) < 2:
            return sample

        image = _to_numpy_hwc(sample["image"])
        label = sample["label"].copy() if isinstance(sample["label"], np.ndarray) else sample["label"].numpy().copy()
        h, w = image.shape[:2]

        donor_idx = random.randint(0, len(self.dataset) - 1)
        donor = self.dataset[donor_idx]
        donor_img = _to_numpy_hwc(donor["image"])
        donor_lbl = donor["label"]
        if isinstance(donor_lbl, torch.Tensor):
            donor_lbl = donor_lbl.numpy()

        if donor_img.shape[:2] != (h, w):
            donor_img_t = torch.from_numpy(donor_img).permute(2, 0, 1).unsqueeze(0)
            donor_img_t = F.interpolate(donor_img_t, size=(h, w), mode="bilinear", align_corners=False)
            donor_img = donor_img_t.squeeze(0).permute(1, 2, 0).numpy()
            donor_lbl_t = torch.from_numpy(donor_lbl).unsqueeze(0).unsqueeze(0).float()
            donor_lbl_t = F.interpolate(donor_lbl_t, size=(h, w), mode="nearest")
            donor_lbl = donor_lbl_t.squeeze().numpy().astype(np.int64)

        props = _mask_region_props(donor_lbl, ignore_bg=0)
        if not props:
            return sample

        random.shuffle(props)
        n_paste = min(self.max_objects, len(props))
        blend_ratio = random.uniform(*self.blend_ratio_range)
        scale = random.uniform(*self.scale_range)

        for prop in props[:n_paste]:
            obj_mask = prop["mask"]

            # Scale the object if scale != 1.0
            if abs(scale - 1.0) > 0.01:
                y1, y2, x1, x2 = prop["bbox"]
                oh, ow = y2 - y1, x2 - x1
                cy, cx = (y1 + y2) / 2, (x1 + x2) / 2
                new_oh, new_ow = int(oh * scale), int(ow * scale)
                if new_oh < 1 or new_ow < 1:
                    continue
                ny1 = max(0, int(cy - new_oh / 2))
                ny2 = min(h, ny1 + new_oh)
                nx1 = max(0, int(cx - new_ow / 2))
                nx2 = min(w, nx1 + new_ow)

                # Extract and resize object region
                obj_img = donor_img[y1:y2, x1:x2]
                obj_m = obj_mask[y1:y2, x1:x2]
                obj_img_t = torch.from_numpy(obj_img).permute(2, 0, 1).unsqueeze(0).float()
                obj_img_t = F.interpolate(obj_img_t, size=(new_oh, new_ow), mode="bilinear", align_corners=False)
                obj_img = obj_img_t.squeeze(0).permute(1, 2, 0).numpy()
                obj_m_t = torch.from_numpy(obj_m.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                obj_m_t = F.interpolate(obj_m_t, size=(new_oh, new_ow), mode="nearest")
                obj_m = obj_m_t.squeeze().numpy().astype(bool)

                rh, rw = min(obj_img.shape[0], ny2 - ny1), min(obj_img.shape[1], nx2 - nx1)
                if rh < 1 or rw < 1:
                    continue
                if blend_ratio > 0:
                    alpha = obj_m[:rh, :rw].astype(np.float32)[..., None] * (1 - blend_ratio)
                    alpha = np.broadcast_to(alpha, (rh, rw, image.shape[-1]))
                    image[ny1:ny1 + rh, nx1:nx1 + rw] = (
                        image[ny1:ny1 + rh, nx1:nx1 + rw] * (1 - alpha)
                        + obj_img[:rh, :rw] * alpha
                    )
                else:
                    image[ny1:ny1 + rh, nx1:nx1 + rw][obj_m[:rh, :rw]] = obj_img[:rh, :rw][obj_m[:rh, :rw]]
                label[ny1:ny1 + rh, nx1:nx1 + rw][obj_m[:rh, :rw]] = donor_lbl[y1:y2, x1:x2].max() * obj_m[:rh, :rw].astype(np.int64)
            else:
                if blend_ratio > 0:
                    alpha = obj_mask.astype(np.float32)[..., None] * (1 - blend_ratio)
                    alpha = np.broadcast_to(alpha, image.shape)
                    image = image * (1 - alpha) + donor_img * alpha
                else:
                    image[obj_mask] = donor_img[obj_mask]
                label[obj_mask] = donor_lbl[obj_mask]

        return {"image": image, "label": label}


# ─── Mosaic ──────────────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("mosaic")
class Mosaic:
    """Mosaic augmentation: combine 4 images into a 2x2 grid.

    Args:
        p: Probability of applying the augmentation.
        mosaic_size: Number of images to combine (default 4 = 2x2).
        offset_range: Center offset range as fraction of image size (0=no offset).
        dataset: Dataset reference for sampling.
    """

    def __init__(self, p=0.5, mosaic_size=4, offset_range=(0.0, 0.2), dataset=None, **kwargs):
        self.p = p
        self.mosaic_size = mosaic_size
        self.offset_range = offset_range
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p or self.dataset is None:
            return sample

        indices = random.sample(range(len(self.dataset)), min(self.mosaic_size - 1, len(self.dataset) - 1))
        if len(indices) < self.mosaic_size - 1:
            return sample

        image = _to_numpy_hwc(sample["image"])
        label = sample["label"] if isinstance(sample["label"], np.ndarray) else sample["label"].numpy()
        h, w = image.shape[:2]
        c = image.shape[-1] if image.ndim == 3 else 1
        half_h, half_w = h // 2, w // 2

        samples = [(image, label)]
        for idx in indices:
            donor = self.dataset[idx]
            d_img = _to_numpy_hwc(donor["image"])
            d_lbl = donor["label"]
            if isinstance(d_lbl, torch.Tensor):
                d_lbl = d_lbl.numpy()
            if d_img.shape[:2] != (half_h, half_w):
                d_img_t = torch.from_numpy(d_img).permute(2, 0, 1).unsqueeze(0) if d_img.ndim == 3 else torch.from_numpy(d_img).unsqueeze(0).unsqueeze(0)
                d_img_t = F.interpolate(d_img_t, size=(half_h, half_w), mode="bilinear", align_corners=False)
                d_img = d_img_t.squeeze(0).permute(1, 2, 0).numpy() if d_img.ndim == 3 else d_img_t.squeeze(0).squeeze(0).numpy()
            d_lbl_t = torch.from_numpy(d_lbl).unsqueeze(0).unsqueeze(0).float()
            d_lbl_t = F.interpolate(d_lbl_t, size=(half_h, half_w), mode="nearest")
            d_lbl = d_lbl_t.squeeze().numpy().astype(np.int64)
            samples.append((d_img, d_lbl))

        canvas_img = np.zeros((half_h * 2, half_w * 2, c), dtype=np.float32) if c > 1 else np.zeros((half_h * 2, half_w * 2), dtype=np.float32)
        canvas_lbl = np.zeros((half_h * 2, half_w * 2), dtype=np.int64)

        # Random center offset
        off_frac = random.uniform(*self.offset_range)
        cy_off = int(random.uniform(-off_frac, off_frac) * h)
        cx_off = int(random.uniform(-off_frac, off_frac) * w)
        mid_h = half_h + cy_off
        mid_w = half_w + cx_off

        positions = [(0, 0), (0, mid_w), (mid_h, 0), (mid_h, mid_w)]
        for (img, lbl), (y, x) in zip(samples, positions):
            sh, sw = img.shape[:2]
            if c > 1:
                canvas_img[y:y + sh, x:x + sw, :img.shape[-1] if img.ndim == 3 else 1] = img if img.ndim == 3 else img[..., None]
            else:
                canvas_img[y:y + sh, x:x + sw] = img[..., 0] if img.ndim == 3 else img
            canvas_lbl[y:y + sh, x:x + sw] = lbl

        y_off = random.randint(0, max(0, canvas_img.shape[0] - h))
        x_off = random.randint(0, max(0, canvas_img.shape[1] - w))
        image = canvas_img[y_off:y_off + h, x_off:x_off + w]
        label = canvas_lbl[y_off:y_off + h, x_off:x_off + w]

        return {"image": image, "label": label}


# ─── PhotometricDistortion ──────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("photometric_distortion")
class PhotometricDistortion:
    """Photometric distortion: random brightness, contrast, saturation, hue.

    Args:
        p: Probability of applying.
        brightness_range: Brightness jitter range (additive).
        contrast_range: Contrast scale range (multiplicative).
        saturation_range: Saturation scale range.
        hue_range: Hue shift range (degrees).
    """

    def __init__(self, p=0.5, brightness_range=(-0.3, 0.3), contrast_range=(0.7, 1.3),
                 saturation_range=(0.7, 1.3), hue_range=(-18, 18), **kwargs):
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = _to_numpy_hwc(sample["image"])
        transforms = [
            self._brightness, self._contrast,
            self._saturation, self._hue,
        ]
        random.shuffle(transforms)
        for fn in transforms:
            image = fn(image)

        image = np.clip(image, 0, 1)
        return {"image": image, "label": sample["label"]}

    def _brightness(self, img):
        return img + random.uniform(*self.brightness_range)

    def _contrast(self, img):
        alpha = random.uniform(*self.contrast_range)
        mean = img.mean()
        return (img - mean) * alpha + mean

    def _saturation(self, img):
        if img.shape[-1] < 3:
            return img
        alpha = random.uniform(*self.saturation_range)
        gray = img.mean(axis=-1, keepdims=True)
        return gray + (img - gray) * alpha

    def _hue(self, img):
        if img.shape[-1] < 3:
            return img
        shift = random.uniform(*self.hue_range) / 360.0
        angle = shift * 2 * math.pi
        cos_v, sin_v = math.cos(angle), math.sin(angle)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        new_r = r * cos_v + g * sin_v + b * (1 - cos_v - sin_v) * 0.333
        new_g = r * (1 - cos_v - sin_v) * 0.333 + g * cos_v + b * sin_v
        new_b = r * sin_v + g * (1 - cos_v - sin_v) * 0.333 + b * cos_v
        return np.stack([new_r, new_g, new_b], axis=-1)


# ─── GridMask ────────────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("grid_mask")
class GridMask:
    """GridMask: drop regular grid cells to force robust feature learning.

    Args:
        p: Probability of applying.
        d_range: Range of grid cell sizes (min_d, max_d) as fraction of image.
        ratio_range: Range of masked area ratio within each cell (0-1).
        rotate_range: Random rotation angle range for the grid.
    """

    def __init__(self, p=0.5, d_range=(0.05, 0.15), ratio_range=(0.3, 0.7),
                 rotate_range=(0, 0), **kwargs):
        self.p = p
        self.d_range = d_range
        self.ratio_range = ratio_range
        self.rotate_range = rotate_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = sample["image"]
        if isinstance(image, np.ndarray):
            h, w = image.shape[0], image.shape[1]
        else:
            h, w = image.shape[-2], image.shape[-1]

        d = int(max(h, w) * random.uniform(*self.d_range))
        if d < 2:
            return sample
        ratio = random.uniform(*self.ratio_range)
        l = int(d * ratio)

        mask = np.ones((h + d, w + d), dtype=np.float32)
        for i in range(0, h + d, d):
            for j in range(0, w + d, d):
                mask[i:i + l, j:j + l] = 0

        y_off = random.randint(0, d - 1)
        x_off = random.randint(0, d - 1)
        mask = mask[y_off:y_off + h, x_off:x_off + w]

        if isinstance(image, torch.Tensor):
            mask_t = torch.from_numpy(mask)
            if image.dim() == 3 and image.shape[0] in (1, 3):
                mask_t = mask_t.unsqueeze(0)  # (1, H, W) for CHW
            image = image * mask_t
        else:
            if image.ndim == 3:
                image = image * mask[..., None]
            else:
                image = image * mask

        return {"image": image, "label": sample["label"]}


# ─── CLAHE ───────────────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("clahe")
class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization.

    Args:
        p: Probability of applying.
        clip_limit_range: Contrast limiting threshold range, randomly sampled each call.
        tile_size_range: Tile grid size range, randomly sampled each call.
    """

    def __init__(self, p=0.5, clip_limit_range=(1.0, 5.0), tile_size_range=(4, 16), **kwargs):
        self.p = p
        self.clip_limit_range = clip_limit_range
        self.tile_size_range = tile_size_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = _to_numpy_hwc(sample["image"])
        clip_limit = random.uniform(*self.clip_limit_range)
        tile_size = random.randint(self.tile_size_range[0], self.tile_size_range[1])

        if image.ndim == 3:
            for c in range(image.shape[-1]):
                image[..., c] = self._apply_clahe(image[..., c], clip_limit, tile_size)
        else:
            image = self._apply_clahe(image, clip_limit, tile_size)

        return {"image": np.clip(image, 0, 1), "label": sample["label"]}

    def _apply_clahe(self, channel: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
        h, w = channel.shape
        ts = tile_size
        result = np.zeros_like(channel)
        for i in range(0, h, ts):
            for j in range(0, w, ts):
                tile = channel[i:min(i + ts, h), j:min(j + ts, w)]
                mn, mx = tile.min(), tile.max()
                if mx - mn < 1e-8:
                    result[i:min(i + ts, h), j:min(j + ts, w)] = tile
                    continue
                tile_norm = ((tile - mn) / (mx - mn) * 255).astype(np.uint8)
                hist, _ = np.histogram(tile_norm, bins=256, range=(0, 256))
                clip = int(clip_limit * tile.size / 256)
                excess = np.sum(np.maximum(hist - clip, 0))
                hist = np.minimum(hist, clip)
                hist += excess // 256
                cdf = hist.cumsum().astype(np.float64)
                cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8) * 255
                mapped = cdf[tile_norm] / 255.0 * (mx - mn) + mn
                result[i:min(i + ts, h), j:min(j + ts, w)] = mapped
        return result


# ─── Geometric transforms (using torch for interpolation) ────────────────────


class _GeometricBase:
    """Base class for geometric transforms that need torch interpolation.

    Handles HWC numpy <-> CHW torch conversion transparently.
    """

    def _apply_grid(self, image, label, grid):
        """Apply affine grid to image (bilinear) and label (nearest).

        Args:
            image: HWC numpy or CHW tensor
            label: HW numpy or tensor
            grid: (1, H, W, 2) affine grid

        Returns:
            (image_out, label_out) in same format as input
        """
        was_np = isinstance(image, np.ndarray)
        img_t = _img_to_chw_torch(image)
        lab_t = _lab_to_4d_torch(label)

        img_out = F.grid_sample(img_t, grid, mode="bilinear",
                                padding_mode="reflection", align_corners=False)
        lab_out = F.grid_sample(lab_t, grid, mode="nearest",
                                padding_mode="zeros", align_corners=False)

        return _from_chw_to_hwc(img_out, was_np), _from_4d_to_hw(lab_out, was_np)


# ─── RandomAffine ────────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("random_affine")
class RandomAffine(_GeometricBase):
    """Random affine transformation: rotation + translation + scale + shear.

    Args:
        p: Probability of applying.
        degrees_range: Rotation angle range in degrees.
        translate_range: Translation range as fraction of image size.
        scale_range: Scale range.
        shear_range: Shear angle range in degrees.
    """

    def __init__(self, p=0.5, degrees_range=(-15, 15), translate_range=(0.0, 0.1),
                 scale_range=(0.9, 1.1), shear_range=(-5, 5), **kwargs):
        self.p = p
        self.degrees_range = degrees_range
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.shear_range = shear_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = sample["image"]
        label = sample["label"]
        was_np = isinstance(image, np.ndarray)

        img_t = _img_to_chw_torch(image)
        h, w = img_t.shape[-2], img_t.shape[-1]

        angle = random.uniform(*self.degrees_range)
        tx = random.uniform(*self.translate_range) * w
        ty = random.uniform(*self.translate_range) * h
        sx = random.uniform(*self.scale_range)
        sy = random.uniform(*self.scale_range)
        shx = math.tan(math.radians(random.uniform(*self.shear_range)))
        shy = math.tan(math.radians(random.uniform(*self.shear_range)))

        theta = math.radians(angle)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        m11 = cos_a * sx + shy * sin_a * sy
        m12 = -sin_a * sy + shy * cos_a * sx
        m21 = sin_a * sx + shx * cos_a * sy
        m22 = cos_a * sy + shx * sin_a * sx
        m13 = 2 * tx / w
        m23 = 2 * ty / h

        aff = torch.tensor([[m11, m12, m13], [m21, m22, m23]], dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(aff, img_t.shape, align_corners=False)

        img_out, lab_out = self._apply_grid(image, label, grid)
        return {"image": img_out, "label": lab_out}


# ─── RandomPerspective ──────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("random_perspective")
class RandomPerspective:
    """Random perspective transformation.

    Args:
        p: Probability of applying.
        distortion_scale_range: Maximum distortion range as fraction of image size.
    """

    def __init__(self, p=0.5, distortion_scale_range=(0.05, 0.15), **kwargs):
        self.p = p
        self.distortion_scale_range = distortion_scale_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = sample["image"]
        label = sample["label"]
        was_np = isinstance(image, np.ndarray)

        img_t = _img_to_chw_torch(image)
        h, w = img_t.shape[-2], img_t.shape[-1]

        d = random.uniform(*self.distortion_scale_range)
        src = torch.tensor([[0, 0], [w, 0], [w, h], [0, h]], dtype=torch.float32)
        dst = src.clone()
        dst[0] += torch.tensor([random.uniform(0, d * w), random.uniform(0, d * h)])
        dst[1] += torch.tensor([random.uniform(-d * w, 0), random.uniform(0, d * h)])
        dst[2] += torch.tensor([random.uniform(-d * w, 0), random.uniform(-d * h, 0)])
        dst[3] += torch.tensor([random.uniform(0, d * w), random.uniform(-d * h, 0)])

        H = self._compute_homography(src, dst)
        if H is None:
            return sample

        grid = self._perspective_grid(H, h, w).unsqueeze(0)

        lab_t = _lab_to_4d_torch(label)
        img_out = F.grid_sample(img_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        lab_out = F.grid_sample(lab_t, grid, mode="nearest", padding_mode="zeros", align_corners=False)

        return {"image": _from_chw_to_hwc(img_out, was_np), "label": _from_4d_to_hw(lab_out, was_np)}

    @staticmethod
    def _compute_homography(src, dst):
        A = []
        for i in range(4):
            x1, y1 = src[i]
            x2, y2 = dst[i]
            A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
            A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
        A = torch.tensor(A, dtype=torch.float32)
        try:
            U, S, V = torch.linalg.svd(A)
            H = V[-1].view(3, 3)
            H = H / (H[2, 2] + 1e-8)
            return H
        except Exception:
            return None

    @staticmethod
    def _perspective_grid(H, h, w):
        ys = torch.linspace(-1, 1, h)
        xs = torch.linspace(-1, 1, w)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ones = torch.ones_like(grid_x)
        coords = torch.stack([grid_x, grid_y, ones], dim=-1)
        H_inv = torch.inverse(H)
        flat = coords.reshape(-1, 3)
        src_coords = flat @ H_inv.T
        src_coords = src_coords[:, :2] / (src_coords[:, 2:3] + 1e-8)
        src_coords[:, 0] = src_coords[:, 0] / (w / 2) - 1
        src_coords[:, 1] = src_coords[:, 1] / (h / 2) - 1
        return src_coords.reshape(h, w, 2)


# ─── RandomErasing ───────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("random_erasing")
class RandomErasing:
    """Random Erasing: randomly erase rectangular regions (image only).

    Args:
        p: Probability of applying.
        scale_range: Range of erased area as fraction of total image area.
        ratio_range: Range of aspect ratio of erased region.
        fill_value: Value to fill erased region ("random" or float).
        max_count: Maximum number of erasing regions.
    """

    def __init__(self, p=0.5, scale_range=(0.02, 0.15), ratio_range=(0.3, 3.3),
                 fill_value="random", max_count=1, **kwargs):
        self.p = p
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.fill_value = fill_value
        self.max_count = max_count

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = sample["image"]
        was_np = isinstance(image, np.ndarray)
        if was_np:
            h, w = image.shape[0], image.shape[1]
        else:
            h, w = image.shape[-2], image.shape[-1]
        area = h * w

        image = image.copy() if was_np else image.clone()

        for _ in range(self.max_count):
            target_area = random.uniform(*self.scale_range) * area
            log_ratio = (math.log(self.ratio_range[0]), math.log(self.ratio_range[1]))
            aspect = math.exp(random.uniform(*log_ratio))
            rh = int(round(math.sqrt(target_area * aspect)))
            rw = int(round(math.sqrt(target_area / aspect)))
            if rh >= h or rw >= w:
                continue
            y = random.randint(0, h - rh)
            x = random.randint(0, w - rw)

            if was_np:
                if self.fill_value == "random":
                    fill = np.random.randn(rh, rw, image.shape[-1] if image.ndim == 3 else 1).astype(np.float32)
                    if image.ndim == 2:
                        fill = fill.reshape(rh, rw)
                else:
                    fill = np.full((rh, rw) + (image.shape[-1:] if image.ndim == 3 else ()), self.fill_value, dtype=np.float32)
                image[y:y + rh, x:x + rw] = fill
            else:
                if self.fill_value == "random":
                    fill = torch.randn(image.shape[:-2] + (rh, rw))
                else:
                    fill = torch.full(image.shape[:-2] + (rh, rw), self.fill_value)
                image[..., y:y + rh, x:x + rw] = fill

        return {"image": image, "label": sample["label"]}


# ─── RandomSolarize ──────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("random_solarize")
class RandomSolarize:
    """Random Solarize: invert pixels above a threshold.

    Args:
        p: Probability of applying.
        threshold_range: Pixel value threshold range for inversion (0-1 range),
            randomly sampled each call.
    """

    def __init__(self, p=0.3, threshold_range=(0.3, 0.7), **kwargs):
        self.p = p
        self.threshold_range = threshold_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        threshold = random.uniform(*self.threshold_range)
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            image = image.clone()
            mask = image > threshold
            image[mask] = 1.0 - image[mask]
        else:
            image = image.copy()
            mask = image > threshold
            image[mask] = 1.0 - image[mask]

        return {"image": image, "label": sample["label"]}


# ─── ColorJitter ─────────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("color_jitter")
class ColorJitter:
    """Comprehensive color jittering with random order.

    Args:
        p: Probability of applying.
        brightness_range: Brightness adjustment range.
        contrast_range: Contrast adjustment range.
        saturation_range: Saturation adjustment range.
        hue_range: Hue shift range.
    """

    def __init__(self, p=0.5, brightness_range=(-0.2, 0.2), contrast_range=(0.8, 1.2),
                 saturation_range=(0.8, 1.2), hue_range=(-0.1, 0.1), **kwargs):
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = _to_numpy_hwc(sample["image"])

        order = [0, 1, 2, 3]
        random.shuffle(order)
        for i in order:
            if i == 0:
                image = image + random.uniform(*self.brightness_range)
            elif i == 1:
                image = image * random.uniform(*self.contrast_range)
            elif i == 2 and image.shape[-1] >= 3:
                factor = random.uniform(*self.saturation_range)
                gray = image.mean(axis=-1, keepdims=True)
                image = gray + (image - gray) * factor
            elif i == 3 and image.shape[-1] >= 3:
                shift = random.uniform(*self.hue_range) * 180
                angle = math.radians(shift)
                cos_v, sin_v = math.cos(angle), math.sin(angle)
                r, g, b = image[..., 0].copy(), image[..., 1].copy(), image[..., 2].copy()
                image[..., 0] = r * cos_v + g * sin_v
                image[..., 1] = -r * sin_v + g * cos_v

        return {"image": np.clip(image, 0, 1), "label": sample["label"]}


# ─── Posterize ───────────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("posterize")
class Posterize:
    """Reduce the number of bits per channel.

    Args:
        p: Probability of applying.
        bits_range: Range of bits to keep (1-8), randomly sampled each call.
    """

    def __init__(self, p=0.3, bits_range=(2, 6), **kwargs):
        self.p = p
        self.bits_range = bits_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample
        bits = random.randint(max(1, self.bits_range[0]), min(8, self.bits_range[1]))
        levels = 2 ** bits
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            image = torch.floor(image * levels) / levels
        else:
            image = np.floor(image * levels) / levels
        return {"image": image, "label": sample["label"]}


# ─── Sharpness ───────────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("sharpness")
class Sharpness:
    """Random sharpness adjustment using unsharp masking.

    Args:
        p: Probability of applying.
        factor_range: Range of sharpness factor.
    """

    def __init__(self, p=0.3, factor_range=(0.5, 2.0), **kwargs):
        self.p = p
        self.factor_range = factor_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = sample["image"]
        factor = random.uniform(*self.factor_range)
        was_np = isinstance(image, np.ndarray)

        img_t = _img_to_chw_torch(image)
        kernel = torch.ones(1, 1, 3, 3) / 9.0
        c = img_t.shape[1]
        kernel = kernel.expand(c, -1, -1, -1)
        blurred = F.conv2d(img_t, kernel, padding=1, groups=c)
        img_t = img_t + factor * (img_t - blurred)
        img_t = torch.clamp(img_t, 0, 1)

        return {"image": _from_chw_to_hwc(img_t, was_np), "label": sample["label"]}


# ─── ChannelDropout ──────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("channel_dropout")
class ChannelDropout:
    """Randomly drop one or more color channels.

    Args:
        p: Probability of applying.
        drop_count_range: Range of number of channels to drop per call.
        fill_value: Value to fill dropped channels.
    """

    def __init__(self, p=0.3, drop_count_range=(1, 1), fill_value=0.0, **kwargs):
        self.p = p
        self.drop_count_range = drop_count_range
        self.fill_value = fill_value

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = sample["image"]
        if isinstance(image, torch.Tensor):
            image = image.clone()
            if image.dim() == 3 and image.shape[0] > 1:
                n_ch = image.shape[0]
                n_drop = random.randint(self.drop_count_range[0], min(self.drop_count_range[1], n_ch - 1))
                drop_indices = random.sample(range(n_ch), n_drop)
                for idx in drop_indices:
                    image[idx] = self.fill_value
        else:
            image = image.copy()
            if image.ndim == 3 and image.shape[-1] > 1:
                n_ch = image.shape[-1]
                n_drop = random.randint(self.drop_count_range[0], min(self.drop_count_range[1], n_ch - 1))
                drop_indices = random.sample(range(n_ch), n_drop)
                for idx in drop_indices:
                    image[..., idx] = self.fill_value

        return {"image": image, "label": sample["label"]}


# ─── CoarseDropout ───────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("coarse_dropout")
class CoarseDropout:
    """Drop large rectangular blocks.

    Args:
        p: Probability of applying.
        num_holes_range: Range of number of holes to drop.
        hole_height_range: Range of hole height as fraction of image height.
        hole_width_range: Range of hole width as fraction of image width.
        fill_value: Value to fill holes.
    """

    def __init__(self, p=0.5, num_holes_range=(1, 8),
                 hole_height_range=(0.02, 0.15), hole_width_range=(0.02, 0.15),
                 fill_value=0.0, **kwargs):
        self.p = p
        self.num_holes_range = num_holes_range
        self.hole_height_range = hole_height_range
        self.hole_width_range = hole_width_range
        self.fill_value = fill_value

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = sample["image"]
        was_np = isinstance(image, np.ndarray)
        h, w = (image.shape[0], image.shape[1]) if was_np else (image.shape[-2], image.shape[-1])

        image = image.copy() if was_np else image.clone()
        n_holes = random.randint(*self.num_holes_range)

        for _ in range(n_holes):
            rh = int(h * random.uniform(*self.hole_height_range))
            rw = int(w * random.uniform(*self.hole_width_range))
            if rh < 1 or rw < 1:
                continue
            y = random.randint(0, max(0, h - rh))
            x = random.randint(0, max(0, w - rw))
            if was_np:
                image[y:y + rh, x:x + rw] = self.fill_value
            else:
                image[..., y:y + rh, x:x + rw] = self.fill_value

        return {"image": image, "label": sample["label"]}


# ─── GaussianBlur ────────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("gaussian_blur")
class GaussianBlurAug:
    """Apply random Gaussian blur.

    Args:
        p: Probability of applying.
        kernel_range: Range of kernel sizes (must be odd).
        sigma_range: Range of sigma values.
    """

    def __init__(self, p=0.3, kernel_range=(3, 7), sigma_range=(0.1, 2.0), **kwargs):
        self.p = p
        self.kernel_range = kernel_range
        self.sigma_range = sigma_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        image = sample["image"]
        was_np = isinstance(image, np.ndarray)
        ks = random.choice([k for k in range(self.kernel_range[0], self.kernel_range[1] + 1) if k % 2 == 1])
        sigma = random.uniform(*self.sigma_range)

        ax = torch.arange(-ks // 2 + 1., ks // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        img_t = _img_to_chw_torch(image)
        c = img_t.shape[1]
        k = kernel.view(1, 1, ks, ks).expand(c, -1, -1, -1)
        img_t = F.conv2d(img_t, k, padding=ks // 2, groups=c)

        return {"image": _from_chw_to_hwc(img_t, was_np), "label": sample["label"]}


# ─── GammaCorrection ─────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("gamma_correction")
class GammaCorrectionAug:
    """Random gamma correction.

    Args:
        p: Probability of applying.
        gamma_range: Range of gamma values.
    """

    def __init__(self, p=0.3, gamma_range=(0.7, 1.5), **kwargs):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample
        gamma = random.uniform(*self.gamma_range)
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            mn, mx = image.min(), image.max()
            if mx - mn > 0:
                image = ((image - mn) / (mx - mn)) ** gamma * (mx - mn) + mn
        else:
            mn, mx = image.min(), image.max()
            if mx - mn > 0:
                image = ((image - mn) / (mx - mn)) ** gamma * (mx - mn) + mn
        return {"image": image, "label": sample["label"]}


# ─── ElasticDeform ───────────────────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("elastic_deform")
class ElasticDeformAug(_GeometricBase):
    """Random elastic deformation.

    Args:
        p: Probability of applying.
        alpha_range: Deformation magnitude range, randomly sampled each call.
        sigma_range: Gaussian smoothing sigma range, randomly sampled each call.
    """

    def __init__(self, p=0.3, alpha_range=(20, 80), sigma_range=(3, 7), **kwargs):
        self.p = p
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    def _gaussian_filter(self, x, sigma):
        ks = int(4 * sigma + 1)
        if ks % 2 == 0:
            ks += 1
        ax = torch.arange(-ks // 2 + 1., ks // 2 + 1.)
        gauss = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel = (gauss / gauss.sum()).view(1, 1, -1)
        x = F.conv1d(x.view(1, 1, -1), kernel, padding=ks // 2).view(x.shape)
        return x

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        alpha = random.uniform(*self.alpha_range)
        sigma = random.uniform(*self.sigma_range)

        image = sample["image"]
        label = sample["label"]
        was_np = isinstance(image, np.ndarray)

        img_t = _img_to_chw_torch(image)
        h, w = img_t.shape[-2], img_t.shape[-1]

        dx = torch.rand(h, w) * 2 - 1
        dy = torch.rand(h, w) * 2 - 1
        for i in range(h):
            dx[i] = self._gaussian_filter(dx[i], sigma)
            dy[i] = self._gaussian_filter(dy[i], sigma)
        dx = dx * alpha / w
        dy = dy * alpha / h

        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        grid = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)

        img_out, lab_out = self._apply_grid(image, label, grid)
        return {"image": img_out, "label": lab_out}


# ─── Simple spatial transforms ──────────────────────────────────────────────


@AUGMENTATION_REGISTRY.register("horizontal_flip")
class HorizontalFlip:
    """Random horizontal flip. Args: p."""

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, label = sample["image"], sample["label"]
            if isinstance(image, np.ndarray):
                image = np.flip(image, axis=1).copy()  # HWC: flip along W (axis=1)
                label = np.flip(label, axis=1).copy() if label.ndim == 2 else np.flip(label, axis=1).copy()
            else:
                image = torch.flip(image, dims=[-1])
                label = torch.flip(label, dims=[-1])
            return {"image": image, "label": label}
        return sample


@AUGMENTATION_REGISTRY.register("vertical_flip")
class VerticalFlip:
    """Random vertical flip. Args: p."""

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, label = sample["image"], sample["label"]
            if isinstance(image, np.ndarray):
                image = np.flip(image, axis=0).copy()  # HWC: flip along H (axis=0)
                label = np.flip(label, axis=0).copy()
            else:
                image = torch.flip(image, dims=[-2])
                label = torch.flip(label, dims=[-2])
            return {"image": image, "label": label}
        return sample


@AUGMENTATION_REGISTRY.register("random_rotate90")
class RandomRotate90Aug:
    """Random 90/180/270 degree rotation. Args: p."""

    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            k = random.randint(1, 3)
            image, label = sample["image"], sample["label"]
            if isinstance(image, np.ndarray):
                # HWC: rotate axes (0, 1) which are H, W
                image = np.rot90(image, k, axes=(0, 1)).copy()
                label = np.rot90(label, k, axes=(0, 1)).copy()
            else:
                # CHW or HW: rotate last two dims
                image = torch.rot90(image, k, dims=[-2, -1])
                label = torch.rot90(label, k, dims=[-2, -1])
            return {"image": image, "label": label}
        return sample


@AUGMENTATION_REGISTRY.register("random_rotate")
class RandomRotateAug(_GeometricBase):
    """Random rotation within an arbitrary angle range.

    Args:
        p: Probability of applying.
        degrees_range: Range of rotation angles in degrees, e.g. (-45, 45) or (0, 180).
    """

    def __init__(self, p=0.5, degrees_range=(-15, 15), **kwargs):
        self.p = p
        self.degrees_range = degrees_range

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        angle = random.uniform(*self.degrees_range)
        if abs(angle) < 1e-6:
            return sample

        image, label = sample["image"], sample["label"]
        img_t = _img_to_chw_torch(image)
        h, w = img_t.shape[-2], img_t.shape[-1]

        theta = math.radians(angle)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        aff = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(aff, img_t.shape, align_corners=False)

        img_out, lab_out = self._apply_grid(image, label, grid)
        return {"image": img_out, "label": lab_out}


@AUGMENTATION_REGISTRY.register("random_scale")
class RandomScaleAug:
    """Random scaling with resize back. Args: p, scale_range."""

    def __init__(self, p=0.5, scale_range=(0.8, 1.2), **kwargs):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        scale = random.uniform(*self.scale_range)
        image = sample["image"]
        label = sample["label"]
        was_np = isinstance(image, np.ndarray)

        img_t = _img_to_chw_torch(image)
        lab_t = _lab_to_4d_torch(label)
        h, w = img_t.shape[-2], img_t.shape[-1]
        new_h, new_w = int(h * scale), int(w * scale)

        img_t = F.interpolate(img_t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        img_t = F.interpolate(img_t, size=(h, w), mode="bilinear", align_corners=False)
        lab_t = F.interpolate(lab_t, size=(new_h, new_w), mode="nearest")
        lab_t = F.interpolate(lab_t, size=(h, w), mode="nearest")

        return {"image": _from_chw_to_hwc(img_t, was_np), "label": _from_4d_to_hw(lab_t, was_np)}


@AUGMENTATION_REGISTRY.register("gaussian_noise")
class GaussianNoiseAug:
    """Add Gaussian noise. Args: p, std_range."""

    def __init__(self, p=0.3, std_range=(0.01, 0.08), **kwargs):
        self.p = p
        self.std_range = std_range

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        std = random.uniform(*self.std_range)
        image = sample["image"]
        if isinstance(image, np.ndarray):
            noise = np.random.normal(0, std, image.shape).astype(np.float32)
        else:
            noise = torch.randn_like(image) * std
        return {"image": image + noise, "label": sample["label"]}


@AUGMENTATION_REGISTRY.register("brightness_contrast")
class BrightnessContrastAug:
    """Random brightness and contrast. Args: p, brightness_range, contrast_range."""

    def __init__(self, p=0.3, brightness_range=(-0.2, 0.2), contrast_range=(0.8, 1.2), **kwargs):
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, sample):
        if random.random() > self.p:
            return sample
        image = sample["image"]
        b = random.uniform(*self.brightness_range)
        c = random.uniform(*self.contrast_range)
        mean = image.mean()
        image = (image - mean) * c + mean + b
        return {"image": image, "label": sample["label"]}


# ─── Pipeline builder ────────────────────────────────────────────────────────


def build_augmentation_pipeline(aug_config: list, img_size: int = 224, dataset=None) -> "Compose":
    """Build augmentation pipeline from YAML config.

    Args:
        aug_config: List of dicts with 'name' and optional 'params'.
            Example:
                - name: horizontal_flip
                  params: {p: 0.5}
                - name: copy_paste
                  params: {p: 0.3}
        img_size: Target image size.
        dataset: Dataset reference (for sample-level augmentations).

    Returns:
        Compose transform.
    """
    from medseg.datasets.transforms import Compose, Resize, Normalize

    transforms = [Resize(img_size)]

    for aug_cfg in aug_config:
        name = aug_cfg["name"]
        params = aug_cfg.get("params", {}) or {}
        aug_cls = AUGMENTATION_REGISTRY.get(name)
        aug = aug_cls(**params)
        if hasattr(aug, "set_dataset") and dataset is not None:
            aug.set_dataset(dataset)
        transforms.append(aug)

    transforms.append(Normalize())
    return Compose(transforms)


def list_available_augmentations() -> list:
    """Return list of all registered augmentation names."""
    return AUGMENTATION_REGISTRY.list_available()
