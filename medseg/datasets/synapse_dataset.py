"""Synapse / ACDC loader for TransUNet-style preprocessed data.

Supported on-disk layouts
-------------------------
**Train / val (2D slices)** — TransUNet default (one sample per file):

- ``.npz`` with ``image`` / ``label`` shaped ``(H, W)`` (e.g. ``case0005_slice000.npz``)
- Optional ``.npz`` with stacked slices ``(D, H, W)`` (legacy / custom packs)

**Test / val volumes (3D)**:

- ``.h5`` or ``.npz`` with ``image`` / ``label`` shaped ``(D, H, W)`` (e.g. ``case0001.npy.h5``)

Grayscale CT/MRI is replicated to 3 channels before returning tensors so encoders
with ``in_channels: 3`` work out of the box. Spatial resize to ``img_size`` is
handled by ``get_train_transforms`` / ``get_val_transforms``.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)


class VolumeLayout(str, Enum):
    """How ``image`` is stored inside a single file."""

    SLICE_2D = "slice_2d"          # (H, W) — TransUNet train_npz / val_npz
    VOLUME_3D = "volume_3d"        # (D, H, W)
    SLICE_2D_HWC = "slice_2d_hwc"  # (H, W, C) with C in {1, 3}


def _classify_shape(shape: tuple) -> VolumeLayout:
    if len(shape) == 2:
        return VolumeLayout.SLICE_2D
    if len(shape) == 3:
        if shape[-1] in (1, 3):
            return VolumeLayout.SLICE_2D_HWC
        if shape[1] == shape[2]:
            return VolumeLayout.VOLUME_3D
        raise ValueError(
            f"Unrecognized 3D shape {shape}. "
            "Expected (D,H,W) with H==W, or (H,W,C) with C in {{1,3}}."
        )
    raise ValueError(
        f"Expected shape (H,W), (H,W,C), or (D,H,W), got {shape}"
    )


def _classify_volume_layout(image: np.ndarray) -> VolumeLayout:
    return _classify_shape(tuple(image.shape))


def _volume_slice_count_from_shape(shape: tuple) -> int:
    layout = _classify_shape(shape)
    if layout in (VolumeLayout.SLICE_2D, VolumeLayout.SLICE_2D_HWC):
        return 1
    return int(shape[0])


def _volume_slice_count(image: np.ndarray) -> int:
    return _volume_slice_count_from_shape(tuple(image.shape))


def _is_volume_3d(image: np.ndarray, label: np.ndarray) -> bool:
    """True when arrays are ``(D, H, W)`` volumes (not HWC)."""
    if image.ndim != 3 or label.ndim != 3:
        return False
    if image.shape[-1] in (1, 3):
        return False
    return image.shape[1] == image.shape[2] and label.shape == image.shape


def _to_hw(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize raw arrays to 2D ``(H, W)`` before transforms."""
    img_layout = _classify_volume_layout(image)
    if img_layout == VolumeLayout.SLICE_2D_HWC:
        if image.shape[-1] == 1:
            image = image[..., 0]
        else:
            image = np.mean(image, axis=-1)
    if label.ndim == 3:
        if label.shape[-1] == 1:
            label = label[..., 0]
        elif label.shape[0] == label.shape[1]:
            raise ValueError(
                f"Label shape {label.shape} does not match 2D slice layout; "
                "use (H,W) or (H,W,1)."
            )
        else:
            label = label[0]
    if image.ndim != 2 or label.ndim != 2:
        raise ValueError(
            f"After normalization expected 2D image/label, got "
            f"image={image.shape}, label={label.shape}"
        )
    if image.shape[:2] != label.shape[:2]:
        raise ValueError(
            f"Image/label spatial mismatch: image={image.shape}, label={label.shape}"
        )
    return image, label


def _read_volume_slice(
    image: np.ndarray, label: np.ndarray, slice_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    layout = _classify_volume_layout(image)
    if layout in (VolumeLayout.SLICE_2D, VolumeLayout.SLICE_2D_HWC):
        if slice_idx not in (-1, 0):
            raise IndexError(
                f"2D slice file only has one sample (slice_idx={slice_idx})."
            )
        return _to_hw(image, label)
    if slice_idx >= 0:
        return _to_hw(image[slice_idx], label[slice_idx])
    return image.astype(np.float32), label.astype(np.int64)


def _arrays_to_model_tensors(image, label) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert post-transform arrays/tensors to ``(3,H,W)`` float + ``(H,W)`` long."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=0)
        elif image.ndim == 3 and image.shape[-1] in (1, 3):
            image = image.transpose(2, 0, 1)
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
        else:
            raise ValueError(f"Cannot convert image array with shape {image.shape}")
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(np.asarray(label)).long()
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        label = label.long() if isinstance(label, torch.Tensor) else torch.from_numpy(label).long()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    return image, label


def _discover_sample_names(root_dir: str) -> List[str]:
    names: List[str] = []
    for fname in sorted(os.listdir(root_dir)):
        if fname.endswith(".npy.h5"):
            names.append(fname[: -len(".npy.h5")])
        elif fname.endswith(".h5"):
            names.append(fname[: -len(".h5")])
        elif fname.endswith(".npz"):
            names.append(fname[: -len(".npz")])
        elif fname.endswith(".npy"):
            names.append(fname[: -len(".npy")])
    return names


def _resolve_file_paths(root_dir: str, name: str) -> Tuple[Optional[str], Optional[str]]:
    candidates = [
        os.path.join(root_dir, name + ext)
        for ext in (".npy.h5", ".h5", ".npz", ".npy")
    ]
    for path in candidates:
        if os.path.exists(path):
            if path.endswith(".h5") or path.endswith(".npy.h5"):
                return path, None
            return None, path
    return None, None


def inspect_volume_file(path: str) -> dict:
    """Return layout metadata for one ``.npz`` / ``.h5`` file (for verification)."""
    if path.endswith((".h5", ".npy.h5")):
        if h5py is None:
            raise ImportError("h5py is required to read .h5 volumes")
        with h5py.File(path, "r") as f:
            image_shape = tuple(f["image"].shape)
            label_shape = tuple(f["label"].shape)
            layout = _classify_shape(image_shape)
            return {
                "path": path,
                "layout": layout.value,
                "image_shape": image_shape,
                "label_shape": label_shape,
                "num_slices": _volume_slice_count_from_shape(image_shape),
            }
    data = np.load(path)
    if "image" not in data or "label" not in data:
        raise KeyError(f"{path} must contain 'image' and 'label' keys")
    image = data["image"]
    label = data["label"]
    layout = _classify_volume_layout(image)
    return {
        "path": path,
        "layout": layout.value,
        "image_shape": tuple(image.shape),
        "label_shape": tuple(label.shape),
        "num_slices": _volume_slice_count(image),
    }


def verify_dataset_root(
    root_dir: str,
    *,
    split: str = "train",
    expect_min_files: int = 1,
    sample_limit: int = 5,
) -> dict:
    """Validate a Synapse/ACDC directory and return a summary dict.

    Raises ``FileNotFoundError`` / ``ValueError`` on hard failures.
    """
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    names = _discover_sample_names(root_dir)
    if len(names) < expect_min_files:
        raise FileNotFoundError(
            f"{root_dir}: expected at least {expect_min_files} volume files, "
            f"found {len(names)}"
        )

    samples = []
    total_slices = 0
    layouts_seen = set()

    for name in names[:sample_limit]:
        h5_path, npz_path = _resolve_file_paths(root_dir, name)
        path = h5_path or npz_path
        if path is None:
            continue
        info = inspect_volume_file(path)
        samples.append(info)
        layouts_seen.add(info["layout"])
        if split == "train":
            total_slices += info["num_slices"]
        else:
            total_slices += 1

    if split == "train":
        for name in names[sample_limit:]:
            _, npz_path = _resolve_file_paths(root_dir, name)
            h5_path, _ = _resolve_file_paths(root_dir, name)
            path = h5_path or npz_path
            if path:
                total_slices += inspect_volume_file(path)["num_slices"]

    # Heuristic: TransUNet train folders should be mostly 2D npz files.
    if split == "train" and VolumeLayout.VOLUME_3D.value in layouts_seen:
        logger.warning(
            "%s: train split contains 3D volume files. "
            "TransUNet train_npz uses one (H,W) slice per .npz file.",
            root_dir,
        )

    summary = {
        "root_dir": root_dir,
        "split": split,
        "num_files": len(names),
        "num_samples": total_slices if split == "train" else len(names),
        "layouts": sorted(layouts_seen),
        "samples": samples,
    }
    logger.info(
        "Verified %s [%s]: %d files -> %d samples, layouts=%s",
        root_dir,
        split,
        len(names),
        summary["num_samples"],
        summary["layouts"],
    )
    return summary


class SynapseDataset(Dataset):
    """Synapse / ACDC dataset (TransUNet preprocessed layout).

    Args:
        root_dir: Directory with ``.npz`` (2D slices or 3D stacks) or ``.h5`` volumes.
        split: ``'train'`` expands volumes into slices; ``'val'``/``'test'`` load
            whole volumes when ``slice_idx == -1``.
        list_file: Optional text file of sample names (one per line, no extension).
        transform: Callable applied to ``{'image', 'label'}`` dicts.
        img_size: Target size (used by default transforms in ``train.py``).
        verify: If ``True``, run :func:`verify_dataset_root` at construction time.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        list_file=None,
        transform=None,
        img_size=224,
        verify: bool = False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        if list_file is not None and os.path.exists(list_file):
            with open(list_file, "r", encoding="utf-8") as f:
                self.sample_list = [line.strip() for line in f if line.strip()]
        else:
            self.sample_list = _discover_sample_names(root_dir)

        if not self.sample_list:
            raise FileNotFoundError(
                f"No .npz/.h5 files found under {root_dir}. "
                "See configs/intro_to_datasets/synapse.yaml for TransUNet layout."
            )

        self.slices: List[Tuple[str, int]] = []
        if split == "train":
            for name in self.sample_list:
                h5_path, npz_path = _resolve_file_paths(root_dir, name)
                if h5_path is not None:
                    if h5py is None:
                        raise ImportError("h5py is required to read .h5 training volumes")
                    with h5py.File(h5_path, "r") as f:
                        num_slices = _volume_slice_count_from_shape(tuple(f["image"].shape))
                    for s in range(num_slices):
                        self.slices.append((name, s))
                elif npz_path is not None:
                    with np.load(npz_path) as data:
                        num_slices = _volume_slice_count(data["image"])
                    for s in range(num_slices):
                        self.slices.append((name, s))
                else:
                    logger.warning("Skipping missing sample %s under %s", name, root_dir)
        else:
            for name in self.sample_list:
                self.slices.append((name, -1))

        if not self.slices:
            raise ValueError(f"No samples indexed from {root_dir} (split={split})")

        if verify:
            verify_dataset_root(root_dir, split=split)

    def __len__(self):
        return len(self.slices)

    def _load_data(self, name, slice_idx):
        h5_path, npz_path = _resolve_file_paths(self.root_dir, name)

        if h5_path is not None:
            if h5py is None:
                raise ImportError("h5py is required to read .h5 volumes")
            with h5py.File(h5_path, "r") as f:
                image = f["image"]
                label = f["label"]
                if slice_idx >= 0:
                    image, label = _to_hw(image[slice_idx], label[slice_idx])
                else:
                    image = image[:]
                    label = label[:]
        elif npz_path is not None:
            with np.load(npz_path) as data:
                image, label = _read_volume_slice(
                    data["image"], data["label"], slice_idx
                )
        else:
            raise FileNotFoundError(
                f"Cannot find {name}.{{h5,npz}} under {self.root_dir}"
            )

        return image.astype(np.float32), label.astype(np.int64)

    def __getitem__(self, idx):
        name, slice_idx = self.slices[idx]
        image, label = self._load_data(name, slice_idx)

        if _is_volume_3d(
            np.asarray(image) if not isinstance(image, np.ndarray) else image,
            np.asarray(label) if not isinstance(label, np.ndarray) else label,
        ):
            raise ValueError(
                f"Sample '{name}' is a 3D volume {tuple(image.shape)}. "
                "2D augmentations and train.py validation expect slice data "
                "(TransUNet train_npz / val_npz). Use test_vol_h5 with test.py "
                "volume mode for 3D evaluation, or point data.val_dir to 2D "
                "val_npz slices."
            )

        if self.transform is not None:
            sample = self.transform({"image": image, "label": label})
            image, label = sample["image"], sample["label"]

        image, label = _arrays_to_model_tensors(image, label)
        return {"image": image, "label": label, "case_name": name}
