"""Synapse dataset - 3D volumes converted to 2D slices."""

import os
import numpy as np
try:
    import h5py
except ImportError:
    h5py = None
import torch
from torch.utils.data import Dataset


class SynapseDataset(Dataset):
    """Synapse multi-organ dataset.

    Expects data in HDF5 format with 'image' and 'label' keys,
    or .npz files with 'image' and 'label' keys.

    Args:
        root_dir: Path to the data directory containing .h5 or .npz files.
        split: 'train' or 'test'.
        list_file: Text file listing case names (one per line).
        transform: Optional transforms to apply.
        img_size: Target image size for resizing.
    """
    def __init__(self, root_dir, split='train', list_file=None, transform=None, img_size=224):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size

        if list_file is not None and os.path.exists(list_file):
            with open(list_file, 'r') as f:
                self.sample_list = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover files
            self.sample_list = []
            for fname in sorted(os.listdir(root_dir)):
                if fname.endswith(('.h5', '.npz', '.npy')):
                    self.sample_list.append(fname.replace('.h5', '').replace('.npz', '').replace('.npy', ''))

        self.slices = []
        if split == 'train':
            # For training, expand 3D volumes to 2D slices
            for name in self.sample_list:
                h5_path = os.path.join(root_dir, name + '.h5')
                npz_path = os.path.join(root_dir, name + '.npz')
                if os.path.exists(h5_path):
                    with h5py.File(h5_path, 'r') as f:
                        num_slices = f['image'].shape[0]
                    for s in range(num_slices):
                        self.slices.append((name, s))
                elif os.path.exists(npz_path):
                    data = np.load(npz_path)
                    num_slices = data['image'].shape[0]
                    for s in range(num_slices):
                        self.slices.append((name, s))
        else:
            # For test, load entire volumes
            for name in self.sample_list:
                self.slices.append((name, -1))  # -1 means load all

    def __len__(self):
        return len(self.slices)

    def _load_data(self, name, slice_idx):
        h5_path = os.path.join(self.root_dir, name + '.h5')
        npz_path = os.path.join(self.root_dir, name + '.npz')

        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                if slice_idx >= 0:
                    image = f['image'][slice_idx]
                    label = f['label'][slice_idx]
                else:
                    image = f['image'][:]
                    label = f['label'][:]
        elif os.path.exists(npz_path):
            data = np.load(npz_path)
            if slice_idx >= 0:
                image = data['image'][slice_idx]
                label = data['label'][slice_idx]
            else:
                image = data['image']
                label = data['label']
        else:
            raise FileNotFoundError(f"Cannot find {name}.h5 or {name}.npz in {self.root_dir}")

        return image.astype(np.float32), label.astype(np.int64)

    def __getitem__(self, idx):
        name, slice_idx = self.slices[idx]
        image, label = self._load_data(name, slice_idx)

        if self.transform is not None:
            sample = self.transform({"image": image, "label": label})
            image, label = sample["image"], sample["label"]

        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=0)  # H,W -> 3,H,W
            elif image.ndim == 3 and image.shape[-1] in [1, 3]:
                image = image.transpose(2, 0, 1)  # H,W,C -> C,H,W
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).long()

        return {"image": image, "label": label, "case_name": name}
