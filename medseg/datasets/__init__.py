"""Dataset modules.

Note:
    The preferred image-mask segmentation dataset is ``GenericDataset``
    (in ``generic_dataset.py``). It supports arbitrary ``num_classes``
    (1 foreground class for binary, or 2/3/4/... for multi-class) — mask
    pixel values are used directly as class indices.
"""

from .synapse_dataset import SynapseDataset
from .generic_dataset import GenericDataset
from .unlabeled_dataset import UnlabeledDataset
from .transforms import get_train_transforms, get_val_transforms

# Advanced augmentation methods
from . import advanced_aug

# Semi-supervised datasets
from .semi_datasets import SemiSupervisedDataset, PairedLabeledUnlabeledDataset

# Weakly supervised datasets
from .weakly_supervised_datasets import (
    WeaklySupervisedDataset,
    BoxSupervisedDataset,
    ImageLabelDataset,
    CAMDataset
)

# Domain adaptation datasets
from .domain_adaptation_datasets import (
    DomainAdaptationDataset,
    SourceTargetDataset,
    SourceFreeDataset
)

# Text-guided datasets (image + caption)
from .text_image_dataset import (
    TextImageDataset,
    QaTaCOV19Dataset,
    MosMedPlusDataset,
)

__all__ = [
    'SynapseDataset',
    'GenericDataset',
    'UnlabeledDataset',
    'SemiSupervisedDataset',
    'PairedLabeledUnlabeledDataset',
    'WeaklySupervisedDataset',
    'BoxSupervisedDataset',
    'ImageLabelDataset',
    'CAMDataset',
    'DomainAdaptationDataset',
    'SourceTargetDataset',
    'SourceFreeDataset',
    'TextImageDataset',
    'QaTaCOV19Dataset',
    'MosMedPlusDataset',
    'get_train_transforms',
    'get_val_transforms',
]
