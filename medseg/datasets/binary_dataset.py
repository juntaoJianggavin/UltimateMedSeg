"""Deprecated module. Use medseg.datasets.GenericDataset instead.

Kept solely to avoid ImportError in third-party code that imported the old path.
"""
from .generic_dataset import GenericDataset  # noqa: F401
__all__ = ["GenericDataset"]
