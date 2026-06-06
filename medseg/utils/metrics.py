"""Evaluation metrics for medical image segmentation."""

import numpy as np
import torch


def dice_coefficient(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Compute per-class Dice coefficient."""
    dice_scores = {}
    for c in range(1, num_classes):  # skip background
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            dice_scores[c] = 1.0 if target_c.sum() == 0 else 0.0
        else:
            dice_scores[c] = (2.0 * intersection) / union
    return dice_scores


def iou_score(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Compute per-class IoU (Jaccard index)."""
    iou_scores = {}
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.float32)
        target_c = (target == c).astype(np.float32)
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        if union == 0:
            iou_scores[c] = 1.0 if target_c.sum() == 0 else 0.0
        else:
            iou_scores[c] = intersection / union
    return iou_scores


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Compute per-class 95th percentile Hausdorff distance."""
    try:
        from medpy.metric.binary import hd95
    except ImportError:
        return {c: float("nan") for c in range(1, num_classes)}

    hd_scores = {}
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)
        if pred_c.sum() == 0 or target_c.sum() == 0:
            hd_scores[c] = float("nan")
        else:
            hd_scores[c] = hd95(pred_c, target_c)
    return hd_scores


def compute_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Compute all metrics."""
    return {
        "dice": dice_coefficient(pred, target, num_classes),
        "iou": iou_score(pred, target, num_classes),
        "hd95": hausdorff_distance_95(pred, target, num_classes),
    }
