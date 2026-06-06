"""Utilities for semi-supervised segmentation methods.

Provides EMA update, ramp-up scheduling, pseudo-label generation,
and strong augmentation pipelines.

Reference implementation: https://github.com/HiLab-git/SSL4MIS
"""

import copy
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float):
    """Update *teacher* parameters as exponential moving average of *student*.

    teacher_param = decay * teacher_param + (1 - decay) * student_param
    """
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)
    # Also update buffers (e.g. BatchNorm running stats)
    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        t_buf.data.copy_(s_buf.data)


def create_ema_model(model: nn.Module) -> nn.Module:
    """Create a deep-copied EMA teacher model with frozen gradients."""
    teacher = copy.deepcopy(model)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    return teacher


# ---------------------------------------------------------------------------
# Ramp-up scheduling
# ---------------------------------------------------------------------------

def sigmoid_rampup(current: int, rampup_length: int) -> float:
    """Sigmoid-shaped ramp-up from 0 to 1 over *rampup_length* epochs.

    Returns 1.0 when rampup_length <= 0.
    """
    if rampup_length <= 0:
        return 1.0
    current = max(0.0, min(float(current), float(rampup_length)))
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


def linear_rampup(current: int, rampup_length: int) -> float:
    """Linear ramp-up from 0 to 1."""
    if rampup_length <= 0:
        return 1.0
    return min(1.0, float(current) / float(rampup_length))


def get_current_consistency_weight(epoch: int, consistency_weight: float,
                                   rampup_epochs: int) -> float:
    """Compute consistency weight at given epoch with sigmoid ramp-up."""
    return consistency_weight * sigmoid_rampup(epoch, rampup_epochs)


# ---------------------------------------------------------------------------
# Pseudo-label generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def pseudo_label_with_threshold(logits: torch.Tensor, threshold: float = 0.95):
    """Generate pseudo-labels from logits with confidence thresholding.

    Args:
        logits: (B, C, H, W) raw model outputs.
        threshold: Minimum softmax confidence to keep a pseudo-label.

    Returns:
        pseudo_labels: (B, H, W) long tensor, -1 where confidence < threshold.
        mask: (B, H, W) bool tensor, True where label is valid.
    """
    probs = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = probs.max(dim=1)
    mask = max_probs.ge(threshold)
    pseudo_labels[~mask] = -1  # mark as ignore
    return pseudo_labels, mask


@torch.no_grad()
def pseudo_label_hard(logits: torch.Tensor):
    """Generate hard pseudo-labels (argmax) without thresholding.

    Args:
        logits: (B, C, H, W) raw model outputs.

    Returns:
        pseudo_labels: (B, H, W) long tensor.
    """
    return logits.argmax(dim=1)


# ---------------------------------------------------------------------------
# Strong augmentation for images (tensor-level, applied on GPU)
# ---------------------------------------------------------------------------

class StrongAugmentation(nn.Module):
    """Tensor-level strong augmentation for semi-supervised consistency.

    Applies a random subset of: color jitter, Gaussian noise, Gaussian blur,
    and CutOut.  Operates on (B, C, H, W) float tensors in [0, 1].
    """

    def __init__(self, img_size: int = 224):
        super().__init__()
        self.img_size = img_size

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) float tensor."""
        x = x.clone()
        B = x.shape[0]

        # 1. Color jitter (brightness + contrast)
        if random.random() < 0.8:
            brightness = 1.0 + random.uniform(-0.4, 0.4)
            x = x * brightness
            contrast = 1.0 + random.uniform(-0.4, 0.4)
            mean = x.mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * contrast + mean
            x = x.clamp(0.0, 1.0)

        # 2. Gaussian noise
        if random.random() < 0.5:
            std = random.uniform(0.01, 0.1)
            noise = torch.randn_like(x) * std
            x = (x + noise).clamp(0.0, 1.0)

        # 3. Gaussian blur
        if random.random() < 0.5:
            ks = random.choice([3, 5])
            sigma = random.uniform(0.1, 2.0)
            padding = ks // 2
            # Create Gaussian kernel
            ax = torch.arange(ks, dtype=x.dtype, device=x.device) - padding
            kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
            kernel = kernel / kernel.sum()
            kernel_2d = kernel[None, :] * kernel[:, None]
            kernel_2d = kernel_2d.expand(x.shape[1], 1, ks, ks)
            x = F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])

        # 4. CutOut
        if random.random() < 0.5:
            cut_size = int(self.img_size * random.uniform(0.1, 0.3))
            for b in range(B):
                cy = random.randint(0, x.shape[2] - 1)
                cx = random.randint(0, x.shape[3] - 1)
                y1 = max(0, cy - cut_size // 2)
                y2 = min(x.shape[2], cy + cut_size // 2)
                x1 = max(0, cx - cut_size // 2)
                x2 = min(x.shape[3], cx + cut_size // 2)
                x[b, :, y1:y2, x1:x2] = 0.0

        return x


def get_strong_augmentation(img_size: int = 224) -> StrongAugmentation:
    """Create strong augmentation module for semi-supervised training."""
    return StrongAugmentation(img_size)


# ---------------------------------------------------------------------------
# Gaussian input noise (NeurIPS 2017 Mean Teacher input perturbation)
# ---------------------------------------------------------------------------

class GaussianInputNoise(nn.Module):
    """Additive Gaussian input noise.

    This is the original Mean Teacher (Tarvainen & Valpola, NeurIPS 2017)
    perturbation:  x' = x + N(0, sigma^2).  Combined with model-internal
    Dropout it gives the "Gaussian input noise + dropout" recipe described
    in the paper (Sec. 3.1).  No color jitter, no blur, no cutout.
    """

    def __init__(self, std: float = 0.15, clamp: bool = True):
        super().__init__()
        self.std = float(std)
        self.clamp = bool(clamp)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0.0:
            return x
        noise = torch.randn_like(x) * self.std
        out = x + noise
        if self.clamp:
            out = out.clamp(0.0, 1.0)
        return out


def get_input_noise(noise_type: str = "gaussian",
                    img_size: int = 224,
                    std: float = 0.15) -> nn.Module:
    """Return an input-noise module by name.

    Args:
        noise_type: ``"gaussian"`` (Mean Teacher paper default) or
            ``"strong_aug"`` (heavier color/blur/cutout pipeline).
        img_size: Image spatial size (only used by ``"strong_aug"``).
        std: Std-dev for ``"gaussian"``.

    Raises:
        ValueError: For unknown noise types.  No silent fallback.
    """
    if noise_type == "gaussian":
        return GaussianInputNoise(std=std)
    if noise_type == "strong_aug":
        return StrongAugmentation(img_size)
    raise ValueError(
        f"Unknown consistency_noise '{noise_type}'. "
        f"Expected one of: 'gaussian', 'strong_aug'."
    )
