# Reference: https://github.com/yassouali/CCT
# Paper:     https://arxiv.org/abs/2003.09082
"""Cross Consistency Training (CCT) semi-supervised segmentation.

Ouali et al., "Semi-Supervised Semantic Segmentation Needs Strong,
Varied Perturbations", BMVC 2020.

Main decoder produces predictions; auxiliary decoders with different
perturbations enforce consistency on unlabeled data.

Faithful-to-paper notes:
  * In the official ``yassouali/CCT`` repo each auxiliary decoder is a
    **mini-decoder**: a perturbation applied to the encoder/bottleneck
    feature, followed by Conv-BN-ReLU -> Upsample x2 -> Conv-BN-ReLU ->
    Upsample x2 -> 1x1 segmentation head.  This file mirrors that
    structure rather than the older 1x1-only head used in earlier
    revisions of this project.
  * Auxiliary decoders consume the **bottleneck feature** (the deepest
    encoder output) so the mini-decoder genuinely re-decodes the image
    through its own perturbed path.  The "fallback" output-perturbation
    branch (for special architectures that do not expose .encoder /
    .bottleneck) raises rather than silently degrading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

from .base import BaseSemiMethod
from .utils import get_current_consistency_weight


# ---------------------------------------------------------------------------
# Feature perturbations (operate on intermediate feature maps)
# ---------------------------------------------------------------------------

class _Identity(nn.Module):
    def forward(self, x):
        return x


class DropoutPerturbation(nn.Module):
    """Channel-wise feature dropout (Dropout2d)."""
    def __init__(self, drop_rate: float = 0.5):
        super().__init__()
        self.drop = nn.Dropout2d(p=drop_rate)

    def forward(self, x):
        return self.drop(x)


class FeatureNoisePerturbation(nn.Module):
    """Multiplicative uniform noise: x * (1 + U(-r, r)).

    yassouali/CCT calls this ``F-Noise``; range r=0.3 is the published
    default.
    """
    def __init__(self, noise_range: float = 0.3):
        super().__init__()
        self.noise_range = float(noise_range)

    def forward(self, x):
        if not self.training or self.noise_range <= 0.0:
            return x
        noise = x.new_empty(x.shape).uniform_(-self.noise_range, self.noise_range)
        return x * (1.0 + noise)


class FeatureDropPerturbation(nn.Module):
    """Spatial feature drop: zero out random spatial regions (1x1 mask).

    Same shape as DropOut but applied per spatial position rather than per
    channel — matches yassouali/CCT's ``FeatureDrop`` recipe.
    """
    def __init__(self, drop_rate: float = 0.3):
        super().__init__()
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        if not self.training:
            return x
        B, _, H, W = x.shape
        mask = (torch.rand(B, 1, H, W, device=x.device) > self.drop_rate).float()
        return x * mask


class VATStylePerturbation(nn.Module):
    """Learned (input-space) perturbation: small depthwise conv + IN + LReLU.

    Approximates the per-feature VAT-style perturbation block in
    yassouali/CCT (a small learnable transformation that nudges the
    feature toward an adversarial direction without solving the full VAT
    objective).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),
            nn.InstanceNorm2d(in_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _build_perturbation(ptype: str, in_channels: int) -> nn.Module:
    if ptype == 'dropout':
        return DropoutPerturbation(drop_rate=0.5)
    if ptype == 'feature_noise':
        return FeatureNoisePerturbation(noise_range=0.3)
    if ptype == 'feature_drop':
        return FeatureDropPerturbation(drop_rate=0.3)
    if ptype == 'vat':
        return VATStylePerturbation(in_channels)
    if ptype == 'identity':
        return _Identity()
    raise ValueError(
        f"Unknown CCT perturbation type '{ptype}'. "
        f"Available: dropout / feature_noise / feature_drop / vat / identity."
    )


# ---------------------------------------------------------------------------
# Mini-decoder for auxiliary branches  (yassouali/CCT style)
# ---------------------------------------------------------------------------

class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AuxMiniDecoder(nn.Module):
    """Mini-decoder = Perturbation -> [ConvBNReLU -> Upsample x2] x2 -> 1x1.

    Matches the spirit of yassouali/CCT's auxiliary decoders, which
    upsample the deep bottleneck feature through two conv stages before
    producing class logits.
    """

    def __init__(self, in_channels: int, num_classes: int,
                 perturbation: nn.Module, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = max(in_channels // 2, num_classes)
        out_channels = max(mid_channels // 2, num_classes)
        self.perturbation = perturbation
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block1 = _ConvBNReLU(in_channels, mid_channels, k=3)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block2 = _ConvBNReLU(mid_channels, out_channels, k=3)
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.perturbation(x)
        x = self.block1(self.up1(x))
        x = self.block2(self.up2(x))
        return self.head(x)


# ---------------------------------------------------------------------------
# CCT Method
# ---------------------------------------------------------------------------

class CrossConsistencyTraining(BaseSemiMethod):
    """Cross Consistency Training (CCT).

    Each auxiliary branch is a mini-decoder operating on the encoder's
    deepest feature (or the bottleneck output), with a different
    perturbation injected before its conv-upsample stack.

    Args:
        model: Primary segmentation model.  Must expose ``.encoder`` and
            ``.bottleneck`` (the project's :class:`SegmentationModel`
            satisfies this).
        device: Torch device.
        n_auxiliary: Number of auxiliary perturbed decoders (default 3).
        perturbation_types: List of perturbation names.  When None, cycles
            through ``[dropout, feature_noise, feature_drop, vat]``.
        consistency_weight: Maximum consistency weight (default 1.0).
        rampup_epochs: Ramp-up epochs for consistency weight (default 40).
        img_size: Image spatial size.

    Raises:
        TypeError: If ``model`` does not expose the encoder / bottleneck /
            decoder / head layout required by the mini-decoder
            construction.  No silent fallback.
    """

    _DEFAULT_PTYPES = ('dropout', 'feature_noise', 'feature_drop', 'vat')

    def __init__(self, model: nn.Module, device: torch.device,
                 n_auxiliary: int = 3,
                 perturbation_types: List[str] = None,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.n_auxiliary = int(n_auxiliary)
        if perturbation_types is None:
            perturbation_types = [
                self._DEFAULT_PTYPES[i % len(self._DEFAULT_PTYPES)]
                for i in range(self.n_auxiliary)
            ]
        self.perturbation_types = perturbation_types
        self.aux_decoders: nn.ModuleList = None  # built in build()
        self._num_classes: int = -1
        self._feat_channels: int = -1

    # ----------------------------------------------------------------- build
    def _detect_layout(self) -> None:
        """Verify the model exposes the modular layout required by CCT.

        Raises a TypeError (no silent fallback) so users get an informative
        error rather than a degraded "output-perturbation only" mode.
        """
        if not (hasattr(self.model, 'encoder')
                and hasattr(self.model, 'bottleneck')
                and hasattr(self.model, 'head')
                and hasattr(self.model.head, 'conv')):
            raise TypeError(
                "CrossConsistencyTraining requires the standard "
                "SegmentationModel layout (.encoder + .bottleneck + "
                ".decoder + .head with a Conv2d 'head.conv').  Got "
                f"{type(self.model).__name__}, which does not expose this "
                "interface."
            )
        self._num_classes = self.model.head.conv.out_channels

        # Probe bottleneck output channels (the aux mini-decoders operate
        # on this deep feature).
        in_ch = 3
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                in_ch = m.in_channels
                break
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, self.img_size, self.img_size,
                                device=self.device)
            feats = self.model.encoder(dummy)
            btn = self.model.bottleneck(feats[-1])
        if was_training:
            self.model.train()
        if btn.dim() != 4:
            raise TypeError(
                "CCT expects a 4D bottleneck feature (B,C,H,W); got shape "
                f"{tuple(btn.shape)}."
            )
        self._feat_channels = btn.shape[1]

    def build(self) -> None:
        self._detect_layout()
        self.aux_decoders = nn.ModuleList()
        for ptype in self.perturbation_types[:self.n_auxiliary]:
            perturbation = _build_perturbation(ptype, self._feat_channels)
            self.aux_decoders.append(
                AuxMiniDecoder(
                    in_channels=self._feat_channels,
                    num_classes=self._num_classes,
                    perturbation=perturbation,
                )
            )
        self.aux_decoders.to(self.device)

    def extra_params(self):
        if self.aux_decoders is None:
            return []
        return list(self.aux_decoders.parameters())

    # --------------------------------------------------------- forward util
    def _main_forward(self, x: torch.Tensor):
        """Run the main branch and also return the deep bottleneck feature
        used as input to the aux mini-decoders.
        """
        feats = self.model.encoder(x)
        btn = self.model.bottleneck(feats[-1])
        decoded = self.model.decoder(btn, feats[:-1])
        main_pred = self.model.head(decoded)
        if main_pred.shape[2:] != x.shape[2:]:
            main_pred = F.interpolate(
                main_pred, size=x.shape[2:],
                mode='bilinear', align_corners=False,
            )
        return btn, main_pred

    # ---------------------------------------------------------- train_step
    def train_step(
        self,
        labeled_batch: Dict[str, Any],
        unlabeled_batch: Dict[str, Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        total_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()
        self.aux_decoders.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)

        # --- Supervised loss on labeled data (main decoder only) ---
        pred_l = self.model(images_l)
        if isinstance(pred_l, (list, tuple)):
            pred_l = pred_l[0]
        sup_loss = criterion(pred_l, labels)

        # --- Consistency on unlabeled data ---
        btn_u, main_pred_u = self._main_forward(images_u)

        # Main prediction softmax serves as target (detach: no grad to main).
        with torch.no_grad():
            main_soft = F.softmax(main_pred_u.detach(), dim=1)

        consistency_loss = images_u.new_zeros(())
        for aux_dec in self.aux_decoders:
            aux_pred = aux_dec(btn_u)
            if aux_pred.shape[2:] != main_soft.shape[2:]:
                aux_pred = F.interpolate(
                    aux_pred, size=main_soft.shape[2:],
                    mode='bilinear', align_corners=False,
                )
            consistency_loss = consistency_loss + F.mse_loss(
                F.softmax(aux_pred, dim=1), main_soft)
        consistency_loss = consistency_loss / max(len(self.aux_decoders), 1)

        w = get_current_consistency_weight(epoch, self.consistency_weight,
                                           self.rampup_epochs)
        total_loss = sup_loss + w * consistency_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.aux_decoders.parameters()),
            max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": consistency_loss.item(),
            "w": w,
        }

    def get_eval_model(self) -> nn.Module:
        return self.model
