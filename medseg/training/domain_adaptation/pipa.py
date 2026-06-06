# PiPa: Pixel- and Patch-wise Pairing for Cross-Domain Semantic Segmentation (ACM MM 2023)
# Reference: https://github.com/chen742/PiPa
# Paper: https://arxiv.org/abs/2211.07609
# Implemented from paper formulas; not a copy of the official repo.
"""PiPa adds two complementary contrastive objectives on top of any UDA
pseudo-labelling pipeline:

    L_pix : per-pixel InfoNCE *within an image*
            positives = pixels of the same (pseudo-)class, same image
            negatives = pixels of other classes, same image

    L_pat : per-patch InfoNCE *across images in the same batch*
            positives = patches whose dominant class agrees, different image
            negatives = patches whose dominant class differs

Both terms are NT-Xent / InfoNCE (Eq. 4-7 of the paper):

    L = - mean_i log [ exp(z_i . z+_i / tau) / sum_j exp(z_i . z_j / tau) ]

The features ``z`` are L2-normalised class-probability vectors derived
from the student logits — this matches the "projection-free" variant of
PiPa (Sec. 4.4, Ablation B) and avoids needing a separate projection MLP
in the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("pipa")
class PiPaLoss(nn.Module):
    """Pixel + Patch contrastive loss for UDA segmentation.

    Chen et al., ACM MM 2023.
    Reference (not copied): https://github.com/chen742/PiPa

    Args:
        pixel_weight: scalar on L_pix.
        patch_weight: scalar on L_pat.
        temperature: tau in the InfoNCE denominator (paper default 0.1).
        pixels_per_image: how many anchor pixels to sample per image
            (uniform across pseudo-classes present in the image). The
            paper recommends 256-512; we default to 256 for speed.
        patch_size: side length (pixels) of the patches used by L_pat.
        confidence_threshold: pixels whose teacher confidence is below
            this value are excluded from sampling, matching the paper's
            "reliable anchor" filter (Sec. 4.3).
        num_classes: number of segmentation classes.
    """

    def __init__(
        self,
        pixel_weight: float = 0.1,
        patch_weight: float = 0.1,
        temperature: float = 0.1,
        pixels_per_image: int = 256,
        patch_size: int = 16,
        confidence_threshold: float = 0.5,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"PiPa temperature must be positive, got {temperature}")
        if pixels_per_image <= 0:
            raise ValueError(
                f"PiPa pixels_per_image must be positive, got {pixels_per_image}"
            )
        if patch_size <= 0:
            raise ValueError(f"PiPa patch_size must be positive, got {patch_size}")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"PiPa confidence_threshold must be in [0, 1], got "
                f"{confidence_threshold}"
            )
        self.pixel_weight = pixel_weight
        self.patch_weight = patch_weight
        self.temperature = temperature
        self.pixels_per_image = pixels_per_image
        self.patch_size = patch_size
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _info_nce(
        anchors: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Standard supervised InfoNCE over a single bank of L2-normalised
        feature vectors ``anchors`` with class assignments ``labels``.

        Args:
            anchors: (N, D), each row L2-normalised.
            labels:  (N,), class id per anchor.
        """
        N = anchors.shape[0]
        if N < 2:
            return anchors.new_zeros(())
        sim = (anchors @ anchors.t()) / temperature           # (N, N)
        # Subtract per-row max for numerical stability (does not change
        # log-softmax) and mask the self-similarity diagonal *after* the
        # logsumexp via a finite large-negative value to avoid the
        # ``0 * (-inf) = NaN`` trap when later multiplied by the bool mask.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        diag = torch.eye(N, device=anchors.device, dtype=torch.bool)
        neg_inf = torch.full_like(sim, -1e9)
        sim = torch.where(diag, neg_inf, sim)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)     # (N, N) bool
        same = same & (~diag)
        # Skip anchors with no positive in the bank.
        has_pos = same.any(dim=1)
        if not has_pos.any():
            return anchors.new_zeros(())
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_log_prob = (same.float() * log_prob).sum(dim=1) / same.float().sum(dim=1).clamp_min(1.0)
        return -pos_log_prob[has_pos].mean()

    @torch.no_grad()
    def _sample_pixels(
        self,
        feat: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
    ):
        """Uniform per-image, per-class sampling of pixel anchors.

        Returns (anchors, anchor_labels) for the *single* image.
        """
        C, H, W = feat.shape
        feat_flat = feat.permute(1, 2, 0).reshape(-1, C)
        lbl_flat = labels.reshape(-1)
        val_flat = valid.reshape(-1)
        idx_pool = torch.nonzero(val_flat, as_tuple=False).flatten()
        if idx_pool.numel() == 0:
            return None, None
        if idx_pool.numel() > self.pixels_per_image:
            sel = idx_pool[torch.randperm(idx_pool.numel(), device=idx_pool.device)[: self.pixels_per_image]]
        else:
            sel = idx_pool
        return feat_flat[sel], lbl_flat[sel]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        target_pred: torch.Tensor,
        teacher_pred: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if target_pred is None:
            raise ValueError("PiPaLoss requires target_pred.")
        B, C, H, W = target_pred.shape

        # Features: L2-normalised softmax vectors (projection-free variant).
        prob = F.softmax(target_pred, dim=1)
        feat = F.normalize(prob, dim=1, eps=1e-6)

        # Pseudo-labels + reliability mask from teacher (or detached student).
        with torch.no_grad():
            ref = teacher_pred if teacher_pred is not None else target_pred.detach()
            prob_T = F.softmax(ref, dim=1)
            conf_T, pseudo_T = prob_T.max(dim=1)
            valid = conf_T >= self.confidence_threshold

        # -------- L_pix: per-image contrast ---------------------------
        pix_losses = []
        for b in range(B):
            a, lab = self._sample_pixels(feat[b], pseudo_T[b], valid[b])
            if a is None or lab.unique().numel() < 2:
                continue
            pix_losses.append(self._info_nce(a, lab, self.temperature))
        if pix_losses:
            l_pix = torch.stack(pix_losses).mean()
        else:
            l_pix = target_pred.new_zeros(())

        # -------- L_pat: cross-image patch contrast -------------------
        # Average-pool features and pseudo-labels into patch tokens.
        ps = max(1, min(self.patch_size, min(H, W) // 2 or 1))
        patch_feat = F.avg_pool2d(feat, kernel_size=ps)
        patch_feat = F.normalize(patch_feat, dim=1, eps=1e-6)
        # Patch label = majority class of the pseudo-labels in that patch.
        # We approximate "majority" with a one-hot pool + argmax.
        one_hot = F.one_hot(pseudo_T, num_classes=C).permute(0, 3, 1, 2).float()
        patch_cls = F.avg_pool2d(one_hot, kernel_size=ps).argmax(dim=1)  # (B, h, w)
        patch_valid = F.avg_pool2d(valid.float().unsqueeze(1), kernel_size=ps).squeeze(1) > 0.5

        Bp, Cf, hp, wp = patch_feat.shape
        pa_feat = patch_feat.permute(0, 2, 3, 1).reshape(-1, Cf)
        pa_lbl = patch_cls.reshape(-1)
        pa_val = patch_valid.reshape(-1)
        if pa_val.any() and pa_lbl[pa_val].unique().numel() >= 2:
            l_pat = self._info_nce(pa_feat[pa_val], pa_lbl[pa_val], self.temperature)
        else:
            l_pat = target_pred.new_zeros(())

        total = self.pixel_weight * l_pix + self.patch_weight * l_pat
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
