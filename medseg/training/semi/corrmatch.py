# CorrMatch (CVPR 2024)
# Reference: https://github.com/BBBBchan/CorrMatch
# Paper: https://arxiv.org/abs/2306.04300
# Implemented from paper formulas; not a copy of the official repo.
"""CorrMatch: Label Propagation via Patch-wise Correlation Matching.

Sun et al., "CorrMatch: Label Propagation via Correlation Matching for
Semi-Supervised Semantic Segmentation", CVPR 2024.

Paper recipe — implemented from the equations alone:

  Given a feature map ``F in R^{B x C x h x w}`` extracted before the
  segmentation head, the patch-wise correlation matrix is

      F_hat = L2-normalise(F) along channel axis,         (Eq. 3)
      S = F_hat^T @ F_hat in R^{B x N x N},   N = h*w     (Eq. 4)

  where ``S[b, i, j]`` is the cosine similarity between spatial position
  ``i`` and position ``j`` of sample ``b``.

  Label propagation (paper Sec. 3.2):
    Let p(i) in [0, 1]^C be the teacher's softmax at pixel ``i`` and let
    conf(i) = max_c p(i, c).  A pixel ``j`` is *low-confidence* iff
    conf(j) < tau_low.  For each such j we look at its top-K most-correlated
    high-confidence neighbours

        N_K(j) = top-K { i : conf(i) >= tau_high } by S[b, j, i],

    keeping only neighbours whose correlation is above ``tau_corr``.
    The propagated label is the *correlation-weighted majority* of those
    neighbours' pseudo-labels:

        y_hat(j) = argmax_c  sum_{i in N_K(j)}  S[b, j, i] * 1[y_hat(i) == c]

    and the propagated mask is enabled iff at least one valid neighbour
    survives.

  The final unsupervised loss is per-pixel CE between the student's
  strong-view prediction and the (high-confidence union propagated)
  pseudo-label, masked by the union mask.  Supervised loss on labeled
  data is unchanged.

The correlation tensor is computed at the segmentation feature resolution
(typically ``input // 4``) to keep memory at ``B * (hw)^2`` floats; we
downsample the teacher / mask / pseudo-label to that resolution and
upsample the propagated label back to input resolution for the CE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight,
    get_strong_augmentation,
)


# ---------------------------------------------------------------------------
# Correlation-based label propagation (Eq. 3-6 of the paper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _patch_correlation(feat: torch.Tensor) -> torch.Tensor:
    """Patch-wise cosine-correlation matrix.

    Args:
        feat: (B, C, h, w)
    Returns:
        S: (B, hw, hw) cosine similarity in [-1, 1].
    """
    B, C, h, w = feat.shape
    flat = feat.reshape(B, C, h * w)
    flat = F.normalize(flat, dim=1, eps=1e-6)  # L2 along channel
    return torch.bmm(flat.transpose(1, 2), flat)  # (B, N, N)


@torch.no_grad()
def correlation_propagation(
    feat: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau_high: float,
    tau_low: float,
    tau_corr: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propagate high-confidence pseudo-labels to low-confidence pixels.

    The teacher logits and the feature must share spatial size ``(h, w)``;
    the caller is expected to downsample as needed.

    Returns:
        propagated_label: (B, h, w) long, -1 where no propagation succeeded.
        union_mask:       (B, h, w) bool, True where the (union of high-conf
                          + successfully-propagated) pixels live.
    """
    B, _, h, w = teacher_logits.shape
    N = h * w
    probs = F.softmax(teacher_logits, dim=1)
    conf, hard = probs.max(dim=1)                      # (B, h, w)
    hard_flat = hard.view(B, N)
    conf_flat = conf.view(B, N)

    high_mask = conf_flat.ge(tau_high)                 # (B, N)
    low_mask = conf_flat.lt(tau_low)                   # (B, N)

    # Start from the high-confidence pseudo-label; everything else = -1
    out = torch.full_like(hard_flat, fill_value=-1)
    out[high_mask] = hard_flat[high_mask]
    union = high_mask.clone()

    S = _patch_correlation(feat)                       # (B, N, N)
    # Mask correlations to high-conf donors only
    donor_mask = high_mask.unsqueeze(1)                # (B, 1, N)
    S_masked = S.masked_fill(~donor_mask, float('-inf'))

    # For each query pixel, take top-K most-correlated donors
    k = min(int(top_k), N)
    top_corr, top_idx = S_masked.topk(k=k, dim=2)      # (B, N, K)
    top_valid = (top_corr > tau_corr) & torch.isfinite(top_corr)

    # Gather donors' hard labels: (B, N, K)
    donor_labels = torch.gather(
        hard_flat.unsqueeze(1).expand(-1, N, -1), 2, top_idx)
    # Number of classes
    C = int(probs.shape[1])

    # Correlation-weighted vote.  one_hot: (B, N, K, C)
    one_hot = F.one_hot(donor_labels.clamp(min=0), num_classes=C).float()
    weights = top_corr.clamp(min=0.0).unsqueeze(-1) * top_valid.float().unsqueeze(-1)
    votes = (one_hot * weights).sum(dim=2)             # (B, N, C)

    has_donor = top_valid.any(dim=2)                   # (B, N)
    voted = votes.argmax(dim=2)                        # (B, N)

    # Apply only to low-confidence pixels that received at least one donor
    target = low_mask & has_donor
    out[target] = voted[target]
    union = union | target

    return out.view(B, h, w), union.view(B, h, w)


# ---------------------------------------------------------------------------
# CorrMatch method
# ---------------------------------------------------------------------------

class CorrMatch(BaseSemiMethod):
    """CorrMatch: correlation-based pseudo-label propagation.

    Args:
        model: Student model.  Must expose ``.encoder``, ``.bottleneck``,
            ``.decoder``, ``.head`` (the project's standard
            SegmentationModel).  No fallback — raises if missing.
        device: Torch device.
        ema_decay: EMA decay for the teacher (default 0.999).
        tau_high: High-confidence threshold (paper default 0.95).
        tau_low:  Low-confidence threshold below which propagation is
                  attempted (paper default 0.7).
        tau_corr: Minimum correlation to accept a donor (paper default 0.85).
        top_k:    Number of correlated donors per query pixel (paper default 8).
        corr_size: Resolution at which the correlation matrix is computed.
                  Lower = cheaper memory; default 56 (i.e. 224/4).
        consistency_weight: Max unsupervised loss weight.
        rampup_epochs: Sigmoid ramp-up epochs.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 ema_decay: float = 0.999,
                 tau_high: float = 0.95,
                 tau_low: float = 0.70,
                 tau_corr: float = 0.85,
                 top_k: int = 8,
                 corr_size: int = 56,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        if not (0.0 < tau_low <= tau_high <= 1.0):
            raise ValueError(
                f"Require 0 < tau_low <= tau_high <= 1, got "
                f"tau_low={tau_low}, tau_high={tau_high}.")
        if not (-1.0 <= tau_corr <= 1.0):
            raise ValueError(f"tau_corr must be in [-1, 1], got {tau_corr}.")
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}.")
        if int(corr_size) <= 0:
            raise ValueError(f"corr_size must be > 0, got {corr_size}.")
        self.ema_decay = ema_decay
        self.tau_high = float(tau_high)
        self.tau_low = float(tau_low)
        self.tau_corr = float(tau_corr)
        self.top_k = int(top_k)
        self.corr_size = int(corr_size)
        self.teacher: nn.Module = None
        self.strong_aug = None

    def build(self) -> None:
        # Fail loudly if the model is not the SegmentationModel layout the
        # method depends on — no silent fallback to "logits-as-features".
        if not (hasattr(self.model, 'encoder')
                and hasattr(self.model, 'bottleneck')
                and hasattr(self.model, 'decoder')
                and hasattr(self.model, 'head')
                and hasattr(self.model.head, 'conv')):
            raise TypeError(
                "CorrMatch requires SegmentationModel(.encoder + .bottleneck "
                "+ .decoder + .head with Conv2d 'head.conv').  Got "
                f"{type(self.model).__name__}.")
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        self.strong_aug = get_strong_augmentation(self.img_size)

    # ----- internal: decoder feature + logits without head's final upsample
    def _decoder_feat_and_logits(self, model: nn.Module, x: torch.Tensor):
        feats = model.encoder(x)
        btn = model.bottleneck(feats[-1])
        decoded = model.decoder(btn, feats[:-1])
        logits = model.head(decoded)
        return decoded, logits

    @staticmethod
    def _take_first(out):
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

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
        self.teacher.eval()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)
        H, W = images_u.shape[-2:]

        # --- Supervised loss on labeled data ---
        pred_l = self._take_first(self.model(images_l))
        sup_loss = criterion(pred_l, labels)

        # --- Teacher pass on weak view: collect decoder feature + logits ---
        with torch.no_grad():
            t_feat, t_logits = self._decoder_feat_and_logits(self.teacher, images_u)
            # Downsample to corr_size for the correlation matrix
            cs = min(self.corr_size, t_feat.shape[-1], t_feat.shape[-2])
            t_feat_ds = F.interpolate(
                t_feat, size=(cs, cs), mode='bilinear', align_corners=False)
            t_logits_ds = F.interpolate(
                t_logits, size=(cs, cs), mode='bilinear', align_corners=False)

            prop_label_ds, union_mask_ds = correlation_propagation(
                t_feat_ds, t_logits_ds,
                tau_high=self.tau_high,
                tau_low=self.tau_low,
                tau_corr=self.tau_corr,
                top_k=self.top_k,
            )
            # Upsample to input resolution for CE.  Use nearest for labels;
            # mask must also be nearest to preserve {0, 1}.
            prop_label = F.interpolate(
                prop_label_ds.unsqueeze(1).float(),
                size=(H, W), mode='nearest').squeeze(1).long()
            union_mask = F.interpolate(
                union_mask_ds.unsqueeze(1).float(),
                size=(H, W), mode='nearest').squeeze(1).bool()

        # --- Student on strong view ---
        images_u_s = self.strong_aug(images_u)
        s_pred = self._take_first(self.model(images_u_s))

        # --- Masked CE supervised by propagated label ---
        prop_target = prop_label.clone()
        prop_target[~union_mask] = -1
        ce_pix = F.cross_entropy(s_pred, prop_target, ignore_index=-1, reduction='none')
        denom = union_mask.float().sum().clamp(min=1.0)
        unsup_loss = (ce_pix * union_mask.float()).sum() / denom

        w = get_current_consistency_weight(epoch, self.consistency_weight, self.rampup_epochs)
        total_loss = sup_loss + w * unsup_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": unsup_loss.item(),
            "w": w,
            "propagated_ratio": union_mask.float().mean().item(),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
