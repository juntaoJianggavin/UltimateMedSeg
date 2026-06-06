# WPGSeg (CVPR 2024)
# Reference: https://github.com/Barrett-python/WPS-SAM
# Paper: https://arxiv.org/abs/2403.14186
# Implemented from paper formulas; not a copy of the official repo.
"""WPGSeg — Weakly-supervised Prompt-Guided Segmentation with SAM.

CVPR 2024 family of "prompt-driven SAM pseudo-mask" methods (Barrett /
WPS-SAM style). The loss recipe is the canonical one shared across
WPGSeg / WPS-SAM / PSAM (CVPR 2024 cluster of SAM-augmented WSSS):
Paper: https://arxiv.org/abs/2403.14186
Reference repository: https://github.com/Barrett-python/WPS-SAM

Why it exists.
    Recent WSSS work observed that pixel-level pseudo-masks distilled
    from a Segment Anything Model (SAM), prompted by the discriminative
    pixels of a CAM (peaks → point prompts; or text → SAM2 boxes), are
    significantly cleaner than the CAM itself. The training loss is
    therefore a confidence-weighted cross-entropy on the SAM-distilled
    pseudo-mask, combined with the standard image-level multi-label BCE:

        L = L_cls( image-level )
          + lambda_seg * mean_{(x,y) : c_sam > tau_conf}
                          conf(x,y) * CE( P(x,y) , y_sam(x,y) )
          + lambda_consist * KL( P_orig || P_aug )      [optional]

    The dataloader is responsible for running SAM offline and emitting
    ``sam_pseudo_mask`` (B, H, W) integers in [0, C-1] with -1 for
    "ignore", together with a per-pixel confidence map ``sam_confidence``
    in [0, 1]. The consistency term, if enabled, requires a second
    prediction tensor from an augmented input (matching the multi-view
    SAM-WSSS recipe of WPS-SAM Sec. 3.4).

    This implementation reproduces the loss formula only; SAM, the
    prompt generation, and pseudo-mask caching live outside the loss.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("wpgseg")
class WPGSegLoss(nn.Module):
    """WPGSeg classification + SAM-pseudo CE + optional KL consistency.

    Args:
        lambda_seg: Weight on the SAM-pseudo cross-entropy (default 1.0).
        cls_weight: Weight on the image-level multi-label BCE
            classification head.
        lambda_consist: Weight on the symmetric KL consistency between
            two augmented predictions (default 0.1). Set to 0 to drop.
        conf_thresh: Per-pixel SAM confidence below which the pixel is
            ignored in the segmentation CE term (default 0.5).
        ignore_index: Label value in the SAM pseudo-mask treated as
            "no supervision" (default -1).
        fg_channel_start: Index of the first foreground channel; channel
            0 is treated as background when fg_channel_start == 1.
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        cls_weight: float = 1.0,
        lambda_consist: float = 0.1,
        conf_thresh: float = 0.5,
        ignore_index: int = -1,
        fg_channel_start: int = 1,
        **kwargs,
    ):
        super().__init__()
        if not 0.0 <= conf_thresh <= 1.0:
            raise ValueError(
                f"conf_thresh must be in [0, 1] (got {conf_thresh})"
            )
        self.lambda_seg = lambda_seg
        self.cls_weight = cls_weight
        self.lambda_consist = lambda_consist
        self.conf_thresh = conf_thresh
        self.ignore_index = ignore_index
        self.fg_channel_start = fg_channel_start

    # ------------------------------------------------------------------
    @staticmethod
    def _sym_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        """0.5 * (KL(p||q) + KL(q||p)) — both averaged over pixels."""
        log_p = F.log_softmax(p_logits, dim=1)
        log_q = F.log_softmax(q_logits, dim=1)
        p = log_p.exp()
        q = log_q.exp()
        kl_pq = (p * (log_p - log_q)).sum(dim=1).mean()
        kl_qp = (q * (log_q - log_p)).sum(dim=1).mean()
        return 0.5 * (kl_pq + kl_qp)

    def forward(
        self,
        predictions: torch.Tensor,
        sam_pseudo_mask: torch.Tensor,
        sam_confidence: Optional[torch.Tensor] = None,
        image_labels: Optional[torch.Tensor] = None,
        predictions_aug: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) semantic logits.
            sam_pseudo_mask: (B, H, W) integer labels distilled from SAM
                (``ignore_index`` for pixels SAM declined to label).
            sam_confidence: (B, H, W) or (B, 1, H, W) per-pixel SAM
                confidence in [0, 1]. If None, defaults to 1 for labelled
                pixels.
            image_labels: (B, C_fg) binary multi-label tags. If supplied,
                a multi-label BCE on GAP(predictions[fg]) is added.
            predictions_aug: Optional (B, C, H, W) second prediction from
                a strongly augmented version of the same input. Used only
                when ``lambda_consist > 0``.
            labeled_loss: Optional dense supervised loss to add.
        """
        if predictions.dim() != 4:
            raise ValueError(
                f"predictions must be (B, C, H, W); got "
                f"{tuple(predictions.shape)}"
            )
        B, C, H, W = predictions.shape

        # Bring sam_pseudo_mask to (B, H, W).
        if sam_pseudo_mask.dim() == 4 and sam_pseudo_mask.shape[1] == 1:
            sam_pseudo_mask = sam_pseudo_mask.squeeze(1)
        if sam_pseudo_mask.shape[-2:] != (H, W):
            sam_pseudo_mask = F.interpolate(
                sam_pseudo_mask.float().unsqueeze(1),
                size=(H, W),
                mode="nearest",
            ).squeeze(1).long()
        else:
            sam_pseudo_mask = sam_pseudo_mask.long()

        # Confidence map → (B, H, W) in [0, 1].
        if sam_confidence is None:
            conf = (sam_pseudo_mask != self.ignore_index).float()
        else:
            conf = sam_confidence
            if conf.dim() == 4 and conf.shape[1] == 1:
                conf = conf.squeeze(1)
            if conf.shape[-2:] != (H, W):
                conf = F.interpolate(
                    conf.unsqueeze(1).float(), size=(H, W), mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            conf = conf.clamp(0.0, 1.0)

        total = predictions.new_zeros(())

        # (1) Image-level multi-label BCE on GAP of fg channels.
        if image_labels is not None and self.cls_weight > 0:
            fg_logits = predictions[:, self.fg_channel_start:]
            if fg_logits.shape[1] != image_labels.shape[1]:
                # Truncate/pad to match (mirrors EPSLoss policy).
                n = min(fg_logits.shape[1], image_labels.shape[1])
                fg_logits = fg_logits[:, :n]
                image_labels = image_labels[:, :n]
            gap = fg_logits.mean(dim=(2, 3))
            cls_loss = F.binary_cross_entropy_with_logits(
                gap, image_labels.float()
            )
            total = total + self.cls_weight * cls_loss

        # (2) Confidence-weighted CE on SAM pseudo-mask.
        if self.lambda_seg > 0:
            valid = (sam_pseudo_mask != self.ignore_index) & (conf > self.conf_thresh)
            if valid.any():
                # log P at the pseudo-class.
                log_p = F.log_softmax(predictions, dim=1)
                # Gather per-pixel log-prob of the pseudo-class. Clamp
                # ignore_index away to avoid OOB indexing.
                target = sam_pseudo_mask.clamp(min=0)
                gathered = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
                w = (conf * valid.float())
                ce_term = -(gathered * w).sum() / w.sum().clamp_min(1.0)
                total = total + self.lambda_seg * ce_term

        # (3) Multi-view KL consistency.
        if predictions_aug is not None and self.lambda_consist > 0:
            if predictions_aug.shape != predictions.shape:
                raise ValueError(
                    f"predictions_aug shape {tuple(predictions_aug.shape)} "
                    f"must match predictions {tuple(predictions.shape)}."
                )
            total = total + self.lambda_consist * self._sym_kl(
                predictions, predictions_aug
            )

        if labeled_loss is not None:
            total = total + labeled_loss
        return total
