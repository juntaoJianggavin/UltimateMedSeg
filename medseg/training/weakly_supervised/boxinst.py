# Reference: https://github.com/aim-uofa/AdelaiDet
# Paper: https://arxiv.org/abs/2012.02310
"""BoxInst — High-Performance Instance Segmentation with Box Annotations.

Tian et al., CVPR 2021.
Paper: https://arxiv.org/abs/2012.02310
Official repository: https://github.com/aim-uofa/AdelaiDet
    (AdelaiDet/adet/modeling/condinst/condinst.py — ``compute_pairwise_term``
     and ``compute_project_term``)

Unlike the BoxSupervisedLoss already in ``losses.py`` (which treats the
whole box as a foreground rectangle pseudo-label — the BoxSup 2015 recipe),
BoxInst supervises the predicted mask through two box-derived terms only:

    L_box = L_proj + L_pair

  (1) Projection loss (Sec. 3.1 of the paper).
      For each ground-truth box B, the 1-D max-projection of the predicted
      mask along the X axis must match the box's 1-D projection along X
      (a rectangular pulse equal to 1 inside the column range of B and 0
      outside). The same is required along Y. Dice loss is computed on the
      two 1-D vectors, summed.

  (2) Pairwise affinity loss (Sec. 3.2).
      For every pair of pixels within a small KxK window, an "affinity"
      indicator is computed from the predicted log-probabilities:
        p_y = log P(y_i = y_j) = log( P_i * P_j + (1 - P_i)(1 - P_j) )
      The loss is then -log p_y averaged over pairs whose colour similarity
      ``exp(-||I_i - I_j|| / tau)`` exceeds ``color_thresh``. This pushes
      label agreement only between pixels that look alike — exactly the
      colour-affinity term in BoxInst.

This module re-implements both terms directly from the paper formulae; no
code is lifted from the AdelaiDet repository.
"""

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("boxinst")
class BoxInstLoss(nn.Module):
    """BoxInst projection + colour-affinity pairwise loss.

    Args:
        projection_weight: Weight on the X/Y dice projection loss
            (BoxInst default 1.0).
        pairwise_weight: Weight on the colour-similarity pairwise term
            (BoxInst default 1.0; warm-up to 1.0 over a few iters in the
            official repo — we omit warm-up and let the user schedule it).
        pairwise_size: Window size K for the pairwise term (default 3 → 8
            neighbours per pixel after excluding self).
        pairwise_dilation: Dilation for the pairwise window (default 2).
        color_thresh: Similarity threshold above which a pair is
            "same-colour" and contributes to the pairwise loss
            (BoxInst default 0.3, on Lab-space cosine similarity).
        color_tau: Temperature for the colour Gaussian
            ``exp(-||I_i - I_j|| / tau)`` (default 0.5, paper uses Lab).
        fg_channel_start: First channel treated as foreground in
            ``predictions``. Multi-class softmax outputs (C > 1) are
            reduced to a single fg map by summing channels >= this index.
            With ``fg_channel_start = 1`` (default) channel 0 = bg.
        eps: Numerical floor.
    """

    def __init__(
        self,
        projection_weight: float = 1.0,
        pairwise_weight: float = 1.0,
        pairwise_size: int = 3,
        pairwise_dilation: int = 2,
        color_thresh: float = 0.3,
        color_tau: float = 0.5,
        fg_channel_start: int = 1,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        if pairwise_size < 3 or pairwise_size % 2 == 0:
            raise ValueError(
                f"pairwise_size must be odd and >=3, got {pairwise_size}"
            )
        self.projection_weight = projection_weight
        self.pairwise_weight = pairwise_weight
        self.pairwise_size = pairwise_size
        self.pairwise_dilation = max(pairwise_dilation, 1)
        self.color_thresh = color_thresh
        self.color_tau = color_tau
        self.fg_channel_start = fg_channel_start
        self.eps = eps

    # ------------------------------------------------------------------
    # Foreground probability map
    # ------------------------------------------------------------------
    def _fg_prob(self, predictions: torch.Tensor) -> torch.Tensor:
        """Reduce (B, C, H, W) logits → (B, 1, H, W) foreground prob."""
        if predictions.shape[1] == 1:
            return torch.sigmoid(predictions)
        prob = F.softmax(predictions, dim=1)
        if self.fg_channel_start >= prob.shape[1]:
            raise ValueError(
                f"fg_channel_start={self.fg_channel_start} exceeds C={prob.shape[1]}"
            )
        fg = prob[:, self.fg_channel_start:].sum(dim=1, keepdim=True)
        return fg.clamp(self.eps, 1.0 - self.eps)

    # ------------------------------------------------------------------
    # (1) Projection loss — dice along X and Y for each box
    # ------------------------------------------------------------------
    @staticmethod
    def _dice_1d(pred: torch.Tensor, gt: torch.Tensor, eps: float) -> torch.Tensor:
        """Soft dice on 1-D vectors. Matches official BoxInst
        ``dice_coefficient``: denominator uses x² + target² (not x + target).

        Official: loss = 1 - 2*intersection / (x**2.sum() + target**2.sum() + eps)
        """
        num = 2.0 * (pred * gt).sum(dim=-1) + eps
        den = (pred ** 2).sum(dim=-1) + (gt ** 2).sum(dim=-1) + eps
        return 1.0 - num / den

    def _projection_loss(
        self,
        fg_prob: torch.Tensor,            # (B, 1, H, W)
        boxes_per_image: Sequence,        # list[B] of (N_b, 4) tensors xyxy
    ) -> torch.Tensor:
        """Sum of 1-D dice losses on max-projections inside each box.

        Inside each ``box = [x1, y1, x2, y2]``:
            * crop predicted fg-prob to box and max over rows → (W_box,)
              vs ground-truth all-ones vector of the same length;
            * crop and max over columns → (H_box,) vs all-ones vector;
            * dice(pred_proj, ones).
        Boxes are processed independently and averaged across the batch.
        """
        device = fg_prob.device
        B, _, H, W = fg_prob.shape
        losses: List[torch.Tensor] = []

        for b in range(B):
            boxes_b = boxes_per_image[b]
            if boxes_b is None or len(boxes_b) == 0:
                continue
            if not torch.is_tensor(boxes_b):
                boxes_b = torch.as_tensor(boxes_b, device=device)
            else:
                boxes_b = boxes_b.to(device)
            if boxes_b.numel() == 0:
                continue
            boxes_b = boxes_b.view(-1, 4)

            for bx in boxes_b:
                x1, y1, x2, y2 = bx.long().tolist()
                x1 = max(0, min(x1, W - 1))
                x2 = max(x1 + 1, min(x2, W))
                y1 = max(0, min(y1, H - 1))
                y2 = max(y1 + 1, min(y2, H))

                crop = fg_prob[b, 0, y1:y2, x1:x2]   # (h, w)
                # max-projection along Y → 1-D vector of length w
                proj_x = crop.max(dim=0).values
                # max-projection along X → 1-D vector of length h
                proj_y = crop.max(dim=1).values

                gt_x = torch.ones_like(proj_x)
                gt_y = torch.ones_like(proj_y)

                losses.append(self._dice_1d(proj_x, gt_x, self.eps))
                losses.append(self._dice_1d(proj_y, gt_y, self.eps))

        if not losses:
            return fg_prob.new_zeros(())
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # (2) Pairwise colour-affinity loss
    # ------------------------------------------------------------------
    def _unfold(self, x: torch.Tensor) -> torch.Tensor:
        """Sliding KxK window, dilated. Returns (B, C, K*K, H, W)."""
        k = self.pairwise_size
        d = self.pairwise_dilation
        padding = (k // 2) * d
        # nn.functional.unfold returns (B, C*k*k, H*W); reshape carefully
        B, C, H, W = x.shape
        patches = F.unfold(x, kernel_size=k, dilation=d, padding=padding)
        patches = patches.view(B, C, k * k, H, W)
        return patches

    def _color_similarity(self, images: torch.Tensor) -> torch.Tensor:
        """exp(-||I_center - I_neighbour|| / tau) over a KxK window.

        Returns (B, K*K - 1, H, W) — the centre offset is excluded.
        """
        patches = self._unfold(images)                  # (B, C, K^2, H, W)
        center = patches[:, :, patches.shape[2] // 2:patches.shape[2] // 2 + 1]
        diff = patches - center                         # broadcast
        dist = diff.pow(2).sum(dim=1).sqrt()            # (B, K^2, H, W)

        sim = torch.exp(-dist / self.color_tau)         # (B, K^2, H, W)
        # Drop centre slot.
        c = sim.shape[1] // 2
        keep = torch.cat([sim[:, :c], sim[:, c + 1:]], dim=1)
        return keep

    def _pairwise_loss(
        self,
        fg_prob: torch.Tensor,            # (B, 1, H, W)
        images: torch.Tensor,             # (B, 3, H, W) — paper uses Lab
        boxes_per_image: Sequence,        # to build the in-box mask
    ) -> torch.Tensor:
        """BoxInst pairwise term, restricted to in-box pixels.

        Loss = -log P(y_i = y_j) summed over neighbour pairs (i, j) whose
        colour similarity > color_thresh and whose centre pixel lies inside
        at least one ground-truth box (BoxInst uses ``mask_out`` to ignore
        outside-box regions; we follow the same convention)."""
        B, _, H, W = fg_prob.shape

        # log P(same label) for foreground prob p_i, p_j ∈ (eps, 1-eps)
        log_p_center = torch.log(fg_prob)
        log_1mp_center = torch.log(1.0 - fg_prob)

        # Neighbour log-probs via unfold of the same maps.
        log_p_nb = self._unfold(log_p_center)            # (B, 1, K^2, H, W)
        log_1mp_nb = self._unfold(log_1mp_center)
        c = log_p_nb.shape[2] // 2
        log_p_nb = torch.cat([log_p_nb[:, :, :c], log_p_nb[:, :, c + 1:]], dim=2)
        log_1mp_nb = torch.cat([log_1mp_nb[:, :, :c], log_1mp_nb[:, :, c + 1:]], dim=2)

        # Same-label log-prob (numerically stable logaddexp).
        log_same = torch.logaddexp(
            log_p_center.unsqueeze(2) + log_p_nb,
            log_1mp_center.unsqueeze(2) + log_1mp_nb,
        )                                                # (B, 1, K^2-1, H, W)

        # Colour similarity gate.
        sim = self._color_similarity(images)             # (B, K^2-1, H, W)
        sim_gate = (sim >= self.color_thresh).float().unsqueeze(1)

        # In-box mask (B, 1, H, W). Pixels outside every box are masked out.
        in_box = torch.zeros(B, 1, H, W, device=fg_prob.device)
        for b in range(B):
            boxes_b = boxes_per_image[b]
            if boxes_b is None or len(boxes_b) == 0:
                continue
            if not torch.is_tensor(boxes_b):
                boxes_b = torch.as_tensor(boxes_b, device=fg_prob.device)
            else:
                boxes_b = boxes_b.to(fg_prob.device)
            if boxes_b.numel() == 0:
                continue
            for bx in boxes_b.view(-1, 4):
                x1, y1, x2, y2 = bx.long().tolist()
                x1 = max(0, min(x1, W - 1))
                x2 = max(x1 + 1, min(x2, W))
                y1 = max(0, min(y1, H - 1))
                y2 = max(y1 + 1, min(y2, H))
                in_box[b, 0, y1:y2, x1:x2] = 1.0

        weight = sim_gate * in_box.unsqueeze(2)          # (B, 1, K^2-1, H, W)
        denom = weight.sum().clamp_min(1.0)
        # We want -log P(same), weighted, mean over valid pairs.
        loss = -(log_same * weight).sum() / denom
        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        predictions: torch.Tensor,
        images: torch.Tensor,
        boxes: Union[torch.Tensor, Sequence],
        labeled_loss: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) semantic logits. C=1 is treated as
                sigmoid foreground; C>1 uses softmax and sums channels
                ``[fg_channel_start:]`` as the foreground probability.
            images: (B, 3, H, W). The paper recommends CIELAB but RGB is
                accepted; tune ``color_tau`` accordingly.
            boxes: Per-image bounding boxes. Either a (B, N, 4) tensor (with
                possibly padded -1 rows) or a list of length B of (N_b, 4)
                tensors / arrays in [x1, y1, x2, y2] format.
            labeled_loss: Optional pre-computed supervised CE/dice to add
                (useful in mixed weak/strong batches).
        """
        fg_prob = self._fg_prob(predictions)

        if torch.is_tensor(boxes) and boxes.dim() == 3:
            # Convert (B, N, 4) → list[B] of (N_b, 4) by stripping -1 rows.
            boxes_per_image = [
                bx[(bx >= 0).all(dim=-1)] for bx in boxes
            ]
        else:
            boxes_per_image = list(boxes)

        proj = self._projection_loss(fg_prob, boxes_per_image)
        pair = self._pairwise_loss(fg_prob, images, boxes_per_image)

        total = self.projection_weight * proj + self.pairwise_weight * pair
        if labeled_loss is not None:
            total = total + labeled_loss
        return total
