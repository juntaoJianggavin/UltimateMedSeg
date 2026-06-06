"""Interactive / click-based segmentation losses.

Includes:
    - fBRSLoss: helper class with static ``encode_clicks`` method
      (port of the GPU branch of DistMaps from the official RITM code).
    - iSegLoss: iterative refinement with click feedback
      (Wang et al., ECCV 2018).
    - ClickSupervisionLoss: click-based supervision combining positive/
      negative click signals, entropy regularisation, spatial smoothness,
      and optional box prior
      (Bearman et al., ECCV 2016 / Papadopoulos et al., ICCV 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from medseg.registry import LOSS_REGISTRY


# ---------------------------------------------------------------------------
# Low-level helper
# ---------------------------------------------------------------------------

def _dist_maps_single(
    points: torch.Tensor,
    coord_rows: torch.Tensor,
    coord_cols: torch.Tensor,
    H: int,
    W: int,
    norm_radius: float,
    use_disks: bool,
) -> torch.Tensor:
    """Compute distance/disk map for one set of points (one batch element).

    Faithful port of the GPU branch of
    ``DistMaps.get_coord_features`` from the official RITM code.

    Args:
        points: (K, 2) with ``[x, y]`` (column, row).  May be empty.
        coord_rows, coord_cols: (H, W) meshgrid arrays.
        H, W: spatial dimensions.
        norm_radius: normalisation radius.
        use_disks: binary disk mode vs. distance-transform mode.

    Returns:
        (1, H, W) distance or disk map.
    """
    device = points.device

    if points.numel() == 0:
        return torch.full((1, H, W), 1e6, device=device)

    pt_rows = points[:, 1].view(-1, 1, 1)   # y coords
    pt_cols = points[:, 0].view(-1, 1, 1)   # x coords

    invalid = (points[:, 0] < 0) | (points[:, 1] < 0)  # (K,)

    dr = coord_rows.unsqueeze(0) - pt_rows    # (K, H, W)
    dc = coord_cols.unsqueeze(0) - pt_cols    # (K, H, W)

    if not use_disks:
        dr = dr / norm_radius
        dc = dc / norm_radius

    sq_dist = dr * dr + dc * dc               # (K, H, W)

    if invalid.any():
        sq_dist[invalid.view(-1, 1, 1).expand_as(sq_dist)] = 1e6

    min_sq_dist = sq_dist.min(dim=0)[0]       # (H, W)

    if use_disks:
        result = (min_sq_dist <= norm_radius ** 2).float()
    else:
        result = min_sq_dist.sqrt().mul(2).tanh()

    return result.unsqueeze(0)                # (1, H, W)


# ---------------------------------------------------------------------------
# fBRSLoss – click-encoding helper (was referenced but never defined)
# ---------------------------------------------------------------------------

class fBRSLoss:
    """Utility class providing click-encoding for interactive segmentation.

    This class was previously referenced by ``iSegLoss`` and
    ``ClickSupervisionLoss`` but was never defined.  The static
    ``encode_clicks`` method converts raw click coordinates into
    positive / negative distance (or disk) maps that can be used as
    spatial supervision signals.
    """

    @staticmethod
    def encode_clicks(
        clicks: torch.Tensor,
        spatial_shape: Tuple[int, int],
        norm_radius: float = 3.0,
        use_disks: bool = True,
    ) -> torch.Tensor:
        """Encode click coordinates into positive/negative distance maps.

        Args:
            clicks: (B, N, 3) tensor where each click is ``[x, y, is_pos]``.
                    ``is_pos > 0`` marks a positive (foreground) click;
                    coordinates with ``x < 0`` or ``y < 0`` are ignored.
            spatial_shape: ``(H, W)`` of the output maps.
            norm_radius: radius for the disk / normalisation.
            use_disks: if ``True``, produce binary disk maps; otherwise
                       smooth distance-transform maps.

        Returns:
            (B, 2, H, W) tensor — channel 0 = positive map, channel 1 =
            negative map.
        """
        H, W = spatial_shape
        B = clicks.shape[0]
        device = clicks.device

        rows = torch.arange(H, device=device).float()
        cols = torch.arange(W, device=device).float()
        coord_rows, coord_cols = torch.meshgrid(rows, cols, indexing="ij")

        pos_maps: List[torch.Tensor] = []
        neg_maps: List[torch.Tensor] = []

        for b in range(B):
            batch_clicks = clicks[b]  # (N, 3)
            valid = (batch_clicks[:, 0] >= 0) & (batch_clicks[:, 1] >= 0)
            valid_clicks = batch_clicks[valid]  # (K, 3)

            pos_mask = valid_clicks[:, 2] > 0
            pos_pts = valid_clicks[pos_mask, :2]    # (Kp, 2)  [x, y]
            neg_pts = valid_clicks[~pos_mask, :2]   # (Kn, 2)  [x, y]

            pos_map = _dist_maps_single(
                pos_pts, coord_rows, coord_cols, H, W, norm_radius, use_disks
            )
            neg_map = _dist_maps_single(
                neg_pts, coord_rows, coord_cols, H, W, norm_radius, use_disks
            )

            pos_maps.append(pos_map)
            neg_maps.append(neg_map)

        pos_batch = torch.stack(pos_maps, dim=0)  # (B, 1, H, W)
        neg_batch = torch.stack(neg_maps, dim=0)  # (B, 1, H, W)

        # Convert distance maps to binary masks:
        # disk mode → 1 inside radius; distance mode → 1 where value < 1
        pos_binary = (pos_batch < 1.0).float() if not use_disks else pos_batch
        neg_binary = (neg_batch < 1.0).float() if not use_disks else neg_batch

        return torch.cat([pos_binary, neg_binary], dim=1)  # (B, 2, H, W)


# ---------------------------------------------------------------------------
# iSegLoss
# ---------------------------------------------------------------------------

@LOSS_REGISTRY.register("iseg")
class iSegLoss(nn.Module):
    """iSeg: iterative refinement with click feedback.

    Wang et al., "Interactive Image Segmentation with Latent Diversity",
    ECCV 2018.
    """

    def __init__(
        self,
        max_rounds: int = 5,
        click_radius: int = 3,
        round_decay: float = 0.8,
        **kwargs,
    ):
        super().__init__()
        self.max_rounds = max_rounds
        self.click_radius = click_radius
        self.round_decay = round_decay

    def _simulate_clicks(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = predictions.shape
        device = predictions.device
        pred_cls = predictions.argmax(dim=1)
        clicks_list = []
        for b in range(B):
            err = (pred_cls[b] != target[b]).float()
            if err.sum() == 0:
                clicks_list.append(torch.zeros(1, 3, device=device) - 1)
                continue
            yy, xx = torch.where(err > 0.5)
            if len(yy) == 0:
                clicks_list.append(torch.zeros(1, 3, device=device) - 1)
                continue
            cy, cx = yy.float().mean().long(), xx.float().mean().long()
            is_pos = (target[b, cy, cx] > 0).float()
            click = torch.tensor([[cx.float(), cy.float(), is_pos]], device=device)
            clicks_list.append(click)
        max_n = max(c.shape[0] for c in clicks_list)
        padded = torch.zeros(B, max_n, 3, device=device) - 1
        for b, c in enumerate(clicks_list):
            padded[b, :c.shape[0]] = c
        return padded

    def forward(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor,
        clicks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if clicks is None:
            clicks = self._simulate_clicks(predictions, target)
        click_maps = fBRSLoss.encode_clicks(clicks, predictions.shape[2:],
                                            norm_radius=self.click_radius,
                                            use_disks=True)
        loss = F.cross_entropy(predictions, target.long(), ignore_index=-1)
        prob = F.softmax(predictions, dim=1)
        pos_mask = click_maps[:, :1]
        neg_mask = click_maps[:, 1:]
        fg_prob = prob[:, 1:].sum(dim=1, keepdim=True) if predictions.shape[1] > 1 else prob[:, :1]
        aux = -(pos_mask * torch.log(fg_prob + 1e-8)).mean()
        aux = aux - (neg_mask * torch.log(1 - fg_prob + 1e-8)).mean()
        loss = loss + 0.5 * aux
        return loss


# ---------------------------------------------------------------------------
# ClickSupervisionLoss
# ---------------------------------------------------------------------------

@LOSS_REGISTRY.register("click_supervision")
class ClickSupervisionLoss(nn.Module):
    """Click-based supervision for interactive segmentation.

    Combines positive/negative click supervision, entropy regularisation
    on unlabelled regions, spatial smoothness, and optional box prior.

    Bearman et al., ECCV 2016 / Papadopoulos et al., ICCV 2017.
    """

    def __init__(
        self,
        click_radius: int = 3,
        entropy_weight: float = 0.1,
        smoothness_weight: float = 0.2,
        box_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.click_radius = click_radius
        self.entropy_weight = entropy_weight
        self.smoothness_weight = smoothness_weight
        self.box_weight = box_weight

    def forward(
        self,
        predictions: torch.Tensor,
        clicks: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = predictions.shape
        prob = F.softmax(predictions, dim=1)
        loss = torch.tensor(0.0, device=predictions.device)

        if clicks is not None:
            click_maps = fBRSLoss.encode_clicks(clicks, (H, W),
                                                norm_radius=self.click_radius,
                                                use_disks=True)
            pos_mask = click_maps[:, :1]
            neg_mask = click_maps[:, 1:]
            fg_prob = prob[:, 1:].sum(dim=1, keepdim=True) if C > 1 else prob[:, :1]
            click_loss = -(pos_mask * torch.log(fg_prob + 1e-8)).mean()
            click_loss = click_loss - (neg_mask * torch.log(1 - fg_prob + 1e-8)).mean()
            loss = loss + click_loss

            click_region = (click_maps.sum(dim=1, keepdim=True) > 0).float()
            unlabelled = 1.0 - click_region
            entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1, keepdim=True)
            loss = loss + self.entropy_weight * (unlabelled * entropy).mean()

        fg = prob[:, 1:].sum(dim=1) if C > 1 else prob[:, 0]
        grad_h = (fg[:, 1:, :] - fg[:, :-1, :]).abs().mean()
        grad_w = (fg[:, :, 1:] - fg[:, :, :-1]).abs().mean()
        loss = loss + self.smoothness_weight * (grad_h + grad_w)

        if boxes is not None:
            box_mask = torch.zeros(B, 1, H, W, device=predictions.device)
            for b in range(B):
                if boxes[b].numel() >= 4:
                    x1, y1, x2, y2 = boxes[b].long()
                    box_mask[b, 0, y1:y2, x1:x2] = 1.0
            outside = 1.0 - box_mask
            bg_prob = prob[:, :1]
            loss = loss + self.box_weight * (outside * (-torch.log(bg_prob + 1e-8))).mean()

        if target is not None:
            loss = loss + F.cross_entropy(predictions, target.long(), ignore_index=-1)
        return loss
