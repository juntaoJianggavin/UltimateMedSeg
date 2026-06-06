# AllSpark (CVPR 2024)
# Reference: https://github.com/xmed-lab/AllSpark
# Paper: https://arxiv.org/abs/2403.01818
# Implemented from paper formulas; not a copy of the official repo.
"""AllSpark: All-pairs Reborn Token Integration for Semi-Supervised Seg.

Wang et al., "AllSpark: Reborn Labeled Features from Unlabeled in
Transformer for Semi-Supervised Semantic Segmentation", CVPR 2024.

The core component is the *AllSpark* cross-attention block (paper Eq. 3-5):

    Given a batch of labeled and unlabeled decoder features
    F_l in R^{B_l x N x C}, F_u in R^{B_u x N x C} (flattened tokens),
    AllSpark "rebirth"s the labeled tokens by cross-attending them to
    the *unlabeled* tokens — the labeled features query a key/value
    bank built from the unlabeled features:

        Q = F_l W_q
        K = F_u W_k                                            (cross batch)
        V = F_u W_v
        F_l_reborn = LayerNorm( F_l + Attn(Q, K, V) )
        F_l_reborn = LayerNorm( F_l_reborn + FFN(F_l_reborn) )

    The reborn labeled features then go through the segmentation head and
    are supervised by the *true* labels.  The unlabeled features go
    through the same head and are supervised by an EMA-teacher
    pseudo-label (with a FixMatch-style confidence threshold).

    Intuitively the cross-attention forces the labeled supervision to flow
    along inter-batch token correspondences, regularising the encoder so
    its labeled features cannot "memorise" labels independently of the
    unlabeled distribution.

This file implements the block + training loop from the paper equations
alone — no GitHub source code is referenced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight,
    pseudo_label_with_threshold,
)


# ---------------------------------------------------------------------------
# AllSpark cross-attention block
# ---------------------------------------------------------------------------

class AllSparkBlock(nn.Module):
    """Cross-batch cross-attention: labeled queries, unlabeled K/V.

    Operates on (B, C, H, W) feature maps; flattens to tokens internally.
    """

    def __init__(self, channels: int, num_heads: int = 8,
                 mlp_ratio: float = 2.0, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"AllSpark: channels ({channels}) must be divisible by "
                f"num_heads ({num_heads}).")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels, bias=True)
        self.k_proj = nn.Linear(channels, channels, bias=True)
        self.v_proj = nn.Linear(channels, channels, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(channels, channels, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        hidden = max(int(channels * mlp_ratio), channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, channels),
            nn.Dropout(proj_drop),
        )

    def _attend(self, q_tokens, kv_tokens):
        """q_tokens: (Bq, N, C); kv_tokens: (Bk, N, C).

        Cross-batch: every query attends over the concatenation of *all*
        unlabeled tokens — i.e. K/V is a bank of size (Bk * N, C).  This
        is the "all-pairs" semantic the paper's name calls out.
        """
        Bq, N, C = q_tokens.shape
        Bk, Nk, _ = kv_tokens.shape

        q = self.q_proj(q_tokens).view(Bq, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (Bq, h, N, d)

        bank = kv_tokens.reshape(1, Bk * Nk, C).expand(Bq, -1, -1)
        k = self.k_proj(bank).view(Bq, Bk * Nk, self.num_heads, self.head_dim)
        v = self.v_proj(bank).view(Bq, Bk * Nk, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (Bq, h, N, Bk*Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v                                  # (Bq, h, N, d)
        out = out.permute(0, 2, 1, 3).reshape(Bq, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out

    def forward(self, feat_l: torch.Tensor, feat_u: torch.Tensor) -> torch.Tensor:
        """Rebirth labeled features by cross-attending to unlabeled.

        Args:
            feat_l: (Bl, C, H, W) labeled features.
            feat_u: (Bu, C, H, W) unlabeled features.
        Returns:
            reborn (Bl, C, H, W) features (residual added).
        """
        Bl, C, H, W = feat_l.shape
        Bu, Cu, Hu, Wu = feat_u.shape
        if (C != Cu) or (H != Hu) or (W != Wu):
            raise ValueError(
                "AllSparkBlock requires identical channel/spatial dims for "
                f"labeled and unlabeled features; got {feat_l.shape} vs "
                f"{feat_u.shape}.")
        N = H * W
        tokens_l = feat_l.flatten(2).transpose(1, 2)  # (Bl, N, C)
        tokens_u = feat_u.flatten(2).transpose(1, 2)  # (Bu, N, C)

        # Pre-norm transformer block
        q_in = self.norm1(tokens_l)
        kv_in = self.norm_kv(tokens_u)
        tokens_l = tokens_l + self._attend(q_in, kv_in)
        tokens_l = tokens_l + self.mlp(self.norm2(tokens_l))

        return tokens_l.transpose(1, 2).reshape(Bl, C, H, W)


# ---------------------------------------------------------------------------
# AllSpark method
# ---------------------------------------------------------------------------

class AllSpark(BaseSemiMethod):
    """AllSpark semi-supervised segmentation.

    The model must expose the standard SegmentationModel layout
    (.encoder / .bottleneck / .decoder / .head with Conv2d 'head.conv'),
    so the AllSpark block can be inserted between the decoder and the
    head.  No silent fallback — raises if the layout is missing.

    Args:
        model: Student model.
        device: Torch device.
        ema_decay: EMA decay for the teacher (default 0.999).
        confidence_threshold: Pseudo-label confidence threshold (default 0.95).
        consistency_weight: Max unsupervised loss weight (default 1.0).
        rampup_epochs: Sigmoid ramp-up epochs (default 40).
        num_heads: Cross-attention head count (default 8).
        mlp_ratio: FFN expansion ratio inside AllSparkBlock (default 2.0).
        attn_drop, proj_drop: dropout rates inside the block.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 ema_decay: float = 0.999,
                 confidence_threshold: float = 0.95,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 num_heads: int = 8,
                 mlp_ratio: float = 2.0,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        self.ema_decay = ema_decay
        self.confidence_threshold = float(confidence_threshold)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.attn_drop = float(attn_drop)
        self.proj_drop = float(proj_drop)
        self.teacher: nn.Module = None
        self.allspark: AllSparkBlock = None
        self._feat_channels: int = -1

    def build(self) -> None:
        if not (hasattr(self.model, 'encoder')
                and hasattr(self.model, 'bottleneck')
                and hasattr(self.model, 'decoder')
                and hasattr(self.model, 'head')
                and hasattr(self.model.head, 'conv')):
            raise TypeError(
                "AllSpark requires SegmentationModel(.encoder + .bottleneck "
                "+ .decoder + .head with Conv2d 'head.conv').  Got "
                f"{type(self.model).__name__}.")
        self._feat_channels = int(self.model.head.conv.in_channels)
        # Pick a head count that divides the channel dim; fail loudly otherwise.
        h = self.num_heads
        while h > 1 and self._feat_channels % h != 0:
            h //= 2
        if self._feat_channels % h != 0:
            raise ValueError(
                f"AllSpark: cannot pick a head count dividing decoder "
                f"channels {self._feat_channels} (requested {self.num_heads}).")
        if h != self.num_heads:
            self.num_heads = h
        self.allspark = AllSparkBlock(
            channels=self._feat_channels,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
        ).to(self.device)

        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)

    def extra_params(self):
        if self.allspark is None:
            return []
        return list(self.allspark.parameters())

    def _decoded(self, model: nn.Module, x: torch.Tensor):
        """Run encoder + bottleneck + decoder, return (decoded_feat, head_logits)."""
        feats = model.encoder(x)
        btn = model.bottleneck(feats[-1])
        decoded = model.decoder(btn, feats[:-1])
        return decoded

    @staticmethod
    def _take_first(out):
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def _final_logits(self, feat: torch.Tensor, target_hw):
        logits = self.model.head(feat)
        if logits.shape[2:] != target_hw:
            logits = F.interpolate(
                logits, size=target_hw, mode='bilinear', align_corners=False)
        return logits

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
        self.allspark.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)
        H, W = images_l.shape[-2:]

        # --- Pseudo-label from EMA teacher on weak unlabeled view ---
        with torch.no_grad():
            t_logits = self._take_first(self.teacher(images_u))
            pseudo, mask = pseudo_label_with_threshold(
                t_logits, self.confidence_threshold)

        # --- Forward student decoder for both labeled and unlabeled in
        #     the same train step (parameters shared, BN sees both halves).
        decoded_l = self._decoded(self.model, images_l)
        decoded_u = self._decoded(self.model, images_u)

        # Spatial sizes must match for the AllSpark block; resample if not.
        if decoded_l.shape[2:] != decoded_u.shape[2:]:
            decoded_u_for_attn = F.interpolate(
                decoded_u, size=decoded_l.shape[2:],
                mode='bilinear', align_corners=False)
        else:
            decoded_u_for_attn = decoded_u

        # --- AllSpark rebirth: labeled features are refined by cross-
        #     attending to unlabeled features.
        decoded_l_reborn = self.allspark(decoded_l, decoded_u_for_attn)

        # --- Supervised loss on REBORN labeled features
        logits_l = self._final_logits(decoded_l_reborn, (H, W))
        sup_loss = criterion(logits_l, labels)

        # --- Unsupervised CE on unlabeled (head shared) ---
        logits_u = self._final_logits(decoded_u, (H, W))
        ce_pix = F.cross_entropy(logits_u, pseudo, ignore_index=-1, reduction='none')
        denom = mask.float().sum().clamp(min=1.0)
        unsup_loss = (ce_pix * mask.float()).sum() / denom

        w = get_current_consistency_weight(epoch, self.consistency_weight, self.rampup_epochs)
        total_loss = sup_loss + w * unsup_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.allspark.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": unsup_loss.item(),
            "w": w,
            "conf_ratio": mask.float().mean().item(),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        # At inference there is no AllSpark rebirth (no labeled query / unlabeled bank
        # pairing), so we evaluate the EMA teacher's plain forward — same convention
        # as MeanTeacher and UniMatch in this codebase.
        return self.teacher
