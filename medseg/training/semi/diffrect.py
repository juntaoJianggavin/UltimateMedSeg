# DiffRect (MICCAI 2024)
# Reference: https://github.com/CUHK-AIM-Group/DiffRect
# Paper: https://arxiv.org/abs/2407.09918
# Implemented from paper formulas; not a copy of the official repo.
"""DiffRect: Latent-space Label Rectification via Diffusion.

Liu et al., "DiffRect: Latent Diffusion Label Rectification for
Semi-supervised Medical Image Segmentation", MICCAI 2024.

Conceptual recipe (paper Sec. 3) — implemented from the equations:

  The teacher EMA pseudo-label ``p_pseudo`` is treated as a *noisy*
  version of the unknown clean label ``y_0``.  A conditional diffusion
  network ``g_theta(y_t, image, t)`` is trained to *directly* predict
  the clean label ``y_0`` from a corrupted version, i.e. an
  ``x_0``-prediction variant (paper Eq. 4):

      y_t = sqrt(bar_alpha_t) * y_one_hot + sqrt(1 - bar_alpha_t) * eps
      y_hat_0 = g_theta(y_t, image, t)

  Training (labeled subset, Eq. 6):
      L_diff = CE( y_hat_0 , y_GT )
                                    +  lambda_L1 * | softmax(y_hat_0) - y_one_hot |_1

  Label rectification (unlabeled, Eq. 8):
      Let p_t = teacher's softmax;  treat p_t as a sample at small
      timestep ``t_rect`` (light corruption), then a *single* forward
      pass of g_theta gives the rectified soft label
          p_rect = softmax( g_theta( p_t, image, t_rect ) ).
      Hard target = argmax(p_rect), kept iff max(p_rect) >= tau.

  Why this is distinct from DDFP (also in this repo):
    * DDFP is an eps-prediction DDPM trained with MSE on noise, and uses
      multi-step DDIM sampling to denoise the pseudo-label.
    * DiffRect is an x0-prediction model trained with CE on labels, and
      rectifies with a single forward pass.  The "Latent Diffusion"
      framing of the paper means y_0 is the one-hot label map (so the
      "latent" is the categorical simplex), and the diffusion operates
      directly on label-channels with image conditioning.

The denoiser is a small UNet (label channels are few, so few params
suffice).  The student segmentation model is unchanged.

No GitHub source was copied — everything is implemented from the above
equations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .base import BaseSemiMethod
from .utils import (
    create_ema_model, ema_update,
    get_current_consistency_weight,
)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def _sin_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"time embed dim must be even, got {dim}.")
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32) / float(half)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---------------------------------------------------------------------------
# Tiny x0-prediction conditional UNet
# ---------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_embed):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_embed))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class LabelRectifier(nn.Module):
    """x0-prediction UNet: g_theta(y_t, image, t) -> y_0_logits.

    The output is *logits over classes*, not noise — this is the
    "Label Context Calibration" framing of the paper (predict the
    rectified label directly, then sample softmax).
    """

    def __init__(self, num_classes: int, img_channels: int = 3,
                 base_ch: int = 32, time_dim: int = 64):
        super().__init__()
        self.num_classes = int(num_classes)
        self.time_dim = int(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        in_ch = num_classes + img_channels
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down1 = _Block(base_ch, base_ch * 2, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid = _Block(base_ch * 2, base_ch * 2, time_dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec = _Block(base_ch * 4, base_ch, time_dim)
        self.out = nn.Conv2d(base_ch, num_classes, 3, padding=1)

    def forward(self, y_t: torch.Tensor, image: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        if y_t.shape[1] != self.num_classes:
            raise ValueError(
                f"y_t expected {self.num_classes} channels, got {y_t.shape[1]}.")
        if image.shape[-2:] != y_t.shape[-2:]:
            image = F.interpolate(
                image, size=y_t.shape[-2:],
                mode='bilinear', align_corners=False)
        emb = self.time_mlp(_sin_time_embed(t, self.time_dim))
        x = torch.cat([y_t, image], dim=1)
        x0 = self.in_conv(x)
        d1 = self.down1(x0, emb)
        m = self.mid(self.pool(d1), emb)
        u = self.up(m)
        if u.shape[-2:] != d1.shape[-2:]:
            u = F.interpolate(u, size=d1.shape[-2:],
                              mode='bilinear', align_corners=False)
        d = self.dec(torch.cat([u, d1], dim=1), emb)
        return self.out(d)


# ---------------------------------------------------------------------------
# Linear-beta forward (q-sample) only — no DDIM at inference, single step.
# ---------------------------------------------------------------------------

class _ForwardSchedule:
    def __init__(self, num_timesteps: int, beta_start: float,
                 beta_end: float, device: torch.device):
        if num_timesteps <= 1:
            raise ValueError(f"num_timesteps must be > 1, got {num_timesteps}.")
        self.num_timesteps = int(num_timesteps)
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        bar_alphas = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_bar = torch.sqrt(bar_alphas)
        self.sqrt_one_minus_bar = torch.sqrt(1.0 - bar_alphas)

    def to(self, device):
        self.sqrt_bar = self.sqrt_bar.to(device)
        self.sqrt_one_minus_bar = self.sqrt_one_minus_bar.to(device)
        return self

    def q_sample(self, y0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        sb = self.sqrt_bar[t].view(-1, 1, 1, 1)
        sob = self.sqrt_one_minus_bar[t].view(-1, 1, 1, 1)
        return sb * y0 + sob * noise


# ---------------------------------------------------------------------------
# DiffRect method
# ---------------------------------------------------------------------------

class DiffRect(BaseSemiMethod):
    """DiffRect — diffusion-based pseudo-label rectification.

    Args:
        model: Student segmentation model.
        device: Torch device.
        num_classes: Required — number of segmentation classes; raises if
            missing or < 2 (no silent default).
        ema_decay: EMA decay for the teacher (default 0.999).
        consistency_weight: Max unsupervised loss weight (default 1.0).
        rampup_epochs: Sigmoid ramp-up epochs (default 40).
        diff_loss_weight: Weight on the rectifier's training loss (default 1.0).
        diff_warmup_epochs: Only train the rectifier for the first ``E``
            epochs; the student does not consume rectified labels yet
            (default 5).
        diff_num_timesteps: Number of diffusion timesteps T (default 200).
        diff_rect_t: Timestep at which the teacher pseudo-label is
            injected for the single-pass rectification (default 30 — a
            "light corruption" assumption matching paper Sec. 3.3).
        diff_base_ch: Base channel count for the rectifier UNet (default 32).
        diff_pred_threshold: Confidence threshold for the rectified soft
            label (default 0.5).
        diff_l1_weight: L1 weight on softmax(predicted) vs y_one_hot
            (paper auxiliary, default 0.1).
        beta_start, beta_end: Linear-beta schedule endpoints.
        img_size: Image spatial size.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 num_classes: int = None,
                 ema_decay: float = 0.999,
                 consistency_weight: float = 1.0,
                 rampup_epochs: int = 40,
                 diff_loss_weight: float = 1.0,
                 diff_warmup_epochs: int = 5,
                 diff_num_timesteps: int = 200,
                 diff_rect_t: int = 30,
                 diff_base_ch: int = 32,
                 diff_pred_threshold: float = 0.5,
                 diff_l1_weight: float = 0.1,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 img_size: int = 224, **kwargs):
        super().__init__(model, device, consistency_weight, rampup_epochs, img_size)
        if num_classes is None or int(num_classes) < 2:
            raise ValueError(
                "DiffRect requires `semi.params.num_classes` (>=2) to be "
                "set explicitly so the rectifier knows its channel count; "
                "no silent default is provided.")
        if not (0 < int(diff_rect_t) < int(diff_num_timesteps)):
            raise ValueError(
                f"diff_rect_t must be in (0, diff_num_timesteps); got "
                f"diff_rect_t={diff_rect_t}, T={diff_num_timesteps}.")
        if not (0.0 <= float(diff_pred_threshold) <= 1.0):
            raise ValueError(
                f"diff_pred_threshold must be in [0, 1], got "
                f"{diff_pred_threshold}.")
        self.num_classes = int(num_classes)
        self.ema_decay = float(ema_decay)
        self.diff_loss_weight = float(diff_loss_weight)
        self.diff_warmup_epochs = int(diff_warmup_epochs)
        self.diff_num_timesteps = int(diff_num_timesteps)
        self.diff_rect_t = int(diff_rect_t)
        self.diff_base_ch = int(diff_base_ch)
        self.diff_pred_threshold = float(diff_pred_threshold)
        self.diff_l1_weight = float(diff_l1_weight)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        self.teacher: nn.Module = None
        self.rectifier: LabelRectifier = None
        self.schedule: _ForwardSchedule = None

    def build(self) -> None:
        self.teacher = create_ema_model(self.model)
        self.teacher.to(self.device)
        self.rectifier = LabelRectifier(
            num_classes=self.num_classes,
            img_channels=3,
            base_ch=self.diff_base_ch,
        ).to(self.device)
        self.schedule = _ForwardSchedule(
            num_timesteps=self.diff_num_timesteps,
            beta_start=self.beta_start, beta_end=self.beta_end,
            device=self.device,
        )

    def extra_params(self):
        if self.rectifier is None:
            return []
        return list(self.rectifier.parameters())

    @staticmethod
    def _take_first(out):
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    # ---- single-pass rectification on unlabeled pseudo-label ---------------
    @torch.no_grad()
    def _rectify(self, teacher_soft: torch.Tensor,
                 images_u: torch.Tensor) -> torch.Tensor:
        """Return rectified soft label (B, C, H, W) in the simplex."""
        self.rectifier.eval()
        B = teacher_soft.shape[0]
        t = torch.full((B,), self.diff_rect_t,
                       device=self.device, dtype=torch.long)
        # Light corruption: treat p_teacher as y_t at t = rect_t (noiseless).
        sb = self.schedule.sqrt_bar[self.diff_rect_t]
        y_t = sb * teacher_soft
        logits0 = self.rectifier(y_t, images_u, t)
        return F.softmax(logits0, dim=1)

    # --------------------------------- train_step ---------------------------
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
        self.rectifier.train()

        images_l = labeled_batch['image'].to(self.device)
        labels = labeled_batch['label'].to(self.device)
        images_u = unlabeled_batch['image'].to(self.device)
        H, W = images_l.shape[-2:]

        # ---- Supervised segmentation loss (student) ----
        pred_l = self._take_first(self.model(images_l))
        sup_loss = criterion(pred_l, labels)

        # ---- Train rectifier on (labeled GT one-hot, image) ----
        # y_0 = one_hot(GT)  in (B, C, H, W)
        labels_clamped = labels.clamp(min=0, max=self.num_classes - 1).long()
        y0 = F.one_hot(labels_clamped, num_classes=self.num_classes)
        y0 = y0.permute(0, 3, 1, 2).float()
        B = y0.shape[0]
        t = torch.randint(0, self.diff_num_timesteps, (B,), device=self.device)
        eps = torch.randn_like(y0)
        y_t = self.schedule.q_sample(y0, t, eps)
        logits_x0 = self.rectifier(y_t, images_l, t)
        # CE on labels + L1 between softmax(prediction) and one-hot
        diff_ce = F.cross_entropy(logits_x0, labels_clamped)
        diff_l1 = (F.softmax(logits_x0, dim=1) - y0).abs().mean()
        diff_loss = diff_ce + self.diff_l1_weight * diff_l1

        # ---- Rectified pseudo-label on unlabeled (after warm-up) ----
        in_warmup = epoch < self.diff_warmup_epochs
        if in_warmup:
            unsup_loss = images_u.new_zeros(())
            kept_ratio = 0.0
        else:
            with torch.no_grad():
                t_logits = self._take_first(self.teacher(images_u))
                if t_logits.shape[-2:] != (H, W):
                    t_logits = F.interpolate(
                        t_logits, size=(H, W),
                        mode='bilinear', align_corners=False)
                t_soft = F.softmax(t_logits, dim=1)
                rect = self._rectify(t_soft, images_u)
                conf, hard = rect.max(dim=1)
                mask = conf.ge(self.diff_pred_threshold)
                target = hard.clone()
                target[~mask] = -1

            stud_pred = self._take_first(self.model(images_u))
            if stud_pred.shape[-2:] != target.shape[-2:]:
                stud_pred = F.interpolate(
                    stud_pred, size=target.shape[-2:],
                    mode='bilinear', align_corners=False)
            ce_pix = F.cross_entropy(
                stud_pred, target, ignore_index=-1, reduction='none')
            denom = mask.float().sum().clamp(min=1.0)
            unsup_loss = (ce_pix * mask.float()).sum() / denom
            kept_ratio = float(mask.float().mean().item())

        w = get_current_consistency_weight(
            epoch, self.consistency_weight, self.rampup_epochs)
        total_loss = sup_loss + w * unsup_loss + self.diff_loss_weight * diff_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.rectifier.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        return {
            "loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": float(unsup_loss.item()),
            "diff_loss": float(diff_loss.item()),
            "w": w,
            "kept_ratio": kept_ratio,
            "warmup": int(in_warmup),
        }

    def update(self, epoch: int) -> None:
        ema_update(self.teacher, self.model, self.ema_decay)

    def get_eval_model(self) -> nn.Module:
        return self.teacher
