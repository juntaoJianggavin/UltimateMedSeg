"""FDA: Fourier Domain Adaptation for Semantic Segmentation (CVPR 2020).

# Paper: https://arxiv.org/abs/2004.05498
# Reference: https://github.com/YanchaoYang/FDA

Algorithm summary (from the paper):
    FDA performs *image-level* style alignment by swapping the low-frequency
    component of the source amplitude spectrum with that of a (randomly
    paired) target image. Concretely, for a source image x_s and target x_t:

        F_s = FFT(x_s);   F_t = FFT(x_t)
        |F_s|, ph_s = magnitude/phase of F_s;  |F_t|, _ = magnitude of F_t
        |F'_s|  =  M_beta * |F_t|  +  (1 - M_beta) * |F_s|
        x'_s   =  iFFT( |F'_s|  *  exp(i * ph_s) )

    where M_beta is a centred low-frequency rectangular mask of bandwidth
    beta in [0, 1] (paper default beta ≈ 0.01). The training objective is

        L_FDA = L_ce(model(x'_s), y_s)               (Eq. 4)
              + lambda_ent * L_ent(model(x_t))       (Eq. 6, Charbonnier
                                                       penalised entropy)

    L_ent is the Charbonnier-penalised entropy of the target prediction:
        L_ent = mean( I_x ** (2 * eta) ) ** (1 / (2 * eta))
    with I_x the normalised per-pixel entropy and eta = 2.0 in the paper.

NOTE on integration:
    The training loop does a single forward of the model on source/target
    images before calling this loss. To stay faithful to the paper, you
    should pre-stylise source images at the *dataset / dataloader* level by
    calling :py:meth:`FDALoss.fda_amplitude_swap` on each (source, target)
    pair before they enter the model. The supervised CE that the trainer
    then computes on ``source_pred`` is automatically the FDA L_ce term.
    The Charbonnier-penalised target entropy (Eq. 6) is computed inside
    this loss from the trainer-provided ``target_pred``.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("fda")
class FDALoss(nn.Module):
    """Fourier Domain Adaptation training loss.

    Yang & Soatto, CVPR 2020.
    Reference (not copied): https://github.com/YanchaoYang/FDA

    Args:
        entropy_weight: weight on the Charbonnier-penalised target entropy
            (Eq. 6 of the paper, ``lambda_ent``).
        eta: Charbonnier exponent. Paper uses 2.0.
        beta: low-frequency bandwidth of the amplitude-swap mask, in
            [0, 1] (paper default 0.01).
        num_classes: number of segmentation classes (used to *normalise*
            the per-pixel entropy to [0, 1] before the Charbonnier penalty).
    """

    def __init__(
        self,
        entropy_weight: float = 0.005,
        eta: float = 2.0,
        beta: float = 0.01,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < beta <= 1.0):
            raise ValueError(f"FDA beta must be in (0, 1], got {beta}")
        if eta <= 0:
            raise ValueError(f"FDA eta must be positive, got {eta}")
        self.entropy_weight = entropy_weight
        self.eta = eta
        self.beta = beta
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Amplitude-swap helper (paper Sec. 3.1)
    # ------------------------------------------------------------------
    @staticmethod
    def fda_amplitude_swap(
        src_img: torch.Tensor,
        tgt_img: torch.Tensor,
        beta: float = 0.01,
    ) -> torch.Tensor:
        """Replace the low-frequency amplitude of ``src_img`` with ``tgt_img``'s.

        Both tensors must be 4-D (B, C, H, W) and share spatial size.
        Returns the *stylised* source image with the same shape.

        Implementation follows the paper formulas (FFT_swap):
            mask M_beta is a centred rectangle of side 2 * b + 1, where
                b = floor(beta * min(H, W) / 2).
        """
        if src_img.dim() != 4 or tgt_img.dim() != 4:
            raise ValueError(
                f"fda_amplitude_swap expects 4-D tensors, got "
                f"src={tuple(src_img.shape)} tgt={tuple(tgt_img.shape)}"
            )
        if src_img.shape[-2:] != tgt_img.shape[-2:]:
            raise ValueError(
                f"FDA amplitude swap requires matching spatial size; got "
                f"src H,W={tuple(src_img.shape[-2:])} vs "
                f"tgt H,W={tuple(tgt_img.shape[-2:])}"
            )

        # 2-D FFT (per channel) and shift the zero-frequency to the centre.
        F_src = torch.fft.fft2(src_img, dim=(-2, -1))
        F_tgt = torch.fft.fft2(tgt_img, dim=(-2, -1))
        F_src = torch.fft.fftshift(F_src, dim=(-2, -1))
        F_tgt = torch.fft.fftshift(F_tgt, dim=(-2, -1))

        amp_src = F_src.abs()
        amp_tgt = F_tgt.abs()
        # Replace centre block of source amplitude with target's.
        _, _, H, W = src_img.shape
        b = int(math.floor(beta * min(H, W) / 2.0))
        if b > 0:
            cH, cW = H // 2, W // 2
            h1, h2 = cH - b, cH + b
            w1, w2 = cW - b, cW + b
            amp_src[..., h1:h2, w1:w2] = amp_tgt[..., h1:h2, w1:w2]

        # Recombine with the original source phase and inverse FFT.
        phase_src = torch.angle(F_src)
        F_new = amp_src * torch.exp(1j * phase_src)
        F_new = torch.fft.ifftshift(F_new, dim=(-2, -1))
        x_new = torch.fft.ifft2(F_new, dim=(-2, -1)).real
        return x_new

    # ------------------------------------------------------------------
    # Charbonnier-penalised entropy (paper Eq. 6)
    # ------------------------------------------------------------------
    def _charbonnier_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """L_ent = (mean( I_x^{2*eta} ))^{1/(2*eta)} with I_x normalised."""
        n, c, _, _ = predictions.shape
        prob = F.softmax(predictions, dim=1)
        log_prob = torch.log2(prob + 1e-30)
        norm = math.log2(c) if c > 1 else 1.0
        I_x = -(prob * log_prob).sum(dim=1) / norm   # (B, H, W) in [0, 1]
        # Paper Eq. 6: Charbonnier-style penalty with exponent 2*eta.
        exp = 2.0 * self.eta
        return (I_x.pow(exp) + 1e-12).mean().pow(1.0 / exp)

    # ------------------------------------------------------------------
    # Spectrum-style consistency (auxiliary, training-loop friendly)
    # ------------------------------------------------------------------
    @staticmethod
    def _low_freq_amp_distance(
        src_img: torch.Tensor,
        tgt_img: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """L1 distance between source and target low-frequency amplitudes.

        Drops to ~0 when the dataloader is feeding pre-stylised source
        images; otherwise gives a small monitoring signal of how mis-aligned
        the two spectra are. Used here as a frequency-style monitor, not
        a backprop signal through the model (image-level loss).
        """
        F_src = torch.fft.fftshift(
            torch.fft.fft2(src_img, dim=(-2, -1)), dim=(-2, -1)
        )
        F_tgt = torch.fft.fftshift(
            torch.fft.fft2(tgt_img, dim=(-2, -1)), dim=(-2, -1)
        )
        amp_src, amp_tgt = F_src.abs(), F_tgt.abs()
        _, _, H, W = src_img.shape
        b = int(math.floor(beta * min(H, W) / 2.0))
        if b <= 0:
            return amp_src.new_zeros(())
        cH, cW = H // 2, W // 2
        block_src = amp_src[..., cH - b:cH + b, cW - b:cW + b]
        block_tgt = amp_tgt[..., cH - b:cH + b, cW - b:cW + b]
        return (block_src - block_tgt).abs().mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        target_pred: torch.Tensor,
        source_images: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        labeled_loss: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """
        Args:
            target_pred: target-domain logits, (B, C, H, W). Required.
            source_images / target_images: raw image tensors (B, C, H, W),
                optional. If both are given, a *monitoring-only* low-freq
                amplitude distance is recorded into ``self.last_amp_dist``
                so the user can verify that their pre-stylisation pipeline
                is doing what they expect.
            labeled_loss: supervised CE on source (computed by the trainer);
                added through unchanged.
        """
        if target_pred is None:
            raise ValueError("FDALoss requires target_pred.")

        ent_loss = self._charbonnier_entropy(target_pred)
        total = self.entropy_weight * ent_loss

        # Optional spectrum monitor (no gradient through the model).
        if source_images is not None and target_images is not None:
            with torch.no_grad():
                if source_images.shape[-2:] == target_images.shape[-2:]:
                    self.last_amp_dist = float(
                        self._low_freq_amp_distance(
                            source_images, target_images, self.beta
                        ).item()
                    )

        if labeled_loss is not None:
            total = total + labeled_loss
        return total
