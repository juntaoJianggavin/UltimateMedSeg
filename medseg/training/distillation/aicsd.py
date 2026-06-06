# AICSD: Adaptive Inter-Class Similarity Distillation (TNNLS 2024)
# Reference: https://github.com/AmirMansurian/AICSD
# Paper: https://arxiv.org/abs/2308.04243
# Implemented from paper formulas; not a copy of the official repo.
"""AICSD: Adaptive inter-class similarity distillation for segmentation.

Mansurian et al. compute two complementary inter-class signals from
the per-class spatial probability maps:

  - ICS (Inter-Class Similarity): cosine similarity between every
    pair of class probability vectors, treating each class' spatial
    probability map as a long vector.
  - ICC (Inter-Class Correlation): Pearson correlation across the
    spatial axis between every pair of class probability vectors.

For every sample this gives a (C, C) similarity / correlation matrix
for teacher and for student. Each row of the matrix is softmaxed
(with a temperature) and matched with KL divergence; rows belonging
to more confidently-predicted teacher classes are up-weighted, so
unreliable / rarely-fired classes contribute less to the loss. This
"adaptive" reweighting is the AI part of AICSD.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("aicsd")
class AICSDLoss(nn.Module):
    """Adaptive inter-class similarity / correlation KD for dense pred.

    Args:
        temperature: softmax temperature applied to each row of the
            CxC similarity matrix before KL.
        lambda_ics: weight on the ICS (cosine similarity) term.
        lambda_icc: weight on the ICC (Pearson correlation) term.
        eps: numerical floor for normalisation.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        lambda_ics: float = 1.0,
        lambda_icc: float = 1.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"AICSD temperature must be > 0, got {temperature}."
            )
        if lambda_ics < 0 or lambda_icc < 0:
            raise ValueError(
                f"AICSD lambda weights must be >= 0, got "
                f"lambda_ics={lambda_ics}, lambda_icc={lambda_icc}."
            )
        self.temperature = float(temperature)
        self.lambda_ics = float(lambda_ics)
        self.lambda_icc = float(lambda_icc)
        self.eps = float(eps)

    @staticmethod
    def _flatten_prob(z: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) logits -> (B, C, H*W) class-axis softmax probs."""
        p = F.softmax(z, dim=1)
        B, C, H, W = p.shape
        return p.reshape(B, C, H * W)

    def _ics(self, p: torch.Tensor) -> torch.Tensor:
        # Cosine similarity matrix per sample: (B, C, C).
        p_n = F.normalize(p, p=2, dim=2, eps=self.eps)
        return torch.bmm(p_n, p_n.transpose(1, 2))

    def _icc(self, p: torch.Tensor) -> torch.Tensor:
        # Pearson correlation across the spatial axis.
        mean = p.mean(dim=2, keepdim=True)
        centered = p - mean
        denom = centered.pow(2).sum(dim=2, keepdim=True).clamp(min=self.eps).sqrt()
        normed = centered / denom
        return torch.bmm(normed, normed.transpose(1, 2))

    def _row_kl(
        self,
        mat_s: torch.Tensor,
        mat_t: torch.Tensor,
        row_w: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row temperature-softmax KL, weighted by ``row_w``.

        mat_s / mat_t: (B, C, C). row_w: (B, C) per-row weight from
        teacher confidence; rows are KL-divergences class-by-class.
        """
        B, C, _ = mat_s.shape
        T = self.temperature
        s = mat_s.reshape(B * C, C)
        t = mat_t.reshape(B * C, C)
        log_p_s = F.log_softmax(s / T, dim=1)
        p_t = F.softmax(t / T, dim=1)
        # Per-row KL (sum over class axis), shape (B*C,).
        kl = (p_t * (p_t.clamp(min=self.eps).log() - log_p_s)).sum(dim=1)
        kl = kl.reshape(B, C) * row_w
        # Normalise by the total weight so the loss does not collapse
        # to zero when many rows have low confidence.
        denom = row_w.sum().clamp(min=self.eps)
        return kl.sum() / denom * (T * T)

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            student_output: (B, C, H, W) student logits.
            teacher_output: (B, C, H, W) teacher logits (no grad).
        """
        if student_output.dim() != 4 or teacher_output.dim() != 4:
            raise ValueError(
                "AICSD expects 4D logits (B, C, H, W); got "
                f"student={tuple(student_output.shape)} "
                f"teacher={tuple(teacher_output.shape)}."
            )
        if student_output.shape[-2:] != teacher_output.shape[-2:]:
            teacher_output = F.interpolate(
                teacher_output, size=student_output.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        teacher_output = teacher_output.detach()

        p_s = self._flatten_prob(student_output)
        p_t = self._flatten_prob(teacher_output)

        # Adaptive per-class row weight = teacher's mean per-class probability.
        # A class the teacher rarely fires has a near-zero weight and is
        # effectively dropped from the loss.
        with torch.no_grad():
            row_w = p_t.mean(dim=2)  # (B, C)

        ics_s = self._ics(p_s)
        ics_t = self._ics(p_t)
        icc_s = self._icc(p_s)
        icc_t = self._icc(p_t)

        loss = (
            self.lambda_ics * self._row_kl(ics_s, ics_t, row_w)
            + self.lambda_icc * self._row_kl(icc_s, icc_t, row_w)
        )
        return loss
