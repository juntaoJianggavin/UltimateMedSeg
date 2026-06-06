# Reference: https://github.com/yzd-v/MGD
# Paper: https://arxiv.org/abs/2205.01529
"""MGD: Masked Generative Distillation (ECCV 2022).

Per Yang et al. Sec. 3, a random pixel mask zeros out a fraction
``lambda_mgd`` of the student feature, then a small 3x3-Conv -> ReLU
-> 3x3-Conv generator reconstructs the (full) teacher feature. The
loss is sum-MSE between the reconstruction and the teacher feature,
normalised by batch size and scaled by ``alpha_mgd``.

Paper defaults: ``lambda_mgd = 0.5`` (segmentation, Table 6) and the
detection presets use 0.65; we default to the paper's segmentation
value here.
"""

import torch
import torch.nn as nn
from medseg.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("mgd")
class MGDLoss(nn.Module):
    """Masked Generative Distillation (ECCV 2022).

    Reference: yzd-v/MGD. Implemented from the paper formulas.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        alpha_mgd: float = 0.00002,
        lambda_mgd: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        if not (0.0 < lambda_mgd < 1.0):
            raise ValueError(
                f"MGD lambda_mgd (mask ratio) must be in (0, 1), got {lambda_mgd}."
            )
        self.alpha_mgd = float(alpha_mgd)
        self.lambda_mgd = float(lambda_mgd)

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels,
                                   kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        # Paper Sec. 3.2: 3x3 conv -> ReLU -> 3x3 conv generator.
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
        )

    def get_dis_loss(self, preds_S, preds_T):
        """Strictly reproduce official get_dis_loss."""
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W), device=device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0.0, 1.0)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss

    def forward(self, preds_S, preds_T) -> torch.Tensor:
        """
        Args:
            preds_S: (B, C_s, H, W) student feature, or a list/dict of them.
            preds_T: (B, C_t, H, W) teacher feature, or a list/dict of them.
        """
        # Accept list/dict inputs from the feature-hook plumbing.
        if isinstance(preds_S, (list, tuple)):
            preds_S = preds_S[-1] if preds_S else None
        if isinstance(preds_T, (list, tuple)):
            preds_T = preds_T[-1] if preds_T else None
        if preds_S is None or preds_T is None:
            raise RuntimeError(
                "MGDLoss received None for student or teacher features. "
                "Check that feature_layers in the config matches a real "
                "module name in both models."
            )

        # Spatial alignment so the generator sees matched-shape pairs.
        if preds_S.shape[-2:] != preds_T.shape[-2:]:
            preds_T = nn.functional.interpolate(
                preds_T, size=preds_S.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        if self.align is not None:
            preds_S = self.align(preds_S)

        loss = self.get_dis_loss(preds_S, preds_T.detach()) * self.alpha_mgd
        return loss
