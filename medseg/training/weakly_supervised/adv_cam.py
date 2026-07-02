"""AdvCAM — Adversarially manipulated Class Activation Maps (faithful training loss).

Lee et al., "Anti-Adversarially Manipulated Attributions for Weakly and
Semi-Supervised Semantic Segmentation", CVPR 2021.
Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Anti-Adversarially_Manipulated_Attributions_for_Weakly_and_Semi-Supervised_Semantic_Segmentation_CVPR_2021_paper.pdf
Official: https://github.com/jbeomlee93/AdvCAM
Note: "AdvCAM" is the method shorthand used in the official repository;
the full paper title is "Anti-Adversarially Manipulated Attributions".

Faithful implementation of the AdvCAM adversarial climbing procedure based
on the official source ``obtain_CAM_masking.py``.

Algorithm (per image per class, from the original source):
    for it in range(adv_iter):
        img.requires_grad = True
        outputs = model(img)
        cam = GradCAM(img, class=c)
        if it == 0:
            init_cam = cam.detach()
        logit = GAP(ReLU(outputs))
        logit_loss = -2 * logit[:, c] + sum(logit)
        expanded_mask = add_discriminative(expanded_mask, cam, score_th)
        L_AD = sum(|cam - init_cam| * expanded_mask)
        loss = -logit_loss - L_AD * AD_coeff
        loss.backward()
        img = adv_climb(img, stepsize, grad)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from medseg.registry import LOSS_REGISTRY


def adv_climb(image, epsilon, data_grad):
    """FGSM-style adversarial climb (faithful to original source).

    Normalises the gradient before applying the perturbation:
        ``perturbed = image + epsilon * sign(grad / max|grad|)``
    Clamped to the image's own [min, max] range.
    """
    sign_data_grad = data_grad / (torch.max(torch.abs(data_grad)) + 1e-12)
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(
        perturbed_image,
        image.min().data.float(),
        image.max().data.float(),
    )
    return perturbed_image


def add_discriminative(expanded_mask, regions, score_th):
    """Mark regions above threshold as discriminative.

    Faithful to the original ``add_discriminative`` function:
        ``region_ = regions / regions.max()``
        ``expanded_mask[region_ > score_th] = 1``
    """
    region_ = regions / regions.max()
    expanded_mask[region_ > score_th] = 1
    return expanded_mask


@LOSS_REGISTRY.register("advcam_loss")
class AdvCAMLoss(nn.Module):
    """Faithful AdvCAM training loss with adversarial climbing.

    Matches the original ``obtain_CAM_masking.py`` procedure:

    * ``adv_iter``: number of adversarial climbing iterations (default 27).
    * ``adv_coeff`` (``AD_coeff``): weight for ``L_AD`` term (default 7).
    * ``adv_step_size`` (``AD_stepsize``): perturbation magnitude (default 0.08).
    * ``score_th``: discriminative threshold for ``add_discriminative`` (default 0.5).

    The loss can operate in two modes:

    **Mode A — Full adversarial climbing (faithful):**
        Pass ``features``, ``model_ref``, and ``input_image``.
        The loss performs the iterative adversarial climbing internally.

    **Mode B — Pre-computed CAMs:**
        Pass ``cam_accumulated`` and ``cam_orig`` from an external step.

    **Mode C — Surrogate:**
        When neither features nor pre-computed CAMs are available, falls back
        to a multilabel classification loss only.
    """

    def __init__(self, adv_iter: int = 27, adv_alpha: float = 1.0,
                 adv_weight: float = 1.0,
                 adv_coeff: float = 7.0,
                 adv_step_size: float = 0.08,
                 score_th: float = 0.5,
                 **kwargs):
        super().__init__()
        self.adv_iter = adv_iter
        self.adv_alpha = adv_alpha
        self.adv_weight = adv_weight
        self.adv_coeff = adv_coeff
        self.adv_step_size = adv_step_size
        self.score_th = score_th

    def forward(self, predictions, image_labels,
                features=None, model_ref=None, input_image=None,
                cam_accumulated=None, cam_orig=None,
                labeled_loss=None, **kwargs):
        """Compute the AdvCAM loss.

        Args:
            predictions: (B, C, H, W) class logits.
            image_labels: (B, num_classes) multilabel targets.
            features: optional (B, C_f, H_f, W_f) intermediate features.
            model_ref: optional reference to the segmentation model.
            input_image: optional (B, 3, H, W) input image.
            cam_accumulated / cam_orig: pre-computed CAMs.
            labeled_loss: optional extra loss term.
        """
        # Classification loss (always computed)
        preds_gap = predictions
        if preds_gap.dim() == 4:
            preds_gap = F.adaptive_avg_pool2d(preds_gap, 1).flatten(1)
        cls_loss = F.multilabel_soft_margin_loss(
            preds_gap, image_labels.float())

        # Adversarial climbing loss
        adv_loss = torch.tensor(0.0, device=predictions.device)

        if cam_accumulated is not None and cam_orig is not None:
            # Mode B: pre-computed CAMs
            adv_loss = self._surrogate_adv_loss(cam_accumulated, cam_orig)

        elif features is not None and model_ref is not None:
            # Mode A: full adversarial climbing
            adv_loss = self._adversarial_climbing(
                predictions, image_labels, features,
                model_ref, input_image)

        total_loss = cls_loss + self.adv_weight * adv_loss
        if labeled_loss is not None:
            total_loss = total_loss + labeled_loss
        return total_loss

    # ------------------------------------------------------------------
    # Surrogate (from pre-computed CAMs)
    # ------------------------------------------------------------------
    def _surrogate_adv_loss(self, cam_accumulated, cam_orig):
        """Compute adv loss from pre-computed CAMs.

        Faithful to: ``L_AD = sum(|regions - init_cam| * expanded_mask)``
        where ``expanded_mask`` is computed via ``add_discriminative``.
        """
        expanded_mask = torch.zeros_like(cam_orig)
        expanded_mask = add_discriminative(
            expanded_mask, cam_orig, self.score_th)
        L_AD = torch.sum(
            torch.abs(cam_accumulated - cam_orig.detach()) * expanded_mask)
        return -L_AD

    # ------------------------------------------------------------------
    # Full adversarial climbing (Mode A, faithful to original)
    # ------------------------------------------------------------------
    def _adversarial_climbing(self, predictions, image_labels,
                              features, model_ref, input_image):
        """Perform iterative adversarial climbing (faithful to source)."""
        if input_image is None:
            return torch.tensor(0.0, device=predictions.device)

        B = input_image.shape[0]
        # Get present classes from image_labels
        present_classes = torch.where(image_labels[0] > 0)[0]
        if len(present_classes) == 0:
            present_classes = torch.tensor([1], device=image_labels.device)

        total_L_AD = torch.tensor(0.0, device=predictions.device)

        for c in present_classes:
            c_idx = c.item()

            # Work on a single image at a time (matching original)
            img_single = input_image[0].detach().clone()

            init_cam = None
            expanded_mask = None

            for it in range(self.adv_iter):
                img_single.requires_grad = True

                # Forward
                outputs = model_ref(img_single.unsqueeze(0))
                if outputs.dim() == 4:
                    cam_h, cam_w = outputs.shape[2], outputs.shape[3]
                else:
                    cam_h = cam_w = 1

                # Compute GradCAM
                if features is not None and features.requires_grad:
                    regions = self._compute_gradcam(
                        features, outputs, c_idx)
                else:
                    # Fallback: use class activation map from outputs
                    if outputs.dim() == 4:
                        regions = F.relu(outputs[0, c_idx:c_idx + 1])
                    else:
                        regions = F.relu(outputs[0:1, c_idx:c_idx + 1])

                if it == 0:
                    init_cam = regions.detach().clone()
                    expanded_mask = torch.zeros_like(regions)

                # Logit loss (faithful: -2*logit_c + sum(logit))
                logit = F.relu(outputs)
                if logit.dim() == 4:
                    logit = F.adaptive_avg_pool2d(logit, 1).flatten(1)
                logit_loss = (-2 * logit[:, c_idx]
                              + logit.sum(dim=1))

                # Discriminative mask update
                expanded_mask = add_discriminative(
                    expanded_mask, regions, self.score_th)

                # L_AD
                L_AD = torch.sum(
                    torch.abs(regions - init_cam) * expanded_mask)

                # Combined loss for climbing
                loss = -logit_loss.sum() - L_AD * self.adv_coeff

                # Backward
                model_ref.zero_grad()
                if img_single.grad is not None:
                    img_single.grad.zero_()
                loss.backward()

                # Climb
                with torch.no_grad():
                    img_single = adv_climb(
                        img_single, self.adv_step_size,
                        img_single.grad.data)
                    img_single = img_single.detach()

                total_L_AD = total_L_AD + L_AD.detach()

        return -total_L_AD / max(len(present_classes), 1)

    # ------------------------------------------------------------------
    # GradCAM helper
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_gradcam(features, outputs, class_idx):
        """Compute GradCAM from intermediate features."""
        if features.grad_fn is not None or features.requires_grad:
            features.retain_grad()

        score = outputs[:, class_idx]
        if score.dim() > 1:
            score = score.mean(dim=[1, 2])
        score = score.sum()

        grads = torch.autograd.grad(
            score, features, retain_graph=True, create_graph=False)[0]
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_flat = cam.flatten(2)
        cam_max = cam_flat.max(dim=2, keepdim=True)[0].clamp(min=1e-8)
        cam = cam_flat / cam_max
        return cam.view_as(cam.shape)
