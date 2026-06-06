"""Class Activation Map generator using Grad-CAM.

Selvaraju et al., ICCV 2017.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class CAMGenerator:
    """Generate Class Activation Maps from image-level labels.

    Uses Grad-CAM (Selvaraju et al., ICCV 2017) gradient-weighted feature
    maps to produce coarse class-localisation heatmaps from a classifier.

    ``target_layer``:
        Name of the encoder layer to hook for features+gradients. If left
        as ``None`` the layer is inferred from the encoder class name:
            ResNet / TimmEncoder / GenericTimmEncoder → "encoder.layer4"
            ConvNeXt / ConvNeXtEncoder               → "encoder.stages.3"
            BasicEncoder                              → "encoder.encoder3"
        Any other encoder requires an explicit ``target_layer``; the
        constructor raises ``ValueError`` rather than silently falling back
        to the last module (which is the project's no-mock-fallback rule).
    """

    # encoder-class-name → target-layer mapping, used when target_layer=None.
    _LAYER_RULES = {
        "ResNet": "encoder.layer4",
        "TimmEncoder": "encoder.layer4",
        "GenericTimmEncoder": "encoder.layer4",
        "ConvNeXt": "encoder.stages.3",
        "ConvNeXtEncoder": "encoder.stages.3",
        "BasicEncoder": "encoder.encoder3",
    }

    def __init__(self, model, target_layer: Optional[str] = None):
        """
        Args:
            model: Segmentation/classification model. The encoder must be
                exposed as ``model.encoder``.
            target_layer: Dotted module path. ``None`` triggers inference
                from ``type(model.encoder).__name__``; failure is a hard
                error (no fallback).
        """
        self.model = model
        self.target_layer = target_layer or self._infer_target_layer(model)
        self.features = None
        self.gradients = None

        self._register_hook(self.target_layer)

    @classmethod
    def _infer_target_layer(cls, model) -> str:
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            raise ValueError(
                "CAMGenerator could not find ``model.encoder``; pass an "
                "explicit ``target_layer`` argument."
            )
        cname = type(encoder).__name__
        if cname in cls._LAYER_RULES:
            return cls._LAYER_RULES[cname]
        raise ValueError(
            f"CAMGenerator cannot infer target_layer for encoder of type "
            f"{cname!r}. Pass an explicit ``target_layer`` (e.g. "
            f"'encoder.layer4' for ResNet, 'encoder.stages.3' for ConvNeXt). "
            f"Known rules: {list(cls._LAYER_RULES)}"
        )

    def _register_hook(self, layer_name):
        """Register forward and backward hooks. Hard-errors if name absent."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                def forward_hook(module, input, output):
                    self.features = output

                def backward_hook(module, grad_input, grad_output):
                    self.gradients = grad_output[0]

                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                return
        raise ValueError(
            f"CAMGenerator target_layer {layer_name!r} not found in model. "
            f"Inspect model.named_modules() to pick a valid name."
        )

    def generate_cam(self, image, class_idx):
        """Generate CAM for a specific class.

        Args:
            image: Input image (1, C, H, W)
            class_idx: Target class index

        Returns:
            CAM heatmap (H, W)
        """
        self.model.zero_grad()

        # Forward pass
        output = self.model(image)

        # Backward for target class
        output[0, class_idx].backward()

        # Get weights from gradients
        weights = self.gradients.mean(dim=[2, 3])  # Global average pooling

        # Generate CAM
        cam = torch.zeros(self.features.shape[2:], device=image.device)
        for i, w in enumerate(weights[0]):
            cam += w * self.features[0, i]

        # Normalize
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def generate_batch_cam(self, images, class_indices):
        """Generate CAMs for a batch.

        Args:
            images: Batch of images (B, C, H, W)
            class_indices: List of class indices for each image

        Returns:
            Batch of CAMs (B, H, W)
        """
        cams = []
        for i, (img, cls_idx) in enumerate(zip(images, class_indices)):
            cam = self.generate_cam(img.unsqueeze(0), cls_idx)
            cams.append(cam)

        return torch.stack(cams)
