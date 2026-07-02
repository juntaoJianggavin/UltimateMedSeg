"""Class Activation Map generator using Grad-CAM.

Selvaraju et al., ICCV 2017.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List


class CAMGenerator:
    """Generate Class Activation Maps from image-level labels.

    Uses Grad-CAM (Selvaraju et al., ICCV 2017) gradient-weighted feature
    maps to produce coarse class-localisation heatmaps from a classifier.

    Also provides feature extraction from the target encoder layer for
    weak supervision methods that need backbone features (ReCAM, PSDPM, etc.).

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

    def extract_features(self, images: torch.Tensor, upsample: bool = True) -> torch.Tensor:
        """Run a forward pass and return backbone features from the target layer.

        Args:
            images: Input batch (B, C_in, H, W).
            upsample: If True (default), bilinearly interpolate features to
                match input spatial resolution. If False, return at native
                backbone resolution (e.g. H/32, W/32 for ResNet).

        Returns:
            Features (B, D, H_out, W_out). When upsample=True, H_out/W_out
            match the input image size; otherwise they are the native
            encoder spatial dimensions.
        """
        self.model.zero_grad()
        _ = self.model(images)  # forward pass triggers hook
        features = self.features
        if features.dim() == 2:
            # Flatten case — skip
            return features
        if upsample:
            H, W = images.shape[2], images.shape[3]
            if features.shape[2] != H or features.shape[3] != W:
                features = F.interpolate(
                    features, size=(H, W), mode="bilinear", align_corners=False
                )
        return features

    def extract_features_and_cam(self, images: torch.Tensor,
                                 image_labels: torch.Tensor,
                                 upsample: bool = False) -> tuple:
        """Extract backbone features and class activation maps.

        For segmentation models the decoder head output channels don't
        match the encoder feature channels, so the classic classifier-weight
        CAM formula can't be applied directly.  Instead we use the model's
        own per-class **output logits** (after GAP to native resolution) as
        the CAM — this is the approach used by many weak-supervision
        frameworks when the model is a segmenter rather than a classifier.

        A **single forward pass** produces both features and logits.
        The logits are interpolated to the feature resolution, masked by
        image_labels, and min-max normalised.

        Args:
            images: Batch (B, C_in, H, W).
            image_labels: Multi-label binary tensor (B, C).
            upsample: If True, upsample both features and CAMs to input
                spatial resolution.  If False (default), native resolution.

        Returns:
            (features, cam_map) with shapes (B,D,h,w) and (B,C,h,w).
        """
        B, C = image_labels.shape

        # 1) Single forward pass.
        self.model.zero_grad()
        output = self.model(images)
        features = self.features  # (B, D, h_feat, w_feat)

        H_feat, W_feat = features.shape[2], features.shape[3]

        # 2) Interpolate output logits to feature resolution.
        #    output shape: (B, C_out, H_out, W_out) where C_out = num_classes.
        if output.shape[-2:] != (H_feat, W_feat):
            logits = F.interpolate(
                output, size=(H_feat, W_feat),
                mode="bilinear", align_corners=False,
            )
        else:
            logits = output  # (B, C, h, w)

        # 3) Use logits as CAM (positive = class presence).
        cam_map = F.relu(logits)  # (B, C, h, w)

        # 4) Mask out absent classes.
        mask = image_labels.float().view(B, C, 1, 1)
        cam_map = cam_map * mask

        # 5) Per-(b,c) min-max normalisation to [0, 1].
        flat = cam_map.view(B, C, -1)
        m = flat.amin(dim=2, keepdim=True)
        M = flat.amax(dim=2, keepdim=True)
        cam_map = ((flat - m) / (M - m + 1e-8)).view(B, C, H_feat, W_feat)

        # 6) Optionally upsample.
        if upsample:
            H, W = images.shape[2], images.shape[3]
            if features.shape[2] != H or features.shape[3] != W:
                features = F.interpolate(
                    features, size=(H, W), mode="bilinear", align_corners=False
                )
            cam_map = F.interpolate(
                cam_map, size=(H, W), mode="bilinear", align_corners=False
            )

        return features, cam_map

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

        # Backward for target class (reduce to scalar for GradCAM)
        output[0, class_idx].mean().backward()

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

    def generate_multiclass_cam(self, images: torch.Tensor,
                                image_labels: torch.Tensor) -> torch.Tensor:
        """Generate per-class CAMs for all present classes via Grad-CAM.

        For each image, Grad-CAM is computed for every class whose label is
        present (image_labels[b, c] == 1). The result is a dense tensor of
        shape (B, C, H, W) where C equals the number of classes in
        ``image_labels``.

        Args:
            images: Batch (B, C_in, H, W).
            image_labels: Multi-label binary tensor (B, C).

        Returns:
            CAM logits (B, C, H, W) at input spatial resolution.  Values are
            in [0, 1] after per-(b, c) min-max normalisation.
        """
        B, C = image_labels.shape
        H, W = images.shape[2], images.shape[3]
        cam_map = images.new_zeros(B, C, H, W)

        for b in range(B):
            present = torch.where(image_labels[b] > 0)[0]
            if present.numel() == 0:
                continue
            for c in present.tolist():
                cam = self._grad_cam_single(images[b:b + 1], c)
                cam_map[b, c] = cam

        return cam_map

    def _grad_cam_single(self, image: torch.Tensor,
                         class_idx: int) -> torch.Tensor:
        """Compute Grad-CAM for a single image and class.

        Args:
            image: Single-image batch (1, C_in, H, W).
            class_idx: Target class index (into the model's output channels).

        Returns:
            Normalised CAM (H, W).
        """
        self.model.zero_grad()
        output = self.model(image)

        # Clamp class_idx to output channels
        c_idx = min(class_idx, output.shape[1] - 1)
        output[0, c_idx].backward()

        weights = self.gradients.mean(dim=[2, 3])  # (1, D)

        # Weighted sum of feature maps
        feat = self.features  # (1, D, h, w)
        cam = torch.zeros(feat.shape[2:], device=image.device)
        for i in range(feat.shape[1]):
            cam += weights[0, i] * feat[0, i]

        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Upsample to input spatial size
        H, W = image.shape[2], image.shape[3]
        if cam.shape[0] != H or cam.shape[1] != W:
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        return cam
