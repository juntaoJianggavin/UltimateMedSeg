"""Model builder: assembles encoder + bottleneck + decoder + skip into a full segmentation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from typing import Dict, Any, Optional, List

from .registry import (
    ENCODER_REGISTRY,
    DECODER_REGISTRY,
    SKIP_REGISTRY,
    BOTTLENECK_REGISTRY,
    LOSS_REGISTRY,
)

# Import subpackages to trigger component registration
from .models import encoders  # noqa: F401
from .models import decoders  # noqa: F401
from .models import bottlenecks  # noqa: F401
from .models import skip_connections  # noqa: F401


class IncompatibleEncoderError(ValueError):
    """Raised when a decoder requires a specific encoder that is not provided."""
    pass


class FeatureAdapter(nn.Module):
    """Adapts encoder feature count to match decoder requirements.

    When encoder provides fewer features than decoder needs, generates
    additional downsampled features (deeper stages) via stride-2 convolutions.
    When encoder provides more, takes the deepest (last) features.
    """

    def __init__(self, encoder_channels: List[int], target_stages: int):
        super().__init__()
        n_have = len(encoder_channels)
        self.n_extra = max(0, target_stages - n_have)
        self.target_stages = target_stages

        # Create downsampling convs for extra features (deeper stages)
        self.down_convs = nn.ModuleList()
        if self.n_extra > 0 and n_have > 0:
            last_ch = encoder_channels[-1]
            for i in range(self.n_extra):
                in_ch = last_ch if i == 0 else self.down_convs[-1].out_channels
                out_ch = in_ch * 2
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
                self.down_convs.append(conv)

        # For the "keep fewer" case
        if target_stages < n_have:
            self._keep_start = n_have - target_stages
        else:
            self._keep_start = 0

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.n_extra > 0:
            # Keep all originals, append downsampled extras at the end (deep)
            adapted = list(features)
            x = features[-1]  # deepest feature
            for conv in self.down_convs:
                x = F.relu(conv(x))
                adapted.append(x)
            return adapted
        elif self.target_stages < len(features):
            return list(features[self._keep_start:])
        else:
            return list(features)


class SegmentationModel(nn.Module):
    """Modular segmentation model assembled from registry components.

    Architecture: Input -> Encoder -> Bottleneck -> Decoder (with Skip connections) -> Head -> Output
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        bottleneck: nn.Module,
        head: nn.Module,
        skip_adapter: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.head = head
        self.skip_adapter = skip_adapter

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input image batch (B, C, H, W)
            mask: optional data-side prior mask. If provided, will be set on any
                mask-aware skip module (one that defines ``set_mask``/``clear_mask``)
                before the decoder runs, and cleared afterwards. Use this for
                prior-guided segmentation (atlas, GrabCut prior, previous
                slice's prediction, etc.). For EGE-UNet's internal deep-supervision
                mask flow (where masks come from per-stage seg heads, not from
                the data), use the EGEUNet special_arch directly — that flow is
                bespoke to that architecture and not exposed through this
                generic builder path.
        """
        # Encoder: returns multi-scale features [stage1, stage2, ..., stageN]
        features = self.encoder(x)

        # Bottleneck: processes the deepest feature
        bottleneck_feat = self.bottleneck(features[-1])

        # Inject data-side mask into any mask-aware skip module(s)
        mask_targets = self._collect_mask_targets() if mask is not None else []
        for skip_mod in mask_targets:
            skip_mod.set_mask(mask)

        try:
            # Adapt skip features if encoder/decoder stage count mismatch
            skip_feats = features[:-1]
            if self.skip_adapter is not None:
                skip_feats = self.skip_adapter(skip_feats)
            # Decoder: takes bottleneck output + skip features, returns decoded feature
            decoded = self.decoder(bottleneck_feat, skip_feats)
        finally:
            # Always clear after decoder runs to avoid state leaking across batches
            for skip_mod in mask_targets:
                skip_mod.clear_mask()

        # Segmentation head
        out = self.head(decoded)

        # Upsample to input size if needed
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out

    def _collect_mask_targets(self):
        """Walk the decoder subtree to find any skip module exposing
        ``set_mask``/``clear_mask`` (e.g. GABSkip). Returns a list of modules.
        """
        targets = []
        for sub in self.decoder.modules():
            if hasattr(sub, "set_mask") and callable(sub.set_mask) \
                    and hasattr(sub, "clear_mask") and callable(sub.clear_mask):
                targets.append(sub)
        return targets


class SegmentationHead(nn.Module):
    """Simple segmentation head: Conv 1x1 to num_classes."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Build a segmentation model from config dict.

    Args:
        cfg: Configuration dict with keys: encoder, decoder, skip_connection, bottleneck, num_classes.
             Or an ensemble config: ``{model: {type: ensemble, members: [...]}}``.

    Returns:
        SegmentationModel, special architecture model, or EnsembleModel.
    """
    model_cfg = cfg.get("model", cfg)

    # Ensemble of multiple sub-models (logit averaging)
    model_type = model_cfg.get("type", None)
    if model_type in ("ensemble", "logit_ensemble"):
        from medseg.inference.ensemble import build_ensemble_from_config
        return build_ensemble_from_config(model_cfg, build_model)

    # Check for special architectures first
    arch = model_cfg.get("architecture", None)
    if arch is not None:
        from .models import networks
        return networks.build_special_arch(arch, model_cfg)

    num_classes = model_cfg.get("num_classes", 2)
    img_size = model_cfg.get("img_size", 224)
    # 'native' keyword: defer to encoder's native_img_size attribute
    if img_size == "native":
        enc_cls_for_native = ENCODER_REGISTRY.get(model_cfg["encoder"]["name"])
        img_size = getattr(enc_cls_for_native, "native_img_size", 224)
        print(f"[model_builder] img_size='native' resolved to {img_size} for encoder {model_cfg['encoder']['name']}")

    # Build encoder
    enc_cfg = model_cfg["encoder"]
    enc_name = enc_cfg["name"]
    enc_cls = ENCODER_REGISTRY.get(enc_name)
    encoder = enc_cls(
        pretrained=enc_cfg.get("pretrained", False),
        in_channels=enc_cfg.get("in_channels", 3),
        img_size=img_size,
        **enc_cfg.get("params", {}),
    )

    # Apply freeze policy from yaml if the encoder supports it
    freeze_cfg = enc_cfg.get("freeze_cfg", {}) or {}
    if freeze_cfg and hasattr(encoder, "set_freeze_policy"):
        encoder.set_freeze_policy(
            freeze=freeze_cfg.get("freeze", True),
            unfreeze_last_n=freeze_cfg.get("unfreeze_last_n", 0),
            inference_only=freeze_cfg.get("inference_only", False),
        )

    encoder_channels = encoder.out_channels  # List of channel dims per stage

    # Build bottleneck
    btn_cfg = model_cfg.get("bottleneck", {"name": "none"})
    btn_name = btn_cfg["name"]
    btn_cls = BOTTLENECK_REGISTRY.get(btn_name)
    bottleneck = btn_cls(
        in_channels=encoder_channels[-1],
        **btn_cfg.get("params", {}),
    )
    bottleneck_channels = getattr(bottleneck, "out_channels", encoder_channels[-1])

    # Build decoder
    dec_cfg = model_cfg["decoder"]
    dec_name = dec_cfg["name"]
    dec_cls = DECODER_REGISTRY.get(dec_name)

    # Check if decoder has internal skip mechanism
    has_internal_skip = getattr(dec_cls, "has_internal_skip", False)

    # Build skip connection only for decoders that use external skip
    skip_connection = None
    if not has_internal_skip:
        skip_cfg = model_cfg.get("skip_connection", {"name": "concat"})
        skip_name = skip_cfg["name"]
        skip_cls = SKIP_REGISTRY.get(skip_name)
        skip_connection = skip_cls(**skip_cfg.get("params", {}))

    # Determine if we need a feature adapter for stage count mismatch
    skip_adapter = None
    required_stages = getattr(dec_cls, "required_skip_stages", None)
    skip_channels = encoder_channels[:-1]  # raw skip channels from encoder

    # Check if decoder requires a specific encoder type (incompatible with generic)
    requires_encoder = getattr(dec_cls, "requires_encoder", None)
    if requires_encoder is not None and enc_name != requires_encoder:
        raise IncompatibleEncoderError(
            f"Decoder '{dec_name}' requires encoder '{requires_encoder}' "
            f"but got '{enc_name}'. This decoder is designed for a specific "
            f"encoder architecture and cannot work with generic encoders."
        )

    if required_stages is not None and len(skip_channels) != required_stages:
        skip_adapter = FeatureAdapter(skip_channels, required_stages)
        # Compute adapted channel list for decoder
        n_have = len(skip_channels)
        n_keep = min(n_have, required_stages) if required_stages < n_have else n_have
        adapted_channels = list(skip_channels[-n_keep:]) if n_keep < n_have else list(skip_channels)
        for i in range(max(0, required_stages - n_have)):
            last_ch = skip_channels[-1] if i == 0 else adapted_channels[-1]
            adapted_channels.append(last_ch * 2)
        skip_channels = adapted_channels

    decoder = dec_cls(
        encoder_channels=skip_channels,  # skip connection channels (possibly adapted)
        bottleneck_channels=bottleneck_channels,
        skip_connection=skip_connection,
        img_size=img_size,
        **dec_cfg.get("params", {}),
    )
    decoder_out_channels = getattr(decoder, "out_channels", encoder_channels[0])

    # Segmentation head
    head = SegmentationHead(decoder_out_channels, num_classes)

    return SegmentationModel(encoder, decoder, bottleneck, head, skip_adapter)


def build_model_from_yaml(yaml_path: str) -> nn.Module:
    """Build model from a YAML config file."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return build_model(cfg)
