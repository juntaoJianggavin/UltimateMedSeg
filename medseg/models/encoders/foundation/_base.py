"""Base class + FreezeMixin for foundation-model-backed encoders.

All foundation encoders (CLIP/DINO/DINOv2/DINOv3/SAM/BiomedCLIP/...) should
inherit BaseFoundationEncoder. It provides:
- Lazy pretrained loading with SSL/network fallback.
- A FreezeMixin granting freeze_all / unfreeze_all / unfreeze_last_n_blocks /
  set_freeze_policy methods.
- Standard List[Tensor] output interface (deepest LAST).
"""
# Source: INTERNAL — framework adaptation (this repo).

from __future__ import annotations

from medseg.utils.hf_hub import configure_hf_hub

configure_hf_hub()

import warnings
from typing import List, Optional
import torch
import torch.nn as nn


def load_with_ssl_fallback(load_fn, *args, **kwargs):
    """Try the loader; on SSL failure retry once with unverified context.

    Does NOT silently fall back to random init or a different model. If the
    pretrained weights cannot be loaded, raises RuntimeError with a clear
    message.

    Args:
        load_fn: callable that performs the load (e.g. a model factory).
        *args, **kwargs: forwarded to load_fn. A non-False 'pretrained' kwarg
            indicates pretrained weights are desired.
    """
    import ssl
    # If the caller already disabled pretrained, just call through — no retry needed.
    if kwargs.get("pretrained", True) is False:
        return load_fn(*args, **kwargs)

    try:
        return load_fn(*args, **kwargs)
    except Exception as e1:
        # Retry with unverified SSL context (corporate / local cert miss).
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            # No silent fallback — raise a clear error.
            model_name = ""
            if args:
                model_name = str(args[0])
            elif "model_name" in kwargs:
                model_name = str(kwargs["model_name"])
            raise RuntimeError(
                f"Failed to load pretrained weights for '{model_name}'. "
                f"Initial error: {type(e1).__name__}: {e1}. "
                f"SSL-bypass retry error: {type(e2).__name__}: {e2}. "
                f"Provide a local checkpoint via 'pretrained_path' "
                f"or ensure network access to download the official weights."
            ) from e2
        finally:
            ssl._create_default_https_context = prev


def hf_hub_download_vision_weights(repo_id: str, filename: str = None,
                                   prefix_strip: tuple = ()) -> dict:
    """Auto-download weights from HuggingFace Hub and return a state-dict.

    Uses ``huggingface_hub.hf_hub_download`` to fetch the checkpoint, loads
    it (supports ``.safetensors`` and ``.bin``), and optionally strips common
    key prefixes so the result can be loaded into the backbone.

    Args:
        repo_id: HuggingFace repo identifier, e.g. ``"microsoft/rad-dino"``.
        filename: Specific file to download.  If *None*, tries
            ``"model.safetensors"`` then ``"pytorch_model.bin"``.
        prefix_strip: Tuple of key-prefix strings to strip from the state-dict
            (first match wins, applied per-key).

    Returns:
        A ``dict`` state-dict ready for ``backbone.load_state_dict(..., strict=False)``.

    Raises:
        RuntimeError: If the download or load fails.
    """
    from medseg.utils.hf_hub import hf_hub_download_file

    candidates = [filename] if filename else ["model.safetensors", "pytorch_model.bin"]
    path = None
    last_err = None
    for fn in candidates:
        try:
            path = hf_hub_download_file(repo_id, fn)
            break
        except Exception as e:
            last_err = e
    if path is None:
        raise RuntimeError(
            f"Failed to download weights from '{repo_id}'. "
            f"Tried files: {candidates}. Last error: {last_err}"
        ) from last_err

    # Load the checkpoint.
    if str(path).endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
            state = load_file(path)
        except ImportError:
            raise RuntimeError(
                "safetensors is required to load .safetensors checkpoints. "
                "Install with: pip install safetensors"
            )
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)

    # Unwrap nested dicts.
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    # Strip common vision-tower prefixes.
    if isinstance(state, dict) and prefix_strip:
        cleaned = {}
        for k, v in state.items():
            nk = k
            for pref in prefix_strip:
                if nk.startswith(pref):
                    nk = nk[len(pref):]
                    break
            cleaned[nk] = v
        state = cleaned

    return state


def convert_timm_vit_state_to_hf(state: dict) -> dict:
    """Convert a timm ViT state_dict to HuggingFace ViTModel layout.

    Handles the key-name differences between timm's ``vit_*`` models and
    transformers' ``ViTModel``. Splits fused ``qkv`` weights/biases into
    separate ``query``, ``key``, ``value``.
    """
    hf = {}
    for k, v in state.items():
        nk = k
        nk = nk.replace("cls_token", "embeddings.cls_token")
        nk = nk.replace("pos_embed", "embeddings.position_embeddings")
        nk = nk.replace("patch_embed.proj.", "embeddings.patch_embeddings.projection.")
        nk = nk.replace("blocks.", "encoder.layer.")
        nk = nk.replace(".norm1.", ".layernorm_before.")
        nk = nk.replace(".norm2.", ".layernorm_after.")
        nk = nk.replace(".mlp.fc1.", ".intermediate.dense.")
        nk = nk.replace(".mlp.fc2.", ".output.dense.")
        nk = nk.replace(".attn.proj.", ".output.dense.")
        nk = nk.replace("norm.", "layernorm.")

        if ".attn.qkv." in k:
            dim = v.shape[0] // 3
            base = nk.replace(".attn.qkv.", ".attention.attention.")
            hf[base.replace("qkv", "query") + (".weight" if "weight" in nk else ".bias")] = v[:dim]
            hf[base.replace("qkv", "key") + (".weight" if "weight" in nk else ".bias")] = v[dim:2*dim]
            hf[base.replace("qkv", "value") + (".weight" if "weight" in nk else ".bias")] = v[2*dim:]
            continue

        # Skip timm-specific keys (head, dist_token, etc.)
        if any(skip in k for skip in ("head.", "dist_token", "fc_norm")):
            continue
        hf[nk] = v
    return hf


class FreezeMixin:
    """Mix-in for any encoder to expose freeze controls.

    The encoder must expose:
    - self.backbone (nn.Module)  — the part to freeze
    - optional: self.backbone.blocks/layers/transformer_blocks for partial unfreeze
    """

    def freeze_all(self):
        if hasattr(self, "backbone"):
            for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_all(self):
        if hasattr(self, "backbone"):
            for p in self.backbone.parameters(): p.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int):
        if n <= 0 or not hasattr(self, "backbone"): return
        blocks = None
        for attr in ("blocks", "layers", "transformer_blocks"):
            if hasattr(self.backbone, attr):
                blocks = list(getattr(self.backbone, attr))
                break
        if blocks:
            for blk in blocks[-n:]:
                for p in blk.parameters(): p.requires_grad = True
            for attr in ("norm", "norm_post", "ln_post"):
                if hasattr(self.backbone, attr):
                    m = getattr(self.backbone, attr)
                    if isinstance(m, nn.Module):
                        for p in m.parameters(): p.requires_grad = True

    def set_freeze_policy(self, freeze: bool = True, unfreeze_last_n: int = 0,
                         inference_only: bool = False):
        if freeze: self.freeze_all()
        else: self.unfreeze_all()
        if unfreeze_last_n > 0:
            self.unfreeze_last_n_blocks(unfreeze_last_n)
        if inference_only:
            self.eval()
            for p in self.parameters(): p.requires_grad = False
            return
        # If adapters are injected, keep them trainable
        if getattr(self, "_has_adapters", False):
            for n, p in self.named_parameters():
                if "adapter_attn" in n or "adapter_mlp" in n:
                    p.requires_grad = True

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _Adapter(nn.Module):
    """Houlsby bottleneck adapter: LN -> down -> GELU -> up + residual. Near-zero init."""
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        nn.init.normal_(self.down.weight, std=1e-3); nn.init.zeros_(self.down.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


class _AdapterBlockWrapper(nn.Module):
    """Wrap a transformer block (timm or HF). Inserts adapters after attn and MLP residuals."""
    def __init__(self, block, dim, bottleneck_dim=64):
        super().__init__()
        self.block = block
        self.adapter_attn = _Adapter(dim, bottleneck_dim)
        self.adapter_mlp = _Adapter(dim, bottleneck_dim)

    def forward(self, x, *args, **kwargs):
        # Standard ViT block: norm1, attn, norm2, mlp
        if (hasattr(self.block, 'norm1') and hasattr(self.block, 'attn') and
            hasattr(self.block, 'norm2') and hasattr(self.block, 'mlp')):
            ls1 = getattr(self.block, 'ls1', nn.Identity())
            ls2 = getattr(self.block, 'ls2', nn.Identity())
            dp1 = getattr(self.block, 'drop_path1', getattr(self.block, 'drop_path', nn.Identity()))
            dp2 = getattr(self.block, 'drop_path2', getattr(self.block, 'drop_path', nn.Identity()))
            x = x + dp1(ls1(self.block.attn(self.block.norm1(x))))
            x = self.adapter_attn(x)
            x = x + dp2(ls2(self.block.mlp(self.block.norm2(x))))
            x = self.adapter_mlp(x)
            return x
        # HF-style block: layer_norm1, self_attn, layer_norm2, intermediate+output
        if (hasattr(self.block, 'layer_norm1') and hasattr(self.block, 'self_attn') and
            hasattr(self.block, 'layer_norm2')):
            residual = x
            normed = self.block.layer_norm1(x)
            attn_out = self.block.self_attn(normed)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = residual + attn_out
            x = self.adapter_attn(x)
            residual = x
            normed = self.block.layer_norm2(x)
            if hasattr(self.block, 'intermediate') and hasattr(self.block, 'output'):
                ff = self.block.output(self.block.intermediate(normed))
            elif hasattr(self.block, 'mlp'):
                ff = self.block.mlp(normed)
            else:
                ff = normed
            x = residual + ff
            x = self.adapter_mlp(x)
            return x
        # Generic fallback: just wrap the output
        out = self.block(x, *args, **kwargs)
        if isinstance(out, tuple):
            x_main = out[0]
            x_main = self.adapter_mlp(self.adapter_attn(x_main))
            return (x_main,) + out[1:]
        return self.adapter_mlp(self.adapter_attn(out))


class BaseFoundationEncoder(nn.Module, FreezeMixin):
    """Base for all foundation-model encoders.

    Standard contract:
    - self.out_channels: List[int] — channels at each multi-scale stage (deepest LAST)
    - forward(x) -> List[Tensor]
    - Constructor accepts (in_channels=3, img_size=224, pretrained=True,
      pretrained_path=None, freeze=True, unfreeze_last_n=0, inference_only=False, **kwargs)
    """

    native_img_size: int = 224   # subclasses override (e.g. SAM ViT-H = 1024)

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 pretrained: bool = True, pretrained_path: Optional[str] = None,
                 freeze: bool = True, unfreeze_last_n: int = 0,
                 inference_only: bool = False,
                 use_adapter: bool = False, adapter_dim: int = 64, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self._pretrained = pretrained
        self._pretrained_path = pretrained_path
        self._freeze_cfg = {
            "freeze": freeze, "unfreeze_last_n": unfreeze_last_n,
            "inference_only": inference_only,
            "use_adapter": use_adapter, "adapter_dim": adapter_dim,
        }
        self._has_adapters = False
        # Subclass MUST set self.backbone and self.out_channels in __init__,
        # then call self._apply_freeze_policy()

    def _maybe_inject_adapters(self):
        """Call AFTER self.backbone is built and BEFORE self._apply_freeze_policy()."""
        if not self._freeze_cfg.get("use_adapter", False):
            return
        if not hasattr(self, "backbone"):
            return
        blocks_attr = None
        for attr in ("blocks", "layers", "transformer_blocks"):
            if hasattr(self.backbone, attr):
                blocks_attr = attr; break
        if blocks_attr is None:
            return
        blocks = getattr(self.backbone, blocks_attr)
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential, list)) or len(blocks) == 0:
            return
        dim = getattr(self.backbone, "embed_dim", None) or getattr(self.backbone, "num_features", None)
        if dim is None:
            first = blocks[0]
            for nm in ("norm1", "ln_1", "layer_norm1", "norm"):
                if hasattr(first, nm):
                    d = getattr(first, nm).normalized_shape
                    dim = d[0] if isinstance(d, tuple) else int(d)
                    break
        if dim is None:
            warnings.warn("Adapter injection skipped — could not infer dim.")
            return
        bdim = int(self._freeze_cfg.get("adapter_dim", 64))
        wrapped = [_AdapterBlockWrapper(b, dim, bdim) for b in blocks]
        if isinstance(blocks, nn.ModuleList):
            new_blocks = nn.ModuleList(wrapped)
        else:
            new_blocks = nn.Sequential(*wrapped)
        setattr(self.backbone, blocks_attr, new_blocks)
        self._has_adapters = True

    def _apply_freeze_policy(self):
        cfg = self._freeze_cfg
        self.set_freeze_policy(freeze=cfg["freeze"],
                               unfreeze_last_n=cfg["unfreeze_last_n"],
                               inference_only=cfg["inference_only"])


class HuggingFaceViTWrapper(nn.Module):
    """Wrap a HuggingFace ViT model with a unified interface.

    Provides ``forward_features(x) -> (B, N, C)``, ``embed_dim``,
    ``num_prefix_tokens``, ``patch_embed.patch_size``, and a ``blocks``
    alias so that downstream FPN-from-tokens code, adapter injection, and
    unfreeze controls work for any ViT-family model.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
        cfg = hf_model.config
        self.embed_dim = int(cfg.hidden_size)
        self.num_prefix_tokens = 1  # standard HF ViT: [CLS] at position 0
        # SigLIP models have no CLS token
        if getattr(cfg, "model_type", "") == "siglip" or "siglip" in type(hf_model).__name__.lower():
            self.num_prefix_tokens = 0

        # Expose patch_embed.patch_size for downstream code.
        pe = getattr(hf_model, "embeddings", None)
        if pe is not None and hasattr(pe, "patch_size"):
            ps = pe.patch_size
        else:
            ps = int(getattr(cfg, "patch_size", 16))
        if isinstance(ps, (tuple, list)):
            ps = ps[0]

        class _PE:
            pass
        self.patch_embed = _PE()
        self.patch_embed.patch_size = int(ps)

        # Expose blocks as a unified alias (for adapter injection
        # and unfreeze controls).
        if hasattr(hf_model, "encoder") and hasattr(hf_model.encoder, "layer"):
            # HF ViTModel / Dinov2Model / SiglipVisionModel
            self.blocks = hf_model.encoder.layer
        elif hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "resblocks"):
            # open_clip VisionTransformer
            self.blocks = hf_model.transformer.resblocks
        elif hasattr(hf_model, "layers"):
            # HF CLIPVisionModel: vision_model.encoder.layers
            self.blocks = hf_model.layers
        elif hasattr(hf_model, "encoder_lay"):
            # Some custom HF models
            self.blocks = hf_model.encoder_lay

        # Expose num_features for downstream compatibility.
        self.num_features = self.embed_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return token sequence ``(B, N, C)`` including prefix tokens."""
        out = self.model(x)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)):
            return out[0]
        return out.last_hidden_state

    def get_intermediate_layers(self, x, n):
        """模拟 timm 的 get_intermediate_layers，从指定 block 提取 token。
        Emulate timm's get_intermediate_layers: extract tokens from specified blocks.

        Args:
            x: 输入图像 (B, C, H, W) / Input image tensor.
            n: block 索引列表或整数 / List of block indices or int (last n blocks).

        Returns:
            List of (B, N_patches, C) tensors with prefix tokens stripped.
        """
        # 先做 patch embedding / Run patch embedding
        embeddings = self.model.embeddings(x)
        hidden = embeddings
        outputs = []
        blocks = list(self.blocks)
        target_set = set(n) if isinstance(n, (list, tuple)) else set(range(len(blocks) - n, len(blocks)))
        for i, block in enumerate(blocks):
            hidden = block(hidden)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            if i in target_set:
                # 去掉 prefix tokens / Strip prefix tokens
                out = hidden[:, self.num_prefix_tokens:, :]
                outputs.append(out)
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def load_hf_vit(hf_name: str, pretrained_path: str = None,
                trust_remote_code: bool = False,
                model_cls_name: str = "ViTModel",
                vision_attr: str = None,
                model_obj=None) -> HuggingFaceViTWrapper:
    """Load a ViT from HuggingFace transformers and return a wrapped model.

    Args:
        hf_name: HuggingFace repo id (e.g. ``"owkin/phikon"``).
        pretrained_path: Optional local checkpoint path. If *None*, auto-downloads.
        trust_remote_code: Pass ``True`` for repos with custom model code.
        model_cls_name: Name of the model class to import from ``transformers``.
            Default ``"ViTModel"``.  Use ``"CLIPVisionModel"`` for CLIP-style
            repos, or any other class name.
        vision_attr: If set, extract this attribute from the loaded model
            (e.g. ``"vision_model"`` for CLIPModel, ``"visual"`` for KEEPModel).
        model_obj: If provided, skip loading and wrap this model directly.

    Returns:
        A :class:`HuggingFaceViTWrapper` ready for use as ``self.backbone``.
    """
    if model_obj is not None:
        return HuggingFaceViTWrapper(model_obj)

    try:
        import transformers
    except ImportError:
        raise RuntimeError(
            "transformers is required for HuggingFace ViT loading. "
            "Install with: pip install transformers"
        )

    model_cls = getattr(transformers, model_cls_name, None)
    if model_cls is None:
        raise RuntimeError(
            f"transformers.{model_cls_name} not found. "
            f"Check that the model class is available."
        )

    if pretrained_path:
        model = model_cls.from_pretrained(
            pretrained_path, trust_remote_code=trust_remote_code,
        )
    else:
        model = model_cls.from_pretrained(
            hf_name, trust_remote_code=trust_remote_code,
        )

    if vision_attr:
        model = getattr(model, vision_attr)

    return HuggingFaceViTWrapper(model)


# =====================================================================
# DPT-style multi-block feature projector (替代 FPN-from-tokens)
# DPT-style multi-block feature projector (replaces FPN-from-tokens)
# =====================================================================
#
# 参考 / Reference:
#   Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021
#   https://github.com/isl-org/DPT
#
# 核心思想 / Key idea:
#   从 ViT 的不同 block（如 block 3/6/9/12）各取一次 token 输出，
#   reshape 成 2D feature map 后用不同 stride 的卷积投影成多尺度金字塔。
#   每一级的语义抽象层次真正不同（浅层=纹理，深层=语义）。
#
#   Extract tokens from different ViT blocks (e.g. block 3/6/9/12),
#   reshape to 2D feature maps, then project with different-stride convs
#   to form a multi-scale pyramid. Each level has genuinely different
#   semantic abstraction (shallow=texture, deep=semantics).
# =====================================================================

import torch.nn.functional as F


class DPTHead(nn.Module):
    """DPT-style multi-block projector: 从 ViT 不同 block 取 token 构建真正多尺度金字塔。
    DPT-style multi-block projector: builds a genuine multi-scale pyramid
    from tokens extracted at different ViT blocks.

    用法 / Usage (在子类 __init__ 中):
        self.dpt = DPTHead(embed_dim=768, out_channels=[96, 192, 384, 768])
        self.out_channels = self.dpt.out_channels

    用法 / Usage (在子类 forward 中):
        # 方式1: timm backbone 有 get_intermediate_layers
        # Option 1: timm backbone with get_intermediate_layers
        block_indices = self.dpt.default_block_indices(num_blocks)
        tokens_list = backbone.get_intermediate_layers(x, n=block_indices)
        features = self.dpt(tokens_list, H_patches, W_patches, orig_H, orig_W)

        # 方式2: HF backbone，手动 hook 中间层
        # Option 2: HF backbone, manual intermediate extraction
        tokens_list = [hook_outputs[i] for i in block_indices]
        features = self.dpt(tokens_list, H_patches, W_patches, orig_H, orig_W)

    Args:
        embed_dim: ViT 的 embedding 维度 / ViT embedding dimension.
        out_channels: 4 个尺度的输出通道 / Output channels for 4 scales.
                      默认 [dim//8, dim//4, dim//2, dim]。
        num_prefix_tokens: CLS/register token 数量，需跳过 / Number of prefix tokens to skip.
    """

    def __init__(
        self,
        embed_dim: int,
        out_channels: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens

        if out_channels is None:
            out_channels = [
                max(embed_dim // 8, 1),
                max(embed_dim // 4, 1),
                max(embed_dim // 2, 1),
                embed_dim,
            ]
        assert len(out_channels) == 4, "DPTHead requires exactly 4 output channel sizes"
        self.out_channels = out_channels

        # 4 级 Reassemble: 把每个 block 的 token 投影到目标通道数
        # 4-level Reassemble: project each block's tokens to target channels
        # 浅层(stage0) 上采样 4x → stride /4
        # Shallow (stage0) upsample 4x → stride /4
        self.project0 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[0], 1),
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
        )
        # stage1 上采样 2x → stride /8
        self.project1 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[1], 1),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
        )
        # stage2 保持原始 patch 分辨率 → stride /16
        self.project2 = nn.Conv2d(embed_dim, out_channels[2], 1)
        # stage3 (最深) 下采样 2x → stride /32
        self.project3 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels[3], 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fusion: 每级加一个 3x3 conv 平滑
        # Fusion: 3x3 conv refinement at each level
        self.refine0 = nn.Sequential(nn.Conv2d(out_channels[0], out_channels[0], 3, padding=1, bias=False), nn.BatchNorm2d(out_channels[0]), nn.GELU())
        self.refine1 = nn.Sequential(nn.Conv2d(out_channels[1], out_channels[1], 3, padding=1, bias=False), nn.BatchNorm2d(out_channels[1]), nn.GELU())
        self.refine2 = nn.Sequential(nn.Conv2d(out_channels[2], out_channels[2], 3, padding=1, bias=False), nn.BatchNorm2d(out_channels[2]), nn.GELU())
        self.refine3 = nn.Sequential(nn.Conv2d(out_channels[3], out_channels[3], 3, padding=1, bias=False), nn.BatchNorm2d(out_channels[3]), nn.GELU())

    @staticmethod
    def default_block_indices(num_blocks: int) -> List[int]:
        """计算均匀间隔的 4 个 block 索引（DPT 标准做法）。
        Compute 4 evenly-spaced block indices (standard DPT practice).

        例如 12-block ViT → [2, 5, 8, 11]
        e.g. 12-block ViT → [2, 5, 8, 11]
        """
        step = num_blocks // 4
        return [step - 1, 2 * step - 1, 3 * step - 1, num_blocks - 1]

    def _tokens_to_grid(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """将 (B, N, C) token 序列 reshape 为 (B, C, h, w) 特征图。
        Reshape (B, N, C) token sequence to (B, C, h, w) feature map.

        注意：timm 的 get_intermediate_layers 已自动去掉 CLS/register prefix，
        所以 N 通常刚好等于 h*w。只有 forward_features 的输出才需要去 prefix。
        Note: timm's get_intermediate_layers auto-strips CLS/register prefix,
        so N usually equals h*w. Only forward_features output needs prefix removal.
        """
        if tokens.dim() == 3:
            B, N, C = tokens.shape
            expected = h * w
            # 如果 N > expected，说明还包含 prefix token，去掉
            # If N > expected, prefix tokens are still present — strip them
            if N > expected and self.num_prefix_tokens > 0:
                tokens = tokens[:, self.num_prefix_tokens:, :]
                N = tokens.shape[1]
            if N != expected:
                tokens = tokens[:, :expected, :]
            return tokens.transpose(1, 2).contiguous().view(B, C, h, w)
        return tokens

    def forward(
        self,
        multi_block_tokens: List[torch.Tensor],
        h_patches: int,
        w_patches: int,
        orig_H: int,
        orig_W: int,
    ) -> List[torch.Tensor]:
        """将 4 个 block 的 token 输出转为 4 级金字塔。
        Convert 4 blocks' token outputs to a 4-level feature pyramid.

        Args:
            multi_block_tokens: 长度为 4 的列表，每个 (B, N, C) 或 (B, C, h, w)。
                                List of 4 tensors, each (B, N, C) or (B, C, h, w).
            h_patches: patch grid 高度 / patch grid height.
            w_patches: patch grid 宽度 / patch grid width.
            orig_H: 原始输入图像高度 / original input height.
            orig_W: 原始输入图像宽度 / original input width.

        Returns:
            [f0, f1, f2, f3] — 从浅到深，空间分辨率递减。
            [f0, f1, f2, f3] — shallow to deep, decreasing spatial resolution.
        """
        assert len(multi_block_tokens) == 4, \
            f"DPTHead expects 4 block outputs, got {len(multi_block_tokens)}"

        grids = [self._tokens_to_grid(t, h_patches, w_patches) for t in multi_block_tokens]

        f0 = self.refine0(self.project0(grids[0]))  # 浅层，大分辨率 / shallow, large spatial
        f1 = self.refine1(self.project1(grids[1]))
        f2 = self.refine2(self.project2(grids[2]))
        f3 = self.refine3(self.project3(grids[3]))  # 深层，小分辨率 / deep, small spatial

        # 裁剪到标准金字塔尺寸 / Crop to canonical pyramid sizes
        targets = [
            (max(orig_H // 4, 1), max(orig_W // 4, 1)),
            (max(orig_H // 8, 1), max(orig_W // 8, 1)),
            (max(orig_H // 16, 1), max(orig_W // 16, 1)),
            (max(orig_H // 32, 1), max(orig_W // 32, 1)),
        ]
        out = []
        for feat, (th, tw) in zip([f0, f1, f2, f3], targets):
            if feat.shape[-2:] != (th, tw):
                feat = F.interpolate(feat, size=(th, tw), mode="bilinear", align_corners=False)
            out.append(feat)
        return out
