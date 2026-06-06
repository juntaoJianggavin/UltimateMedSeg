# Reference: https://github.com/ljwztc/CLIP-Driven-Universal-Model
# Paper:     https://arxiv.org/abs/2301.00785
"""CLIP-Driven Universal Model (ICCV 2023) -- 2D adaptation.

Implemented from the paper formulas (Liu et al., "CLIP-Driven Universal
Model for Organ Segmentation and Tumor Detection", ICCV 2023) and the
README of the official repository:
    https://github.com/ljwztc/CLIP-Driven-Universal-Model
        model/Universal_model.py  (Universal_model)
        model/Unet.py             (UNet3D + LUConv/DownTransition/UpTransition)
No source-file copy is performed; modules are re-written from the paper
spec and matched layer-by-layer to the README description.

Algorithm-level 1:1 correspondence is preserved:
    * `organ_embedding` (rand_embedding | word_embedding via CLIP)
    * `controller` 1x1 conv producing per-class dynamic-head params
    * `parse_dynamic_params` slicing into 3 conv layers (8x8 / 8x8 / 8x1)
    * `precls_conv` (channels -> 8) + `GAP` (bottleneck -> 256)
    * Per-class dynamic conv head (`heads_forward`)
    * `encoding_task` helper retained for compatibility

Framework-glue diffs vs. upstream:
    1. **3D -> 2D adaptation.**  Project corpus is mostly 2D (Synapse / QaTa-COV19);
       all `nn.Conv3d`, `nn.MaxPool3d`, `nn.ConvTranspose3d`,
       `nn.AdaptiveAvgPool3d` and `F.conv3d` are replaced 1:1 with their 2D
       counterparts; tensor ranks drop from 5D to 4D accordingly.  No new
       layer is invented; we only collapse the depth dim.  The class is
       therefore named `CLIPDrivenUniversalModel2D` for clarity.
    2. Backbone defaults to a faithful 2D transcription of `UNet3D` (named
       `UNet2D` below).  `swinunetr / dints / unetpp` are skipped because
       their official sources are 3D-specific; users wishing to add a 2D
       Swin-UNETR can plug it in via `backbone='swinunetr'` once a 2D port
       is available.
    3. `forward(image, text=None)` signature for medseg trainer compatibility.
       `text` may optionally be a `(num_classes, 512)` CLIP embedding tensor
       overriding `organ_embedding` for that batch (matches upstream
       `word_embedding` semantics).  When `None`, the model uses the
       registered embedding as-is.
    4. `is_text_guided=True` class attribute.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# UNet2D -- 1:1 transcription of model/Unet.py (3D -> 2D)
# ---------------------------------------------------------------------------


class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    """1:1 of upstream ContBatchNorm3d, but for 4D inputs."""

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )


class LUConv(nn.Module):
    """1:1 of upstream LUConv (3D -> 2D)."""

    def __init__(self, in_chan, out_chan, act):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm2d(out_chan)

        if act == "relu":
            self.activation = nn.ReLU(out_chan)
        elif act == "prelu":
            self.activation = nn.PReLU(out_chan)
        elif act == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            raise ValueError(f"unknown activation: {act}")

    def forward(self, x):
        return self.activation(self.bn1(self.conv1(x)))


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), act)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act)
    else:
        layer1 = LUConv(in_channel, 32 * (2 ** depth), act)
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, act)
    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    """1:1 of upstream DownTransition (3D -> 2D)."""

    def __init__(self, in_channel, depth, act):
        super().__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool2d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    """1:1 of upstream UpTransition (3D -> 2D)."""

    def __init__(self, inChans, outChans, depth, act):
        super().__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans + outChans // 2, depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class UNet2D(nn.Module):
    """1:1 of upstream UNet3D (3D -> 2D); returns (bottleneck, decoder_top)."""

    def __init__(self, in_channels: int = 1, act: str = "relu"):
        super().__init__()
        self.down_tr64 = DownTransition(in_channels, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)

        self.up_tr256 = UpTransition(512, 512, 2, act)
        self.up_tr128 = UpTransition(256, 256, 1, act)
        self.up_tr64 = UpTransition(128, 128, 0, act)

    def forward(self, x):
        out64, skip_out64 = self.down_tr64(x)
        out128, skip_out128 = self.down_tr128(out64)
        out256, skip_out256 = self.down_tr256(out128)
        out512, _ = self.down_tr512(out256)

        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        return out512, out_up_64


# ---------------------------------------------------------------------------
# Universal model -- 1:1 of model/Universal_model.py (3D -> 2D)
# ---------------------------------------------------------------------------


class CLIPDrivenUniversalModel2D(nn.Module):
    """2D adaptation of `Universal_model` from CLIP-Driven Universal Model (ICCV'23).

    Args (mirror upstream where applicable):
        img_size:    input H=W (used only by future swinunetr backbone).
        in_channels: input image channels.
        out_channels: number of organ / lesion classes (one dynamic head per class).
        backbone:    'unet' is the only currently shipped 2D backbone.
        encoding:    'rand_embedding' or 'word_embedding'.
                     - rand_embedding: trainable nn.Embedding(out_channels, 256).
                     - word_embedding: register_buffer (out_channels, 512) + Linear(512,256).
        clip_embedding: optional pre-computed CLIP class embeddings of shape
                        (out_channels, 512); when given with encoding='word_embedding',
                        used to initialise the buffer (upstream behaviour).
    """

    is_text_guided = True

    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 1,
        out_channels: int = 1,
        backbone: str = "unet",
        encoding: str = "rand_embedding",
        clip_embedding: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "unet":
            self.backbone = UNet2D(in_channels=in_channels)
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 8, kernel_size=1),
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise NotImplementedError(
                f"backbone '{backbone}' is not yet ported to 2D in this repo; "
                "currently only 'unet' is available."
            )

        self.encoding = encoding

        # dynamic-head shape (1:1 with upstream)
        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums

        # controller produces (sum(weights+biases)) channels from concat([feat, organ])
        self.controller = nn.Conv2d(
            256 + 256, sum(weight_nums + bias_nums), kernel_size=1, stride=1, padding=0
        )

        if self.encoding == "rand_embedding":
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == "word_embedding":
            # Strict: paper requires pre-computed CLIP text embeddings of the
            # organ / lesion class names. We refuse to silently substitute
            # ``torch.randn`` because doing so would be a non-CLIP model
            # masquerading as CLIP-Universal.
            if clip_embedding is None:
                raise ValueError(
                    "encoding='word_embedding' requires a pre-computed "
                    "(out_channels, 512) CLIP text-embedding tensor; "
                    "got clip_embedding=None. Either pre-compute it with "
                    "CLIP ViT-B/32 and pass via arch_params, or use "
                    "encoding='rand_embedding'."
                )
            if clip_embedding.shape != (out_channels, 512):
                raise ValueError(
                    f"clip_embedding must be ({out_channels}, 512), "
                    f"got {tuple(clip_embedding.shape)}"
                )
            self.register_buffer("organ_embedding", clip_embedding.clone())
            self.text_to_vision = nn.Linear(512, 256)
        else:
            raise ValueError(f"unknown encoding mode: {encoding}")

        self.class_num = out_channels

    # ------------------------------------------------------------------
    # upstream compat helpers
    # ------------------------------------------------------------------

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7), device=task_id.device)
        for i in range(N):
            task_encoding[i, task_id[i]] = 1
        return task_encoding

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(
            torch.split_with_sizes(params, weight_nums + bias_nums, dim=1)
        )

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # 4D weight for 2D conv: (out, in, 1, 1)
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def _resolve_organ_embedding(self, text: Any | None) -> torch.Tensor:
        """Return task encoding tensor of shape (class_num, 256, 1, 1)."""
        if text is not None and isinstance(text, torch.Tensor):
            # caller supplies CLIP class embedding (class_num, 512)
            assert text.shape == (self.class_num, 512), (
                f"text override must be ({self.class_num}, 512), got {tuple(text.shape)}"
            )
            assert hasattr(self, "text_to_vision"), (
                "text override requires encoding='word_embedding'."
            )
            task_encoding = F.relu(self.text_to_vision(text))
            return task_encoding.unsqueeze(-1).unsqueeze(-1)

        if self.encoding == "rand_embedding":
            return self.organ_embedding.weight.unsqueeze(-1).unsqueeze(-1)
        # word_embedding (no override)
        task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
        return task_encoding.unsqueeze(-1).unsqueeze(-1)

    def forward(self, image: torch.Tensor, text: Any | None = None):
        """forward(image, text=None).

        text: optionally a (class_num, 512) tensor that overrides the registered
              organ embedding (mirrors upstream `word_embedding` mechanism).
        Returns logits of shape (B, class_num, H, W).
        """
        # 如果 text=None 且 yaml 里配了 text_prompts，自动使用
        # If text=None and text_prompts configured in yaml, use them automatically
        if text is None and hasattr(self, '_default_text_prompts') and self._default_text_prompts:
            text = self._default_text_prompts
        dec4, out = self.backbone(image)
        task_encoding = self._resolve_organ_embedding(text)  # (C, 256, 1, 1)

        x_feat = self.GAP(dec4)  # (B, 256, 1, 1)
        b = x_feat.shape[0]

        logits_array = []
        for i in range(b):
            x_cond = torch.cat(
                [x_feat[i].unsqueeze(0).repeat(self.class_num, 1, 1, 1), task_encoding],
                dim=1,
            )  # (C, 512, 1, 1)
            params = self.controller(x_cond)  # (C, sum(wb), 1, 1)
            params = params.squeeze(-1).squeeze(-1)  # (C, sum(wb))

            head_inputs = self.precls_conv(out[i].unsqueeze(0))  # (1, 8, H, W)
            head_inputs = head_inputs.repeat(self.class_num, 1, 1, 1)
            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W)

            weights, biases = self.parse_dynamic_params(
                params, 8, self.weight_nums, self.bias_nums
            )
            logits = self.heads_forward(head_inputs, weights, biases, N)  # (1, C, H, W)
            logits_array.append(logits.reshape(1, -1, H, W))

        out = torch.cat(logits_array, dim=0)
        return out
