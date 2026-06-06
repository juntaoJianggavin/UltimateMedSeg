# Reference: https://github.com/Edisonhimself/MediSee
# Paper:     https://arxiv.org/abs/2407.16942
"""MediSee 在 medseg 框架下的薄 wrapper。

核心目标:
1. 真模型构造与权重加载严格依赖 ``_vendor`` 内的上游 MediSee 源码 (按
   论文 / README 说明组装 LLaVA-Med + CLIP-ViT-L/14-336 + MedSAM ViT-B
   + bbox_decoder), 不写任何 mock fallback。
2. 兼容 medseg 现有 ``forward(image, text=None)`` 调用约定, 输出
   ``(B, num_classes, H, W)`` mask logits。
3. **权重必须可用**: ``weights_loader.ensure_all_weights`` 失败直接 raise;
   ``download_weights=False`` 同样 raise (没有 mock 模式: MediSee 是
   17GB MLLM, 任何零参数占位都不是 MediSee)。

注意: 上游 MediSee 推理路径硬编码 ``.cuda()``, 因此本 wrapper 只能在
GPU 环境构造 / 调用。
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from ._path_setup import ensure_vendor_on_path

__all__ = ["MediSeeWrapper"]


class MediSeeWrapper(nn.Module):
    """MediSee (ACM MM 2025) 的 medseg-glue 包装。

    Parameters
    ----------
    in_channels
        与 medseg 其它分割模型对齐, 这里固定 3 通道 (CLIP+SAM 都吃 RGB)。
    num_classes
        输出通道数; MediSee 本质是单 mask 二分类, 默认 1。
    img_size
        逻辑分辨率 (medseg 主流水线使用), 与 MediSee 内部 SAM=1024 / CLIP=336 解耦。
    download_weights
        ``True`` 时自动 ``snapshot_download`` 4 套权重; ``False`` 时进入 mock 模式。
    cache_dir
        HF 缓存目录, ``None`` 走 HF 默认 (``HF_HOME`` / ``~/.cache/huggingface``)。
    precision
        ``fp16`` / ``bf16`` / ``fp32``。
    conv_type
        LLaVA conversation template, MediSee 默认 ``mistral_instruct``。
    max_new_tokens
        evaluate 阶段 LLM 生成长度上限。
    sam_image_size
        SAM 视觉编码器输入分辨率, 上游硬编码 1024。
    clip_image_size
        CLIP 视觉 tower 输入分辨率, 上游硬编码 336。
    """

    is_text_guided: bool = True

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        download_weights: bool = True,
        cache_dir: Optional[str] = None,
        precision: str = "fp16",
        conv_type: str = "mistral_instruct",
        max_new_tokens: int = 32,
        sam_image_size: int = 1024,
        clip_image_size: int = 336,
    ):
        super().__init__()
        if not download_weights:
            raise ValueError(
                "MediSeeWrapper requires download_weights=True. There is no "
                "mock mode: MediSee is a 17 GB MLLM (LLaVA-Med + CLIP-L-336 "
                "+ MedSAM + fine-tuned weights). A zero-mask placeholder is "
                "not MediSee and silently substituting one would invalidate "
                "any reported metric. Either set download_weights=true in "
                "arch_params (with HF_TOKEN if needed) or remove this model "
                "from the run."
            )
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.download_weights = True
        self.cache_dir = cache_dir
        self.precision = precision
        self.conv_type = conv_type
        self.max_new_tokens = max_new_tokens
        self.sam_image_size = sam_image_size
        self.clip_image_size = clip_image_size

        self._built = False
        self._model: Optional[nn.Module] = None
        self._tokenizer = None
        self._clip_processor = None
        self._sam_transform = None
        self._seg_token_idx = -1
        self._box_token_idx = -1

    # ------------------------------------------------------------------
    # build (lazy 真模型构造)
    # ------------------------------------------------------------------
    def build(self) -> None:
        """真正下载权重 + 构造 MediSeeForCausalLM. 重复调用幂等."""
        if self._built:
            return

        ensure_vendor_on_path()

        from .weights_loader import ensure_all_weights

        weights = ensure_all_weights(cache_dir=self.cache_dir)

        # ---- tokenizer ----------------------------------------------------
        import transformers  # type: ignore

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            weights.llava_med_dir,
            cache_dir=None,
            model_max_length=1024,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.add_tokens("[SEG]")
        tokenizer.add_tokens("[BBOX]")
        seg_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        box_idx = tokenizer("[BBOX]", add_special_tokens=False).input_ids[0]

        # ---- MediSee 主模型 ----------------------------------------------
        # vendor 包导入 (sys.path 已注入)
        from model.MediSee import MediSeeForCausalLM  # type: ignore
        from model.llava import conversation as conversation_lib  # type: ignore

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }.get(self.precision, torch.float16)

        model_args = dict(
            train_mask_decoder=True,
            out_dim=256,
            ce_loss_weight=1.0,
            dice_loss_weight=0.5,
            bce_loss_weight=2.0,
            bbox_loss_weight=2.0,
            seg_token_idx=seg_idx,
            box_token_idx=box_idx,
            vision_pretrained=weights.medsam_ckpt,
            vision_tower=weights.clip_dir,
            use_mm_start_end=False,
        )

        m = MediSeeForCausalLM.from_pretrained(
            weights.llava_med_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **model_args,
        )
        m.config.eos_token_id = tokenizer.eos_token_id
        m.config.bos_token_id = tokenizer.bos_token_id
        m.config.pad_token_id = tokenizer.pad_token_id

        m.get_model().initialize_vision_modules(m.get_model().config)
        m.get_model().initialize_MediSee_modules(m.get_model().config)
        m.resize_token_embeddings(len(tokenizer))

        # 加载 MediSee 自身 fine-tuned .bin (HF repo 内可能含多个 shard)
        if weights.medisee_dir and os.path.isdir(weights.medisee_dir):
            sd_collected: dict = {}
            for fname in sorted(os.listdir(weights.medisee_dir)):
                if fname.endswith(".bin") or fname.endswith(".pt"):
                    sd = torch.load(
                        os.path.join(weights.medisee_dir, fname),
                        map_location="cpu",
                    )
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    for k, v in sd.items():
                        sd_collected[k.replace(".base_layer", "")] = v
            if sd_collected:
                m.load_state_dict(sd_collected, strict=False)

        conversation_lib.default_conversation = conversation_lib.conv_templates[
            self.conv_type
        ]

        # CLIP image processor + SAM transform
        from transformers import CLIPImageProcessor  # type: ignore

        clip_processor = CLIPImageProcessor.from_pretrained(weights.clip_dir)

        from model.segment_anything.utils.transforms import ResizeLongestSide  # type: ignore

        sam_transform = ResizeLongestSide(self.sam_image_size)

        # 注册到 module (使 .to(device) 同步生效)
        self._model = m
        self._tokenizer = tokenizer
        self._clip_processor = clip_processor
        self._sam_transform = sam_transform
        self._seg_token_idx = seg_idx
        self._box_token_idx = box_idx
        self._built = True

    # ------------------------------------------------------------------
    # 数据预处理 (single-image)
    # ------------------------------------------------------------------
    def _preprocess_for_sam(self, np_image_uint8):
        import numpy as np

        assert self._sam_transform is not None
        img = self._sam_transform.apply_image(np_image_uint8)
        resize = img.shape[:2]
        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        img_t = (img_t - pixel_mean) / pixel_std
        h, w = img_t.shape[-2:]
        padh = self.sam_image_size - h
        padw = self.sam_image_size - w
        import torch.nn.functional as F

        img_t = F.pad(img_t, (0, padw, 0, padh))
        return img_t, resize

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,
        text: Optional[Union[str, Sequence[str]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.build()
        assert self._model is not None and self._tokenizer is not None

        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype

        b, _, H, W = image.shape

        # text -> list[str]; MediSee is a reasoning MLLM and operates on a
        # natural-language query — None / empty input is rejected.
        if text is None:
            raise ValueError(
                "MediSeeWrapper.forward requires a textual query (str or "
                "Sequence[str] of length B). MediSee performs reasoning-"
                "based pixel perception conditioned on the prompt; running "
                "without a prompt yields a degenerate output."
            )
        if isinstance(text, str):
            text_list = [text] * b
        elif isinstance(text, (list, tuple)):
            if len(text) != b:
                raise ValueError(
                    f"MediSeeWrapper text list length ({len(text)}) must "
                    f"match the image batch ({b})."
                )
            text_list = list(text)
        else:
            raise TypeError(
                "MediSeeWrapper text must be str or Sequence[str], got "
                f"{type(text).__name__}"
            )

        from model.llava import conversation as conversation_lib  # type: ignore
        from model.llava.mm_utils import tokenizer_image_token  # type: ignore
        from utils.utils import DEFAULT_IMAGE_TOKEN  # type: ignore

        out_masks: List[torch.Tensor] = []
        for i in range(b):
            # 0-1 浮点 / 0-255 自动判别
            np_img = image[i].detach().cpu().numpy().transpose(1, 2, 0)
            if np_img.max() <= 1.0:
                np_img = (np_img * 255.0).clip(0, 255).astype("uint8")
            else:
                np_img = np_img.clip(0, 255).astype("uint8")

            sam_img, resize = self._preprocess_for_sam(np_img)
            sam_img = sam_img.to(device=device, dtype=dtype).unsqueeze(0)
            clip_img = self._clip_processor.preprocess(
                np_img, return_tensors="pt"
            )["pixel_values"][0]
            clip_img = clip_img.to(device=device, dtype=dtype).unsqueeze(0)

            conv = conversation_lib.default_conversation.copy()
            conv.messages = []
            conv.append_message(
                conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text_list[i]
            )
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt, self._tokenizer, return_tensors="pt"
            )
            input_ids = input_ids.unsqueeze(0).to(device)

            label_t = torch.zeros((H, W), dtype=torch.long, device=device)

            with torch.no_grad():
                _, pred_masks, _pred_bboxes = self._model.evaluate(
                    images_clip=clip_img,
                    images=sam_img,
                    input_ids=input_ids,
                    resize_list=[resize],
                    label_list=[label_t],
                    max_new_tokens=self.max_new_tokens,
                    tokenizer=self._tokenizer,
                )

            if not (
                pred_masks
                and pred_masks[0].numel() > 0
                and pred_masks[0].shape[0] > 0
            ):
                raise RuntimeError(
                    f"MediSee returned no [SEG] mask for sample {i} "
                    f"(prompt='{text_list[i]}'). The LLM likely did not "
                    "emit the [SEG] token; refine the prompt or check "
                    "the tokenizer / conv_type. Silently substituting a "
                    "zero mask is disabled."
                )
            mask_logits = pred_masks[0][0:1]  # (1, h0, w0)

            if mask_logits.shape[-2:] != (H, W):
                import torch.nn.functional as F

                mask_logits = F.interpolate(
                    mask_logits.unsqueeze(0).float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # broadcast 到 num_classes
            if self.num_classes > 1 and mask_logits.shape[0] == 1:
                mask_logits = mask_logits.expand(self.num_classes, H, W)
            out_masks.append(mask_logits)

        return torch.stack(out_masks, dim=0)  # (B, num_classes, H, W)
