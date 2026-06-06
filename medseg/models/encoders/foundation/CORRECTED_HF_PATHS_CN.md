# 基础编码器 HuggingFace 路径修正

[English](CORRECTED_HF_PATHS.md)

汇总自各编码器的网络调研。每条记录均通过对 `hf-mirror.com`、GitHub API 和/或
权威论文来源的直接 HTTP 探测进行了验证（原生 `web_search` 被工具链屏蔽）。

---

## 1. 已确认的公开 HF 构件（`hf-hub:...`）

| encoder_key | 已验证 hf 路径 | 来源 URL（github/论文） | 说明 |
|---|---|---|---|
| `monet_derm` | `hf-hub:suinleelab/monet` | https://github.com/suinleelab/MONET (Kim et al., *Nature Medicine* 30:1154-1165, 2024) | hf-mirror 上验证 HTTP 200。 |
| `dermclip` | `hf-hub:redlessone/DermLIP_ViT-B-16` | https://github.com/SiyuanYan1/Derm1M (arXiv:2503.14911, ICCV'25 Highlight) | 通过 HF 镜像 API 确认；无 DermCLIP 别名——发布名为 `redlessone/DermLIP_ViT-B-16`。 |
| `keep` | `hf-hub:Astaxanthin/KEEP` | https://github.com/MAGIC-AI4Med/KEEP (arXiv:2412.13126; *Cancer Cell* 2026) | 自定义 `KEEPModel`——使用 `transformers.AutoModel.from_pretrained('Astaxanthin/KEEP', trust_remote_code=True)` 加载。裸 `timm` `hf-hub:` 快捷方式不可用；需下载 `model.safetensors` 并通过 `pretrained_path=` 传入，或从 HF 模型中提取 `.visual` state_dict。 |
| `medsiglip` | `hf-hub:google/medsiglip-448` | https://huggingface.co/google/medsiglip-448 | 镜像上验证 HTTP 200。 |
| `endo_vit` | `hf-hub:egeozsoy/EndoViT` | https://github.com/DominikBatic/EndoViT (official) / https://huggingface.co/egeozsoy/EndoViT / https://doi.org/10.1007/s11548-024-03091-5 | 文件为 `pytorch_model.bin`，键为 `"model"`。非 timm-HF-hub 原生构件——需通过 `huggingface_hub.snapshot_download` 加载。 |
| `colon_gpt_encoder` | `hf-hub:timm/ViT-SO400M-14-SigLIP-384` | https://github.com/ai4colonoscopy/IntelliScope (arXiv:2410.17241) | 此为 ColonGPT 使用的 SigLIP-SO400M/14-384 视觉塔。完整多模态模型在 `hf-hub:ai4colonoscopy/ColonGPT-v1`，为自定义 `ColongptPhi` 格式——需手动提取 `vision_tower.*` 键。 |
| `ctfm` | `hf-hub:project-lighter/ct_fm_feature_extractor` | https://github.com/project-lighter/CT-FM (arXiv:2501.09001) | Pai et al., "CT-FM: A 3D Image-Based Foundation Model for Computed Tomography", 2025。 |
| `raddino` | `hf-hub:microsoft/rad-dino` | https://huggingface.co/microsoft/rad-dino (Pérez-García et al., Microsoft Research, 2024) | 镜像上验证 HTTP 200。 |
| `retfound` | `hf-hub:YukunZhou/RETFound_mae_natureCFP`（眼底/cfp）和 `hf-hub:YukunZhou/RETFound_mae_natureOCT`（oct） | https://github.com/rmaphoh/RETFound_MAE (Zhou et al., *Nature* 2023) | 两者均为门控——需申请访问。 |
| `flair` | `hf-hub:jusiro2/FLAIR` | https://github.com/jusiro/FLAIR | README 文档记载 `FLAIRModel.from_pretrained("jusiro2/FLAIR")`。 |
| `plip` | `hf-hub:vinid/plip` | https://github.com/PathologyFoundation/plip | HF 镜像 API 记录已验证。 |
| `musk` | `hf-hub:xiangjx/musk` | https://github.com/lilab-bcb/MUSK (Xiang et al., *Nature* 2025) | 验证 HTTP 200；目录树包含 README 和权重。 |
| `uni` | `hf-hub:MahmoodLab/UNI` | https://huggingface.co/MahmoodLab/UNI (https://www.nature.com/articles/s41591-024-02857-3) | 镜像上验证 HTTP 200。 |
| `uni2` | `hf-hub:MahmoodLab/UNI2-h` | https://huggingface.co/MahmoodLab/UNI2-h (Chen et al., *Nat. Med.* 2024; https://github.com/mahmoodlab/UNI) | 门控。 |
| `conch` | `hf-hub:MahmoodLab/CONCH` | https://github.com/mahmoodlab/CONCH | 镜像 + GitHub 上验证 HTTP 200。 |
| `phikon` | `hf-hub:owkin/phikon` | https://huggingface.co/owkin/phikon (Filiot et al., medRxiv 10.1101/2023.07.21.23292757) | 镜像上验证 HTTP 200。 |
| `phikon_v2` | `hf-hub:owkin/phikon-v2` | https://huggingface.co/owkin/phikon-v2 (arXiv:2409.09173) | 通过 HF 镜像验证。 |
| `virchow` | `hf-hub:paige-ai/Virchow` | https://huggingface.co/paige-ai/Virchow (arXiv:2309.07778) | 门控。 |
| `virchow2` | `hf-hub:paige-ai/Virchow2` | https://huggingface.co/paige-ai/Virchow2 (arXiv:2408.00738) | 门控。 |
| `prov_gigapath` | `hf-hub:prov-gigapath/prov-gigapath` | https://github.com/prov-gigapath/prov-gigapath (Xu et al., *Nature* 2024) | 镜像上验证 HTTP 200。 |
| `lingshu_vision` | `hf-hub:lingshu-medical-mllm/Lingshu-7B` | https://arxiv.org/abs/2506.07044 ; https://github.com/alibaba-damo-academy/MedEvalKit | 完整 Qwen2.5-VL 检查点——无法直接由 timm 加载。通过 transformers 提取 `model.visual` 并作为 `pretrained_path` 传入。兄弟仓库：`hf-hub:lingshu-medical-mllm/Lingshu-32B`。 |
| `hulumed_vision` | `hf-hub:ZJU-AI4H/Hulu-Med-7B` | https://github.com/ZJUI-AI4H/Hulu-Med (arXiv:2510.08668) | 完整 MLLM——视觉塔已捆绑，需提取。无独立视觉构件。 |
| `healthgpt_vision` | `hf-hub:timm/vit_large_patch14_clip_336.openai` | https://github.com/DCDmllm/HealthGPT (Lin et al. 2025, arXiv:2502.09838) | HealthGPT 直接使用原始 OpenAI CLIP。HealthGPT 特有的 H-LoRA 适配器（仅 LLM 侧）在 `hf-hub:lintw/HealthGPT-M3`；未发布独立视觉塔权重。 |
| `llava_med_vision` | `hf-hub:microsoft/llava-med-v1.5-mistral-7b` | https://github.com/microsoft/LLaVA-Med | 通过 HF 镜像验证。 |
| `medgemma_vision` | `hf-hub:google/medgemma-4b-pt` | https://huggingface.co/google/medgemma-4b-pt | 镜像上验证 HTTP 200（同时提供 `medgemma-4b-it`）。 |
| `qwen25_vl_vision` | `hf-hub:Qwen/Qwen2.5-VL-7B-Instruct` | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct (arXiv:2502.13923) | 镜像上验证 HTTP 200。 |
| `qwen3_vl_vision` | `hf-hub:Qwen/Qwen3-VL-8B-Instruct` | https://github.com/QwenLM/Qwen3-VL | 镜像上验证 HTTP 200。 |

---

## 2. 仅 GitHub 发布（权重通过 GitHub release / Google Drive / CDN 分发，非 HF）

| encoder_key | 已验证 hf 路径 | 来源 URL（github/论文） | 说明 |
|---|---|---|---|
| `panderm` | 仅 GITHUB | https://github.com/SiyuanYan1/PanDerm | 权重通过 Google Drive：`panderm_ll_data6_checkpoint-499.pth`，地址 https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view。裸 PanDerm 编码器无公开 HF Hub 构件。`hf-mirror.com/pandermflow/PanDerm-Large-1` 返回 HTTP 401。 |
| `medclip` | 仅 GITHUB | https://github.com/RyanWangZf/MedCLIP (arXiv:2210.10163) | ViT 权重在 https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip（指针：https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt）。也可通过 `pip install medclip` 安装。 |
| `usfm` | 仅 GITHUB | https://github.com/openmedlab/USFM | 权重：`USFM_latest.pth`，通过 Google Drive https://drive.google.com/file/d/1KRwXZgYterH895Z8EpXpR1L1eSMMJo4q/view。仓库确认活跃（最后推送 2026-04-22）。 |
| `endo_fm` | 仅 GITHUB | https://github.com/med-air/Endo-FM | 权重通过 Google Drive：https://drive.google.com/file/d/1H7B91Ewm4QkZRsnUk1Bn0IQch5P8C7Xl/view?usp=sharing。 |
| `medklip` | 仅 GITHUB | https://github.com/MediaBrain-SJTU/MedKLIP | 官方权重在 Google Drive https://drive.google.com/drive/folders/1HBShH7J_pO8onkzuweDgDq2QPqj6zjG_。第三方 HF 镜像 `huggingface.co/youngzhou12/MedKLIP` 存在但为原始 `.pth`，非 timm hf-hub 构件。 |
| `ark` | 仅 GITHUB | https://github.com/jlianglab/Ark (*Nature* 2025: https://www.nature.com/articles/s41586-025-09079-8 ; MICCAI 2023: arXiv:2310.09507) | 权重通过 Google Form 申请分发 https://forms.gle/qkoDGXNiKRPTDdCe8——无 HF 构件且无公开 CDN；用户需下载 `.pth.tar` 并通过 `pretrained_path=` 传入。注意：旧路径 `jliu288/foundation_ark` 已 404；正确仓库为 `jlianglab/Ark`。 |
| `cxr_clip` | 仅 GITHUB | https://github.com/Soombit-ai/cxr-clip (arXiv:2310.13292, MICCAI 2023) | 权重在 KakaoCDN，如 https://twg.kakaocdn.net/brainrepo/models/cxr-clip/f982386ef38aa3ecd6ce1f8f928e8aa8/r50_m.tar。原 `kakaobrain/cxr-clip` 已转移至 `Soombit-ai/cxr-clip`。 |
| `visionfm` | 仅 GITHUB | https://github.com/ABILab-CUHK/VisionFM | 各模态 `.pth` 权重通过 README 中的 Google Drive 链接获取（如眼底：https://drive.google.com/file/d/13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4/view）。 |

---

## 3. 无公开权重（仅论文）

| encoder_key | 已验证 hf 路径 | 来源 URL（github/论文） | 说明 |
|---|---|---|---|
| `sonoclip` | 无 | — | HF 镜像 API 搜索 `sonoclip`（模型和数据集）返回 0 结果。未找到匹配的 GitHub 仓库。无公开构件。 |
| `busi_mae` | 无 | — | 无可验证的论文或仓库以 "BUSI-MAE" 品牌发布基础模型检查点。多篇 SSL 论文描述了在 BUSI 上微调 MAE 的方法，但均未以此名称发布基础模型检查点。 |
| `eyediff` | 无 | https://arxiv.org/abs/2411.10004 | 仅论文。仅引用 `huggingface/diffusers` DreamBooth 训练脚本；无已发布检查点，无含权重的 GitHub 仓库。 |

---

## 4. 调研结论不明确

（无——所有 39 条记录均已归入上述三类之一。）
