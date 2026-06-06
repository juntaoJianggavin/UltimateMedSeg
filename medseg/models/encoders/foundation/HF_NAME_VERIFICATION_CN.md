# 基础模型 HF 构件验证状态

[English](HF_NAME_VERIFICATION.md)

**日期：** 2026-06-03（已更新）

本文档已被 `CORRECTED_HF_PATHS.md` 取代，后者包含完整的已验证列表。
`medseg/encoders/foundation/` 中的所有编码器现在遵循严格策略：
`pretrained=True` 但未设置 `pretrained_path` 时将抛出
`NotImplementedError`（不会静默回退到其他权重）。

完整的已验证 HF 构件表请参见 `CORRECTED_HF_PATHS_CN.md`。

---

## 1. 已确认（构件存在且可访问/门控）

| encoder_key       | HF 构件                                  | 门控？ | 证据                                                                                  |
|-------------------|----------------------------------------------|--------|-------------------------------------------------------------------------------------------|
| huatuogpt_vision  | FreedomIntelligence/HuatuoGPT-Vision-7B      | 否     | HF API（通过 hf-mirror.com）返回 `id=FreedomIntelligence/HuatuoGPT-Vision-7B`           |

---

## 2. 未找到或不可验证（无实际构件路径；需要 `pretrained_path`）

| encoder_key          | 声称的名称（已移除）                                                  | 状态        |
|----------------------|-----------------------------------------------------------------------------|---------------|
| uni                  | hf-hub:MahmoodLab/UNI                                                       | 不可验证  |
| uni2                 | hf-hub:MahmoodLab/UNI-2                                                     | 不可验证  |
| conch                | hf-hub:MahmoodLab/CONCH                                                     | 不可验证  |
| phikon               | hf-hub:owkin/phikon                                                         | 不可验证  |
| phikon_v2            | hf-hub:owkin/phikon-v2                                                      | 不可验证  |
| virchow              | hf-hub:paige-ai/Virchow                                                     | 不可验证  |
| virchow2             | hf-hub:paige-ai/Virchow2                                                    | 不可验证  |
| prov_gigapath        | hf-hub:prov-gigapath/prov-gigapath                                          | 不可验证  |
| musk                 | hf-hub:xiangjx/musk                                                         | 不可验证  |
| plip                 | hf-hub:vinid/plip                                                           | 不可验证  |
| panderm              | hf-hub:pandermflow/PanDerm-Large-1                                          | 不可验证  |
| monet_derm           | hf-hub:suinleelab/monet                                                     | 不可验证  |
| dermclip             | hf-hub:dermclip/dermclip-vit-base-patch16-224                               | 不可验证  |
| biomedclip           | hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224             | 不可验证  |
| medclip              | hf-hub:RyanWW/MedCLIP-ViT                                                   | 不可验证  |
| keep                 | vit_base_patch16_clip_224（timm 名称，非 HF 仓库；无 MAGIC-AI4Med/KEEP） | 不可验证  |
| medsiglip            | hf-hub:google/medsiglip-448                                                 | 不可验证  |
| usfm                 | （仅 timm，无 HF）                                                          | 未找到     |
| sonoclip             | （仅 timm，无 HF）                                                          | 未找到     |
| busi_mae             | （仅 timm，无 HF）                                                          | 未找到     |
| endo_fm              | （仅 timm，无 HF）                                                          | 未找到     |
| endo_vit             | （仅 timm，无 HF）                                                          | 未找到     |
| colon_gpt_encoder    | （仅 timm，无 HF）                                                          | 未找到     |
| raddino              | hf-hub:microsoft/rad-dino                                                   | 不可验证  |
| medklip              | （仅 timm，无 HF）                                                          | 未找到     |
| ark                  | （仅 timm，无 HF）                                                          | 未找到     |
| cxr_clip             | （仅 timm，无 HF）                                                          | 未找到     |
| ctfm                 | （仅 timm，无 HF）                                                          | 未找到     |
| retfound             | hf-hub:rmaphoh/RETFound_oct_meh                                             | 不可验证  |
| flair                | （仅 timm，无 HF）                                                          | 未找到     |
| visionfm             | （仅 timm，无 HF）                                                          | 未找到     |
| eyediff              | hf-hub:CrazyBrick/EyeDiff                                                   | 不可验证  |
| llava_med_vision     | microsoft/llava-med-v1.5-mistral-7b                                         | 不可验证  |
| medgemma_vision      | google/medgemma-4b-it                                                       | 不可验证  |
| qwen25_vl_vision     | Qwen/Qwen2.5-VL-7B-Instruct                                                 | 不可验证  |
| qwen3_vl_vision      | Qwen/Qwen3-VL-7B                                                            | 不可验证  |
| lingshu_vision       | lingshu-medical/lingshu-7b                                                  | 不可验证  |
| hulumed_vision       | HuLuMed/HuLuMed-7B                                                          | 不可验证  |
| healthgpt_vision     | lintw/HealthGPT-M3                                                          | 不可验证  |

---

## 3. 说明

- 本仓库之前为部分编码器提交了**虚构的 HF 名称**。这些名称被硬编码为默认值，
  会导致静默下载失败（或更严重地，加载了恰好共享该路径的无关构件）。
- 这些编码器现在在 `pretrained=True` 且未设置 `pretrained_path` 时
  **抛出 `NotImplementedError`。** 它们不会静默回退到其他检查点。
- 设置 `pretrained=False` 将产生**随机初始化的 ViT-B/16**（或等效骨干）。
  这**不是真正的基础模型**，不得在任何实验中将其作为基础模型报告。
- 要使用真实模型：从**原始论文/项目页面**获取权重（如实验室 GitHub release、
  已获授权的门控 HF 仓库，或论文中的直接下载链接），并通过
  `pretrained_path=...` 传入本地路径。
- 仅 `huatuogpt_vision`（第 1 节）具有已验证的、可直接加载的 HF 构件。
