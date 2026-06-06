# Foundation Model HF Artifact Verification Status

[中文文档](HF_NAME_VERIFICATION_CN.md)

**Date:** 2026-06-03 (updated)

This document has been superseded by `CORRECTED_HF_PATHS.md` which contains
the full verified list. All encoders in `medseg/encoders/foundation/` now
follow a strict policy: `pretrained=True` without `pretrained_path` raises
`NotImplementedError` (no silent fallback to other weights).

See `CORRECTED_HF_PATHS.md` for the complete verified HF artifact table.

---

## 1. CONFIRMED (artifact exists and is accessible / gated)

| encoder_key       | HF artifact                                  | gated? | evidence                                                                                  |
|-------------------|----------------------------------------------|--------|-------------------------------------------------------------------------------------------|
| huatuogpt_vision  | FreedomIntelligence/HuatuoGPT-Vision-7B      | no     | HF API (via hf-mirror.com) returns `id=FreedomIntelligence/HuatuoGPT-Vision-7B`           |

---

## 2. NOT_FOUND or UNVERIFIABLE (no real artifact path; `pretrained_path` required)

| encoder_key          | claimed name (now removed)                                                  | status        |
|----------------------|-----------------------------------------------------------------------------|---------------|
| uni                  | hf-hub:MahmoodLab/UNI                                                       | UNVERIFIABLE  |
| uni2                 | hf-hub:MahmoodLab/UNI-2                                                     | UNVERIFIABLE  |
| conch                | hf-hub:MahmoodLab/CONCH                                                     | UNVERIFIABLE  |
| phikon               | hf-hub:owkin/phikon                                                         | UNVERIFIABLE  |
| phikon_v2            | hf-hub:owkin/phikon-v2                                                      | UNVERIFIABLE  |
| virchow              | hf-hub:paige-ai/Virchow                                                     | UNVERIFIABLE  |
| virchow2             | hf-hub:paige-ai/Virchow2                                                    | UNVERIFIABLE  |
| prov_gigapath        | hf-hub:prov-gigapath/prov-gigapath                                          | UNVERIFIABLE  |
| musk                 | hf-hub:xiangjx/musk                                                         | UNVERIFIABLE  |
| plip                 | hf-hub:vinid/plip                                                           | UNVERIFIABLE  |
| panderm              | hf-hub:pandermflow/PanDerm-Large-1                                          | UNVERIFIABLE  |
| monet_derm           | hf-hub:suinleelab/monet                                                     | UNVERIFIABLE  |
| dermclip             | hf-hub:dermclip/dermclip-vit-base-patch16-224                               | UNVERIFIABLE  |
| biomedclip           | hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224             | UNVERIFIABLE  |
| medclip              | hf-hub:RyanWW/MedCLIP-ViT                                                   | UNVERIFIABLE  |
| keep                 | vit_base_patch16_clip_224 (timm name, not an HF repo; no MAGIC-AI4Med/KEEP) | UNVERIFIABLE  |
| medsiglip            | hf-hub:google/medsiglip-448                                                 | UNVERIFIABLE  |
| usfm                 | (timm_only, no HF)                                                          | NOT_FOUND     |
| sonoclip             | (timm_only, no HF)                                                          | NOT_FOUND     |
| busi_mae             | (timm_only, no HF)                                                          | NOT_FOUND     |
| endo_fm              | (timm_only, no HF)                                                          | NOT_FOUND     |
| endo_vit             | (timm_only, no HF)                                                          | NOT_FOUND     |
| colon_gpt_encoder    | (timm_only, no HF)                                                          | NOT_FOUND     |
| raddino              | hf-hub:microsoft/rad-dino                                                   | UNVERIFIABLE  |
| medklip              | (timm_only, no HF)                                                          | NOT_FOUND     |
| ark                  | (timm_only, no HF)                                                          | NOT_FOUND     |
| cxr_clip             | (timm_only, no HF)                                                          | NOT_FOUND     |
| ctfm                 | (timm_only, no HF)                                                          | NOT_FOUND     |
| retfound             | hf-hub:rmaphoh/RETFound_oct_meh                                             | UNVERIFIABLE  |
| flair                | (timm_only, no HF)                                                          | NOT_FOUND     |
| visionfm             | (timm_only, no HF)                                                          | NOT_FOUND     |
| eyediff              | hf-hub:CrazyBrick/EyeDiff                                                   | UNVERIFIABLE  |
| llava_med_vision     | microsoft/llava-med-v1.5-mistral-7b                                         | UNVERIFIABLE  |
| medgemma_vision      | google/medgemma-4b-it                                                       | UNVERIFIABLE  |
| qwen25_vl_vision     | Qwen/Qwen2.5-VL-7B-Instruct                                                 | UNVERIFIABLE  |
| qwen3_vl_vision      | Qwen/Qwen3-VL-7B                                                            | UNVERIFIABLE  |
| lingshu_vision       | lingshu-medical/lingshu-7b                                                  | UNVERIFIABLE  |
| hulumed_vision       | HuLuMed/HuLuMed-7B                                                          | UNVERIFIABLE  |
| healthgpt_vision     | lintw/HealthGPT-M3                                                          | UNVERIFIABLE  |

---

## 3. NOTES

- This repository previously committed **fabricated HF names** for several of
  the encoders listed in section 2. Those names were hard-coded as defaults
  and would have caused silent download failures (or, worse, loading of an
  unrelated artifact happening to share the path).
- Those encoders now **raise `NotImplementedError` when `pretrained=True` and
  no `pretrained_path` is set.** They will not silently fall back to a
  different checkpoint.
- Setting `pretrained=False` yields a **random-init ViT-B/16** (or equivalent
  backbone). This is **NOT the real foundation model** and must not be
  reported as such in any experiment.
- To use the real model: obtain the weights from the **original paper / project
  page** (e.g. the lab's GitHub release, a gated HF repo to which you have
  been granted access, or a direct download link in the publication) and pass
  the local path via `pretrained_path=...`.
- Only `huatuogpt_vision` (section 1) has a verified, directly-loadable HF
  artifact at the recorded identifier.
