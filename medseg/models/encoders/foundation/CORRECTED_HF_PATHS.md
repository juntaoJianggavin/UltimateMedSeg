# Corrected HuggingFace Paths for Foundation Encoders

[中文文档](CORRECTED_HF_PATHS_CN.md)

Aggregated from per-encoder web research. Each entry was verified via direct HTTP probes against `hf-mirror.com`, the GitHub API, and/or canonical paper sources (native `web_search` was blocked by the harness).

---

## 1. Confirmed public HF artifact (`hf-hub:...`)

| encoder_key | verified_hf_path | source URL (github/paper) | notes |
|---|---|---|---|
| `monet_derm` | `hf-hub:suinleelab/monet` | https://github.com/suinleelab/MONET (Kim et al., *Nature Medicine* 30:1154-1165, 2024) | Verified HTTP 200 on hf-mirror. |
| `dermclip` | `hf-hub:redlessone/DermLIP_ViT-B-16` | https://github.com/SiyuanYan1/Derm1M (arXiv:2503.14911, ICCV'25 Highlight) | Confirmed via HF mirror API; no DermCLIP alias — published under `redlessone/DermLIP_ViT-B-16`. |
| `keep` | `hf-hub:Astaxanthin/KEEP` | https://github.com/MAGIC-AI4Med/KEEP (arXiv:2412.13126; *Cancer Cell* 2026) | Custom `KEEPModel` — load with `transformers.AutoModel.from_pretrained('Astaxanthin/KEEP', trust_remote_code=True)`. The bare `timm` `hf-hub:` shortcut does NOT work; download `model.safetensors` and pass as `pretrained_path=`, or extract `.visual` state_dict from the HF model. |
| `medsiglip` | `hf-hub:google/medsiglip-448` | https://huggingface.co/google/medsiglip-448 | Verified HTTP 200 on mirror. |
| `endo_vit` | `hf-hub:egeozsoy/EndoViT` | https://github.com/DominikBatic/EndoViT (official) / https://huggingface.co/egeozsoy/EndoViT / https://doi.org/10.1007/s11548-024-03091-5 | File is `pytorch_model.bin` with key `"model"`. NOT a timm-HF-hub native artifact — load via `huggingface_hub.snapshot_download`. |
| `colon_gpt_encoder` | `hf-hub:timm/ViT-SO400M-14-SigLIP-384` | https://github.com/ai4colonoscopy/IntelliScope (arXiv:2410.17241) | This is the SigLIP-SO400M/14-384 vision tower used by ColonGPT. Full multimodal model at `hf-hub:ai4colonoscopy/ColonGPT-v1` in a custom `ColongptPhi` format — `vision_tower.*` keys must be extracted manually. |
| `ctfm` | `hf-hub:project-lighter/ct_fm_feature_extractor` | https://github.com/project-lighter/CT-FM (arXiv:2501.09001) | Pai et al., "CT-FM: A 3D Image-Based Foundation Model for Computed Tomography", 2025. |
| `raddino` | `hf-hub:microsoft/rad-dino` | https://huggingface.co/microsoft/rad-dino (Pérez-García et al., Microsoft Research, 2024) | Verified HTTP 200 on mirror. |
| `retfound` | `hf-hub:YukunZhou/RETFound_mae_natureCFP` (fundus/cfp) and `hf-hub:YukunZhou/RETFound_mae_natureOCT` (oct) | https://github.com/rmaphoh/RETFound_MAE (Zhou et al., *Nature* 2023) | Both gated — request access required. |
| `flair` | `hf-hub:jusiro2/FLAIR` | https://github.com/jusiro/FLAIR | README documents `FLAIRModel.from_pretrained("jusiro2/FLAIR")`. |
| `plip` | `hf-hub:vinid/plip` | https://github.com/PathologyFoundation/plip | Verified HF mirror API record. |
| `musk` | `hf-hub:xiangjx/musk` | https://github.com/lilab-bcb/MUSK (Xiang et al., *Nature* 2025) | Verified HTTP 200; tree contains README + weights. |
| `uni` | `hf-hub:MahmoodLab/UNI` | https://huggingface.co/MahmoodLab/UNI (https://www.nature.com/articles/s41591-024-02857-3) | Verified HTTP 200 on mirror. |
| `uni2` | `hf-hub:MahmoodLab/UNI2-h` | https://huggingface.co/MahmoodLab/UNI2-h (Chen et al., *Nat. Med.* 2024; https://github.com/mahmoodlab/UNI) | Gated. |
| `conch` | `hf-hub:MahmoodLab/CONCH` | https://github.com/mahmoodlab/CONCH | Verified HTTP 200 on mirror + GitHub. |
| `phikon` | `hf-hub:owkin/phikon` | https://huggingface.co/owkin/phikon (Filiot et al., medRxiv 10.1101/2023.07.21.23292757) | Verified HTTP 200 on mirror. |
| `phikon_v2` | `hf-hub:owkin/phikon-v2` | https://huggingface.co/owkin/phikon-v2 (arXiv:2409.09173) | Verified via HF mirror. |
| `virchow` | `hf-hub:paige-ai/Virchow` | https://huggingface.co/paige-ai/Virchow (arXiv:2309.07778) | Gated. |
| `virchow2` | `hf-hub:paige-ai/Virchow2` | https://huggingface.co/paige-ai/Virchow2 (arXiv:2408.00738) | Gated. |
| `prov_gigapath` | `hf-hub:prov-gigapath/prov-gigapath` | https://github.com/prov-gigapath/prov-gigapath (Xu et al., *Nature* 2024) | Verified HTTP 200 on mirror. |
| `lingshu_vision` | `hf-hub:lingshu-medical-mllm/Lingshu-7B` | https://arxiv.org/abs/2506.07044 ; https://github.com/alibaba-damo-academy/MedEvalKit | Full Qwen2.5-VL checkpoint — not directly loadable by timm. Extract `model.visual` via transformers and pass as `pretrained_path`. Sibling: `hf-hub:lingshu-medical-mllm/Lingshu-32B`. |
| `hulumed_vision` | `hf-hub:ZJU-AI4H/Hulu-Med-7B` | https://github.com/ZJUI-AI4H/Hulu-Med (arXiv:2510.08668) | Full MLLM — vision tower is bundled and must be extracted. No standalone vision artifact. |
| `healthgpt_vision` | `hf-hub:timm/vit_large_patch14_clip_336.openai` | https://github.com/DCDmllm/HealthGPT (Lin et al. 2025, arXiv:2502.09838) | HealthGPT uses stock OpenAI CLIP unchanged. HealthGPT-specific H-LoRA adapters (LLM side only) at `hf-hub:lintw/HealthGPT-M3`; no separate vision-tower weights published. |
| `llava_med_vision` | `hf-hub:microsoft/llava-med-v1.5-mistral-7b` | https://github.com/microsoft/LLaVA-Med | Verified via HF mirror. |
| `medgemma_vision` | `hf-hub:google/medgemma-4b-pt` | https://huggingface.co/google/medgemma-4b-pt | Verified HTTP 200 on mirror (also `medgemma-4b-it`). |
| `qwen25_vl_vision` | `hf-hub:Qwen/Qwen2.5-VL-7B-Instruct` | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct (arXiv:2502.13923) | Verified HTTP 200 on mirror. |
| `qwen3_vl_vision` | `hf-hub:Qwen/Qwen3-VL-8B-Instruct` | https://github.com/QwenLM/Qwen3-VL | Verified HTTP 200 on mirror. |

---

## 2. GitHub-only release (weights distributed via GitHub release / Google Drive / CDN, not HF)

| encoder_key | verified_hf_path | source URL (github/paper) | notes |
|---|---|---|---|
| `panderm` | GITHUB_ONLY | https://github.com/SiyuanYan1/PanDerm | Weights via Google Drive: `panderm_ll_data6_checkpoint-499.pth` at https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view. No public HF Hub artifact for the bare PanDerm encoder. `hf-mirror.com/pandermflow/PanDerm-Large-1` returned HTTP 401. |
| `medclip` | GITHUB_ONLY | https://github.com/RyanWangZf/MedCLIP (arXiv:2210.10163) | ViT weights at https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip (pointer: https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt). Also installable via `pip install medclip`. |
| `usfm` | GITHUB_ONLY | https://github.com/openmedlab/USFM | Weights: `USFM_latest.pth` via Google Drive https://drive.google.com/file/d/1KRwXZgYterH895Z8EpXpR1L1eSMMJo4q/view. Repo confirmed live (last push 2026-04-22). |
| `endo_fm` | GITHUB_ONLY | https://github.com/med-air/Endo-FM | Weights via Google Drive: https://drive.google.com/file/d/1H7B91Ewm4QkZRsnUk1Bn0IQch5P8C7Xl/view?usp=sharing. |
| `medklip` | GITHUB_ONLY | https://github.com/MediaBrain-SJTU/MedKLIP | Official weights on Google Drive https://drive.google.com/drive/folders/1HBShH7J_pO8onkzuweDgDq2QPqj6zjG_. Third-party HF mirror of a retrained ckpt at `huggingface.co/youngzhou12/MedKLIP` exists but is a raw `.pth`, not a timm hf-hub artifact. |
| `ark` | GITHUB_ONLY | https://github.com/jlianglab/Ark (*Nature* 2025: https://www.nature.com/articles/s41586-025-09079-8 ; MICCAI 2023: arXiv:2310.09507) | Weights distributed by request via Google Form https://forms.gle/qkoDGXNiKRPTDdCe8 — no HF artifact and no public CDN; user must download `.pth.tar` and pass `pretrained_path=`. Note: legacy `jliu288/foundation_ark` path is 404; correct repo is `jlianglab/Ark`. |
| `cxr_clip` | GITHUB_ONLY | https://github.com/Soombit-ai/cxr-clip (arXiv:2310.13292, MICCAI 2023) | Weights on KakaoCDN, e.g. https://twg.kakaocdn.net/brainrepo/models/cxr-clip/f982386ef38aa3ecd6ce1f8f928e8aa8/r50_m.tar. Original `kakaobrain/cxr-clip` was transferred to `Soombit-ai/cxr-clip`. |
| `visionfm` | GITHUB_ONLY | https://github.com/ABILab-CUHK/VisionFM | Per-modality `.pth` weights via Google Drive links in the README (e.g. Fundus: https://drive.google.com/file/d/13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4/view). |

---

## 3. No public weights (paper-only)

| encoder_key | verified_hf_path | source URL (github/paper) | notes |
|---|---|---|---|
| `sonoclip` | NONE | — | HF mirror API returned 0 results for `sonoclip` (models and datasets). No matching GitHub repo found. No public artifact. |
| `busi_mae` | NONE | — | No verifiable paper or repository for a model branded "BUSI-MAE". Generic MAE-on-BUSI fine-tuning is described in several SSL papers but none release a foundation-model checkpoint under this name. |
| `eyediff` | NONE | https://arxiv.org/abs/2411.10004 | Paper-only. Only references `huggingface/diffusers` DreamBooth training script; no released checkpoint, no GitHub repo with weights. |

---

## 4. Research inconclusive

(none — all 39 entries resolved into one of the three categories above.)
