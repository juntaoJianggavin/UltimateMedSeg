"""Text-guided UNet variants (faithful 2D ports of official sources).

This subpackage hosts segmentation models that fuse a textual prompt
(class name / radiology caption / organ word embedding) with a U-shape
visual backbone end-to-end. Each model is a paper-formula port of an
officially published implementation; see the per-file header for the
exact source URL.

Per project policy: only 2D-native methods are included. 3D-native
methods (SegVol, Hermes, UniSeg, MA-SAM, ZePT, MedSAM2, etc.) were
removed because a 2D adaptation cannot be 99% faithful to a 3D paper.

Currently included:

- ``tganet``           Tomar et al., MICCAI 2022, "TGANet: Text-guided
                       Attention for Improved Polyp Segmentation"
                       https://github.com/nikhilroxtomar/TGANet
- ``lvit``             Li et al., TMI 2023, "LViT: Language meets Vision
                       Transformer in Medical Image Segmentation"
                       https://github.com/HUANGLIZI/LViT
- ``languide``         Zhong et al., MICCAI 2023, "Ariadne's Thread"
                       https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023
- ``clip_universal``   Liu et al., ICCV 2023, "CLIP-Driven Universal
                       Model for Organ Segmentation and Tumor Detection"
                       https://github.com/ljwztc/CLIP-Driven-Universal-Model
                       (2D port of the official UNet variant)
- ``cris``             Wang et al., CVPR 2022, "CRIS: CLIP-Driven
                       Referring Image Segmentation"
                       https://github.com/DerrickWang005/CRIS.pytorch
- ``biomedparse``      Zhao et al., Nature Methods 2024, BiomedParse
                       https://github.com/microsoft/BiomedParse
- ``tpro``             Zhang et al., MICCAI 2023, "TPRO"
                       https://github.com/shijun18/TPRO
- ``salip``            Aleem et al., BMVC 2024, "SaLIP"
                       https://github.com/aleemsidra/SaLIP
- ``causal_clipseg``   Chen et al., MICCAI 2024, "CausalCLIPSeg"
                       https://github.com/WUTCM-Lab/CausalCLIPSeg
- ``medclip_sam``      Koleilat et al., MICCAI 2024, "MedCLIP-SAM"
                       https://github.com/HealthX-Lab/MedCLIP-SAM
Prompt-guided (non-text) models like SAM-Med2D and LiteMedSAM have been
moved to ``medseg/networks/sam/`` where the rest of the SAM family lives.
"""

from medseg.models.text_unet.tganet import TGANet
from medseg.models.text_unet.lvit import LViT
from medseg.models.text_unet.languide import LanGuideMedSeg
from medseg.models.text_unet.clip_universal import CLIPDrivenUniversalModel2D
from medseg.models.text_unet.cris import CRIS
from medseg.models.text_unet.biomedparse import BiomedParse
from medseg.models.text_unet.tpro import TPRO
from medseg.models.text_unet.salip import SaLIP
from medseg.models.text_unet.causal_clipseg import CausalCLIPSeg
from medseg.models.text_unet.medclip_sam import MedCLIPSAM
from medseg.models.text_unet.tp_drseg import TPDRSeg
from medseg.models.text_unet.cxrclipseg import CXRCLIPSeg


TEXT_UNET_REGISTRY = {
    "tganet": TGANet,
    "lvit": LViT,
    "languide": LanGuideMedSeg,
    "clip_universal": CLIPDrivenUniversalModel2D,
    "cris": CRIS,
    "biomedparse": BiomedParse,
    "tpro": TPRO,
    "salip": SaLIP,
    "causal_clipseg": CausalCLIPSeg,
    "medclip_sam": MedCLIPSAM,
    "tp_drseg": TPDRSeg,
    "cxrclipseg": CXRCLIPSeg,
}


__all__ = [
    "TGANet",
    "LViT",
    "LanGuideMedSeg",
    "CLIPDrivenUniversalModel2D",
    "CRIS",
    "BiomedParse",
    "TPRO",
    "SaLIP",
    "CausalCLIPSeg",
    "MedCLIPSAM",
    "TPDRSeg",
    "CXRCLIPSeg",
    "TEXT_UNET_REGISTRY",
]
