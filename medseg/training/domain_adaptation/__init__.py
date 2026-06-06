"""Domain Adaptation losses for medical image segmentation.

Each method links to a verifiable GitHub reference implementation in its
own file's header. Methods whose implementation diverged from the official
source by more than a small amount were removed (see git history).

Active methods:
    - source_only  : Lower-bound baseline (CE+Dice on source, no DA)
                     https://github.com/valeoai/ADVENT  (baseline convention)
    - advent       : Vu et al., CVPR 2019
                     https://github.com/valeoai/ADVENT
    - dann         : Ganin et al., JMLR 2016
                     https://github.com/fungtion/DANN
    - tent         : Wang et al., ICLR 2021
                     https://github.com/DequanWang/tent
    - dpl          : Chen et al., MICCAI 2021
                     https://github.com/cchen-cc/SFDA-DPL
    - cbmt         : Class-Balanced Mean Teacher (ADA4MIA benchmark)
                     https://github.com/whq-xxh/ADA4MIA
    - fda          : Yang & Soatto, CVPR 2020
                     https://github.com/YanchaoYang/FDA
    - crst         : Zou et al., ICCV 2019
                     https://github.com/yzou2/CRST
    - pixmatch     : Melas-Kyriazi & Manrai, CVPR 2021
                     https://github.com/lukemelas/pixmatch
    - mic          : Hoyer et al., CVPR 2023 (Masked Image Consistency)
                     https://github.com/lhoyer/MIC
    - daformer_fd  : Hoyer et al., CVPR 2022 (DAFormer PL CE + RCS + L_FD proxy)
                     https://github.com/lhoyer/DAFormer
    - hrda         : Hoyer et al., ECCV 2022 (multi-resolution scale attention)
                     https://github.com/lhoyer/HRDA
    - pipa         : Chen et al., ACM MM 2023 (pixel + patch InfoNCE)
                     https://github.com/chen742/PiPa
    - ddb          : Du et al., CVPR 2023 (dual-domain decoupled bridging)
                     https://github.com/xinyuelll/DDB
    - sepico       : Xie et al., TPAMI 2023 (semantic-guided pixel contrast
                     with persistent class prototypes + distributional KL)
                     https://github.com/BIT-DA/SePiCo
    - diga         : Shen et al., CVPR 2023 (distillation-guided adaptation;
                     symmetric KL + class-balanced soft-CE refinement)
                     https://github.com/BIT-DA/DiGA
    - micdrop      : Hoyer et al., ECCV 2024 (Masked Image Consistency +
                     complementary feature dropout)
                     https://github.com/lhoyer/MICDrop
    - semivl_da    : Karazija et al., ECCV 2024 (vision-language guided
                     self-training; class-text prototype InfoNCE adapted
                     to UDA, no-CLIP "trainable prototype" variant)
                     https://github.com/google-research/semivl
"""

from .advent import AdvEntLoss
from .tent import TentLoss
from .dpl import DPLLoss
from .cbmt import CBMTLoss
from .dann import DANNLoss
from .source_only import SourceOnlyLoss
from .fda import FDALoss
from .crst import CRSTLoss
from .pixmatch import PixMatchLoss
from .mic import MICLoss
from .daformer import DAFormerFDLoss
from .hrda import HRDALoss
from .pipa import PiPaLoss
from .ddb import DDBLoss
from .sepico import SePiCoLoss
from .diga import DiGALoss
from .micdrop import MICDropLoss
from .semivl import SemiVLLoss

__all__ = [
    'AdvEntLoss',
    'TentLoss',
    'DPLLoss',
    'CBMTLoss',
    'DANNLoss',
    'SourceOnlyLoss',
    'FDALoss',
    'CRSTLoss',
    'PixMatchLoss',
    'MICLoss',
    'DAFormerFDLoss',
    'HRDALoss',
    'PiPaLoss',
    'DDBLoss',
    'SePiCoLoss',
    'DiGALoss',
    'MICDropLoss',
    'SemiVLLoss',
]
