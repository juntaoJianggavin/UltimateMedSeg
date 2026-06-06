"""Knowledge Distillation for Medical Image Segmentation.

Each method links to a verifiable GitHub reference implementation in its
own file's header. Methods that diverged substantially from the official
source (or fabricated/non-existent papers) were removed.

Active methods:
    - unet_distillation  Hinton et al. logit/feature/attention/multi-scale KD
                         https://arxiv.org/abs/1503.02531
    - vanilla_kd         Hinton et al., NeurIPS workshop 2014
                         https://github.com/peterliht/knowledge-distillation-pytorch
    - hint_distillation  Romero et al., FitNets, ICLR 2015
                         https://github.com/adri-romsor/FitNets
    - attention_mimicry  Simplified attention mimicry baseline (kept for back-compat)
    - at                 Zagoruyko & Komodakis, Attention Transfer, ICLR 2017
                         https://github.com/szagoruyko/attention-transfer
    - fsp                Yim et al., A Gift from KD (FSP), CVPR 2017
                         https://github.com/yoshitomo-matsubara/torchdistill
    - nst                Huang & Wang, Neuron Selectivity Transfer, 2017
                         https://github.com/HobbitLong/RepDistiller
    - rkd                Park et al., Relational KD, CVPR 2019
                         https://github.com/lenscloth/RKD
    - vid                Ahn et al., Variational Information Distillation, CVPR 2019
                         https://github.com/HobbitLong/RepDistiller
    - dkd                Zhao et al., Decoupled KD, CVPR 2022
                         https://github.com/megvii-research/mdistiller
    - mgd                Yang et al., Masked Generative Distillation, ECCV 2022
                         https://github.com/yzd-v/MGD
    - dist               Huang et al., DIST, NeurIPS 2022
                         https://github.com/hunto/DIST_KD
    - cirkd_minibatch    Yang et al., CIRKD, CVPR 2022
                         https://github.com/winycg/CIRKD
    - cwd                Shu et al., Channel-wise Distillation, ICCV 2021
                         https://github.com/irfanICMLL/TorchDistiller
    - review_kd          Chen et al., Knowledge Review, CVPR 2021
                         https://github.com/dvlab-research/ReviewKD
    - simkd              Chen et al., SimKD, CVPR 2022
                         https://github.com/DefangChen/SimKD
    - norm_kd            Liu et al., NORM (normalised logits KD), ICLR 2023
                         https://github.com/xyliu7/NORM
    - sdd                Wei et al., Scale Decoupled Distillation, CVPR 2024
                         https://github.com/shicaiwei123/SDD-CVPR2024
    - aicsd              Mansurian et al., Adaptive Inter-Class Similarity
                         Distillation, TNNLS 2024
                         https://github.com/AmirMansurian/AICSD
    - logit_std_kd       Sun et al., Logit Standardization in KD, CVPR 2024
                         https://github.com/sunshangquan/logit-standardization-KD
    - ttm_kd             Zheng & Yang, Transformed Teacher Matching, ICLR 2024
                         https://github.com/zkxufo/TTM
    - ctkd               Li et al., Curriculum Temperature for KD, AAAI 2023
                         https://github.com/zhengli97/CTKD
    - mlkd               Jin et al., Multi-Level Logit Distillation, CVPR 2023
                         https://github.com/Jin-Ying/Multi-Level-Logit-Distillation
    - anatomy_kd         Anatomy-aware KD (medical, self-contained formulation)
    - boundary_kd        Boundary-aware KD (medical, self-contained formulation)
    - multi_organ_kd     Multi-organ class-balanced KD (medical)
    - cross_modality_kd  Cross-modality KD (medical, CT/MRI)
"""

from .feature_extractor import FeatureExtractor

from .unet_distillation import UNetDistillationLoss
from .vanilla_kd import VanillaKDLoss
from .hint_distillation import HintDistillationLoss
from .attention_mimicry import AttentionMimicryLoss
from .at import ATLoss
from .fsp import FSPLoss
from .nst import NSTLoss
from .rkd import RKDLoss
from .vid import VIDLoss
from .dkd import DKDLoss
from .mgd import MGDLoss
from .dist_kd import DISTLoss
from .cirkd_minibatch import CIRKDMiniBatchLoss
from .cwd import CWDLoss
from .review_kd import ReviewKDLoss
from .simkd import SimKDLoss
from .norm_kd import NormKDLoss
from .sdd import SDDLoss
from .aicsd import AICSDLoss
from .logit_std_kd import LogitStdKDLoss
from .ttm_kd import TTMKDLoss
from .ctkd import CTKDLoss
from .mlkd import MLKDLoss

from .anatomy_kd import AnatomyKDLoss
from .boundary_kd import BoundaryKDLoss
from .multi_organ_kd import MultiOrganKDLoss
from .cross_modality_kd import CrossModalityKDLoss

__all__ = [
    'FeatureExtractor',
    'UNetDistillationLoss',
    'VanillaKDLoss',
    'HintDistillationLoss',
    'AttentionMimicryLoss',
    'ATLoss',
    'FSPLoss',
    'NSTLoss',
    'RKDLoss',
    'VIDLoss',
    'DKDLoss',
    'MGDLoss',
    'DISTLoss',
    'CIRKDMiniBatchLoss',
    'CWDLoss',
    'ReviewKDLoss',
    'SimKDLoss',
    'NormKDLoss',
    'SDDLoss',
    'AICSDLoss',
    'LogitStdKDLoss',
    'TTMKDLoss',
    'CTKDLoss',
    'MLKDLoss',
    'AnatomyKDLoss',
    'BoundaryKDLoss',
    'MultiOrganKDLoss',
    'CrossModalityKDLoss',
]
