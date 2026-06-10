"""Semi-supervised segmentation methods.

Each method here corresponds to a yaml in ``configs/training_paradigms/semi_supervision/`` and to a
verifiable GitHub reference implementation (linked in each file's header).
Methods whose implementation could not be made ~99% faithful to a published
reference were removed; see git history if you need to recover one.

Active methods:
    - mean_teacher         Tarvainen & Valpola, NeurIPS 2017
                           https://github.com/CuriousAI/mean-teacher
    - cps                  Chen et al., CVPR 2021
                           https://github.com/charlesCXK/TorchSemiSeg
    - cct                  Ouali et al., BMVC 2020
                           https://github.com/yassouali/CCT
    - unimatch             Yang et al., CVPR 2023
                           https://github.com/LiheYoung/UniMatch
    - fixmatch             Sohn et al., NeurIPS 2020 (seg variant via SSL4MIS)
                           https://github.com/HiLab-git/SSL4MIS
    - urpc                 Luo et al., MIA 2022
                           https://github.com/HiLab-git/SSL4MIS
    - deep_co_training     Qiao et al., ECCV 2018
                           https://github.com/AlanChou/Deep-Co-Training-for-Semi-Supervised-Image-Recognition
    - flexmatch            Zhang et al., NeurIPS 2021
                           https://github.com/TorchSSL/TorchSSL
    - softmatch            Chen et al., ICLR 2023
                           https://github.com/microsoft/Semi-supervised-learning
    - freematch            Wang et al., ICLR 2023
                           https://github.com/microsoft/Semi-supervised-learning
    - ua_mt                Yu et al., MICCAI 2019 (MC-Dropout uncertainty)
                           https://github.com/yulequan/UA-MT
    - ssl4mis_u            SSL4MIS uncertainty (MC-Dropout) variant
                           https://github.com/HiLab-git/SSL4MIS
    - pi_model             Laine & Aila, ICLR 2017 (Π-model)
                           https://github.com/smlaine2/tempens
    - temporal_ensembling  Laine & Aila, ICLR 2017 (per-sample EMA target)
                           https://github.com/smlaine2/tempens
    - pseudo_label         Lee, ICML Workshop 2013
                           https://github.com/iBelieveCJM/pseudo_label-pytorch
    - ict                  Verma et al., IJCAI 2019 (Interpolation Consistency)
                           https://github.com/vikasverma1077/ICT
    - r_drop               Wu et al., NeurIPS 2021 (Regularized Dropout)
                           https://github.com/dropreg/R-Drop
    - cross_teaching       Luo et al., MIDL 2022 (CNN ↔ Transformer)
                           https://github.com/HiLab-git/SSL4MIS
    - corrmatch            Sun et al., CVPR 2024 (Correlation matching)
                           https://github.com/BBBBchan/CorrMatch
    - allspark             Wang et al., CVPR 2024 (Reborn labeled tokens)
                           https://github.com/xmed-lab/AllSpark
    - diffrect             Liu et al., MICCAI 2024 (Latent-diffusion PL rectification)
                           https://github.com/CUHK-AIM-Group/DiffRect
"""

from .base import BaseSemiMethod
from .mean_teacher import MeanTeacher
from .cps import CrossPseudoSupervision
from .cct import CrossConsistencyTraining
from .unimatch import UniMatch
from .fixmatch import FixMatch
from .urpc import URPC
from .deep_co_training import DeepCoTraining
from .flexmatch import FlexMatch
from .softmatch import SoftMatch
from .freematch import FreeMatch
from .ua_mt import UncertaintyAwareMeanTeacher
from .ssl4mis_u import SSL4MISUncertainty
from .pi_model import PiModel
from .temporal_ensembling import TemporalEnsembling
from .pseudo_label import PseudoLabel
from .ict import InterpolationConsistencyTraining
from .r_drop import RDrop
from .cross_teaching import CrossTeaching
from .corrmatch import CorrMatch
from .allspark import AllSpark
from .diffrect import DiffRect


_SEMI_METHODS = {
    "mean_teacher": MeanTeacher,
    "cps": CrossPseudoSupervision,
    "cct": CrossConsistencyTraining,
    "unimatch": UniMatch,
    "fixmatch": FixMatch,
    "urpc": URPC,
    "deep_co_training": DeepCoTraining,
    "flexmatch": FlexMatch,
    "softmatch": SoftMatch,
    "freematch": FreeMatch,
    "ua_mt": UncertaintyAwareMeanTeacher,
    "ssl4mis_u": SSL4MISUncertainty,
    "pi_model": PiModel,
    "temporal_ensembling": TemporalEnsembling,
    "pseudo_label": PseudoLabel,
    "ict": InterpolationConsistencyTraining,
    "r_drop": RDrop,
    "cross_teaching": CrossTeaching,
    "corrmatch": CorrMatch,
    "allspark": AllSpark,
    "diffrect": DiffRect,
}


__all__ = [
    "BaseSemiMethod",
    "MeanTeacher",
    "CrossPseudoSupervision",
    "CrossConsistencyTraining",
    "UniMatch",
    "FixMatch",
    "URPC",
    "DeepCoTraining",
    "FlexMatch",
    "SoftMatch",
    "FreeMatch",
    "UncertaintyAwareMeanTeacher",
    "SSL4MISUncertainty",
    "PiModel",
    "TemporalEnsembling",
    "PseudoLabel",
    "InterpolationConsistencyTraining",
    "RDrop",
    "CrossTeaching",
    "CorrMatch",
    "AllSpark",
    "DiffRect",
    "build_semi_method",
    "_SEMI_METHODS",
]


def build_semi_method(semi_cfg: dict, model, device, img_size: int = 224) -> BaseSemiMethod:
    """Build a semi-supervised method from config."""
    method_name = semi_cfg.get("method", "mean_teacher")
    if method_name not in _SEMI_METHODS:
        available = ", ".join(sorted(_SEMI_METHODS.keys()))
        raise KeyError(f"Unknown semi method '{method_name}'. Available: [{available}]")
    params = semi_cfg.get("params", {})
    cls = _SEMI_METHODS[method_name]
    method = cls(model=model, device=device, img_size=img_size, **params)
    method.build()
    return method
