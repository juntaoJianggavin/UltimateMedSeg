"""Weakly Supervised Segmentation methods for medical images.

Retained methods (each maps to a documented paper, the implementation is
the version actually exercised by ``train_weakly_supervised.py``):

    - BoxSupervised : Box-only mask + foreground/background CE
                      (BoxSup / BoxInst family, MIL projection variant)
    - CAM           : Class Activation Mapping with Grad-CAM hooks
                      (Zhou et al., CVPR 2016 / Selvaraju et al., ICCV 2017)
    - MIL           : Multi-instance learning from image-level labels
    - EMPseudoLabel : EM refinement of pseudo masks from weak labels
    - Point         : Bearman et al., ECCV 2016 point supervision
    - GatedCRF      : Obukhov et al., NeurIPS 2019 differentiable CRF
    - Affinity      : Pixel affinity propagation (AffinityNet style)
    - TreeEnergy    : Tree-structured energy minimization
    - SEAM          : Wang et al., CVPR 2020 self-supervised equivariant attention
    - PuzzleCAM     : Jo & Yu, ICIP 2021 puzzle piece matching
    - AdvCAM        : Lee et al., CVPR 2021 adversarial complementary erasing
    - MCTformer     : Xu et al., CVPR 2022 multi-class token transformer
    - SAMGuidedWeak : SAM-guided pseudo-mask refinement
    - iSeg / ClickSupervision : interactive click-based supervision

Methods that previously lived here (GrabCut, fBRS, RITM, SimpleClick,
Scribble, SeCo, DuPL, CTI, WeCLIP, S2C, DiG, PCSS, GazeMedSeg, SimTxtSeg,
ExCEL, IRNet, AuxSeg) were removed because their implementations diverged
substantially from the originating papers (missing GMM/graph-cut/feature
back-prop/frozen CLIP/SAM/text encoders) and could not be exercised
faithfully without adding entire model branches and dataloader streams.
"""

from .cam_generator import CAMGenerator
from .box_supervised import BoxSupervisedLoss
from .cam import CAMLoss
from .mil import MILLoss
from .em_pseudo_label import EMPseudoLabelLoss
from .point_supervised import PointSupervisedLoss
from .gated_crf import GatedCRFLoss
from .affinity import AffinityLoss
from .tree_energy import TreeEnergyLoss
from .seam import SEAMLoss
from .puzzle_cam import PuzzleCAMLoss
from .adv_cam import AdvCAMLoss
from .mctformer import MCTformerLoss
from .sam_guided_weak import SAMGuidedWeakLoss
from .interactive import fBRSLoss, iSegLoss, ClickSupervisionLoss
from .eps import EPSLoss
from .boxinst import BoxInstLoss
from .scribble_sup import ScribbleSupLoss
from .recam import ReCAMLoss
from .toco import ToCoLoss
from .lpcam import LPCAMLoss
from .mars import MARSLoss
from .bacon import BACoNLoss
from .wpgseg import WPGSegLoss
from .dupl import DuPLLoss
from .more import MoReLoss
from .psdpm import PSDPMLoss
from .semples import SemPLeSLoss

__all__ = [
    'CAMGenerator',
    'BoxSupervisedLoss',
    'CAMLoss',
    'MILLoss',
    'EMPseudoLabelLoss',
    'PointSupervisedLoss',
    'GatedCRFLoss',
    'AffinityLoss',
    'TreeEnergyLoss',
    'SEAMLoss',
    'PuzzleCAMLoss',
    'AdvCAMLoss',
    'MCTformerLoss',
    'SAMGuidedWeakLoss',
    'fBRSLoss',
    'iSegLoss',
    'ClickSupervisionLoss',
    'EPSLoss',
    'BoxInstLoss',
    'ScribbleSupLoss',
    'ReCAMLoss',
    'ToCoLoss',
    'LPCAMLoss',
    'MARSLoss',
    'BACoNLoss',
    'WPGSegLoss',
    'DuPLLoss',
    'MoReLoss',
    'PSDPMLoss',
    'SemPLeSLoss',
]
