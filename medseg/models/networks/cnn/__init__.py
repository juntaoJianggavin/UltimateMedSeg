"""CNN-based complete segmentation architectures."""

from .unet3plus import UNet3Plus
from .lv_unet import LVUNet
from .ege_unet import EGEUNet
from .malunet import MALUNet
from .lite_unet import LiteUNet
from .mk_unet import MKUNet
from .u_lite import ULite
from .acc_unet import ACCUNet
from .cmunext import CMUNeXt
from .mew_unet import MEWUNet
# Note: ultralbm_unet was moved to ../mamba/ (its architecture is Mamba-based);
# it is no longer re-exported from this CNN subpackage.

# Re-export: DoubleUNet's canonical home is CNN (file lives under transformer/
# for historical reasons but architecturally it is CNN-based).
from medseg.models.networks.transformer.double_unet import DoubleUNet

# Self-contained ports from GitHub (all building blocks inline)
from .dcsaunet_model import DCSAUNet
from .cfanet_model import CFANet
from .attention_unet import AttentionUNet
from .unetpp_model import UNetPP
from .multiresunet import MultiResUNet
from .scseunet import SCSEUNet
from .resunet_a import ResUNetA
from .sa_unet import SAUNet
from .pan_model import PAN
from .denseunet import DenseUNet
from .linknet_model import LinkNet
from .pspnet_model import PSPNet
from .resunetpp_model import ResUNetPP
from .fr_unet_model import FRUNet
from .mednext_model import MedNeXt
from .nnunet_2d import NNUNet2D
from .r2unet import R2UNet
from .kiunet import KiUNet
from .aau_net import AAUNet
from .cmu_net import CMUNet
from .dscnet import DSCNet
from .stu_net import STUNet
from .dconnnet import DconnNet

# Domain-specific ports (2024-2026)
from .polyper import Polyper
from .hovernet_lite import HoverNetLite

__all__ = [
    "UNet3Plus", "LVUNet", "EGEUNet", "MALUNet",
    "LiteUNet", "MKUNet", "ULite",
    "ACCUNet", "CMUNeXt",
    "DoubleUNet",
    # Self-contained GitHub ports
    "DCSAUNet", "CFANet",
    "AttentionUNet", "UNetPP", "MultiResUNet", "SCSEUNet", "ResUNetA",
    "SAUNet", "PAN", "DenseUNet", "LinkNet", "PSPNet",
    "ResUNetPP", "FRUNet", "MedNeXt",
    "NNUNet2D", "R2UNet", "KiUNet",
    "AAUNet", "CMUNet", "DSCNet",
    "STUNet", "DconnNet",
    # Domain-specific ports (2024-2026)
    "Polyper",
]
