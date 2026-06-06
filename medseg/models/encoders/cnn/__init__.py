"""CNN-family encoders."""

import sys as _sys

from . import basic_encoder
_sys.modules['medseg.models.encoders.basic_encoder'] = basic_encoder
from . import dcsaunet_encoder
_sys.modules['medseg.models.encoders.dcsaunet_encoder'] = dcsaunet_encoder
from . import cfanet_encoder
_sys.modules['medseg.models.encoders.cfanet_encoder'] = cfanet_encoder
from . import mednext_encoder
_sys.modules['medseg.models.encoders.mednext_encoder'] = mednext_encoder
from . import convnext_encoder
_sys.modules['medseg.models.encoders.convnext_encoder'] = convnext_encoder
from . import efficientnetv2_encoder
_sys.modules['medseg.models.encoders.efficientnetv2_encoder'] = efficientnetv2_encoder
from . import attention_unet_encoder
_sys.modules['medseg.models.encoders.attention_unet_encoder'] = attention_unet_encoder
from . import r2unet_encoder
_sys.modules['medseg.models.encoders.r2unet_encoder'] = r2unet_encoder
from . import mew_encoder
_sys.modules['medseg.models.encoders.mew_encoder'] = mew_encoder
from . import lv_unet_encoder
_sys.modules['medseg.models.encoders.lv_unet_encoder'] = lv_unet_encoder
from . import malunet_encoder
_sys.modules['medseg.models.encoders.malunet_encoder'] = malunet_encoder
from . import ege_unet_encoder
_sys.modules['medseg.models.encoders.ege_unet_encoder'] = ege_unet_encoder
