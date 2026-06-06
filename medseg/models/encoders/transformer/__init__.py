"""Transformer-family encoders."""

import sys as _sys

from . import transunet_encoder
_sys.modules['medseg.models.encoders.transunet_encoder'] = transunet_encoder
from . import swinunet_encoder
_sys.modules['medseg.models.encoders.swinunet_encoder'] = swinunet_encoder
from . import ucransnet_encoder
_sys.modules['medseg.models.encoders.ucransnet_encoder'] = ucransnet_encoder
from . import missformer_encoder
_sys.modules['medseg.models.encoders.missformer_encoder'] = missformer_encoder
from . import medt_encoder
_sys.modules['medseg.models.encoders.medt_encoder'] = medt_encoder
from . import mctrans_encoder
_sys.modules['medseg.models.encoders.mctrans_encoder'] = mctrans_encoder
from . import hiformer_encoder
_sys.modules['medseg.models.encoders.hiformer_encoder'] = hiformer_encoder
from . import daeformer_encoder
_sys.modules['medseg.models.encoders.daeformer_encoder'] = daeformer_encoder
from . import fatnet_encoder
_sys.modules['medseg.models.encoders.fatnet_encoder'] = fatnet_encoder
from . import h2former_encoder
_sys.modules['medseg.models.encoders.h2former_encoder'] = h2former_encoder
from . import scaleformer_encoder
_sys.modules['medseg.models.encoders.scaleformer_encoder'] = scaleformer_encoder
from . import mtunet_encoder
_sys.modules['medseg.models.encoders.mtunet_encoder'] = mtunet_encoder
from . import vit_pyramid_encoder
_sys.modules['medseg.models.encoders.vit_pyramid_encoder'] = vit_pyramid_encoder
from . import pvtv2_encoder
_sys.modules['medseg.models.encoders.pvtv2_encoder'] = pvtv2_encoder
from . import segformer_mit_encoder
_sys.modules['medseg.models.encoders.segformer_mit_encoder'] = segformer_mit_encoder
from . import maxvit_encoder
_sys.modules['medseg.models.encoders.maxvit_encoder'] = maxvit_encoder
from . import cswin_encoder
_sys.modules['medseg.models.encoders.cswin_encoder'] = cswin_encoder
from . import mobile_u_vit_encoder
_sys.modules['medseg.models.encoders.mobile_u_vit_encoder'] = mobile_u_vit_encoder
