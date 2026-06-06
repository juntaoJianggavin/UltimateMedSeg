"""Mamba-family encoders."""

import sys as _sys

# Order matters: vmunet_encoder must be importable as
# ``medseg.models.encoders.vmunet_encoder`` before umamba_encoder loads, because the
# latter triggers ``medseg.models.networks.mamba.mamba_unet`` which imports vmunet
# via its legacy absolute path.
from . import vmunet_encoder
_sys.modules['medseg.models.encoders.vmunet_encoder'] = vmunet_encoder
from . import mamba_pure_encoder
_sys.modules['medseg.models.encoders.mamba_pure_encoder'] = mamba_pure_encoder
from . import umamba_encoder
_sys.modules['medseg.models.encoders.umamba_encoder'] = umamba_encoder
from . import lkm_encoder
_sys.modules['medseg.models.encoders.lkm_encoder'] = lkm_encoder
from . import vm_unet_v2_encoder
_sys.modules['medseg.models.encoders.vm_unet_v2_encoder'] = vm_unet_v2_encoder
from . import log_vmamba_encoder
from . import vmkla_encoder
from . import lightm_encoder
_sys.modules['medseg.models.encoders.lightm_encoder'] = lightm_encoder
from . import ultralight_vm_encoder
_sys.modules['medseg.models.encoders.ultralight_vm_encoder'] = ultralight_vm_encoder
from . import ultralbm_encoder
_sys.modules['medseg.models.encoders.ultralbm_encoder'] = ultralbm_encoder
