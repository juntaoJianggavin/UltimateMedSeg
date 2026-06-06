"""RWKV-family encoders."""

import sys as _sys

# rwkv_encoder first: rir_zigzag_encoder may resolve it via the legacy alias.
from . import rwkv_encoder
_sys.modules['medseg.models.encoders.rwkv_encoder'] = rwkv_encoder
from . import rir_zigzag_encoder
_sys.modules['medseg.models.encoders.rir_zigzag_encoder'] = rir_zigzag_encoder
from . import u_rwkv_encoder
_sys.modules['medseg.models.encoders.u_rwkv_encoder'] = u_rwkv_encoder
from . import md_rwkv_encoder
_sys.modules['medseg.models.encoders.md_rwkv_encoder'] = md_rwkv_encoder
