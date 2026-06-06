"""Dermatology foundation encoders."""

import sys as _sys
import warnings as _warnings
for _stem in ('dermclip_encoder', 'monet_derm_encoder', 'derm_foundation_encoder', 'panderm_encoder'):
    try:
        _mod = __import__('medseg.models.encoders.foundation.dermatology.' + _stem, fromlist=[_stem])
        _sys.modules['medseg.models.encoders.foundation.' + _stem] = _mod
        _sys.modules['medseg.models.encoders.' + _stem] = _mod
        globals()[_stem] = _mod
    except ImportError as _e:
        _warnings.warn(
            f"Dermatology encoder '{_stem}' not available: {_e}. "
            f"Install the required dependencies to use this encoder."
        )
