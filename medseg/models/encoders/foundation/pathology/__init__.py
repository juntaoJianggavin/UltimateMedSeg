"""Pathology foundation encoders."""
import sys as _sys
for _stem in ('phikon_encoder', 'musk_encoder', 'plip_encoder', 'phikon_v2_encoder', 'uni_encoder', 'path_foundation_encoder'):
    try:
        _mod = __import__(f'medseg.models.encoders.foundation.pathology.{_stem}', fromlist=[_stem])
        _sys.modules[f'medseg.models.encoders.foundation.{_stem}'] = _mod
        _sys.modules[f'medseg.models.encoders.{_stem}'] = _mod
        globals()[_stem] = _mod
    except ImportError as e:
        import warnings; warnings.warn(f'Could not import {_stem}: {e}')
