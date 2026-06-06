"""KAN / MLP encoders (UKAN, Rolling-UNet, UNeXt, etc.)."""

import sys as _sys

for _stem in ('ukan_encoder', 'rolling_unet_encoder', 'unext_encoder', 'wav_kan_encoder'):
    try:
        _mod = __import__(f'medseg.models.encoders.kan_mlp.{_stem}', fromlist=[_stem])
        _sys.modules[f'medseg.models.encoders.{_stem}'] = _mod
        globals()[_stem] = _mod
    except ImportError:
        pass
