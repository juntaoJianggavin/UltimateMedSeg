"""Encoder modules.

Encoders are organised into family sub-packages (cnn, transformer, mamba,
rwkv, linear_attn, wrapper). Importing this package triggers registration of
every encoder in the global ``ENCODER_REGISTRY``.

For backwards compatibility, every encoder module is also exposed as a
top-level attribute of ``medseg.models.encoders`` and registered in ``sys.modules``
under its original path ``medseg.models.encoders.<stem>``. This keeps absolute
imports such as ``from medseg.models.encoders.vmunet_encoder import SS2D`` working
unchanged after the reorganisation.

NOTE: the legacy ``sys.modules['medseg.models.encoders.<stem>']`` aliases are set up
eagerly inside each sub-package's ``__init__.py`` (right after each module
finishes loading). This is required so that cross-package imports — e.g.
``umamba_encoder`` triggers ``networks.mamba.mamba_unet`` which does
``from medseg.models.encoders.vmunet_encoder import ...`` — resolve while
``medseg.models.encoders`` is still being initialised.
"""

from . import cnn as _cnn
from . import transformer as _tx
from . import mamba as _mb
from . import rwkv as _rw
from . import linear_attn as _la
from . import wrapper as _wr
from . import kan_mlp as _km
from . import foundation as _fo

# Re-export every encoder module as a top-level attribute of
# ``medseg.models.encoders`` so legacy ``medseg.models.encoders.<stem>`` attribute access
# (not just ``import``) keeps working.
for _pkg, _stems in [
    (_cnn, [
        'basic_encoder',
        'dcsaunet_encoder',
        'cfanet_encoder',
        'mednext_encoder',
        'convnext_encoder',
        'efficientnetv2_encoder',
        'lv_unet_encoder',
        'malunet_encoder',
        'ege_unet_encoder',
    ]),
    (_tx, [
        'transunet_encoder',
        'swinunet_encoder',
        'ucransnet_encoder',
        'missformer_encoder',
        'medt_encoder',
        'mctrans_encoder',
        'hiformer_encoder',
        'daeformer_encoder',
        'fatnet_encoder',
        'h2former_encoder',
        'scaleformer_encoder',
        'mtunet_encoder',
        'vit_pyramid_encoder',
        'pvtv2_encoder',
        'segformer_mit_encoder',
        'maxvit_encoder',
    ]),
    (_mb, [
        'vmunet_encoder',
        'umamba_encoder',
        'mamba_pure_encoder',
    ]),
    (_rw, [
        'rwkv_encoder',
        'rir_zigzag_encoder',
    ]),
    (_la, [
        'retnet_encoder',
        'linformer_encoder',
        'performer_encoder',
        'ttt_encoder',
        'xlstm_encoder',
    ]),
    (_wr, [
        'timm_encoder',
    ]),
    (_km, [
        'ukan_encoder',
        'rolling_unet_encoder',
        'unext_encoder',
    ]),
    # NOTE: foundation encoders (sam_vit_encoder, clip_encoder, ...) install
    # their own sys.modules['medseg.models.encoders.<stem>'] aliases from inside each
    # family sub-package (general/pathology/...). Do NOT re-register them here.
]:
    for _stem in _stems:
        _attr = getattr(_pkg, _stem, None)
        if _attr is not None:
            globals()[_stem] = _attr

# Also pull foundation encoder stems into our top-level namespace for
# attribute-style access (medseg.models.encoders.sam_vit_encoder). The sys.modules
# alias was already installed by the family sub-package __init__.
import sys as _sys
for _stem in ('sam_vit_encoder', 'clip_encoder', 'dino_encoder', 'dinov2_encoder',
              'dinov3_encoder', 'biomedclip_encoder', 'medclip_encoder',
              'conch_encoder', 'uni_encoder', 'usfm_encoder', 'ctfm_encoder',
              'raddino_encoder', 'radimagenet_encoder'):
    _m = _sys.modules.get('medseg.models.encoders.' + _stem)
    if _m is not None:
        globals()[_stem] = _m

del _pkg, _stems, _stem, _attr, _m, _sys
