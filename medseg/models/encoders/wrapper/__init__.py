"""Wrapper-family encoders."""

import sys as _sys

from . import timm_encoder
_sys.modules['medseg.models.encoders.timm_encoder'] = timm_encoder
