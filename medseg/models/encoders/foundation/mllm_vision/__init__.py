"""MLLM vision encoders (empty for now)."""
import sys as _sys
for _stem in ('llava_med_vision_encoder', 'hulumed_vision_encoder', 'qwen3_vl_vision_encoder', 'lingshu_vision_encoder', 'qwen25_vl_vision_encoder', 'medgemma_vision_encoder', 'healthgpt_vision_encoder', 'huatuogpt_vision_encoder'):
    try:
        _mod = __import__(f'medseg.models.encoders.foundation.mllm_vision.{_stem}', fromlist=[_stem])
        _sys.modules[f'medseg.models.encoders.foundation.{_stem}'] = _mod
        _sys.modules[f'medseg.models.encoders.{_stem}'] = _mod
        globals()[_stem] = _mod
    except ImportError as e:
        import warnings; warnings.warn(f'Could not import {_stem}: {e}')
