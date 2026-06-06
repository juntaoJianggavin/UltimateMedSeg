"""线性注意力机制 encoder (RetNet / Linformer / Performer / TTT / xLSTM)。
Linear attention encoders (RetNet / Linformer / Performer / TTT / xLSTM)."""
import sys as _sys
for _stem in ('retnet_encoder', 'linformer_encoder', 'performer_encoder', 'ttt_encoder', 'xlstm_encoder'):
    try:
        _mod = __import__(f'medseg.models.encoders.linear_attn.{_stem}', fromlist=[_stem])
        _sys.modules[f'medseg.models.encoders.{_stem}'] = _mod
        globals()[_stem] = _mod
    except ImportError:
        pass
