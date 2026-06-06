"""跳跃连接模块 / Skip connection modules.

分类 / Categories:
    basic/        基础 (concat / add / dense)
    attention/    注意力 (AG / CAB / SAB / SCSE / CBAM / Gating / GRU / GAB / SC-Att / TA-MoSC)
    transformer/  Transformer (CrossAttn / TransFusion / AggAttn / MISSFormer / UCTrans)
    mamba/        Mamba (SK-VM++)
    fusion/       CNN融合 (BiFusion / Deformable / MultiScale / FeatureRefine / CCM / SDI)
"""
from . import basic, attention, transformer, mamba, fusion
import sys as _sys
for _pkg, _stems in [
    (basic,       ['basic_skip','dense_skip']),
    (attention,   ['attention_gate_skip','cab_skip','sab_skip','scse_skip','cbam_skip','gating_skip','gru_gate_skip','gab_skip','sc_att_bridge_skip','ta_mosc_skip']),
    (transformer, ['cross_attn_skip','transformer_fusion_skip','aggregation_attention_skip','missformer_bridge_skip','uctrans_skip']),
    (mamba,       ['skvmpp_skip']),
    (fusion,      ['bifusion_skip','deformable_skip','multiscale_skip','feature_refine_skip','ccm_skip','sdi_skip']),
]:
    for _stem in _stems:
        _mod = getattr(_pkg, _stem)
        _sys.modules[f'medseg.models.skip_connections.{_stem}'] = _mod
        globals()[_stem] = _mod
