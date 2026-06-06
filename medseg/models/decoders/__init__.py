"""解码器模块 / Decoder modules.

分类 / Categories:
    basic/       基础上采样 (UNet / bilinear / deconv / depthwise-sep)
    dense/       密集连接 (UNet++)
    cascade/     级联 (CASCADE / EMCAD / CFM / G-CASCADE / EDLDNet / MERIT)
    pyramid/     金字塔聚合 (UPerNet)
    mlp/         MLP (SegFormer MLP / MLP decoder)
    specific/    特定网络专属 (TransUNet CUP / FAT-Net / HiFormer / etc.)
    transformer/ Transformer (DAEFormer / MTUNet / nnFormer / SwinUNet / UCTransNet)
    attention/   注意力 (Attention gate / HAM / Lawin)
    mamba/       Mamba (VM-UNet)
"""
from . import basic, dense, cascade, pyramid, mlp, specific, transformer, attention, mamba
import sys as _sys
for _pkg, _stems in [
    (basic,       ['unet_decoder','bilinear_decoder','deconv_decoder','dw_sep_decoder']),
    (dense,       ['unetpp_decoder','unet3plus_decoder']),
    (cascade,     ['cascade_decoder','cascade_full_decoder','cascade_emcad_decoder','cfm_decoder','emcad_decoder','edldnet_decoder','gcascade_decoder','merit_decoder']),
    (pyramid,     ['upernet_decoder']),
    (mlp,         ['mlp_decoder','segformer_decoder']),
    (specific,    ['cfanet_decoder','dcsaunet_decoder','rwkv_unet_decoder','kiunet_decoder','transunet_decoder','fatnet_decoder','h2former_decoder','hiformer_decoder','missformer_decoder','scaleformer_decoder','malunet_decoder','ege_unet_decoder']),
    (transformer, ['daeformer_decoder','mtunet_decoder','nnformer_decoder','swinunet_decoder','uctransnet_decoder']),
    (attention,   ['attention_decoder','ham_decoder','lawin_decoder']),
    (mamba,       ['vmunet_decoder']),
]:
    for _stem in _stems:
        _mod = getattr(_pkg, _stem)
        _sys.modules[f'medseg.models.decoders.{_stem}'] = _mod
        globals()[_stem] = _mod
