"""Networks: categorized complete segmentation architectures.

Subcategories:
    - transformer/ : Transformer-based (ViT, Swin, etc.)
    - cnn/         : CNN-based (convolutional UNet variants)
    - rwkv/        : RWKV-based
    - mamba/       : Mamba / State-Space Model based
    - kan_mlp/     : KAN / MLP / LSTM based
    - other/       : Non-standard architectures (TTT, hybrid, etc.)
"""

# ── CNN ────────────────────────────────────────────────────────────────────────
from .cnn import (
    UNet3Plus, LVUNet, EGEUNet, MALUNet,
    LiteUNet, MKUNet, ULite,
    ACCUNet, CMUNeXt, MEWUNet,
    DoubleUNet,
    # Self-contained GitHub ports
    DCSAUNet, CFANet,
    AttentionUNet, UNetPP, MultiResUNet, SCSEUNet, ResUNetA,
    SAUNet, PAN, DenseUNet, LinkNet, PSPNet,
    ResUNetPP, FRUNet, MedNeXt,
    NNUNet2D, R2UNet, KiUNet,
    AAUNet, CMUNet, DSCNet,
    STUNet, DconnNet,
    # Domain-specific ports (2024-2026)
    Polyper,
    # Pathology (CNN-based)
    HoverNetLite,
)

# ── Mamba / SSM ────────────────────────────────────────────────────────────────
from .mamba import (
    MambaUNet, HVMUNet, LightMUNet, SwinUMamba,
    UMambaBot, UMambaEnc, UltraLightVMUNet,
    VMUNet, VMUNetV2, LKMUNet, LoGVMamba, VMKLAUNet,
    UltraLBMUNet, NnMamba2D,
    PolypMamba, HCMamba,
    # Domain-specific ports (2024-2026)
    MUCMNet, ACMambaSeg, SkinMamba, SerpMamba,
    MambaVesselNetPP, UUMamba, ViMUNet, DCMNet, DermoMamba,
)

# ── RWKV ───────────────────────────────────────────────────────────────────────
from .rwkv import URWKV, RWKVUNet, MDRWKVUNet, RIRZigzag

# ── KAN / MLP / LSTM ──────────────────────────────────────────────────────────
from .kan_mlp import UKAN, WavKANUNet
from .kan_mlp.rolling_unet import RollingUNet, RollingUNet_M, RollingUNet_L
from .kan_mlp.unext import UNeXt

# ── Transformer (new, populated below as files are added) ──────────────────────
from .transformer import (
    DATransUNet, DSTransUNet, UCTransNet, MobileUViT,
    CSWinUNet, FCBFormer, PVTUNet, TransNetR,
    # Self-contained GitHub ports
    TransUNet, SwinUNet, MedT, DAEFormer, MISSFormer,
    H2Former, HiFormer, MCTrans, MTUNet, ScaleFormer,
    FATNet, UCTransNetEnc,
    NNFormer2D, TransFuse, LeViTUNet, TransAttUNet,
    PolypPVT, CASCADE,
    HSNet, SSFormer, LDNet, ESFPNet, MIST,
    # Domain-specific ports (2024-2026)
    SEPNet, CTNet,
    # Pathology
    NuLite, TransNuSeg,
)

# ── SAM family ─────────────────────────────────────────────────────────────
from .sam import (
    MedSAM, SAMUS,
    SAMViTBase, SAMViTLarge, MobileSAM, SAM2,
    SAMMed2D, MedicalSAMAdapter, SAMed, AutoSAM,
)


# ── 线性注意力机制 (TTT / xLSTM) ──────────────────────────────────────
# ── Linear Attention Models (TTT / xLSTM) ─────────────────────────
from .linear_attn import (
    TTTUNet, XLSTMUNetBot, XLSTMUNetEnc, UVixLSTM,
)

# Text-guided + UNet end-to-end fusion models (forward(image, text=None))
# 文本引导分割模型（forward(image, text=None)）
from medseg.models.text_unet import (
    TGANet,
    LViT,
    LanGuideMedSeg,
    CLIPDrivenUniversalModel2D,
    CRIS,
    BiomedParse,
    TPRO,
    SaLIP,
    CausalCLIPSeg,
    MedCLIPSAM,
    TPDRSeg,
    CXRCLIPSeg,
)

# Prompt-guided SAM wrappers (点/框 prompt，非文本引导)
from medseg.models.networks.sam import SAMMed2DWrapper, LiteMedSAM

# MLLM + segmentation paradigm (LLM-as-decoder), forward(image, text=None)
from medseg.inference.mllm import MediSeeWrapper


_SPECIAL_ARCHS = {
    # CNN
    "rolling_unet": RollingUNet,
    "rolling_unet_s": RollingUNet,
    "rolling_unet_m": RollingUNet_M,
    "rolling_unet_l": RollingUNet_L,
    "unet3plus": UNet3Plus,
    "unext": UNeXt,
    "lv_unet": LVUNet,
    "ege_unet": EGEUNet,
    "malunet": MALUNet,
    "lite_unet": LiteUNet,
    "mk_unet": MKUNet,
    "ttt_unet": TTTUNet,
    "u_lite": ULite,
    "ultralbm_unet": UltraLBMUNet,
    "acc_unet": ACCUNet,
    "cmunext": CMUNeXt,
    # Mamba / SSM
    "mamba_unet": MambaUNet,
    "h_vmunet": HVMUNet,
    "lightm_unet": LightMUNet,
    "swin_umamba": SwinUMamba,
    "umamba_bot": UMambaBot,
    "umamba_enc": UMambaEnc,
    "ultralight_vmunet": UltraLightVMUNet,
    "vm_unet": VMUNet,
    "vm_unet_v2": VMUNetV2,
    "lkm_unet": LKMUNet,
    "log_vmamba": LoGVMamba,
    "vmkla_unet": VMKLAUNet,
    # RWKV
    "u_rwkv": URWKV,
    "rwkv_unet": RWKVUNet,
    "md_rwkv_unet": MDRWKVUNet,
    # KAN / MLP / LSTM
    "ukan": UKAN,
    "mew_unet": MEWUNet,
    "wav_kan_unet": WavKANUNet,
    "xlstm_unet_bot": XLSTMUNetBot,
    "xlstm_unet_enc": XLSTMUNetEnc,
    # Transformer
    "da_transunet": DATransUNet,
    "ds_transunet": DSTransUNet,
    "uctransnet_full": UCTransNet,
    "mobile_u_vit": MobileUViT,
    "cswin_unet": CSWinUNet,
    "fcbformer": FCBFormer,
    "pvt_unet": PVTUNet,
    "double_unet": DoubleUNet,
    "transnetr": TransNetR,
    # GitHub ports (transformer)
    "transunet": TransUNet,
    "swinunet": SwinUNet,
    "medt": MedT,
    "daeformer": DAEFormer,
    "missformer": MISSFormer,
    "h2former": H2Former,
    "hiformer": HiFormer,
    "mctrans": MCTrans,
    "mtunet": MTUNet,
    "scaleformer": ScaleFormer,
    "fatnet": FATNet,
    "uctransnet_enc": UCTransNetEnc,
    "nnformer_2d": NNFormer2D,
    "transfuse": TransFuse,
    "levit_unet": LeViTUNet,
    "transatt_unet": TransAttUNet,
    "polyp_pvt": PolypPVT,
    "nnmamba_2d": NnMamba2D,
    "cascade": CASCADE,
    # GitHub ports (cnn)
    "dcsaunet": DCSAUNet,
    "cfanet": CFANet,
    "attention_unet": AttentionUNet,
    "unetpp": UNetPP,
    "multiresunet": MultiResUNet,
    "scseunet": SCSEUNet,
    "resunet_a": ResUNetA,
    "sa_unet": SAUNet,
    "pan": PAN,
    "denseunet": DenseUNet,
    "linknet": LinkNet,
    "pspnet": PSPNet,
    "resunetpp": ResUNetPP,
    "fr_unet": FRUNet,
    "mednext": MedNeXt,
    "nnunet_2d": NNUNet2D,
    "r2unet": R2UNet,
    "kiunet": KiUNet,
    "aau_net": AAUNet,
    "cmu_net": CMUNet,
    "dscnet": DSCNet,
    # GitHub ports (transformer, newly added)
    "hsnet": HSNet,
    "ssformer": SSFormer,
    "ldnet": LDNet,
    "esfpnet": ESFPNet,
    "mist": MIST,
    # GitHub ports (rwkv)
    "rir_zigzag": RIRZigzag,
    # Newly added (SAM-style transformer / CNN / Mamba)
    "medsam": MedSAM,
    "stu_net": STUNet,
    "polyp_mamba": PolypMamba,
    "dconnnet": DconnNet,
    "hc_mamba": HCMamba,
    "samus": SAMUS,
    "sam_b": SAMViTBase,
    "sam_l": SAMViTLarge,
    "mobile_sam": MobileSAM,
    "sam2": SAM2,
    "sam_med2d": SAMMed2D,
    "medical_sam_adapter": MedicalSAMAdapter,
    "samed": SAMed,
    "auto_sam": AutoSAM,
    # Domain-specific ports (2024-2026)
    # Polyp
    "sepnet": SEPNet,
    "ctnet": CTNet,
    "polyper": Polyper,
    # Ultrasound
    "dcm_net": DCMNet,
    "uu_mamba": UUMamba,
    "vim_unet": ViMUNet,
    # Pathology
    "u_vixlstm": UVixLSTM,
    "nulite": NuLite,
    "transnuseg": TransNuSeg,
    "hovernet_lite": HoverNetLite,
    # Skin
    "mucm_net": MUCMNet,
    "ac_mambaseg": ACMambaSeg,
    "skin_mamba": SkinMamba,
    "dermomamba": DermoMamba,
    # Retinal
    "serp_mamba": SerpMamba,
    "mamba_vesselnet_pp": MambaVesselNetPP,
}

# Models below take a (image, text=None) forward signature and use
# bespoke constructor kwargs (no canonical in_channels/num_classes injection).
# We therefore route their construction through a separate path that simply
# forwards ``arch_params`` plus a few common keys (num_classes, img_size,
# in_channels) when the constructor accepts them.
_TEXT_UNET_ARCHS = {
    "tganet": TGANet,
    "lvit": LViT,
    "languide": LanGuideMedSeg,
    "clip_universal": CLIPDrivenUniversalModel2D,
    # 2D text-guided medical segmentation methods (2022-2024 venues).
    # 3D-native methods (SegVol/Hermes/UniSeg/MA-SAM/ZePT/MedSAM2) were
    # excluded per project policy: a 2D adaptation cannot be 99% faithful
    # to a 3D paper.
    "cris": CRIS,
    "biomedparse": BiomedParse,
    "tpro": TPRO,
    "salip": SaLIP,
    "causal_clipseg": CausalCLIPSeg,
    "medclip_sam": MedCLIPSAM,
    "tp_drseg": TPDRSeg,
    "cxrclipseg": CXRCLIPSeg,
    # MLLM + segmentation
    "medisee": MediSeeWrapper,
}

# Prompt-guided (点/框 prompt) 走 _SPECIAL_ARCHS
# Prompt-guided (point/box) models go through _SPECIAL_ARCHS
_SPECIAL_ARCHS["sammed2d_wrapper"] = SAMMed2DWrapper
_SPECIAL_ARCHS["lite_medsam"] = LiteMedSAM


def _build_text_unet(arch_name: str, cfg: dict):
    """Build a text+UNet model.

    Common cfg keys (encoder.in_channels / num_classes / img_size) are
    forwarded only when the target constructor accepts them, so that each
    upstream signature stays untouched.
    """
    import inspect

    cls = _TEXT_UNET_ARCHS[arch_name]
    params = dict(cfg.get("arch_params", {}))

    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys())

    # Map canonical keys -> constructor kwarg names when accepted
    in_channels = cfg.get("encoder", {}).get("in_channels", None)
    num_classes = cfg.get("num_classes", None)
    img_size = cfg.get("img_size", None)

    if in_channels is not None and "in_channels" in accepted and "in_channels" not in params:
        params["in_channels"] = in_channels
    if num_classes is not None:
        if "num_classes" in accepted and "num_classes" not in params:
            params["num_classes"] = num_classes
        elif "out_channels" in accepted and "out_channels" not in params:
            params["out_channels"] = num_classes
    if img_size is not None and "img_size" in accepted and "img_size" not in params:
        params["img_size"] = img_size

    # text_prompts 不传给构造函数，而是构造后挂到模型上
    # text_prompts is not passed to __init__, but attached after construction
    text_prompts = params.pop("text_prompts", None)

    # 只传构造函数真正接受的参数
    # Only pass kwargs the constructor actually accepts
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not has_var_kw:
        params = {k: v for k, v in params.items() if k in accepted}

    model = cls(**params)

    # 把 yaml 里的 text_prompts 挂到模型上
    # 模型 forward(image, text=None) 时，如果 text=None 就自动用这个
    # Attach text_prompts from yaml; model.forward uses it when text=None
    if text_prompts is not None:
        model._default_text_prompts = text_prompts

    return model


def build_special_arch(arch_name: str, cfg: dict):
    """Build a special architecture model."""
    if arch_name in _TEXT_UNET_ARCHS:
        return _build_text_unet(arch_name, cfg)
    if arch_name not in _SPECIAL_ARCHS:
        available = ", ".join(sorted(
            list(_SPECIAL_ARCHS.keys()) + list(_TEXT_UNET_ARCHS.keys())
        ))
        raise KeyError(f"'{arch_name}' not found. Available: [{available}]")
    cls = _SPECIAL_ARCHS[arch_name]
    # Respect pretrained flag from encoder config so smoke tests can disable
    # weight downloads for special architectures.
    enc_cfg = cfg.get("encoder", {})
    pretrained = enc_cfg.get("pretrained", True)

    # Resolve img_size (handle 'native' keyword)
    img_size = cfg.get("img_size", 224)
    if img_size == "native" or not isinstance(img_size, int):
        # Try to get native size from encoder registry
        enc_name = enc_cfg.get("name", "")
        try:
            from medseg.registry import ENCODER_REGISTRY
            enc_cls = ENCODER_REGISTRY.get(enc_name)
            img_size = getattr(enc_cls, "native_img_size", 224)
        except Exception:
            img_size = 224
        if not isinstance(img_size, int):
            img_size = 224

    build_kwargs = dict(
        in_channels=enc_cfg.get("in_channels", 3),
        num_classes=cfg.get("num_classes", 2),
        img_size=img_size,
        pretrained=pretrained,
    )
    build_kwargs.update(cfg.get("arch_params", {}))
    try:
        return cls(**build_kwargs)
    except TypeError:
        # Some architectures don't accept pretrained kwarg
        build_kwargs.pop("pretrained", None)
        return cls(**build_kwargs)
