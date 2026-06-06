"""通用组件注册表 / Generic registry for modular components."""


class Registry:
    """通用注册表，用于 encoder / decoder / skip / bottleneck / loss 等模块化组件。
    A generic registry for encoders, decoders, skip connections, etc."""

    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, key: str):
        """装饰器：将一个类注册到指定 key 下。
        Decorator to register a class under a given key."""
        def decorator(cls):
            if key in self._registry:
                raise KeyError(f"'{key}' is already registered in {self.name}")
            self._registry[key] = cls
            return cls
        return decorator

    def get(self, key: str):
        """按 key 获取已注册的类。支持 timm_ 前缀动态查找。
        Get a registered class by key. Supports timm_ prefix dynamic lookup."""
        if key in self._registry:
            return self._registry[key]

        # 动态 timm 支持：以 timm_ 开头的 encoder 名自动创建 wrapper 类
        # Dynamic timm support: encoder names starting with timm_ are
        # auto-resolved to a TimmEncoder wrapper without pre-registration.
        if self.name == "encoders" and key.startswith("timm_") and key != "timm":
            timm_model_name = key[5:]  # 去掉 "timm_" 前缀 / strip "timm_" prefix
            return self._make_timm_class(key, timm_model_name)

        available = ", ".join(sorted(self._registry.keys())[:20])
        hint = ""
        if self.name == "encoders":
            hint = (
                " Tip: any timm model can be used with the 'timm_' prefix, "
                "e.g. 'timm_resnet50', 'timm_efficientnet_b7', 'timm_swin_base_patch4_window12_384'."
            )
        raise KeyError(
            f"'{key}' not found in {self.name}. "
            f"Available (first 20): [{available}].{hint}"
        )

    def _make_timm_class(self, registry_key: str, timm_model_name: str):
        """动态创建并缓存一个 timm encoder 类。
        Dynamically create and cache a timm encoder class."""
        from medseg.models.encoders.wrapper.timm_encoder import TimmEncoder

        class _DynTimmEnc(TimmEncoder):
            def __init__(self, pretrained=False, in_channels=3, img_size=224, **kwargs):
                super().__init__(
                    model_name=timm_model_name,
                    pretrained=pretrained,
                    in_channels=in_channels,
                    img_size=img_size,
                    **kwargs,
                )

        _DynTimmEnc.__name__ = f"Timm_{timm_model_name}"
        _DynTimmEnc.__qualname__ = f"Timm_{timm_model_name}"

        # 缓存到注册表，下次不再重复创建 / Cache for future lookups
        self._registry[registry_key] = _DynTimmEnc
        return _DynTimmEnc

    def list_available(self):
        """返回已注册 key 的排序列表 / Return sorted list of registered keys."""
        return sorted(self._registry.keys())

    def __contains__(self, key: str):
        if key in self._registry:
            return True
        # timm_ 前缀的总是视为存在 / timm_ prefixed keys always exist
        if self.name == "encoders" and key.startswith("timm_") and key != "timm":
            return True
        return False

    def __len__(self):
        return len(self._registry)


ENCODER_REGISTRY = Registry("encoders")
DECODER_REGISTRY = Registry("decoders")
SKIP_REGISTRY = Registry("skip_connections")
BOTTLENECK_REGISTRY = Registry("bottlenecks")
LOSS_REGISTRY = Registry("losses")
AUGMENTATION_REGISTRY = Registry("augmentations")
