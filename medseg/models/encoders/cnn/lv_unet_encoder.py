"""LV-UNet Encoder: MobileNetV3-Large + VanillaNet-style Block mid-stages.
LV-UNet 编码器：MobileNetV3-Large 预训练主干 + VanillaNet 风格 Block 中间层。

Extracts 6 multi-scale feature maps:
  提取 6 个多尺度特征图：
  - 3 from MobileNetV3-Large (firstconv+encoder1/2/3)
    来自 MobileNetV3-Large 的 3 个阶段
  - 3 from VanillaNet Block mid-stages
    来自 VanillaNet Block 中间下采样阶段

out_channels: [16, 24, 80, 160, 240, 480] (default dims)
"""
# Reference: https://github.com/juntaoJianggavin/LV-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from timm.models.layers import trunc_normal_

from medseg.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# Series-informed activation / 序列激活函数 (from VanillaNet / LV-UNet)
# ---------------------------------------------------------------------------

class SeriesActivation(nn.ReLU):
    """Series-informed activation: ReLU followed by depthwise conv + BN.
    序列激活：ReLU 后接深度卷积 + BN。

    In deploy mode, BN is fused into the conv weights for faster inference.
    部署模式下，BN 融合进卷积权重以加速推理。
    """
    def __init__(self, dim, act_num=3, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.weight = nn.Parameter(
            torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num
        trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return F.conv2d(
                super().forward(x),
                self.weight, self.bias,
                padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return self.bn(F.conv2d(
                super().forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def switch_to_deploy(self):
        kernel = self.weight
        bn = self.bn
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        self.weight.data = kernel * t
        self.bias = nn.Parameter(
            bn.bias + (0 - bn.running_mean) * bn.weight / std)
        del self.bn
        self.deploy = True


# ---------------------------------------------------------------------------
# VanillaNet-style Block / VanillaNet 风格下采样 Block (from LV-UNet)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """VanillaNet-style downsampling block: Conv1x1 -> BN -> LeakyReLU -> Conv1x1 -> BN -> MaxPool -> activation.
    VanillaNet 风格下采样块：Conv1x1 -> BN -> LeakyReLU -> Conv1x1 -> BN -> MaxPool -> 激活。
    """
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False):
        super().__init__()
        self.act_learn = 0
        self.deploy = deploy
        if deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6),
            )
        self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        self.act = SeriesActivation(dim_out, act_num, deploy=deploy)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = F.leaky_relu(x, self.act_learn)
            x = self.conv2(x)
        x = self.pool(x)
        x = self.act(x)
        return x

    def switch_to_deploy(self):
        kernel1, bias1 = self._fuse_bn(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel1
        self.conv1[0].bias.data = bias1
        kernel2, bias2 = self._fuse_bn(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(
            kernel2.transpose(1, 3),
            self.conv1[0].weight.data.squeeze(3).squeeze(2)
        ).transpose(1, 3)
        self.conv.bias.data = bias2 + (
            self.conv1[0].bias.data.view(1, -1, 1, 1) * kernel2
        ).sum(3).sum(2).sum(1)
        del self.conv1, self.conv2
        self.act.switch_to_deploy()
        self.deploy = True

    @staticmethod
    def _fuse_bn(conv, bn):
        kernel = conv.weight
        bias = conv.bias
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias + (bias - bn.running_mean) * bn.weight / std


# ---------------------------------------------------------------------------
# LV-UNet Encoder / LV-UNet 编码器
# ---------------------------------------------------------------------------

@ENCODER_REGISTRY.register("lv_unet")
class LVUNetEncoder(nn.Module):
    """LV-UNet encoder: MobileNetV3-Large backbone + VanillaNet Block mid-stages.
    LV-UNet 编码器：MobileNetV3-Large 主干 + VanillaNet Block 中间下采样层。

    Architecture / 架构:
      - firstconv: stride-2 conv (3 -> 16), output at H/2
        首层卷积（3 -> 16），输出 H/2
      - encoder1: MobileNetV3 stages 1-2 (16 -> 24), output at H/2
        MobileNetV3 第1-2层（16 -> 24），输出 H/2
      - encoder2: MobileNetV3 stages 3-5 (24 -> 40), output at H/4
        MobileNetV3 第3-5层（24 -> 40），输出 H/4
      - encoder3: MobileNetV3 stages 6-9 (40 -> 80), output at H/8
        MobileNetV3 第6-9层（40 -> 80），输出 H/8
      - Block stages: VanillaNet downsample x3, output at H/16, H/32, H/64
        VanillaNet Block 下采样 x3，输出 H/16, H/32, H/64

    Returns 6 multi-scale feature maps.
    返回 6 个多尺度特征图。

    Args:
        in_channels: Input image channels (default 3). / 输入通道数。
        dims: Channel dims for Block mid-stages [stage0_in, s1, s2, s3].
              Block 中间层通道数。
        act_num: Number of activation terms. / 激活项数量。
        strides: Downsampling strides for mid stages. / 中间层下采样步长。
        deploy: Use fused BN for inference. / 是否使用融合 BN 推理模式。
        pretrained: Use pretrained MobileNetV3 weights. / 是否使用预训练权重。
    """

    def __init__(self, in_channels: int = 3, img_size: int = 224,
                 dims=None, act_num: int = 1, strides=None,
                 deploy: bool = False, pretrained: bool = True, **kwargs):
        super().__init__()
        if dims is None:
            dims = [80, 160, 240, 480]
        if strides is None:
            strides = [2, 2, 2]

        self.deploy = deploy

        # --- MobileNetV3-Large encoder / MobileNetV3-Large 编码器 ---
        try:
            from torchvision import models
            if pretrained:
                try:
                    from torchvision.models import MobileNet_V3_Large_Weights
                    mobile = models.mobilenet_v3_large(
                        weights=MobileNet_V3_Large_Weights.DEFAULT)
                except (ImportError, AttributeError):
                    mobile = models.mobilenet_v3_large(pretrained=True)
            else:
                mobile = models.mobilenet_v3_large(pretrained=False)

            self.firstconv = mobile.features[0]       # 3 -> 16, stride 2
            self.encoder1 = nn.Sequential(             # 16 -> 24
                mobile.features[1], mobile.features[2])
            self.encoder2 = nn.Sequential(             # 24 -> 40
                mobile.features[3], mobile.features[4], mobile.features[5])
            self.encoder3 = nn.Sequential(             # 40 -> 80
                mobile.features[6], mobile.features[7],
                mobile.features[8], mobile.features[9])
        except Exception:
            # Fallback: lightweight conv encoder if torchvision unavailable
            # 回退：torchvision 不可用时使用轻量卷积编码器
            self.firstconv = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False),
                nn.BatchNorm2d(16), nn.Hardswish())
            self.encoder1 = nn.Sequential(
                nn.Conv2d(16, 24, 3, 1, 1, bias=False),
                nn.BatchNorm2d(24), nn.ReLU(inplace=True))
            self.encoder2 = nn.Sequential(
                nn.Conv2d(24, 40, 3, 2, 1, bias=False),
                nn.BatchNorm2d(40), nn.ReLU(inplace=True))
            self.encoder3 = nn.Sequential(
                nn.Conv2d(40, 80, 3, 2, 1, bias=False),
                nn.BatchNorm2d(80), nn.ReLU(inplace=True))

        # --- VanillaNet mid stages (downsample) / VanillaNet 中间下采样层 ---
        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            self.stages.append(Block(
                dims[i], dims[i + 1], act_num=act_num,
                stride=strides[i], deploy=deploy))

        # Encoder output channels (shallow -> deep):
        # 编码器输出通道（由浅到深）：
        #   e1=24, e2=40, e3=80, then Block outputs dims[1:]
        self.out_channels: List[int] = [24, 40, 80] + list(dims[1:])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode input to multi-scale feature maps (shallow to deep).
        将输入编码为多尺度特征图（由浅到深）。

        Returns: [e1(24ch,H/2), e2(40ch,H/4), e3(80ch,H/8),
                  mid0(160ch,H/16), mid1(240ch,H/32), mid2(480ch,H/64)]
        """
        features: List[torch.Tensor] = []

        # MobileNetV3 stages / MobileNetV3 阶段
        x = self.firstconv(x)       # H/2, 16ch
        e1 = self.encoder1(x)       # H/2, 24ch
        features.append(e1)
        e2 = self.encoder2(e1)      # H/4, 40ch
        features.append(e2)
        e3 = self.encoder3(e2)      # H/8, 80ch
        features.append(e3)

        # VanillaNet Block mid stages / VanillaNet Block 中间下采样
        feat = e3
        for stage in self.stages:
            feat = stage(feat)
            features.append(feat)

        return features

    def change_act(self, m):
        """Set leaky ReLU slope for deep training technique.
        设置 leaky ReLU 斜率用于深度训练技巧。"""
        for stage in self.stages:
            stage.act_learn = m

    def switch_to_deploy(self):
        """Fuse BN into conv weights for faster inference.
        融合 BN 到卷积权重以加速推理。"""
        for stage in self.stages:
            stage.switch_to_deploy()
        self.deploy = True
