"""LV-UNet: Lightweight VanillaNet-style UNet for Medical Image Segmentation.

Faithful reimplementation from:
  https://github.com/juntaoJianggavin/LV-UNet  (BIBM 2024)

Key idea: MobileNetV3-Large pretrained encoder + VanillaNet-style Block/UpBlock
decoder with series-informed activation (depthwise conv activation).
Supports deploy mode (BN fusion for faster inference).
"""
# Source: https://github.com/juntaoJianggavin/LV-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from medseg.utils.timm_compat import trunc_normal_


# ---------------------------------------------------------------------------
# Series-informed activation (from VanillaNet / LV-UNet)
# ---------------------------------------------------------------------------

class SeriesActivation(nn.ReLU):
    """Series-informed activation: ReLU followed by depthwise conv + BN.

    In deploy mode, BN is fused into the conv weights for faster inference.
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
# VanillaNet-style Block and UpBlock (from LV-UNet)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """VanillaNet-style downsampling block: Conv1x1→BN→LeakyReLU→Conv1x1→BN→MaxPool→activation."""
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


class UpBlock(nn.Module):
    """VanillaNet-style upsampling block: Conv1x1→BN→LeakyReLU→Conv1x1→BN→Upsample→activation."""
    def __init__(self, dim, dim_out, act_num=3, factor=2, deploy=False):
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
        self.upsample = nn.Upsample(scale_factor=factor, mode='bilinear',
                                     align_corners=False)
        self.act = SeriesActivation(dim_out, act_num, deploy=deploy)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            x = F.leaky_relu(x, self.act_learn)
            x = self.conv2(x)
        x = self.upsample(x)
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
# LV-UNet
# ---------------------------------------------------------------------------

class LVUNet(nn.Module):
    """LV-UNet: MobileNetV3-Large encoder + VanillaNet-style decoder.

    Architecture:
      - Encoder: MobileNetV3-Large pretrained features (3 stages)
      - Mid stages: VanillaNet Block x3 (downsample)
      - Decoder up_stages1: UpBlock x3 (with skip from mid stages)
      - Decoder up_stages2: UpBlock x3 (with skip from encoder stages)
      - Final: UpBlock + 1x1 conv head

    Args:
        in_channels: Input channels (default: 3).
        num_classes: Number of output classes (default: 2).
        img_size: Input image size (default: 224).
        dims: Channel dims for mid stages [stage0, stage1, stage2, stage3].
        dims2: Channel dims for up_stages2 [us0, us1, us2, us3].
        act_num: Number of activation terms in series-informed activation.
        strides: Downsampling strides for mid stages.
        deploy: If True, use fused BN for faster inference.
        pretrained: Whether to use pretrained MobileNetV3 weights.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 dims=None, dims2=None, act_num=1, strides=None,
                 deploy=False, pretrained=True, deep_supervision=False, **kwargs):
        super().__init__()
        if dims is None:
            dims = [80, 160, 240, 480]
        if dims2 is None:
            dims2 = [80, 40, 24, 16]
        if strides is None:
            strides = [2, 2, 2]

        self.deploy = deploy
        self.deep_supervision = deep_supervision

        # --- MobileNetV3-Large encoder ---
        try:
            from torchvision import models
        except ImportError as e:
            raise RuntimeError(
                "LV-UNet requires torchvision for the MobileNet-V3 encoder. "
                "Install with: pip install torchvision"
            ) from e
        if pretrained:
            try:
                from torchvision.models import MobileNet_V3_Large_Weights
                mobile = models.mobilenet_v3_large(
                    weights=MobileNet_V3_Large_Weights.DEFAULT)
            except (ImportError, AttributeError):
                mobile = models.mobilenet_v3_large(pretrained=True)
        else:
            mobile = models.mobilenet_v3_large(pretrained=False)

        self.firstconv = mobile.features[0]
        self.encoder1 = nn.Sequential(
            mobile.features[1], mobile.features[2])
        self.encoder2 = nn.Sequential(
            mobile.features[3], mobile.features[4], mobile.features[5])
        self.encoder3 = nn.Sequential(
            mobile.features[6], mobile.features[7],
            mobile.features[8], mobile.features[9])

        # --- VanillaNet mid stages (downsample) ---
        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            self.stages.append(Block(
                dims[i], dims[i + 1], act_num=act_num,
                stride=strides[i], deploy=deploy))

        # --- Decoder up_stages1 (mirror mid stages) ---
        self.up_stages1 = nn.ModuleList()
        for i in range(len(strides)):
            self.up_stages1.append(UpBlock(
                dims[3 - i], dims[2 - i], act_num=act_num,
                factor=strides[2 - i], deploy=deploy))

        # --- Decoder up_stages2 (mirror encoder stages) ---
        self.up_stages2 = nn.ModuleList()
        for i in range(3):
            self.up_stages2.append(UpBlock(
                dims2[i], dims2[i + 1], act_num=act_num,
                factor=2, deploy=deploy))

        # --- Final head ---
        self.depth = len(strides)
        self.final = nn.ModuleList()
        self.final.append(UpBlock(
            dims2[-1], dims2[-1], act_num=act_num, factor=2, deploy=deploy))
        self.final.append(nn.Conv2d(dims2[-1], num_classes, 1))

        # Deep supervision side output heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(dims[2 - i], num_classes, 1) for i in range(self.depth)
            ] + [
                nn.Conv2d(dims2[i + 1], num_classes, 1) for i in range(3)
            ])

    def forward(self, x):
        input_size = x.shape[2:]

        # MobileNetV3 encoder
        x = self.firstconv(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # VanillaNet mid stages (downsample)
        encoder_feats = []
        feat = e3
        for i in range(self.depth):
            encoder_feats.append(feat)
            feat = self.stages[i](feat)

        # Decoder up_stages1 (with skip from mid stages)
        ds_intermediates = []
        for i in range(self.depth):
            feat = self.up_stages1[i](feat)
            skip = encoder_feats[self.depth - 1 - i]
            if feat.shape[2:] != skip.shape[2:]:
                feat = F.interpolate(feat, size=skip.shape[2:],
                                     mode='bilinear', align_corners=False)
            feat = feat + skip
            if self.training and self.deep_supervision:
                ds_intermediates.append(feat)

        # Decoder up_stages2 (with skip from encoder)
        feat = self.up_stages2[0](feat)
        if feat.shape[2:] != e2.shape[2:]:
            feat = F.interpolate(feat, size=e2.shape[2:],
                                 mode='bilinear', align_corners=False)
        feat = feat + e2
        if self.training and self.deep_supervision:
            ds_intermediates.append(feat)

        feat = self.up_stages2[1](feat)
        if feat.shape[2:] != e1.shape[2:]:
            feat = F.interpolate(feat, size=e1.shape[2:],
                                 mode='bilinear', align_corners=False)
        feat = feat + e1
        if self.training and self.deep_supervision:
            ds_intermediates.append(feat)

        feat = self.up_stages2[2](feat)
        if self.training and self.deep_supervision:
            ds_intermediates.append(feat)

        # Final
        for module in self.final:
            feat = module(feat)

        # Ensure output matches input size
        if feat.shape[2:] != input_size:
            feat = F.interpolate(feat, size=input_size,
                                 mode='bilinear', align_corners=False)

        if self.training and self.deep_supervision:
            aux_outputs = []
            for i, inter in enumerate(ds_intermediates):
                aux = self.ds_heads[i](inter)
                if aux.shape[2:] != input_size:
                    aux = F.interpolate(aux, size=input_size,
                                        mode='bilinear', align_corners=False)
                aux_outputs.append(aux)
            return [feat] + aux_outputs

        return feat

    def change_act(self, m):
        """Set leaky ReLU slope for deep training technique."""
        for i in range(self.depth):
            self.stages[i].act_learn = m
            self.up_stages1[i].act_learn = m
        for i in range(3):
            self.up_stages2[i].act_learn = m
        if hasattr(self.final[0], 'act_learn'):
            self.final[0].act_learn = m

    def switch_to_deploy(self):
        """Fuse BN into conv weights for faster inference."""
        for i in range(self.depth):
            self.stages[i].switch_to_deploy()
            self.up_stages1[i].switch_to_deploy()
        for i in range(3):
            self.up_stages2[i].switch_to_deploy()
        self.final[0].switch_to_deploy()
        self.deploy = True
