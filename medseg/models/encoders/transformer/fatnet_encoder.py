"""FAT-Net Encoder: faithful port from https://github.com/SZUcsh/FAT-Net

Reference: Wu et al., "FAT-Net: Feature Adaptive Transformers for Automated Skin Lesion Segmentation"
All class/attribute names match the original for pretrained weight loading.
"""
# Source: https://github.com/SZUcsh/FAT-Net

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision import models as resnet_model

from medseg.registry import ENCODER_REGISTRY


# ============= FAMBlock (from FAT_Net.py) =============
class FAMBlock(nn.Module):
    """Feature Adaptive Module Block."""
    def __init__(self, channels):
        super(FAMBlock, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1
        return out


# ============= SEBlock (from FAT_Net.py) =============
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.mul(x, y)
        return y


# ============= DecoderBottleneckLayer (from FAT_Net.py) =============
class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


@ENCODER_REGISTRY.register("fatnet")
class FATNetEncoder(nn.Module):
    """FAT-Net Encoder: ResNet34 + DeiT-Tiny dual path.
    Faithful to https://github.com/SZUcsh/FAT-Net
    
    Note: Original uses torch.hub to load deit_tiny_distilled_patch16_224.
    For flexibility, we support passing pretrained DeiT or using random init.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        pretrained_path: str = None,
        use_deit_pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        # ResNet34 encoder
        resnet = resnet_model.resnet34(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

        # Transformer branch
        try:
            if use_deit_pretrained:
                transformer = torch.hub.load('facebookresearch/deit:main',
                                             'deit_tiny_distilled_patch16_224', pretrained=True)
            else:
                transformer = torch.hub.load('facebookresearch/deit:main',
                                             'deit_tiny_distilled_patch16_224', pretrained=False)
            self.patch_embed = transformer.patch_embed
            self.transformers = nn.ModuleList([transformer.blocks[i] for i in range(12)])
        except Exception:
            # Fallback: create simple transformer blocks
            from timm.models.vision_transformer import PatchEmbed, Block
            self.patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=192)
            self.transformers = nn.ModuleList([
                Block(dim=192, num_heads=3, mlp_ratio=4., qkv_bias=True) for _ in range(12)])

        # Fusion layers
        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        # FAM blocks
        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        self._out_channels = [64, 128, 256, 512]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        b, c, h, w = x.shape

        # ResNet encoding
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        # Transformer path
        emb = self.patch_embed(x)
        for i in range(12):
            emb = self.transformers[i](emb)
        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 192, 14, 14)
        feature_tf = self.conv_seq_img(feature_tf)

        # Fusion
        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        # Apply FAM blocks
        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)

        return [e1, e2, e3, feature_out]
