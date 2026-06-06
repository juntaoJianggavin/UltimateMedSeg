"""FAT-Net: Feature Adaptive Transformers for Automated Skin Lesion Segmentation.

Faithful port from github.com/SZUcsh/FAT-Net (Medical Image Analysis 2022).

Official architecture:
    - CNN Encoder: ResNet34 pretrained backbone (conv1 + bn1 + relu + layer1-4)
    - Transformer Encoder: DeiT (facebookresearch/deit) 12 transformer blocks
    - Bottleneck: concat CNN+Transformer features -> SE attention -> conv reduce
    - Skip connections: FAM (Feature Adaptive Module) blocks on each level
    - Decoder: DecoderBottleneckLayer (1x1 reduce + transpose conv up + 1x1)
    - Final: ConvTranspose2d 4x upsample + 2 conv layers

Reference:
    FAT-Net: Feature Adaptive Transformers for Automated Skin Lesion
    Segmentation. Medical Image Analysis, 2022.
"""
# Source: https://github.com/SZUcsh/FAT-Net

import torch
from torchvision import models as resnet_model
from torch import nn
import torch.nn.functional as F

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


class FAMBlock(nn.Module):
    """Feature Adaptive Module block (official: conv3+relu, conv1+relu, sum)."""
    def __init__(self, channels):
        super(FAMBlock, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1
        return out


class DecoderBottleneckLayer(nn.Module):
    """Decoder bottleneck (official implementation).

    1x1 reduce -> norm -> relu -> transpose conv up (or bilinear) ->
    1x1 project -> norm -> relu
    """
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3,
                    stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True,
                                  mode="bilinear")

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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (official implementation)."""
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
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y


class FAT_Net(nn.Module):
    """FAT-Net official architecture.

    ResNet34 CNN encoder + DeiT transformer encoder, SE attention at
    bottleneck, FAM blocks on skip connections, DecoderBottleneck decoder.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super(FAT_Net, self).__init__()

        # DeiT-tiny distilled transformer encoder.
        transformer = load_with_ssl_fallback(
            torch.hub.load,
            'facebookresearch/deit:main',
            'deit_tiny_distilled_patch16_224',
            pretrained=True)
        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )
        self._use_deit = True

        resnet = load_with_ssl_fallback(
            resnet_model.resnet34, pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512,
                                      kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512,
                                kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        emb = self.patch_embed(x)
        for i in range(12):
            emb = self.transformers[i](emb)
        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 192, h // 16, w // 16)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)

        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out


class FATNet(nn.Module):
    """FAT-Net wrapper with standard interface.

    Args:
        in_channels: Input channels (default 3).
        num_classes: Output segmentation classes (default 2).
        img_size: Input spatial size (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        self.model = FAT_Net(n_channels=in_channels, n_classes=num_classes)

    def forward(self, x):
        out = self.model(x)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear",
                                align_corners=True)
        return out
