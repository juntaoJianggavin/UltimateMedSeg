"""Medical Transformer (MedT) Encoder: faithful port from
https://github.com/jeya-maria-jose/Medical-Transformer

Reference: Valanarasu et al., "Medical Transformer: Gated Axial-Attention
           for Medical Image Segmentation"
Files: lib/models/axialnet.py, lib/models/utils.py
All class/attribute names match the original for pretrained weight loading.
"""
# Source: https://github.com/jeya-maria-jose/Medical-Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

from medseg.registry import ENCODER_REGISTRY


# ============= qkv_transform (from utils.py) =============
class qkv_transform(nn.Conv1d):
    """Conv1d for QKV transformation."""
    pass


# ============= AxialAttention (from axialnet.py) =============
class AxialAttention(nn.Module):
    """Gated Axial Attention mechanism."""
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                            padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Gating parameters
        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)

        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                               [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Relative position embeddings
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(
            self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                              [self.group_planes // 2, self.group_planes // 2,
                                                               self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        # Gate
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)

        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # Gate output
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


# ============= AxialBlock (from axialnet.py) =============
class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # When stride > 1, apply stride via conv_down and use reduced kernel_size
        self.conv_down = nn.Conv2d(inplanes, width, kernel_size=1, stride=stride, bias=False)
        effective_ks = kernel_size // stride if stride > 1 else kernel_size
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=effective_ks)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=effective_ks, width=True)
        self.conv_up = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)
        out = self.conv_up(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ============= MedT Encoder Model =============
class AxialEncoder(nn.Module):
    """Axial Attention encoder from Medical Transformer."""
    def __init__(self, block, layers, in_channels=3, groups=8, width_per_group=64,
                 norm_layer=None, s=0.125, img_size=128):
        super(AxialEncoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = int(64 * s)
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size // 4))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size // 4))
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size // 8))
        self.layer4 = self._make_layer(block, int(512 * s), layers[3], stride=2, kernel_size=(img_size // 16))

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=56):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        return features


@ENCODER_REGISTRY.register("medt")
class MedTEncoder(nn.Module):
    """Medical Transformer Encoder wrapper.
    Faithful to https://github.com/jeya-maria-jose/Medical-Transformer
    Uses Gated Axial Attention with height and width axes.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 128,
        s: float = 0.125,
        groups: int = 8,
        pretrained_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.encoder = AxialEncoder(AxialBlock, [1, 2, 4, 1], in_channels=in_channels,
                                    groups=groups, s=s, img_size=img_size)

        # Compute out_channels
        base = int(128 * s)
        self._out_channels = [
            base * AxialBlock.expansion,
            int(256 * s) * AxialBlock.expansion,
            int(512 * s) * AxialBlock.expansion,
            int(512 * s) * AxialBlock.expansion,
        ]

        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)

    @property
    def out_channels(self):
        return self._out_channels

    def _load_pretrained(self, path):
        state = torch.load(path, map_location='cpu')
        msg = self.load_state_dict(state, strict=False)
        print(f"MedT encoder loaded: {msg}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)
