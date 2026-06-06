"""MedT – self-contained port from github.com/jeya-maria-jose/Medical-Transformer.

Standard interface:
    model = MedT(in_channels=3, num_classes=2, img_size=224)
    out = model(x)  # -> (B, num_classes, H, W)
"""
# Source: https://github.com/jeya-maria-jose/Medical-Transformer

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── qkv_transform (from utils.py, just an nn.Conv1d alias) ──────────────────
class qkv_transform(nn.Conv1d):
    """Conv1d alias for axial attention qkv projection."""


# ── Axial Attention modules ──────────────────────────────────────────────────
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.qkv_transform = qkv_transform(
            in_planes, out_planes * 2, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)
        self.relative = nn.Parameter(
            torch.randn(self.group_planes * 2, kernel_size * 2 - 1),
            requires_grad=True)
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
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(
            qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
            [self.group_planes // 2, self.group_planes // 2,
             self.group_planes], dim=2)
        all_emb = torch.index_select(
            self.relative, 1, self.flatten_index).view(
                self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_emb, k_emb, v_emb = torch.split(
            all_emb, [self.group_planes // 2, self.group_planes // 2,
                      self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_emb)
        kr = torch.einsum('bgci,cij->bgij', k, k_emb).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked = torch.cat([qk, qr, kr], dim=1)
        stacked = self.bn_similarity(stacked).view(
            N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_emb)
        stacked_out = torch.cat([sv, sve], dim=-1).view(
            N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_out).view(
            N, W, self.out_planes, 2, H).sum(dim=-2)
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)
        if self.stride > 1:
            output = self.pooling(output)
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(
            0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialAttention_dynamic(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.width = width
        self.qkv_transform = qkv_transform(
            in_planes, out_planes * 2, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)
        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.relative = nn.Parameter(
            torch.randn(self.group_planes * 2, kernel_size * 2 - 1),
            requires_grad=True)
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
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(
            qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
            [self.group_planes // 2, self.group_planes // 2,
             self.group_planes], dim=2)
        all_emb = torch.index_select(
            self.relative, 1, self.flatten_index).view(
                self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_emb, k_emb, v_emb = torch.split(
            all_emb, [self.group_planes // 2, self.group_planes // 2,
                      self.group_planes], dim=0)
        qr = torch.mul(torch.einsum('bgci,cij->bgij', q, q_emb), self.f_qr)
        kr = torch.mul(
            torch.einsum('bgci,cij->bgij', k, k_emb).transpose(2, 3),
            self.f_kr)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked = torch.cat([qk, qr, kr], dim=1)
        stacked = self.bn_similarity(stacked).view(
            N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked, dim=3)
        sv = torch.mul(
            torch.einsum('bgij,bgcj->bgci', similarity, v), self.f_sv)
        sve = torch.mul(
            torch.einsum('bgij,cij->bgci', similarity, v_emb), self.f_sve)
        stacked_out = torch.cat([sv, sve], dim=-1).view(
            N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_out).view(
            N, W, self.out_planes, 2, H).sum(dim=-2)
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)
        if self.stride > 1:
            output = self.pooling(output)
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(
            0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.width = width
        self.qkv_transform = qkv_transform(
            in_planes, out_planes * 2, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 1)
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
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(
            qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
            [self.group_planes // 2, self.group_planes // 2,
             self.group_planes], dim=2)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        sim = self.bn_similarity(qk).reshape(
            N * W, 1, self.groups, H, H).sum(dim=1).contiguous()
        similarity = F.softmax(sim, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sv = sv.reshape(N * W, self.out_planes, H).contiguous()
        output = self.bn_output(sv).reshape(
            N, W, self.out_planes, 1, H).sum(dim=-2).contiguous()
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)
        if self.stride > 1:
            output = self.pooling(output)
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(
            0, math.sqrt(1. / self.in_planes))


# ── Axial Blocks ─────────────────────────────────────────────────────────────
class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_dynamic(
            width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_dynamic(
            width, width, groups=groups, kernel_size=kernel_size,
            stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv_down(x)))
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)
        out = self.bn2(self.conv_up(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_wopos(
            width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_wopos(
            width, width, groups=groups, kernel_size=kernel_size,
            stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv_down(x)))
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)
        out = self.bn2(self.conv_up(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ── MedT model ───────────────────────────────────────────────────────────────
class _medt_net(nn.Module):
    def __init__(self, block, block_2, layers, num_classes=2,
                 groups=8, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None,
                 s=0.125, img_size=128, imgchan=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(imgchan, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, 128, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, int(128 * s), layers[0], kernel_size=(img_size // 2))
        self.layer2 = self._make_layer(
            block, int(256 * s), layers[1], stride=2,
            kernel_size=(img_size // 2),
            dilate=replace_stride_with_dilation[0])
        self.decoder4 = nn.Conv2d(int(512 * s), int(256 * s),
                                  kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(int(256 * s), int(128 * s),
                                  kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv2d(int(128 * s), num_classes,
                                kernel_size=1, stride=1, padding=0)
        # patch branch
        self.conv1_p = nn.Conv2d(imgchan, self.inplanes, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        self.conv2_p = nn.Conv2d(self.inplanes, 128, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.conv3_p = nn.Conv2d(128, self.inplanes, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)
        self.relu_p = nn.ReLU(inplace=True)
        img_size_p = img_size // 4
        # Compute spatial dims and cap kernel sizes for patch branch
        sp = img_size_p // 2  # spatial after conv1_p (stride=2)
        ks1 = min(img_size_p // 2, sp)
        sp2 = sp // 2  # after layer2_p stride=2
        ks2 = min(img_size_p // 2, sp)
        sp3 = sp2 // 2  # after layer3_p stride=2
        ks3 = min(img_size_p // 4, sp2)
        ks4 = min(img_size_p // 8, sp3)
        self.layer1_p = self._make_layer(
            block_2, int(128 * s), layers[0], kernel_size=ks1)
        self.layer2_p = self._make_layer(
            block_2, int(256 * s), layers[1], stride=2,
            kernel_size=ks2,
            dilate=replace_stride_with_dilation[0])
        self.layer3_p = self._make_layer(
            block_2, int(512 * s), layers[2], stride=2,
            kernel_size=ks3,
            dilate=replace_stride_with_dilation[1])
        self.layer4_p = self._make_layer(
            block_2, int(1024 * s), layers[3], stride=1,
            kernel_size=ks4,
            dilate=replace_stride_with_dilation[2])
        self.decoder1_p = nn.Conv2d(int(1024 * 2 * s), int(1024 * 2 * s),
                                    kernel_size=3, stride=2, padding=1)
        self.decoder2_p = nn.Conv2d(int(1024 * 2 * s), int(512 * 2 * s),
                                    kernel_size=3, stride=1, padding=1)
        self.decoder3_p = nn.Conv2d(int(512 * 2 * s), int(256 * 2 * s),
                                    kernel_size=3, stride=1, padding=1)
        self.decoder4_p = nn.Conv2d(int(256 * 2 * s), int(128 * 2 * s),
                                    kernel_size=3, stride=1, padding=1)
        self.decoder5_p = nn.Conv2d(int(128 * 2 * s), int(128 * s),
                                    kernel_size=3, stride=1, padding=1)
        self.decoderf = nn.Conv2d(int(128 * s), int(128 * s),
                                  kernel_size=3, stride=1, padding=1)
        # Store patch_size for dynamic patch extraction
        self._patch_size = img_size // 4

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1,
                    dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups=self.groups, base_width=self.base_width,
                            dilation=previous_dilation, norm_layer=norm_layer,
                            kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer,
                                kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        xin = x.clone()
        # Global branch
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x = F.relu(F.interpolate(self.decoder4(x2), scale_factor=(2, 2),
                                 mode='bilinear'))
        x = torch.add(x, x1)
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2, 2),
                                 mode='bilinear'))
        # Local (patch) branch
        x_loc = x.clone()
        ps = self._patch_size
        for i in range(0, 4):
            for j in range(0, 4):
                x_p = xin[:, :, ps * i:ps * (i + 1), ps * j:ps * (j + 1)]
                x_p = self.relu_p(self.bn1_p(self.conv1_p(x_p)))
                x_p = self.relu_p(self.bn2_p(self.conv2_p(x_p)))
                x_p = self.relu_p(self.bn3_p(self.conv3_p(x_p)))
                x1_p = self.layer1_p(x_p)
                x2_p = self.layer2_p(x1_p)
                x3_p = self.layer3_p(x2_p)
                x4_p = self.layer4_p(x3_p)
                x_p = F.relu(F.interpolate(self.decoder1_p(x4_p),
                             size=x4_p.shape[2:], mode='bilinear',
                             align_corners=False))
                x_p = torch.add(x_p, x4_p)
                x_p = F.relu(F.interpolate(self.decoder2_p(x_p),
                             size=x3_p.shape[2:], mode='bilinear',
                             align_corners=False))
                x_p = torch.add(x_p, x3_p)
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p),
                             size=x2_p.shape[2:], mode='bilinear',
                             align_corners=False))
                x_p = torch.add(x_p, x2_p)
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p),
                             size=x1_p.shape[2:], mode='bilinear',
                             align_corners=False))
                x_p = torch.add(x_p, x1_p)
                x_p = F.relu(F.interpolate(self.decoder5_p(x_p),
                             size=(ps, ps), mode='bilinear',
                             align_corners=False))
                x_loc[:, :, ps * i:ps * (i + 1),
                      ps * j:ps * (j + 1)] = x_p
        x = torch.add(x, x_loc)
        x = F.relu(self.decoderf(x))
        return self.adjust(F.relu(x))


class MedT(nn.Module):
    """MedT wrapper with standard interface.

    Args:
        in_channels (int): Number of input channels (default: 3).
        num_classes (int): Number of output classes (default: 2).
        img_size (int): Input image size (default: 224).
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        self.model = _medt_net(
            AxialBlock_dynamic, AxialBlock_wopos, [1, 2, 4, 1],
            num_classes=num_classes, s=0.125,
            img_size=img_size, imgchan=in_channels)

    def forward(self, x):
        return self.model(x)
