"""Basic neural network layers for graph convolution.

Source: https://github.com/SLDGroup/G-CASCADE (lib/gcn_lib/torch_nn.py)
Adapted from ViG (Huawei, 2022).
"""

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """Get activation layer by name."""
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'hswish':
        return nn.Hardswish(inplace)
    else:
        raise NotImplementedError(f'activation layer [{act}] is not found')


def norm_layer(norm, nc):
    """Get normalization layer by name."""
    norm = norm.lower()
    if norm == 'batch':
        return nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        return nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(f'normalization layer [{norm}] is not found')


class BasicConv(Seq):
    """Multi-layer conv sequence with optional norm/act/dropout."""

    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.,
                 kernel_size=1, padding=0, groups=4):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], kernel_size,
                            padding=padding, bias=bias, groups=groups))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))
        super().__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def batched_index_select(x, idx):
    """Fetch neighbor features from given neighbor indices.

    Args:
        x: (B, C, N, 1) input feature tensor.
        idx: (2, B, N, k) edge index tensor.

    Returns:
        (B, C, N, k) neighbor features.
    """
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)
    x = x.transpose(2, 1)
    x = x.contiguous().view(batch_size * num_vertices_reduced, -1)
    feature = x[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims)
    feature = feature.permute(0, 3, 1, 2).contiguous()
    return feature
