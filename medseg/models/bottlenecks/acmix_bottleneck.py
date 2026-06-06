"""ACmix bottleneck: 1:1 faithful port of the official LeapLabTHU/ACmix.

Reference: Pan et al., "On the Integration of Self-Attention and Convolution",
           CVPR 2022. https://arxiv.org/abs/2111.14556
Official code: https://github.com/LeapLabTHU/ACmix  (file: ACmix.py)

Key idea (faithfully reproduced):
    Both self-attention and a k_conv x k_conv convolution can be decomposed
    into the SAME stage 1 (three 1x1 projections producing q/k/v) followed
    by different stage 2 aggregations.  ACmix fuses the two paths with two
    learnable scalars ``rate1`` and ``rate2`` (initialised to 0.5).

Stage 1 (shared):
    q = conv1(x), k = conv2(x), v = conv3(x)     # all 1x1, out_channels each

Stage 2a — Self-attention (unfold-based, with relative positional encoding
``conv_p``):
    att = softmax( q @ (k + pe) over a k_att x k_att neighbourhood )
    out_att = att @ v

Stage 2b — Convolution (shift-based):
    f_all  = fc( cat[q,k,v] )                    # 3*head -> k_conv^2
    out_conv = dep_conv( f_conv )                # depth-wise k_conv conv
                                                 # initialised as identity
                                                 # shift kernels.

Output: ``rate1 * out_att + rate2 * out_conv``.

Bottleneck wrapper:
    ``ACmixBottleneck`` keeps in_channels == out_channels (no spatial down-
    sampling), wraps the official ACmix block with BN + ReLU and an additive
    residual so it plugs into the project's bottleneck registry interface.
"""
# Source: INTERNAL — framework adaptation (this repo).

import math

import torch
import torch.nn as nn

from medseg.registry import BOTTLENECK_REGISTRY


# ---------------- helpers (1:1 with official ACmix.py) ----------------

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


def position(H, W, device):
    """2-D normalised coordinates in [-1, 1].  device-agnostic version of the
    official `position()` helper (which hard-coded ``.cuda()``)."""
    loc_w = torch.linspace(-1.0, 1.0, W, device=device).unsqueeze(0).repeat(H, 1)
    loc_h = torch.linspace(-1.0, 1.0, H, device=device).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride_(x, stride):
    """Sub-sample by ``stride`` along H, W (matches official ``stride``)."""
    return x[:, :, ::stride, ::stride]


# ---------------- ACmix block (1:1 with official ACmix.py) ----------------

class ACmix(nn.Module):
    """Faithful port of the ACmix module (LeapLabTHU/ACmix/ACmix.py)."""

    def __init__(self, in_planes, out_planes, kernel_att=7, head=4,
                 kernel_conv=3, stride=1, dilation=1):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = nn.Parameter(torch.Tensor(1))
        self.rate2 = nn.Parameter(torch.Tensor(1))
        self.head_dim = out_planes // head

        # Shared 1x1 stage-1 projections (q / k / v).
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        # Positional encoder (2-D coord -> head_dim).
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        # Self-attention unfold.
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att,
                                padding=0, stride=self.stride)
        self.softmax = nn.Softmax(dim=1)

        # Conv branch: 3*head -> k_conv^2 score map -> identity-shift dep_conv.
        self.fc = nn.Conv2d(3 * self.head,
                            self.kernel_conv * self.kernel_conv,
                            kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(
            self.kernel_conv * self.kernel_conv * self.head_dim,
            out_planes, kernel_size=self.kernel_conv, bias=True,
            groups=self.head_dim, padding=1, stride=stride,
        )

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        # dep_conv is initialised as a set of identity-shift kernels:
        # the i-th input channel-group only sees the i-th spatial location of
        # its k_conv x k_conv neighbourhood, exactly matching a standard
        # k_conv x k_conv convolution at initialisation.
        kernel = torch.zeros(
            self.kernel_conv * self.kernel_conv,
            self.kernel_conv, self.kernel_conv,
        )
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride

        # ------------------ self-attention branch ------------------
        pe = self.conv_p(position(h, w, x.device))

        q_att = q.reshape(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.reshape(b * self.head, self.head_dim, h, w)
        v_att = v.reshape(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride_(q_att, self.stride)
            q_pe = stride_(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).reshape(
            b * self.head, self.head_dim,
            self.kernel_att * self.kernel_att, h_out, w_out,
        )
        unfold_rpe = self.unfold(self.pad_att(pe)).reshape(
            1, self.head_dim,
            self.kernel_att * self.kernel_att, h_out, w_out,
        )

        att = (q_att.unsqueeze(2) *
               (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).reshape(
            b * self.head, self.head_dim,
            self.kernel_att * self.kernel_att, h_out, w_out,
        )
        out_att = (att.unsqueeze(1) * out_att).sum(2).reshape(
            b, self.out_planes, h_out, w_out,
        )

        # ------------------ convolution branch ---------------------
        f_all = self.fc(torch.cat([
            q.reshape(b, self.head, self.head_dim, h * w),
            k.reshape(b, self.head, self.head_dim, h * w),
            v.reshape(b, self.head, self.head_dim, h * w),
        ], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(
            x.shape[0], -1, x.shape[-2], x.shape[-1],
        )
        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv


# ---------------- registry wrapper ----------------

@BOTTLENECK_REGISTRY.register("acmix")
class ACmixBottleneck(nn.Module):
    """ACmix bottleneck = official ACmix block + BN + ReLU + residual.

    Args:
        in_channels: Number of input channels (== output channels).
        kernel_att: Self-attention neighbourhood size (default 7, official).
        num_heads: Number of attention heads (default 4, official).
        kernel_conv: Convolution branch kernel size (default 3, official).
    """

    def __init__(self, in_channels, kernel_att=7, num_heads=4,
                 kernel_conv=3, **kwargs):
        super().__init__()
        # ACmix requires out_planes % head == 0; pick a safe head count.
        head = num_heads
        while head > 1 and in_channels % head != 0:
            head //= 2

        self.acmix = ACmix(
            in_planes=in_channels, out_planes=in_channels,
            kernel_att=kernel_att, head=head,
            kernel_conv=kernel_conv, stride=1, dilation=1,
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self._out_channels = in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        return self.act(self.norm(self.acmix(x))) + x
