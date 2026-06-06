"""DSCNet – self-contained port of Dynamic Snake Convolution Network.

Reference:
    Qi et al., "Dynamic Snake Convolution based on Topological Geometric
    Constraints for Tubular Structure Segmentation", ICCV 2023.
    https://github.com/YaoleiQi/DSCNet

Architecture: A standard 5-level UNet (encoder channels [32, 64, 128, 256, 512])
where regular 3x3 convolutions in the deeper stages are augmented with a pair
of *Dynamic Snake Convolutions* (DSConv).  DSConv is a deformable 1-D
convolution whose per-position offsets are constrained to form a continuous
"snake" path along either the horizontal (x-axis) or vertical (y-axis)
direction — well suited for thin tubular structures such as vessels or
elongated gland boundaries.

Implementation is pure-PyTorch (no torchvision deformable-conv op).  The snake
sampling locations are produced by:

    1. predicting K offsets (one per kernel position) along the perpendicular
       axis from a small 3x3 conv,
    2. squashing them with tanh and an *extend_scope* factor,
    3. taking a cumulative sum along the kernel axis to enforce continuity,
       then re-centring on the middle position so the kernel anchor stays put,
    4. building a (B, K, H, W, 2) sampling grid and feeding it through
       ``F.grid_sample`` (bilinear).

The sampled tensor lays out the K kernel positions along the width (morph=0)
or height (morph=1) dimension; a stride-K 1xK / Kx1 conv then collapses them.

To bound activation memory for the K=9 kernel at large input sizes (e.g.
512x512), DSConv is enabled only at levels with spatial resolution <= H/4
(encoder levels 3-5 and matching decoder levels) — top levels fall back to
standard double-conv, which is consistent with the upstream "DSConv replaces
regular convs in selected positions" wording.
"""
# Source: https://github.com/YaoleiQi/DSCNet

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pretrained-loader SSL fallback (required by spec; kept for completeness even
# though this port does not load external weights).
# ---------------------------------------------------------------------------
def _load_with_ssl_fallback(load_fn, *args, **kwargs):
    import ssl, warnings
    try:
        return load_fn(*args, **kwargs)
    except Exception:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            warnings.warn('Pretrained download failed (%s); using random init.' % e2)
            return load_fn(*args, **{**kwargs, 'pretrained': False})
        finally:
            ssl._create_default_https_context = prev


# ---------------------------------------------------------------------------
# Dynamic Snake Convolution (DSConv)
# ---------------------------------------------------------------------------
class _DSConv(nn.Module):
    """Dynamic Snake Convolution.

    A 1-D kernel of length ``kernel_size`` is laid along either the x-axis
    (``morph=0``) or the y-axis (``morph=1``).  At every spatial location the
    K kernel positions are displaced along the perpendicular axis by
    cumulative tanh-squashed offsets, forming a continuous "snake" path.

    Implemented with ``F.grid_sample`` + a stride-K 1-D convolution, so the
    snake-shaped receptive field is materialised explicitly without any
    custom CUDA op.
    """

    def __init__(self, in_ch, out_ch, kernel_size=9, morph=0, extend_scope=1.0):
        super().__init__()
        assert morph in (0, 1)
        self.kernel_size = int(kernel_size)
        self.morph = morph
        self.extend_scope = float(extend_scope)

        # Offset predictor: produces K offsets per location for the
        # perpendicular axis.
        self.offset_conv = nn.Conv2d(in_ch, self.kernel_size, kernel_size=3, padding=1)
        self.offset_bn = nn.BatchNorm2d(self.kernel_size)

        if morph == 0:
            self.dsc_conv = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.kernel_size),
                padding=0,
                bias=False,
            )
        else:
            self.dsc_conv = nn.Conv2d(
                in_ch, out_ch,
                kernel_size=(self.kernel_size, 1),
                stride=(self.kernel_size, 1),
                padding=0,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        K = self.kernel_size

        # (B, K, H, W) tanh-squashed offsets along the perpendicular axis
        offset = self.offset_bn(self.offset_conv(x))
        offset = torch.tanh(offset) * self.extend_scope

        # Snake continuity: cumulative sum then re-centre on the middle.
        offset_cum = torch.cumsum(offset, dim=1)
        center = offset_cum[:, K // 2:K // 2 + 1]
        offset_cum = offset_cum - center

        device, dtype = x.device, x.dtype
        ys = torch.arange(H, device=device, dtype=dtype)
        xs = torch.arange(W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)
        k_idx = torch.arange(K, device=device, dtype=dtype) - (K - 1) / 2  # (K,)

        if self.morph == 0:
            # horizontal kernel; offset displaces along y
            sample_y = grid_y[None, None] + offset_cum                                # (B, K, H, W)
            sample_x = grid_x[None, None] + k_idx[None, :, None, None].expand_as(offset_cum)
        else:
            sample_y = grid_y[None, None] + k_idx[None, :, None, None].expand_as(offset_cum)
            sample_x = grid_x[None, None] + offset_cum

        # Normalise to [-1, 1] for grid_sample.
        denom_w = max(W - 1, 1)
        denom_h = max(H - 1, 1)
        sample_x_n = 2.0 * sample_x / denom_w - 1.0
        sample_y_n = 2.0 * sample_y / denom_h - 1.0
        grid = torch.stack([sample_x_n, sample_y_n], dim=-1)  # (B, K, H, W, 2)

        if self.morph == 0:
            # Place K kernel positions along the width axis.
            grid = grid.permute(0, 2, 3, 1, 4).reshape(B, H, W * K, 2)
        else:
            grid = grid.permute(0, 2, 1, 3, 4).reshape(B, H * K, W, 2)

        sampled = F.grid_sample(
            x, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
        )
        out = self.dsc_conv(sampled)
        out = self.bn(out)
        out = self.act(out)
        return out


# ---------------------------------------------------------------------------
# Encoder/Decoder block: standard 3x3 conv path fused with two DSConv paths
# (along x and y).  use_dsc=False falls back to a plain double-conv.
# ---------------------------------------------------------------------------
class _DSCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=9, use_dsc=True):
        super().__init__()
        self.use_dsc = use_dsc
        self.std = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        if use_dsc:
            self.dsc_x = _DSConv(in_ch, out_ch, kernel_size, morph=0)
            self.dsc_y = _DSConv(in_ch, out_ch, kernel_size, morph=1)
            fused_in = out_ch * 3
        else:
            fused_in = out_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_in, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        s = self.std(x)
        if self.use_dsc:
            f = torch.cat([s, self.dsc_x(x), self.dsc_y(x)], dim=1)
        else:
            f = s
        return self.fuse(f)


# ---------------------------------------------------------------------------
# DSCNet: 5-level UNet with snake-conv-augmented blocks at the deeper stages.
# ---------------------------------------------------------------------------
class DSCNet(nn.Module):
    """Dynamic Snake Convolution Network (ICCV 2023)."""

    def __init__(self, in_channels=3, num_classes=2, img_size=224, kernel_size=9, **kwargs):
        super().__init__()
        self.img_size = int(img_size)
        self.kernel_size = int(kernel_size)
        chs = [32, 64, 128, 256, 512]

        # Top two levels: plain double-conv (DSConv at full resolution would
        # materialise an HxKW sampled tensor that is wasteful for 512x512
        # inputs).  Deeper levels carry the snake-convolution paths.
        self.enc1 = _DSCBlock(in_channels, chs[0], kernel_size, use_dsc=False)
        self.enc2 = _DSCBlock(chs[0],     chs[1], kernel_size, use_dsc=False)
        self.enc3 = _DSCBlock(chs[1],     chs[2], kernel_size, use_dsc=True)
        self.enc4 = _DSCBlock(chs[2],     chs[3], kernel_size, use_dsc=True)
        self.enc5 = _DSCBlock(chs[3],     chs[4], kernel_size, use_dsc=True)
        self.pool = nn.MaxPool2d(2)

        self.up4  = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = _DSCBlock(chs[3] * 2, chs[3], kernel_size, use_dsc=True)
        self.up3  = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = _DSCBlock(chs[2] * 2, chs[2], kernel_size, use_dsc=True)
        self.up2  = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = _DSCBlock(chs[1] * 2, chs[1], kernel_size, use_dsc=False)
        self.up1  = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = _DSCBlock(chs[0] * 2, chs[0], kernel_size, use_dsc=False)

        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        B, C, H_in, W_in = x.shape

        # Pad up so 4 max-pools and 4 transposed-conv2 ups land back exactly.
        ph = (16 - H_in % 16) % 16
        pw = (16 - W_in % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.head(d1)

        # Crop back to the original spatial size.
        if out.shape[-2] != H_in or out.shape[-1] != W_in:
            out = out[..., :H_in, :W_in]
        if out.shape[-2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in), mode='bilinear', align_corners=False)
        return out
