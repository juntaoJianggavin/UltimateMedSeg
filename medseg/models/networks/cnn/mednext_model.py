"""MedNeXt: Transformer-driven Scaling of ConvNets for Medical Segmentation.

Reference:
    Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for
    Medical Image Segmentation", MICCAI 2023.
    https://github.com/MIC-DKFZ/MedNeXt

2D adaptation of the fully ConvNeXt-based UNet architecture with
MedNeXt blocks (depthwise conv + expansion + residual).
"""
# Source: https://github.com/MIC-DKFZ/MedNeXt

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_norm(norm_type, num_channels, n_groups=None):
    """Get normalization layer."""
    if norm_type == 'group':
        # Official MedNeXt uses num_groups = num_channels (per-channel GroupNorm)
        n_groups = n_groups if n_groups else num_channels
        return nn.GroupNorm(n_groups, num_channels)
    return nn.LayerNorm(num_channels)


class _MedNeXtBlock(nn.Module):
    """ConvNeXt-style block for MedNeXt.

    Depthwise conv → LayerNorm/GroupNorm → Linear expansion → GELU →
    Linear contraction → residual connection.
    """

    def __init__(self, in_ch, out_ch, exp_r=4, kernel_size=7,
                 do_res=True, norm_type='group', n_groups=None):
        super().__init__()
        self.do_res = do_res

        # Depthwise convolution
        self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size,
                                  padding=kernel_size // 2,
                                  groups=in_ch)

        # Normalization
        self.norm = _get_norm(norm_type, in_ch, n_groups)

        # Expansion + contraction
        exp_ch = in_ch * exp_r
        self.fc1 = nn.Conv2d(in_ch, exp_ch, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(exp_ch, out_ch, 1)

        # Residual projection if needed
        if do_res and in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        res = x if self.do_res else None

        out = self.dw_conv(x)
        # Normalize: need to handle both GroupNorm and LayerNorm
        if isinstance(self.norm, nn.LayerNorm):
            out = out.permute(0, 2, 3, 1)
            out = self.norm(out)
            out = out.permute(0, 3, 1, 2)
        else:
            out = self.norm(out)

        # Official order: norm → expand_conv → GELU → compress_conv
        out = self.act(self.fc1(out))
        out = self.fc2(out)

        if self.do_res:
            if hasattr(self, 'res_conv'):
                res = self.res_conv(res)
            out = out + res

        return out


class _MedNeXtDownBlock(nn.Module):
    """2x downsampling with MedNeXt block."""

    def __init__(self, in_ch, out_ch, exp_r=4, kernel_size=7,
                 do_res=True, norm_type='group'):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.block = _MedNeXtBlock(out_ch, out_ch, exp_r, kernel_size,
                                    do_res, norm_type)

    def forward(self, x):
        x = self.down(x)
        return self.block(x)


class _MedNeXtUpBlock(nn.Module):
    """2x upsampling with MedNeXt block."""

    def __init__(self, in_ch, out_ch, exp_r=4, kernel_size=7,
                 do_res=True, norm_type='group'):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = _MedNeXtBlock(out_ch, out_ch, exp_r, kernel_size,
                                    do_res, norm_type)

    def forward(self, x):
        x = self.up(x)
        # Official MedNeXt pads after upsample for spatial alignment (2d: pad(1,0,1,0))
        x = F.pad(x, (1, 0, 1, 0))
        return self.block(x)


class MedNeXt(nn.Module):
    """MedNeXt 2D: ConvNeXt-based UNet for medical image segmentation.

    Architecture follows the MedNeXt v1 design with configurable depth,
    expansion ratio, and kernel size.

    Args:
        in_channels: number of input channels.
        num_classes: number of output segmentation classes.
        img_size: spatial size (unused, kept for API consistency).
        model_id: 'S' (small), 'B' (base), 'M' (medium), 'L' (large).
        kernel_size: kernel size for depthwise convolutions.
    """

    _CONFIGS = {
        'S': {'n_channels': 32, 'exp_r': 4, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'B': {'n_channels': 64, 'exp_r': 4, 'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2]},
        'M': {'n_channels': 96, 'exp_r': 4, 'block_counts': [3, 4, 4, 4, 4, 4, 4, 4, 3]},
        'L': {'n_channels': 128, 'exp_r': 4, 'block_counts': [3, 4, 8, 8, 8, 8, 8, 4, 3]},
    }

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 model_id='S', kernel_size=3, **kwargs):
        super().__init__()
        cfg = self._CONFIGS[model_id]
        n_ch = cfg['n_channels']
        exp_r = cfg['exp_r']
        bc = cfg['block_counts']
        ks = kernel_size

        # Stem (official uses kernel_size=1)
        self.stem = nn.Conv2d(in_channels, n_ch, 1)

        # Encoder: each stage keeps spatial, next down block halves
        self.enc1 = self._make_stage(n_ch, n_ch, bc[0], exp_r, ks, down=False)  # /1
        self.down1 = _MedNeXtDownBlock(n_ch, 2 * n_ch, exp_r, ks)               # /2
        self.enc2 = self._make_stage(2 * n_ch, 2 * n_ch, bc[1], exp_r, ks, down=False)
        self.down2 = _MedNeXtDownBlock(2 * n_ch, 4 * n_ch, exp_r, ks)           # /4
        self.enc3 = self._make_stage(4 * n_ch, 4 * n_ch, bc[2], exp_r, ks, down=False)
        self.down3 = _MedNeXtDownBlock(4 * n_ch, 8 * n_ch, exp_r, ks)           # /8
        self.enc4 = self._make_stage(8 * n_ch, 8 * n_ch, bc[3], exp_r, ks, down=False)

        # Bottleneck
        self.bottleneck = self._make_stage(8 * n_ch, 8 * n_ch, bc[4], exp_r, ks,
                                            down=False)                          # /8

        # Decoder: upsample + additive skip + conv (official: x_res + x_up)
        self.up3 = _MedNeXtUpBlock(8 * n_ch, 4 * n_ch, exp_r, ks)               # /4
        self.dec3 = self._make_stage(4 * n_ch, 4 * n_ch, bc[5], exp_r, ks, down=False)

        self.up2 = _MedNeXtUpBlock(4 * n_ch, 2 * n_ch, exp_r, ks)               # /2
        self.dec2 = self._make_stage(2 * n_ch, 2 * n_ch, bc[6], exp_r, ks, down=False)

        self.up1 = _MedNeXtUpBlock(2 * n_ch, n_ch, exp_r, ks)                   # /1
        self.dec1 = self._make_stage(n_ch, n_ch, bc[7], exp_r, ks, down=False)

        # Output
        self.out_conv = nn.Conv2d(n_ch, num_classes, 1)

    @staticmethod
    def _make_stage(in_ch, out_ch, num_blocks, exp_r, ks, down=False):
        layers = []
        if down:
            layers.append(_MedNeXtDownBlock(in_ch, out_ch, exp_r, ks))
            in_ch = out_ch
            num_blocks -= 1
        elif in_ch != out_ch:
            layers.append(_MedNeXtBlock(in_ch, out_ch, exp_r, ks))
            in_ch = out_ch
            num_blocks -= 1
        for _ in range(max(num_blocks, 0)):
            layers.append(_MedNeXtBlock(in_ch, in_ch, exp_r, ks))
        return nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x):
        inp_size = x.shape[2:]

        # Stem
        s = self.stem(x)

        # Encoder (save features BEFORE downsampling)
        e1 = self.enc1(s)          # /1, n_ch
        d1 = self.down1(e1)        # /2
        e2 = self.enc2(d1)         # /2, 2*n_ch
        d2 = self.down2(e2)        # /4
        e3 = self.enc3(d2)         # /4, 4*n_ch
        d3 = self.down3(e3)        # /8
        e4 = self.enc4(d3)         # /8, 8*n_ch

        # Bottleneck
        b = self.bottleneck(e4)    # /8

        # Decoder with additive skip connections (official: x_res + x_up)
        d = self.up3(b)                              # /4
        d = self.dec3(d + e3)                        # /4, add skip

        d = self.up2(d)                              # /2
        d = self.dec2(d + e2)                        # /2, add skip

        d = self.up1(d)                              # /1
        d = self.dec1(d + e1)                        # /1, add skip

        out = self.out_conv(d)

        if out.shape[-2:] != torch.Size(inp_size):
            out = F.interpolate(out, size=inp_size, mode='bilinear',
                                align_corners=True)
        return out
