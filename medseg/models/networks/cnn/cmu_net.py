"""
CMU-Net: A Strong ConvMixer-based Medical Ultrasound Image Segmentation Network
(Tang et al., ISBI 2023). Self-contained port for the medseg framework.

Upstream reference:
    https://github.com/FengheTan9/CMU-Net
        src/network/CMUNet.py
        src/network/msag.py

Architecture summary
--------------------
UNet topology with five encoder stages and a ConvMixer-based bottleneck:

    Encoder : conv_block stack with channels [64, 128, 256, 512, 1024]
    Bottleneck : ConvMixerBlock (depth=7, k=7) -- a stack of large-kernel
                 depthwise-conv + pointwise-conv residual blocks that plays the
                 role of a "Multi-axis Aggregation" mixer at the bottleneck.
    Skip connections : refined by MSAG (Multi-Scale Attention Gate) blocks that
                       fuse pointwise / 3x3 / dilated-3x3 convolutions and gate
                       the original feature map.
    Decoder : symmetric up_conv + conv_block path that mirrors the encoder.

Constructor exposes the medseg-standard signature
``(in_channels=3, num_classes=2, img_size=224, **kwargs)``.
"""
# Source: https://github.com/FengheTan9/CMU-Net

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Optional helper kept for fidelity with the SDK pattern (no timm download
# is actually triggered by this network -- CMU-Net trains from scratch).
# ---------------------------------------------------------------------------
def _load_with_ssl_fallback(load_fn, *args, **kwargs):
    import ssl
    import warnings
    try:
        return load_fn(*args, **kwargs)
    except Exception:
        prev = ssl._create_default_https_context
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            return load_fn(*args, **kwargs)
        except Exception as e2:
            warnings.warn(
                "Pretrained download failed (%s); using random init." % e2
            )
            return load_fn(*args, **{**kwargs, "pretrained": False})
        finally:
            ssl._create_default_https_context = prev


# ---------------------------------------------------------------------------
# Building blocks (all private so auto-discovery picks the top-level CMUNet).
# ---------------------------------------------------------------------------
class _Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class _ConvMixerBlock(nn.Module):
    """Stack of depthwise + pointwise conv residual blocks (large kernel)."""

    def __init__(self, dim: int = 1024, depth: int = 7, k: int = 7):
        super().__init__()
        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    _Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                dim,
                                dim,
                                kernel_size=(k, k),
                                groups=dim,
                                padding=(k // 2, k // 2),
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(dim),
                        )
                    ),
                    nn.Conv2d(dim, dim, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ConvBlock(nn.Module):
    """Two 3x3 conv + BN + ReLU layers (the classic UNet double-conv)."""

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _UpConv(nn.Module):
    """2x nearest upsample + 3x3 conv + BN + ReLU."""

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class _MSAG(nn.Module):
    """Multi-Scale Attention Gate used on the skip connections."""

    def __init__(self, channel: int):
        super().__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(
                channel,
                channel,
                kernel_size=3,
                padding=2,
                stride=1,
                dilation=2,
                bias=True,
            ),
            nn.BatchNorm2d(channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(channel * 3, channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        gate = self.relu(torch.cat((x1, x2, x3), dim=1))
        gate = self.voteConv(gate)
        return x + x * gate


# ---------------------------------------------------------------------------
# Top-level network.
# ---------------------------------------------------------------------------
class CMUNet(nn.Module):
    """CMU-Net (Tang et al., ISBI 2023).

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of output segmentation classes.
        img_size: Nominal input resolution; the network is fully convolutional
            and accepts any size that is divisible by 16, but the value is
            stored so downstream framework code can introspect it. The forward
            pass pads inputs up to a multiple of 16 internally and crops the
            logits back to the requested spatial resolution.
        l: depth of the ConvMixer bottleneck (default 7, as in the paper).
        k: kernel size of the ConvMixer depthwise conv (default 7).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: int = 224,
        l: int = 7,
        k: int = 7,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = _ConvBlock(in_channels, 64)
        self.Conv2 = _ConvBlock(64, 128)
        self.Conv3 = _ConvBlock(128, 256)
        self.Conv4 = _ConvBlock(256, 512)
        self.Conv5 = _ConvBlock(512, 1024)
        self.ConvMixer = _ConvMixerBlock(dim=1024, depth=l, k=k)

        # Decoder
        self.Up5 = _UpConv(1024, 512)
        self.Up_conv5 = _ConvBlock(512 * 2, 512)
        self.Up4 = _UpConv(512, 256)
        self.Up_conv4 = _ConvBlock(256 * 2, 256)
        self.Up3 = _UpConv(256, 128)
        self.Up_conv3 = _ConvBlock(128 * 2, 128)
        self.Up2 = _UpConv(128, 64)
        self.Up_conv2 = _ConvBlock(64 * 2, 64)
        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        # Multi-Scale Attention Gates on the skip connections.
        self.msag4 = _MSAG(512)
        self.msag3 = _MSAG(256)
        self.msag2 = _MSAG(128)
        self.msag1 = _MSAG(64)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # The network has four 2x downsamples, so pad to a multiple of 16.
        stride = 16
        pad_h = (stride - h % stride) % stride
        pad_w = (stride - w % stride) % stride
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.ConvMixer(x5)

        x4 = self.msag4(x4)
        x3 = self.msag3(x3)
        x2 = self.msag2(x2)
        x1 = self.msag1(x1)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)

        # Crop back to the original spatial size if we padded.
        if pad_h or pad_w:
            out = out[:, :, :h, :w]

        # Defensive resize in case shape arithmetic ever drifts.
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out


__all__ = ["CMUNet"]
