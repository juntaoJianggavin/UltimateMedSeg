"""LDNet: Lesion-aware Dynamic Kernel Network for Polyp Segmentation.

Reference:
    Ruifei Zhang, Peiwen Lai, Xiang Wan, De-Jun Fan, Feng Gao,
    Xiao-Jian Wu, Guanbin Li.
    "Lesion-aware Dynamic Kernel for Polyp Segmentation."
    MICCAI 2022.
    Upstream code: https://github.com/ReaFly/LDNet

Architecture overview:
    - Encoder: Res2Net-50 (26w_4s) via timm, returning multi-scale features
      at strides {2, 4, 8, 16, 32} with channels {64, 256, 512, 1024, 2048}.
    - Channel reduction conv 1x1 on stages 2..5 (and identity for stage 1).
    - UNet-style decoder with bilinear upsample + skip concatenation.
    - Lesion-aware Dynamic Convolution (LDC): a per-image conv kernel is
      generated from a coarse lesion mask predicted by a lightweight head
      seeded from the global pooled bottleneck feature.
    - At each decoder stage, the kernel is iteratively refined by a
      HeadUpdator that fuses the previous kernel with mask-pooled features.
    - LCA (lesion-aware cross-attention) refines decoder features using the
      coarse mask; ESA (efficient self-attention) refines encoder skips.

Self-contained: only torch and timm are required.
"""
# Source: https://github.com/ReaFly/LDNet

import os

# Keep huggingface_hub timeouts short so construction does not stall.
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '3')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '5')

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from medseg.models.networks.sam.sam_base import load_with_ssl_fallback


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------
class _ConvBlock(nn.Module):
    """Conv + BN + ReLU."""

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _DecoderBlock(nn.Module):
    """ConvBlock x2 followed by bilinear upsample (matches upstream)."""

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = _ConvBlock(in_c, in_c // 4,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = _ConvBlock(in_c // 4, out_c,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


# ---------------------------------------------------------------------------
# Efficient Self-Attention (ESA) and Lesion-aware Cross-Attention (LCA)
# (einops-free port of upstream modules)
# ---------------------------------------------------------------------------
class _PPM(nn.Module):
    """Pyramid Pooling Module producing flattened key/value tokens."""

    def __init__(self, pooling_sizes=(1, 3, 5)):
        super().__init__()
        self.pools = nn.ModuleList(
            [nn.AdaptiveAvgPool2d(output_size=(s, s)) for s in pooling_sizes]
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        return torch.cat([p(x).view(b, c, -1) for p in self.pools], dim=-1)


class _PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _ESALayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, bias=False)
        self.ppm = _PPM((1, 3, 5))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, _, h, w = x.shape
        qkv = self.to_qkv(x)
        inner = qkv.shape[1] // 3
        q, k, v = qkv[:, :inner], qkv[:, inner:2 * inner], qkv[:, 2 * inner:]

        # q: (b, head, h*w, dim_head)
        q = q.view(b, self.heads, self.dim_head, h * w).permute(0, 1, 3, 2)
        # k, v through PPM -> (b, inner, n_kv)
        k = self.ppm(k)
        v = self.ppm(v)
        n_kv = k.shape[-1]
        k = k.view(b, self.heads, self.dim_head, n_kv).permute(0, 1, 3, 2)
        v = v.view(b, self.heads, self.dim_head, n_kv).permute(0, 1, 3, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)  # (b, head, h*w, dim_head)
        out = out.permute(0, 2, 1, 3).reshape(b, h * w, self.heads * self.dim_head)
        return self.to_out(out)


class _ESABlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.0):
        super().__init__()
        self.attn = _ESALayer(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = _PreNorm(dim, _FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (b, h*w, c)
        tokens = self.attn(x) + tokens
        tokens = self.ff(tokens) + tokens
        return tokens.transpose(1, 2).reshape(b, c, h, w)


def _mask_average_pooling(x, mask):
    """Mask-weighted average pooling -> (b, c, 1) feature per sample."""
    mask = torch.sigmoid(mask)
    h, w = x.shape[2], x.shape[3]
    eps = 5e-4
    x_mask = x * mask
    area = F.avg_pool2d(mask, (h, w)) * h * w + eps
    feat = F.avg_pool2d(x_mask, (h, w)) * h * w / area
    return feat.view(x.size(0), x.size(1), -1)


class _LCALayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x, mask):
        b, _, h, w = x.shape
        qkv = self.to_qkv(x)
        inner = qkv.shape[1] // 3
        q, k, v = qkv[:, :inner], qkv[:, inner:2 * inner], qkv[:, 2 * inner:]

        q = q.view(b, self.heads, self.dim_head, h * w).permute(0, 1, 3, 2)

        # Mask must broadcast over per-class channel; if mask has multiple
        # channels, average them so MAP returns a single token per head.
        if mask.shape[1] > 1:
            mask = mask.mean(dim=1, keepdim=True)
        k = _mask_average_pooling(k, mask)
        v = _mask_average_pooling(v, mask)
        n_kv = k.shape[-1]
        k = k.view(b, self.heads, self.dim_head, n_kv).permute(0, 1, 3, 2)
        v = v.view(b, self.heads, self.dim_head, n_kv).permute(0, 1, 3, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, h * w, self.heads * self.dim_head)
        return self.to_out(out)


class _LCABlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=512, dropout=0.0):
        super().__init__()
        self.attn = _LCALayer(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = _PreNorm(dim, _FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x, mask):
        b, c, h, w = x.shape
        # Mask may be at a coarser resolution than x; align to x first.
        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.attn(x, mask) + tokens
        tokens = self.ff(tokens) + tokens
        return tokens.transpose(1, 2).reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# Head Updator (Lesion-aware dynamic kernel refinement)
# ---------------------------------------------------------------------------
class _HeadUpdator(nn.Module):
    def __init__(self, in_channels=64, feat_channels=64,
                 out_channels=None, conv_kernel_size=1):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.num_in = self.feat_channels
        self.num_out = self.feat_channels

        self.pred_transform_layer = nn.Linear(self.in_channels, self.num_in + self.num_out)
        self.head_transform_layer = nn.Linear(self.in_channels, self.num_in + self.num_out)

        self.pred_gate = nn.Linear(self.num_in, self.feat_channels)
        self.head_gate = nn.Linear(self.num_in, self.feat_channels)

        self.pred_norm_in = nn.LayerNorm(self.feat_channels)
        self.head_norm_in = nn.LayerNorm(self.feat_channels)
        self.pred_norm_out = nn.LayerNorm(self.feat_channels)
        self.head_norm_out = nn.LayerNorm(self.feat_channels)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels)
        self.fc_norm = nn.LayerNorm(self.feat_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, feat, head, pred):
        # feat: (B, C, H, W), head: (B, N, C, K, K), pred: (B, N, h, w)
        bs, num_classes = head.shape[:2]

        # Align pred to feat resolution (upsample-by-2 in upstream).
        if pred.shape[-2:] != feat.shape[-2:]:
            pred = F.interpolate(pred, size=feat.shape[-2:], mode='bilinear',
                                 align_corners=False)
        pred = torch.sigmoid(pred)

        # Assemble: pool feat by each class mask -> (B, N, C)
        assemble_feat = torch.einsum('bnhw,bchw->bnc', pred, feat)

        # head: (B, N, C, K, K) -> (B, N, K*K, C)
        head = head.reshape(bs, num_classes, self.in_channels, -1).permute(0, 1, 3, 2)

        assemble_feat = assemble_feat.reshape(-1, self.in_channels)  # (B*N, C)
        bs_num = assemble_feat.size(0)

        pred_feat = self.pred_transform_layer(assemble_feat)  # (B*N, in+out)
        pred_feat_in = pred_feat[:, :self.num_in].view(-1, self.feat_channels)
        pred_feat_out = pred_feat[:, -self.num_out:].view(-1, self.feat_channels)

        head_feat = self.head_transform_layer(head.reshape(bs_num, -1, self.in_channels))
        head_feat_in = head_feat[..., :self.num_in]
        head_feat_out = head_feat[..., -self.num_out:]

        gate_feat = head_feat_in * pred_feat_in.unsqueeze(-2)  # (B*N, K*K, in)

        head_gate = torch.sigmoid(self.head_norm_in(self.head_gate(gate_feat)))
        pred_gate = torch.sigmoid(self.pred_norm_in(self.pred_gate(gate_feat)))

        head_feat_out = self.head_norm_out(head_feat_out)
        pred_feat_out = self.pred_norm_out(pred_feat_out)

        update_head = pred_gate * pred_feat_out.unsqueeze(-2) + head_gate * head_feat_out
        update_head = self.fc_layer(update_head)
        update_head = self.fc_norm(update_head)
        update_head = self.activation(update_head)

        update_head = update_head.reshape(bs, num_classes, -1, self.feat_channels)
        update_head = update_head.permute(0, 1, 3, 2).reshape(
            bs, num_classes, self.feat_channels,
            self.conv_kernel_size, self.conv_kernel_size,
        )
        return update_head


# ---------------------------------------------------------------------------
# Dynamic conv applied per-sample (one kernel per image)
# ---------------------------------------------------------------------------
def _dynamic_conv(feat, head, padding):
    """feat: (B, C, H, W), head: (B, N, C, K, K) -> (B, N, H, W)."""
    bs = feat.size(0)
    n = head.size(1)
    h, w = feat.shape[-2:]
    outs = []
    for t in range(bs):
        outs.append(F.conv2d(feat[t:t + 1], head[t], padding=padding))
    return torch.cat(outs, dim=0).reshape(bs, n, h, w)


# ---------------------------------------------------------------------------
# Res2Net-50 backbone wrapper (timm) with SSL-tolerant pretrained loading
# ---------------------------------------------------------------------------
class _Res2NetBackbone(nn.Module):
    """Wraps timm res2net50_26w_4s as a multi-stage feature extractor.

    Returns features at strides {2, 4, 8, 16, 32} with channels
    {64, 256, 512, 1024, 2048}.
    """

    def __init__(self, in_chans=3, pretrained=True):
        super().__init__()
        self.model = load_with_ssl_fallback(
            timm.create_model,
            'res2net50_26w_4s',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=in_chans,
        )
        # Cache channels for downstream construction.
        self.out_channels = self.model.feature_info.channels()

    def forward(self, x):
        feats = self.model(x)  # list of 5 tensors
        return feats  # [e1, e2, e3, e4, e5]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class LDNet(nn.Module):
    """LDNet with Res2Net-50 backbone and Lesion-aware Dynamic Kernels.

    Args:
        in_channels: number of input image channels.
        num_classes: number of segmentation classes (final logit channels).
        img_size: input spatial resolution (used only as a hint).
        pretrained: whether to attempt loading timm pretrained weights.
        unified_channels: channel width used for dynamic kernel and unify
            convs (default 64, matches upstream).
        conv_kernel_size: spatial extent of the dynamic conv kernel.
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 pretrained=True, unified_channels=64, conv_kernel_size=1,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.unified_channels = unified_channels
        self.conv_kernel_size = conv_kernel_size

        # Encoder
        self.backbone = _Res2NetBackbone(in_chans=in_channels, pretrained=pretrained)
        c1, c2, c3, c4, c5 = self.backbone.out_channels  # 64, 256, 512, 1024, 2048

        # Channel reduction
        self.reduce2 = nn.Conv2d(c2, 64, 1)
        self.reduce3 = nn.Conv2d(c3, 128, 1)
        self.reduce4 = nn.Conv2d(c4, 256, 1)
        self.reduce5 = nn.Conv2d(c5, 512, 1)

        # Decoder
        self.decoder5 = _DecoderBlock(in_c=512, out_c=512)
        self.decoder4 = _DecoderBlock(in_c=512 + 256, out_c=256)
        self.decoder3 = _DecoderBlock(in_c=256 + 128, out_c=128)
        self.decoder2 = _DecoderBlock(in_c=128 + 64, out_c=64)
        self.decoder1 = _DecoderBlock(in_c=64 + c1, out_c=64)

        # Global context for the initial dynamic kernel
        self.global_pool = nn.Sequential(
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.generate_head = nn.Linear(
            512,
            self.num_classes * self.unified_channels * conv_kernel_size * conv_kernel_size,
        )

        # Kernel refinement modules (one per decoder stage after d5)
        self.head_updators = nn.ModuleList([_HeadUpdator(
            in_channels=unified_channels,
            feat_channels=unified_channels,
            out_channels=unified_channels,
            conv_kernel_size=conv_kernel_size,
        ) for _ in range(4)])

        # Unify-channel convs (project decoder features to `unified_channels`)
        self.unify1 = nn.Conv2d(64, unified_channels, 1)
        self.unify2 = nn.Conv2d(64, unified_channels, 1)
        self.unify3 = nn.Conv2d(128, unified_channels, 1)
        self.unify4 = nn.Conv2d(256, unified_channels, 1)
        self.unify5 = nn.Conv2d(512, unified_channels, 1)

        # ESA on encoder skips (matches upstream channel counts)
        self.esa1 = _ESABlock(dim=c1)   # raw e1 (stem)
        self.esa2 = _ESABlock(dim=64)   # reduced e2
        self.esa3 = _ESABlock(dim=128)  # reduced e3
        self.esa4 = _ESABlock(dim=256)  # reduced e4

        # LCA on decoder features (matches upstream channel counts)
        self.lca1 = _LCABlock(dim=64)
        self.lca2 = _LCABlock(dim=128)
        self.lca3 = _LCABlock(dim=256)
        self.lca4 = _LCABlock(dim=512)

        self.decoder_list = nn.ModuleList([self.decoder4, self.decoder3,
                                           self.decoder2, self.decoder1])
        self.unify_list = nn.ModuleList([self.unify4, self.unify3,
                                         self.unify2, self.unify1])
        self.esa_list = nn.ModuleList([self.esa4, self.esa3,
                                       self.esa2, self.esa1])
        self.lca_list = nn.ModuleList([self.lca4, self.lca3,
                                       self.lca2, self.lca1])

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _pad_to_multiple(x, multiple=32):
        _, _, h, w = x.shape
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if ph == 0 and pw == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
        return x, (ph, pw)

    # ----------------------------------------------------------------- forward
    def forward(self, x):
        in_h, in_w = x.shape[-2:]
        # Res2Net stem requires inputs divisible by 32 for the stride-32 stage.
        x_in, (ph, pw) = self._pad_to_multiple(x, multiple=32)
        bs = x_in.shape[0]

        # Encoder
        e1, e2_, e3_, e4_, e5_ = self.backbone(x_in)

        e2 = self.reduce2(e2_)
        e3 = self.reduce3(e3_)
        e4 = self.reduce4(e4_)
        e5 = self.reduce5(e5_)

        # Decoder bootstrap
        d5 = self.decoder5(e5)
        feat5 = self.unify5(d5)

        # Initial dynamic kernel from global context of bottleneck
        gc = self.global_pool(e5).reshape(bs, -1)
        head = self.generate_head(gc).reshape(
            bs, self.num_classes, self.unified_channels,
            self.conv_kernel_size, self.conv_kernel_size,
        )

        pred = _dynamic_conv(feat5, head, padding=int(self.conv_kernel_size // 2))

        decoder_out = [d5]
        encoder_out = [e4, e3, e2, e1]

        feats = []
        for i in range(4):
            esa_out = self.esa_list[i](encoder_out[i])
            lca_out = self.lca_list[i](decoder_out[-1], pred)
            comb = torch.cat([lca_out, esa_out], dim=1)
            d = self.decoder_list[i](comb)
            decoder_out.append(d)

            feat_i = self.unify_list[i](d)
            feats.append(feat_i)

            head = self.head_updators[i](feat_i, head, pred)
            pred = _dynamic_conv(feat_i, head,
                                 padding=int(self.conv_kernel_size // 2))

        # `pred` is at full input resolution (after padding). Crop the pad and
        # ensure exact output H/W via interpolation if any rounding differs.
        if ph != 0 or pw != 0:
            pred = pred[..., :pred.shape[-2] - ph, :pred.shape[-1] - pw]
        if pred.shape[-2:] != (in_h, in_w):
            pred = F.interpolate(pred, size=(in_h, in_w),
                                 mode='bilinear', align_corners=False)
        return pred


__all__ = ['LDNet']
