"""MCTrans – faithful port from github.com/JiYuanFeng/MCTrans.

Multi-Compound Transformer for Accurate Biomedical Image Segmentation (IJCV 2022).

Architecture:
  - Backbone: ResNet producing 5 features [64, 64, 128, 256, 512]
  - MCTrans center: Conv3x3 projection → DSA (deformable self-attention on levels 2,3,4)
  - Decoder: UNetDecoder with AttBlock (attention-gated skip connections)
  - CA: Cross-attention with learnable proxy tokens for feature enhancement

NOTE: Official MSDeformAttn uses custom CUDA kernels. This port includes a pure
PyTorch implementation using grid_sample as a faithful approximation.
"""
# Source: https://github.com/JiYuanFeng/MCTrans

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_


# ---------------------------------------------------------------------------
# Pure PyTorch Multi-Scale Deformable Attention
# ---------------------------------------------------------------------------
def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Pure PyTorch fallback for MSDeformAttnFunction using grid_sample."""
    N, Len_in, n_heads, d_head = value.shape
    # sampling_locations: (N, Len_q, n_heads, n_levels, n_points, 2) in [0,1]
    # attention_weights: (N, Len_q, n_heads, n_levels, n_points)
    N_, Len_q, _, n_levels, n_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1  # map [0,1] to [-1,1]
    sampling_value_list = []
    for lid, (H_, W_) in enumerate(value_spatial_shapes):
        # v: (N, Len_lid, n_heads, d_head) -> (N*n_heads, d_head, H_, W_)
        v = value_list[lid].permute(0, 2, 3, 1).reshape(N * n_heads, d_head, H_, W_)
        # s: (N, Len_q, n_heads, n_points, 2) -> (N*n_heads, Len_q, n_points, 2)
        s = sampling_grids[:, :, :, lid].permute(0, 2, 1, 3, 4).reshape(N * n_heads, Len_q, n_points, 2)
        v_sampled = F.grid_sample(v, s, mode='bilinear', padding_mode='zeros', align_corners=False)
        # v_sampled: (N*n_heads, d_head, Len_q, n_points)
        sampling_value_list.append(v_sampled)
    sampling_value = torch.stack(sampling_value_list, dim=4)  # (N*n_heads, d_head, Len_q, n_points, n_levels)
    sampling_value = sampling_value.permute(0, 1, 4, 2, 3)  # (N*n_heads, d_head, n_levels, Len_q, n_points)
    # attention: (N, Len_q, n_heads, n_levels, n_points) -> (N*n_heads, 1, n_levels, Len_q, n_points)
    attn = attention_weights.permute(0, 2, 3, 1, 4).reshape(N * n_heads, 1, n_levels, Len_q, n_points)
    output = (sampling_value * attn).sum(-1).sum(2)  # (N*n_heads, d_head, Len_q)
    output = output.reshape(N, n_heads * d_head, Len_q).permute(0, 2, 1)
    return output


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention (pure PyTorch implementation)."""

    def __init__(self, d_model=240, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, spatial_shapes,
                level_start_index, padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            sampling_locations = reference_points[:, :, None, :, None, :2] + \
                sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        output = ms_deform_attn_core_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


# ---------------------------------------------------------------------------
# Transformer layers (DSA + CA)
# ---------------------------------------------------------------------------
class DSALayer(nn.Module):
    def __init__(self, d_model=240, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes,
                level_start_index, padding_mask=None):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, src,
            spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class DSA(nn.Module):
    def __init__(self, att_layer, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([att_layer for _ in range(n_layers)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios,
                pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes,
                           level_start_index, padding_mask)
        return output


class CALayer(nn.Module):
    def __init__(self, d_model=240, d_ffn=1024, dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, src):
        # self attention (seq-first for nn.MultiheadAttention)
        tgt2 = self.self_attn(tgt.transpose(0, 1), tgt.transpose(0, 1),
                               tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # cross attention (also seq-first)
        tgt2 = self.cross_attn(tgt.transpose(0, 1), src.transpose(0, 1),
                                src.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class CA(nn.Module):
    def __init__(self, att_layer, n_layers, n_category=2, d_model=240):
        super().__init__()
        self.layers = nn.ModuleList([att_layer for _ in range(n_layers)])
        self.proxy_embed = nn.Parameter(torch.zeros(1, n_category, d_model))

    def forward(self, src):
        query = None
        B = src.shape[0]
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                query = self.proxy_embed.expand(B, -1, -1)
            else:
                query = query + self.proxy_embed.expand(B, -1, -1)
            query = layer(query, src)
        return query


# ---------------------------------------------------------------------------
# Position embedding (sine)
# ---------------------------------------------------------------------------
def build_sine_position_encoding(mask, hidden_dim):
    """Build 2D sine position embedding from mask."""
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi
    num_pos_feats = hidden_dim // 2
    temperature = 10000
    dim_t = torch.exp(torch.arange(0, num_pos_feats, dtype=torch.float32) *
                       (-math.log(temperature) / num_pos_feats))
    pos_x = x_embed[:, :, :, None] * dim_t
    pos_y = y_embed[:, :, :, None] * dim_t
    pos = torch.cat((pos_y.sin(), pos_x.sin()), dim=3).permute(0, 3, 1, 2)
    return pos


# ---------------------------------------------------------------------------
# Conv helper
# ---------------------------------------------------------------------------
def conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  padding=padding, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


# ---------------------------------------------------------------------------
# ResNet backbone (5 features: [64, 64, 128, 256, 512])
# ---------------------------------------------------------------------------
class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class _ResNetBackbone(nn.Module):
    """ResNet producing 5 features: [c0(64), c1(64), c2(128), c3(256), c4(512)]."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.stem_conv = conv_bn_relu(64, 64, 3, 1, 1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))
        layers = [_BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(_BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))  # /4
        c0 = self.stem_conv(x)            # /4, 64
        c1 = self.layer1(c0)              # /4, 64
        c2 = self.layer2(c1)              # /8, 128
        c3 = self.layer3(c2)              # /16, 256
        c4 = self.layer4(c3)              # /32, 512
        return [c0, c1, c2, c3, c4]


# ---------------------------------------------------------------------------
# Attention-gated decoder blocks
# ---------------------------------------------------------------------------
class AttBlock(nn.Module):
    """Attention gate for skip connections (from official code)."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Align spatial sizes if needed
        if g.shape[2:] != x.shape[2:]:
            x = F.interpolate(x, size=g.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, attention=False):
        super().__init__()
        self.conv1 = conv_bn_relu(in_channels + skip_channels, out_channels)
        self.conv2 = conv_bn_relu(out_channels, out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if attention:
            self.att = AttBlock(F_g=in_channels, F_l=skip_channels, F_int=in_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if hasattr(self, "att"):
                skip = self.att(g=x, x=skip)
            # Align spatial sizes if 2x upsample doesn't perfectly match skip
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear',
                                     align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ---------------------------------------------------------------------------
# MCTrans center (DSA)
# ---------------------------------------------------------------------------
class MCTransCenter(nn.Module):
    """MCTrans center: projects levels (2,3,4) → d_model → DSA → recover."""

    def __init__(self, d_model=240, nhead=8, d_ffn=1024, dropout=0.1,
                 act="relu", n_points=4, n_levels=3, n_sa_layers=6,
                 in_channels=[64, 64, 128, 256, 512], proj_idxs=(2, 3, 4)):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.n_levels = n_levels
        self.proj_idxs = proj_idxs
        self.projs = nn.ModuleList()
        for idx in self.proj_idxs:
            self.projs.append(
                conv_bn_relu(in_channels[idx], d_model, kernel_size=3, padding=1))
        dsa_layer = DSALayer(d_model=d_model, d_ffn=d_ffn, dropout=dropout,
                             activation=act, n_levels=n_levels, n_heads=nhead,
                             n_points=n_points)
        self.dsa = DSA(att_layer=dsa_layer, n_layers=n_sa_layers)
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(self, feats):
        cnn_feats = []
        tran_feats = []
        pos_list = []
        masks = []
        for idx, feat in enumerate(feats):
            if idx not in self.proj_idxs:
                cnn_feats.append(feat)
            else:
                n, c, h, w = feat.shape
                mask = torch.zeros((n, h, w), dtype=torch.bool, device=feat.device)
                masks.append(mask)
                pos_list.append(build_sine_position_encoding(mask, self.d_model))
                tran_feats.append(feat)
        for idx, proj in enumerate(self.projs):
            tran_feats[idx] = proj(tran_feats[idx])
        # Flatten for DSA
        features_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        feature_shapes = []
        for lvl, (feature, mask, pos_embed) in enumerate(
                zip(tran_feats, masks, pos_list)):
            bs, c, h, w = feature.shape
            spatial_shapes.append((h, w))
            feature_shapes.append(feature.shape)
            feature = feature.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            features_flatten.append(feature)
            mask_flatten.append(mask)
        features_flatten = torch.cat(features_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes_t = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=features_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes_t.new_zeros((1,)),
             spatial_shapes_t.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in masks], 1)
        # DSA
        feats_out = self.dsa(features_flatten, spatial_shapes_t,
                              level_start_index, valid_ratios,
                              lvl_pos_embed_flatten, mask_flatten)
        # Recover per-level features
        out = []
        features = feats_out.split(
            spatial_shapes_t.prod(1).tolist(), dim=1)
        for feat, ori_shape in zip(features, spatial_shapes):
            out.append(feat.transpose(1, 2).reshape(
                feat.shape[0], self.d_model, ori_shape[0], ori_shape[1]))
        cnn_feats.extend(out)
        return cnn_feats


# ---------------------------------------------------------------------------
# MCTransAuxHead (training only)
# ---------------------------------------------------------------------------
class MCTransAuxHead(nn.Module):
    def __init__(self, d_model=240, d_ffn=1024, dropout=0.1, act="relu",
                 n_head=8, n_layers=4, num_classes=2,
                 proj_idxs=(2, 3, 4)):
        super().__init__()
        self.proj_idxs = proj_idxs
        ca_layer = CALayer(d_model=d_model, d_ffn=d_ffn, dropout=dropout,
                           activation=act, n_heads=n_head)
        self.ca = CA(att_layer=ca_layer, n_layers=n_layers,
                     n_category=num_classes, d_model=d_model)
        self.head = nn.Sequential(
            nn.Linear(num_classes * d_model, d_model),
            nn.Linear(d_model, num_classes))

    def forward(self, features):
        inputs = [features[idx] for idx in self.proj_idxs]
        inputs_flatten = [item.flatten(2).transpose(1, 2) for item in inputs]
        inputs_flatten = torch.cat(inputs_flatten, 1)
        outputs = self.ca(inputs_flatten)
        logits = self.head(outputs.flatten(1))
        return logits


# ---------------------------------------------------------------------------
# Main MCTrans model
# ---------------------------------------------------------------------------
class MCTrans(nn.Module):
    """Multi-Compound Transformer for medical image segmentation.

    Args:
        in_channels: Number of input image channels (default 3).
        num_classes: Number of output segmentation classes (default 2).
        img_size: Expected input spatial resolution (default 224).
    """

    def __init__(self, in_channels=3, num_classes=2, img_size=224, **kwargs):
        super().__init__()
        d_model = 240
        d_ffn = 1024
        n_heads = 8
        n_sa_layers = 6
        n_ca_layers = 4
        n_points = 4
        n_levels = 3
        enc_channels = [64, 64, 128, 256, 512]
        proj_idxs = (2, 3, 4)
        # Backbone
        self.backbone = _ResNetBackbone(in_channels)
        # MCTrans center (DSA)
        self.center = MCTransCenter(
            d_model=d_model, nhead=n_heads, d_ffn=d_ffn,
            n_sa_layers=n_sa_layers, n_points=n_points, n_levels=n_levels,
            in_channels=enc_channels, proj_idxs=proj_idxs)
        # CA layers (cross-attention with proxy tokens)
        ca_layer = CALayer(d_model=d_model, d_ffn=d_ffn, n_heads=n_heads)
        self.ca = CA(att_layer=ca_layer, n_layers=n_ca_layers,
                     n_category=num_classes, d_model=d_model)
        self.proj_idxs = proj_idxs
        # Feature enhancement via CA proxy tokens
        self.feat_proj = nn.Linear(num_classes * d_model, d_model)
        # UNetDecoder with AttBlock
        dec_in = enc_channels[:2] + [d_model] * n_levels  # [64, 64, 240, 240, 240]
        self.decoder = nn.ModuleList()
        dec_in_rev = dec_in[::-1]
        skip_rev = dec_in_rev[1:]
        for in_c, skip_c in zip(dec_in_rev, skip_rev):
            self.decoder.append(DecBlock(in_c, skip_c, skip_c, attention=True))
        # Final projection head
        self.final_conv = nn.Conv2d(dec_in[0], num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.backbone(x)
        # MCTrans center (DSA on levels 2,3,4)
        features = self.center(features)
        # CA (cross-attention with proxy tokens)
        ca_inputs = [features[idx].flatten(2).transpose(1, 2)
                      for idx in self.proj_idxs]
        ca_input = torch.cat(ca_inputs, 1)
        proxy_out = self.ca(ca_input)  # (B, num_classes, d_model)
        # Feature enhancement: project proxy tokens to enhance features
        proxy_flat = self.feat_proj(proxy_out.flatten(1))  # (B, d_model)
        # Apply as channel-wise modulation to each transformer feature
        for idx in self.proj_idxs:
            feat = features[idx]  # (B, d_model, H, W)
            B = feat.shape[0]
            # Broadcast proxy to spatial dims
            mod = proxy_flat.unsqueeze(-1).unsqueeze(-1).expand_as(feat)
            features[idx] = feat + feat * mod
        # Decode (reverse order: deepest first)
        feats_rev = features[::-1]
        x = feats_rev[0]
        for i, layer in enumerate(self.decoder):
            x = layer(x, feats_rev[i + 1])
        # Upsample to original input resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return self.final_conv(x)
