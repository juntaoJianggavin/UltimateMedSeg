"""MCTrans Encoder: faithful port from https://github.com/JiYuanFeng/MCTrans

Reference: Ji et al., "Multi-Compound Transformer for Accurate Biomedical
           Image Segmentation", MICCAI 2021.
Files referenced (master branch):
  - mctrans/models/encoders/resnet.py         (ResNet-18/34 backbone)
  - mctrans/models/centers/mctrans.py         (MCTrans center module)
  - mctrans/models/trans/transformer.py       (DSA, DSALayer, CA, CALayer)

Architecture:
  ResNet-34 backbone (BasicBlock, arch (3,4,6,3))
   -> 5 stages: stem(64) -> layer1(64) -> layer2(128) -> layer3(256) -> layer4(512)
  MCTrans center:
   1) project last 3 stages (layer2/3/4) to a unified d_model
   2) Transformer Self-Attention (TSA / DSA) over multi-scale tokens
   3) Transformer Cross-Attention (TCA / CA) with learnable proxy_embed
      (used as auxiliary supervision in original repo's segmentation head)

NOTE on MSDeformAttn:
  The official repo uses Multi-Scale Deformable Attention (Deformable DETR
  CUDA op) inside DSALayer.  To keep the project pure-PyTorch and platform
  independent, we replace MSDeformAttn with the standard
  ``nn.MultiheadAttention`` while preserving every other detail of the
  multi-scale flatten / level_embed / position_embedding / split-and-recover
  pipeline exactly as in the original ``MCTrans.forward``.
"""
# Source: https://github.com/JiYuanFeng/MCTrans

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from medseg.registry import ENCODER_REGISTRY


# ====================================================================
#  ResNet-18/34 backbone (1:1 port of mctrans/models/encoders/resnet.py)
# ====================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _make_res_layer(block, inplanes, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion, 1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    layers = [block(inplanes, planes, stride, dilation, downsample)]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, 1, dilation))
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet-18 / ResNet-34 backbone (BasicBlock variant).

    Mirrors mctrans.models.encoders.resnet.ResNet with arch_settings::
        18: (BasicBlock, (2, 2, 2, 2))
        34: (BasicBlock, (3, 4, 6, 3))

    Returns features at 5 stages: stem, layer1, layer2, layer3, layer4
    with channels (64, 64, 128, 256, 512), matching ``filter_nums``.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
    }

    def __init__(self, depth: int = 34, in_channels: int = 3,
                 strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1)):
        super().__init__()
        assert depth in self.arch_settings, f"invalid depth {depth} for resnet"
        block, stage_blocks = self.arch_settings[depth]

        # stem (conv 7x7, stride 2 + maxpool 3x3, stride 2)
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # 4 res-stages
        self.res_layers = nn.ModuleList()
        inplanes = 64
        for i, num_blocks in enumerate(stage_blocks):
            planes = 64 * (2 ** i)
            layer = _make_res_layer(block, inplanes, planes, num_blocks,
                                    stride=strides[i], dilation=dilations[i])
            inplanes = planes * block.expansion
            self.res_layers.append(layer)

        # 5 output channels: stem(64), layer1..4 (64,128,256,512)
        self.out_channels_full = (64, 64, 128, 256, 512)

    def forward(self, x):
        feats = []
        x = self.relu(self.bn1(self.conv1(x)))   # stride 2  -> stem (64ch)
        feats.append(x)
        x = self.maxpool(x)                       # stride 4
        for layer in self.res_layers:
            x = layer(x)
            feats.append(x)
        return feats   # [stem, layer1, layer2, layer3, layer4]


# ====================================================================
#  Sine 2-D position encoding (1:1 from trans/utils.py / DETR convention)
# ====================================================================
class PositionEmbeddingSine(nn.Module):
    """Standard 2-D sine positional embedding used by DETR and MCTrans."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W);  mask: (B, H, W) bool, True=padding
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# ====================================================================
#  DSA (Transformer Self-Attention) — 1:1 logic of trans/transformer.DSA
#  with MSDeformAttn replaced by nn.MultiheadAttention (PyTorch fallback)
# ====================================================================
def _get_activation(act: str):
    if act == "relu":
        return F.relu
    if act == "gelu":
        return F.gelu
    if act == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {act}.")


class DSALayer(nn.Module):
    """Transformer Self-Attention layer.

    Same outer skeleton as ``mctrans.models.trans.transformer.DSALayer``:
    self-attn (with positional embedding via ``with_pos_embed``) -> dropout1 ->
    norm1 -> ffn (linear1 -> act -> dropout2 -> linear2 -> dropout3 -> norm2).
    """

    def __init__(self, d_model=240, d_ffn=1024, dropout=0.1,
                 activation="relu", n_heads=8):
        super().__init__()
        # NOTE: official MCTrans uses MSDeformAttn here — replaced by vanilla
        # nn.MultiheadAttention for pure-PyTorch portability.
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation(activation)
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
        return self.norm2(src)

    def forward(self, src, pos, padding_mask=None):
        q = k = self.with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, src, key_padding_mask=padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class DSA(nn.Module):
    """Stack of DSA layers — identical iteration pattern to the official DSA."""

    def __init__(self, att_layer: DSALayer, n_layers: int):
        super().__init__()
        import copy
        self.layers = nn.ModuleList([copy.deepcopy(att_layer) for _ in range(n_layers)])

    def forward(self, src, pos=None, padding_mask=None):
        out = src
        for layer in self.layers:
            out = layer(out, pos, padding_mask)
        return out


class CALayer(nn.Module):
    """Transformer Cross-Attention layer (1:1 from trans/transformer.CALayer)."""

    def __init__(self, d_model=240, d_ffn=1024, dropout=0.1,
                 activation="relu", n_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, tgt, src):
        # self-attention on proxy queries
        q = k = tgt
        sa, _ = self.self_attn(q.transpose(0, 1), k.transpose(0, 1),
                               tgt.transpose(0, 1))
        tgt = self.norm2(tgt + self.dropout2(sa.transpose(0, 1)))
        # cross-attention: proxy queries attend to multi-scale tokens
        ca, _ = self.cross_attn(tgt.transpose(0, 1),
                                src.transpose(0, 1),
                                src.transpose(0, 1))
        tgt = self.norm1(tgt + self.dropout1(ca.transpose(0, 1)))
        return self.forward_ffn(tgt)


class CA(nn.Module):
    """Stack of CA layers with shared learnable ``proxy_embed`` (1:1 from CA)."""

    def __init__(self, att_layer: CALayer, n_layers: int,
                 n_category: int = 2, d_model: int = 240):
        super().__init__()
        import copy
        self.layers = nn.ModuleList([copy.deepcopy(att_layer) for _ in range(n_layers)])
        self.proxy_embed = nn.Parameter(torch.zeros(1, n_category, d_model))

    def forward(self, src):
        B = src.shape[0]
        query = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                query = self.proxy_embed.expand(B, -1, -1)
            else:
                query = query + self.proxy_embed.expand(B, -1, -1)
            query = layer(query, src)
        return query


# ====================================================================
#  MCTrans center (1:1 logic from mctrans/models/centers/mctrans.py)
# ====================================================================
class MCTransCenter(nn.Module):
    """MCTrans center module: project, multi-scale flatten, DSA, recover."""

    def __init__(self, in_channels, proj_idxs, d_model=240, nhead=8,
                 d_ffn=1024, dropout=0.1, n_sa_layers=6,
                 n_ca_layers=2, n_category=2):
        super().__init__()
        self.proj_idxs = list(proj_idxs)
        self.n_levels = len(self.proj_idxs)
        self.d_model = d_model

        # Conv-3x3 projections to a unified d_model (matches official ConvModule
        # with kernel=3, padding=1, BN, ReLU)
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[idx], d_model, 3, padding=1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True),
            ) for idx in self.proj_idxs
        ])

        dsa_layer = DSALayer(d_model=d_model, d_ffn=d_ffn, dropout=dropout,
                             activation="relu", n_heads=nhead)
        self.dsa = DSA(dsa_layer, n_layers=n_sa_layers)

        ca_layer = CALayer(d_model=d_model, d_ffn=d_ffn, dropout=dropout,
                           activation="relu", n_heads=nhead)
        self.ca = CA(ca_layer, n_layers=n_ca_layers,
                     n_category=n_category, d_model=d_model)

        self.level_embed = nn.Parameter(torch.zeros(self.n_levels, d_model))
        nn.init.normal_(self.level_embed)
        self.position_embedding = PositionEmbeddingSine(d_model // 2,
                                                        normalize=True)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def projection(self, feats):
        cnn_feats, tran_feats, pos_list = [], [], []
        for idx, f in enumerate(feats):
            if idx not in self.proj_idxs:
                cnn_feats.append(f)
            else:
                tran_feats.append(f)
        # apply 3x3 projections, then build sine positional encoding
        for i, proj in enumerate(self.projs):
            tran_feats[i] = proj(tran_feats[i])
            n, _, h, w = tran_feats[i].shape
            mask = torch.zeros((n, h, w), dtype=torch.bool,
                               device=tran_feats[i].device)
            pos_list.append(self.position_embedding(tran_feats[i], mask))
        return cnn_feats, tran_feats, pos_list

    def forward(self, x):
        cnn_feats, trans_feats, pos_embs = self.projection(x)

        feat_flat, pos_flat, shapes = [], [], []
        for lvl, (feat, pos) in enumerate(zip(trans_feats, pos_embs)):
            _, _, h, w = feat.shape
            shapes.append((h, w))
            feat = feat.flatten(2).transpose(1, 2)             # (B, HW, C)
            pos = pos.flatten(2).transpose(1, 2)               # (B, HW, C)
            pos = pos + self.level_embed[lvl].view(1, 1, -1)
            feat_flat.append(feat)
            pos_flat.append(pos)
        feat_flat = torch.cat(feat_flat, dim=1)
        pos_flat = torch.cat(pos_flat, dim=1)

        # multi-scale self-attention
        feat_flat = self.dsa(feat_flat, pos=pos_flat)

        # auxiliary cross-attention with proxy queries (output kept as buffer)
        self.proxy_logits = self.ca(feat_flat)

        # split and recover spatial maps
        sizes = [h * w for h, w in shapes]
        parts = feat_flat.split(sizes, dim=1)
        out_trans = []
        for (h, w), p in zip(shapes, parts):
            B = p.shape[0]
            out_trans.append(p.transpose(1, 2).reshape(B, self.d_model, h, w))

        cnn_feats.extend(out_trans)
        return cnn_feats


# ====================================================================
#  Registered encoder
# ====================================================================
@ENCODER_REGISTRY.register("mctrans")
class MCTransEncoder(nn.Module):
    """MCTrans encoder: ResNet-34 backbone + MCTrans center.

    Faithful port of JiYuanFeng/MCTrans (MICCAI 2021).  The encoder returns
    the four post-stem stages so that it is compatible with any 4-level
    decoder in this project.

    Layout (input H×W=224×224, depth=34):
        layer1 -> H/4   (64ch)   ── kept as plain CNN feature
        layer2 -> H/8   (d_model after MCTrans projection)
        layer3 -> H/16  (d_model)
        layer4 -> H/32  (d_model)

    The stem feature (H/2, 64ch) is dropped to match the [c1,c2,c3,c4]
    contract used by ``BilinearDecoder``.
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        img_size: int = 224,
        depth: int = 34,
        d_model: int = 240,
        nhead: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_sa_layers: int = 6,
        n_ca_layers: int = 2,
        n_category: int = 2,
        pretrained_path: str = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = ResNet(depth=depth, in_channels=in_channels)
        # backbone outputs: [stem(64), layer1(64), layer2(128), layer3(256), layer4(512)]
        # Official proj_idxs=(2,3,4) -> layer2/3/4
        self.center = MCTransCenter(
            in_channels=self.backbone.out_channels_full,
            proj_idxs=(2, 3, 4),
            d_model=d_model,
            nhead=nhead,
            d_ffn=d_ffn,
            dropout=dropout,
            n_sa_layers=n_sa_layers,
            n_ca_layers=n_ca_layers,
            n_category=n_category,
        )

        # We expose the four post-stem stages: layer1 (cnn) + layer2/3/4 (transformer).
        self.out_channels = [64, d_model, d_model, d_model]

        if pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        state = torch.load(path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        msg = self.load_state_dict(state, strict=False)
        print(f"MCTrans encoder loaded: {msg}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)               # 5 stages: stem, l1, l2, l3, l4
        all_feats = self.center(feats)         # cnn_feats + trans_feats (still 5)
        # Drop the stem feature so output matches the 4-level decoder contract.
        return all_feats[1:]                   # [layer1, layer2', layer3', layer4']
