"""TTT-UNet: Enhancing U-Net with Test-Time Training Layers for
Biomedical Image Segmentation.

Faithful reimplementation from:
  https://github.com/rongzhou7/TTT-Unet  (NeurIPS 2024 Workshop)

Based on U-Mamba architecture, replacing MambaLayer with TTTLayer
(Test-Time Training linear layer with RoPE and self-supervised gradient update).

Self-contained TTTLinear implementation (no ``transformers`` dependency).
"""
# Source: https://github.com/rongzhou7/TTT-Unet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for TTT."""
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: (B, num_heads, L, head_dim)
        inv_freq = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        pos_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos_expanded).transpose(1, 2)  # (B, L, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, L, dim)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # (B, 1, L, D)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


# ---------------------------------------------------------------------------
# TTTLinear: Test-Time Training Linear Layer
# ---------------------------------------------------------------------------

class TTTLinear(nn.Module):
    """Test-Time Training (TTT) Linear layer.

    Core idea: uses a learnable linear model W that is updated at test time
    via self-supervised gradient descent on a reconstruction objective.
    Inputs are used as both queries and keys (with RoPE), and the linear
    model produces outputs that approximate the values.

    Simplified self-contained implementation without transformers dependency.

    Args:
        hidden_size: Model dimension.
        num_heads: Number of attention heads.
        mini_batch_size: Mini-batch size for TTT updates.
        base_lr: Base learning rate for test-time training.
        rope_theta: RoPE base frequency.
        conv_kernel: Causal conv kernel size for pre-processing.
    """
    def __init__(self, hidden_size=768, num_heads=12, mini_batch_size=64,
                 base_lr=1.0, rope_theta=10000.0, conv_kernel=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mini_batch_size = mini_batch_size

        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # TTT learnable parameters
        self.W = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.learnable_lr = nn.Parameter(
            torch.ones(self.num_heads, 1, 1) * math.log(base_lr))

        # Layer norms
        self.ln_q = RMSNorm(self.head_dim)
        self.ln_k = RMSNorm(self.head_dim)
        self.post_norm = nn.LayerNorm(hidden_size)

        # Causal conv for pre-processing
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, conv_kernel,
            groups=hidden_size, padding=conv_kernel - 1, bias=True)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, base=rope_theta)

    def forward(self, hidden_states, position_ids=None):
        """
        Args:
            hidden_states: (B, L, D)
            position_ids: (B, L) position indices
        Returns:
            (B, L, D)
        """
        B, L, D = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(L, device=hidden_states.device
                                        ).unsqueeze(0).expand(B, -1)

        # Pre-conv
        h_conv = hidden_states.transpose(1, 2)  # (B, D, L)
        h_conv = self.conv(h_conv)[:, :, :L].transpose(1, 2)  # causal trim

        # Project Q, K, V
        q = self.q_proj(h_conv).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_conv).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, num_heads, L, head_dim)

        # RoPE
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Normalize
        q = self.ln_q(q)
        k = self.ln_k(k)

        # TTT: test-time training via self-supervised gradient update
        output = self._ttt_forward(q, k, v)

        # Output projection
        output = output.transpose(1, 2).reshape(B, L, D)
        output = self.o_proj(output)
        output = self.post_norm(output)
        return output

    def _ttt_forward(self, q, k, v):
        """TTT forward: use mini-batch gradient descent to update W.

        For each mini-batch of tokens, compute reconstruction loss on k→v
        mapping and update W, then apply updated W to q.

        Args:
            q, k, v: (B, num_heads, L, head_dim)
        Returns:
            output: (B, num_heads, L, head_dim)
        """
        B, H, L, d = q.shape
        lr = torch.exp(self.learnable_lr)  # (H, 1, 1)

        # Process in mini-batches
        mb = min(self.mini_batch_size, L)
        n_batches = (L + mb - 1) // mb
        outputs = []

        # Get initial W weight
        W = self.W.weight.unsqueeze(0).unsqueeze(0).expand(
            B, H, d, d).clone()  # (B, H, d, d)

        for i in range(n_batches):
            start = i * mb
            end = min(start + mb, L)

            k_mb = k[:, :, start:end, :]  # (B, H, mb, d)
            v_mb = v[:, :, start:end, :]
            q_mb = q[:, :, start:end, :]

            # Reconstruction: predict v from k using W
            pred = torch.matmul(k_mb, W)  # (B, H, mb, d)

            # Gradient of MSE loss w.r.t. W
            # loss = 0.5 * ||pred - v||^2
            # grad = k^T @ (pred - v) / mb
            residual = pred - v_mb  # (B, H, mb, d)
            grad = torch.matmul(
                k_mb.transpose(-2, -1), residual) / max(end - start, 1)

            # Update W with gradient descent
            W = W - lr.unsqueeze(0) * grad

            # Apply updated W to queries
            out_mb = torch.matmul(q_mb, W)  # (B, H, mb, d)
            outputs.append(out_mb)

        return torch.cat(outputs, dim=2)


# ---------------------------------------------------------------------------
# TTTLayer: wraps TTTLinear for 2D features (like MambaLayer)
# ---------------------------------------------------------------------------

class TTTLayer(nn.Module):
    """TTT layer for 2D features: flatten → TTTLinear → reshape.

    Faithful to TTT-UNet's TTTLayer interface.
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        num_heads = max(1, dim // 64)
        self.ttt = TTTLinear(
            hidden_size=dim, num_heads=num_heads,
            mini_batch_size=64, base_lr=1.0)

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.float()
        B, C = x.shape[:2]
        assert C == self.dim
        img_dims = x.shape[2:]
        n_tokens = img_dims.numel()

        x_flat = x.reshape(B, C, n_tokens).transpose(1, 2)  # (B, N, C)
        x_norm = self.norm(x_flat)

        position_ids = torch.arange(
            n_tokens, device=x.device).unsqueeze(0).expand(B, -1)
        x_ttt = self.ttt(x_norm, position_ids=position_ids)

        return x_ttt.transpose(1, 2).reshape(B, C, *img_dims)


# ---------------------------------------------------------------------------
# UNet building blocks (shared with U-Mamba)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                         if in_ch != out_ch or stride != 1 else nn.Identity())

    def forward(self, x):
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act2(y + self.shortcut(x))


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, features, n_blocks_per_stage=None):
        super().__init__()
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * len(features)
        self.stages = nn.ModuleList()
        ch = in_channels
        for i, f in enumerate(features):
            stride = 1 if i == 0 else 2
            blocks = [ResBlock(ch, f, stride=stride)]
            for _ in range(n_blocks_per_stage[i] - 1):
                blocks.append(ResBlock(f, f))
            self.stages.append(nn.Sequential(*blocks))
            ch = f

    def forward(self, x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class UNetDecoder(nn.Module):
    def __init__(self, features, n_blocks_per_stage=None):
        super().__init__()
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * (len(features) - 1)
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            in_ch = features[i]
            out_ch = features[i + 1]
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            ))
            concat_ch = out_ch * 2
            dec_blocks = [ResBlock(concat_ch, out_ch)]
            for _ in range(n_blocks_per_stage[i] - 1):
                dec_blocks.append(ResBlock(out_ch, out_ch))
            self.blocks.append(nn.Sequential(*dec_blocks))

    def forward(self, bottleneck, skips, return_intermediates=False):
        x = bottleneck
        intermediates = []
        for i, (up, block) in enumerate(zip(self.ups, self.blocks)):
            x = up(x)
            skip = skips[-(i + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            if return_intermediates:
                intermediates.append(x)
        if return_intermediates:
            return x, intermediates
        return x


# ---------------------------------------------------------------------------
# TTT-UNet Bot: TTT at bottleneck (primary variant from paper)
# ---------------------------------------------------------------------------

class TTTUNet(nn.Module):
    """TTT-UNet: UNet with Test-Time Training layer at bottleneck.

    Same architecture as U-Mamba Bot, but replaces MambaLayer with TTTLayer.

    Args:
        in_channels: Input channels.
        num_classes: Output classes.
        img_size: Input image size.
        features: Channel counts per encoder stage.
        n_blocks_per_stage: Number of ResBlocks per encoder stage.
        ttt_d_state: TTT number of heads proxy.
    """
    def __init__(self, in_channels=3, num_classes=2, img_size=224,
                 features=None, n_blocks_per_stage=None,
                 ttt_d_state=16, deep_supervision=False, **kwargs):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]
        if n_blocks_per_stage is None:
            n_blocks_per_stage = [2] * len(features)
        self.deep_supervision = deep_supervision

        self.encoder = UNetEncoder(in_channels, features, n_blocks_per_stage)

        # TTT at bottleneck
        self.ttt_bot = TTTLayer(features[-1], d_state=ttt_d_state)

        # Decoder
        dec_features = list(reversed(features))
        dec_n_blocks = list(reversed(n_blocks_per_stage[:-1]))
        self.decoder = UNetDecoder(dec_features, dec_n_blocks)

        self.head = nn.Conv2d(features[0], num_classes, 1)

        # Deep supervision side output heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(f, num_classes, 1) for f in reversed(features[:-1])
            ])

    def forward(self, x):
        input_size = x.shape[2:]
        skips = self.encoder(x)

        # TTT bottleneck
        bot = self.ttt_bot(skips[-1])

        # Decoder
        if self.training and self.deep_supervision:
            out, intermediates = self.decoder(bot, skips, return_intermediates=True)
        else:
            out = self.decoder(bot, skips)

        out = self.head(out)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear',
                                align_corners=False)

        if self.training and self.deep_supervision:
            aux = []
            for feat, head in zip(intermediates[:-1], self.ds_heads):
                a = head(feat)
                if a.shape[2:] != input_size:
                    a = F.interpolate(a, size=input_size, mode='bilinear', align_corners=False)
                aux.append(a)
            return [out] + aux

        return out
