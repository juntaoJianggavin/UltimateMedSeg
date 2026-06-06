"""2D relative positional embedding for graph convolution.

Source: https://github.com/SLDGroup/G-CASCADE (lib/gcn_lib/pos_embed.py)
"""

import numpy as np


def get_2d_relative_pos_embed(embed_dim, grid_size):
    """Generate 2D relative positional embedding.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Height/width of the feature grid.

    Returns:
        (2*grid_size-1, 2*grid_size-1, embed_dim) relative position embeddings.
    """
    import torch
    from functools import partial

    pos_dim = embed_dim // 4
    omega = np.arange(pos_dim, dtype=np.float64)
    omega /= pos_dim
    omega = 1.0 / (10000 ** omega)

    pos_h = np.arange(grid_size)
    pos_w = np.arange(grid_size)

    out_h = np.einsum('m,d->md', pos_h, omega)
    out_w = np.einsum('m,d->md', pos_w, omega)

    pos_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)
    pos_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)

    pos_h = torch.from_numpy(pos_h).float().unsqueeze(1)
    pos_w = torch.from_numpy(pos_w).float().unsqueeze(0)

    # Build relative position table
    rel_pos_h = torch.cat([pos_h, -pos_h.flip(0)[:-1]], dim=0)
    rel_pos_w = torch.cat([pos_w, -pos_w.flip(1)[:, :-1]], dim=1)

    pos = rel_pos_h.unsqueeze(-1) + rel_pos_w.unsqueeze(-2)
    return pos
