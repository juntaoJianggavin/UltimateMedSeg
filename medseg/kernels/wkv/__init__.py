"""WKV kernel dispatcher for RWKV-style encoders.

This module unifies how every RWKV-based component in the project obtains a
WKV operator:

1. ``load_wkv_cuda(t_max)`` lazily JIT-compiles the official Vision-RWKV CUDA
   op (shipped here as ``wkv_op.cpp`` + ``wkv_cuda.cu``, byte-identical to the
   files used by ``juntaoJianggavin/RWKV-UNet`` and ``txchen-USTC/Zig-RiR``).
   The op is compiled with ``-DTmax=<t_max>`` because its backward kernel
   declares stack arrays of length ``Tmax / TOKEN_SPLIT`` (TOKEN_SPLIT = 32).
2. ``run_wkv(B, T, C, w, u, k, v)`` dispatches to the CUDA op when available
   (with the official analytic backward kernel) or falls back to a vectorised
   PyTorch implementation that PyTorch autograd can differentiate through
   automatically.

This file fixes a long-standing project bug where the previous
``WKV.backward`` in ``medseg/encoders/rwkv_encoder.py`` returned
``torch.zeros_like`` for every input, silently breaking gradient flow through
every RWKV-based encoder (RWKV-UNet, Zig-RiR). With this dispatcher,
gradients are correct on both CPU and CUDA, and the PyTorch fallback now
matches the sign convention of the CUDA kernel (decay applied via ``pp - w``,
not ``pp + w``).
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import torch

__all__ = [
    "load_wkv_cuda",
    "is_cuda_available",
    "get_load_error",
    "wkv_pytorch",
    "run_wkv",
    "WKVCudaFunction",
]


# ---------------------------------------------------------------------------
# CUDA lazy loader
# ---------------------------------------------------------------------------

_CUDA_LOCK = threading.Lock()
_CUDA_OP = None
_CUDA_T_MAX = 0
_CUDA_LOAD_FAILED = False
_CUDA_LOAD_ERROR: Optional[BaseException] = None


def _kernel_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_wkv_cuda(t_max: int = 8192, force: bool = False, verbose: bool = False):
    """Lazily JIT-compile the official Vision-RWKV CUDA op.

    Parameters
    ----------
    t_max : int
        Maximum sequence length the kernel must support. The CUDA backward
        kernel declares stack arrays of length ``Tmax / 32``, so callers
        passing a larger ``T`` must trigger a recompile (this function
        recompiles when the requested ``t_max`` exceeds the cached one).
    force : bool
        Re-attempt compilation even if a previous attempt failed.
    verbose : bool
        Forwarded to ``torch.utils.cpp_extension.load``.

    Returns
    -------
    Optional[module]
        The compiled op module, or ``None`` when CUDA / nvcc / matching GPU
        are unavailable. The first failure is cached so subsequent calls
        return ``None`` immediately (use ``force=True`` to retry).
    """
    global _CUDA_OP, _CUDA_T_MAX, _CUDA_LOAD_FAILED, _CUDA_LOAD_ERROR

    if _CUDA_LOAD_FAILED and not force:
        return None
    if _CUDA_OP is not None and t_max <= _CUDA_T_MAX:
        return _CUDA_OP

    with _CUDA_LOCK:
        if _CUDA_OP is not None and t_max <= _CUDA_T_MAX:
            return _CUDA_OP
        if _CUDA_LOAD_FAILED and not force:
            return None
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA runtime not available")

            from torch.utils.cpp_extension import load  # local import: heavy

            d = _kernel_dir()
            sources = [
                os.path.join(d, "wkv_op.cpp"),
                os.path.join(d, "wkv_cuda.cu"),
            ]
            for src in sources:
                if not os.path.isfile(src):
                    raise FileNotFoundError(f"Missing WKV source: {src}")

            op = load(
                name=f"wkv_t{t_max}",
                sources=sources,
                verbose=verbose,
                extra_cuda_cflags=[
                    "-res-usage",
                    "--use_fast_math",
                    "-O3",
                    "--maxrregcount=60",
                    "--extra-device-vectorization",
                    f"-DTmax={t_max}",
                ],
            )
            _CUDA_OP = op
            _CUDA_T_MAX = t_max
            _CUDA_LOAD_FAILED = False
            _CUDA_LOAD_ERROR = None
            return op
        except Exception as exc:  # pragma: no cover - depends on environment
            _CUDA_LOAD_FAILED = True
            _CUDA_LOAD_ERROR = exc
            if verbose:
                print(f"[wkv] CUDA op compilation failed: {exc!r}; "
                      f"falling back to vectorised PyTorch implementation.")
            return None


def is_cuda_available() -> bool:
    """``True`` when the compiled CUDA op is ready to use."""
    return _CUDA_OP is not None


def get_load_error() -> Optional[BaseException]:
    """Return the exception captured during the most recent failed load."""
    return _CUDA_LOAD_ERROR


# ---------------------------------------------------------------------------
# Vectorised PyTorch fallback (autograd differentiable)
# ---------------------------------------------------------------------------

def wkv_pytorch(B: int, T: int, C: int,
                w: torch.Tensor, u: torch.Tensor,
                k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vectorised PyTorch WKV that matches the Vision-RWKV CUDA semantics.

    Computes (broadcasting ``w, u`` of shape ``(C,)`` over ``(B, T, C)``)::

        y[b, t, c] = ( sum_{i<t} exp(k_i - (t-1-i)*w) * v_i  +  exp(u + k_t) * v_t )
                   / ( sum_{i<t} exp(k_i - (t-1-i)*w)        +  exp(u + k_t)        + eps )

    Implemented with running max for numerical stability so it matches the
    CUDA kernel's ``no = max(o1, k - w*(i-_t))`` convention. Differentiable
    through standard PyTorch autograd; no manual backward is needed.
    """
    if k.dim() != 3 or v.dim() != 3:
        raise ValueError(f"k, v must be (B, T, C); got {k.shape}, {v.shape}")
    if k.shape != v.shape:
        raise ValueError(f"k and v shape mismatch: {k.shape} vs {v.shape}")
    if w.dim() != 1 or u.dim() != 1 or w.shape[0] != C or u.shape[0] != C:
        raise ValueError(
            f"w/u must be 1-D of length C={C}; got {w.shape}, {u.shape}"
        )

    device = v.device
    dtype = v.dtype

    aa = torch.zeros(B, C, device=device, dtype=dtype)
    bb = torch.zeros(B, C, device=device, dtype=dtype)
    pp = torch.full((B, C), -1e38, device=device, dtype=dtype)

    out_slices = []
    for t in range(T):
        kk = k[:, t, :]                # (B, C)
        vv = v[:, t, :]                # (B, C)

        # --- numerator/denominator at step t (current token gets bonus u) ---
        ww = u + kk                    # (B, C)
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        out_t = (e1 * aa + e2 * vv) / (e1 * bb + e2 + 1e-9)
        out_slices.append(out_t)

        # --- recurrence: aa <- exp(-w) * aa + exp(kk) * vv ----------------
        # Equivalent to ``pp_new = max(pp - w, kk)`` (note: pp - w, NOT pp + w
        # as in the previous buggy implementation). This matches the CUDA
        # forward kernel which uses ``no = max(o1, k - w*(i-_t))`` so that w
        # acts as a positive decay rate per step.
        ww2 = pp - w                   # (B, C)
        p2 = torch.maximum(ww2, kk)
        e1_2 = torch.exp(ww2 - p2)
        e2_2 = torch.exp(kk - p2)
        aa = e1_2 * aa + e2_2 * vv
        bb = e1_2 * bb + e2_2
        pp = p2

    return torch.stack(out_slices, dim=1)  # (B, T, C)


# ---------------------------------------------------------------------------
# CUDA-backed autograd Function
# ---------------------------------------------------------------------------

class WKVCudaFunction(torch.autograd.Function):
    """Autograd wrapper around the official Vision-RWKV CUDA op.

    The op writes outputs into pre-allocated tensors (it does not allocate
    them itself), so we mirror the upstream Python wrapper: allocate ``y``
    and gradient tensors with empty memory and let the kernel fill them.
    """

    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        op = _CUDA_OP
        if op is None:
            raise RuntimeError(
                "WKVCudaFunction called but CUDA op is not loaded. "
                "Use run_wkv() which falls back to PyTorch instead."
            )
        ctx.B = B
        ctx.T = T
        ctx.C = C
        w = w.contiguous().float()
        u = u.contiguous().float()
        k = k.contiguous().float()
        v = v.contiguous().float()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device=v.device, dtype=torch.float32,
                        memory_format=torch.contiguous_format)
        op.forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        op = _CUDA_OP
        if op is None:
            raise RuntimeError("WKVCudaFunction backward called without CUDA op.")
        B, T, C = ctx.B, ctx.T, ctx.C
        w, u, k, v = ctx.saved_tensors
        gy = gy.contiguous().float()
        gw = torch.zeros((B, C), device=v.device, dtype=torch.float32)
        gu = torch.zeros((B, C), device=v.device, dtype=torch.float32)
        gk = torch.zeros((B, T, C), device=v.device, dtype=torch.float32)
        gv = torch.zeros((B, T, C), device=v.device, dtype=torch.float32)
        op.backward(B, T, C, w, u, k, v, gy, gw, gu, gk, gv)
        # Reduce per-batch grads on (w, u) -> (C,) to match parameter shape.
        gw = gw.sum(dim=0)
        gu = gu.sum(dim=0)
        return (None, None, None, gw, gu, gk, gv)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def run_wkv(B: int, T: int, C: int,
            w: torch.Tensor, u: torch.Tensor,
            k: torch.Tensor, v: torch.Tensor,
            *, try_cuda: bool = True, t_max: int = 8192) -> torch.Tensor:
    """Compute WKV attention with CUDA acceleration when possible.

    The function first attempts to load (and cache) the CUDA op when the
    inputs live on a CUDA device, and dispatches to it via the autograd
    ``Function`` above. On CPU - or when the CUDA op cannot be compiled -
    it falls back to ``wkv_pytorch`` whose gradient is computed by autograd
    automatically. Both paths produce mathematically equivalent results
    (modulo float32 rounding).

    Parameters
    ----------
    try_cuda : bool
        Set to ``False`` to force the PyTorch path (useful for unit tests).
    t_max : int
        Forwarded to ``load_wkv_cuda`` on the first compilation.
    """
    use_cuda = (
        try_cuda
        and v.is_cuda and k.is_cuda and w.is_cuda and u.is_cuda
        and load_wkv_cuda(t_max=max(t_max, T)) is not None
    )
    if use_cuda:
        return WKVCudaFunction.apply(
            B, T, C, w.float(), u.float(), k.float(), v.float()
        )
    return wkv_pytorch(B, T, C, w.float(), u.float(), k.float(), v.float())
