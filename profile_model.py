"""Model profiling tool: measure FLOPs, parameter count, and FPS.

Only counts parameters that are actually used during the forward pass.
Uses forward hooks to track which modules are called and accumulate FLOPs.

Usage:
    # Profile a single config (FLOPs + params)
    python3 profile_model.py --config configs/synapse/unet_resnet34.yaml

    # Profile with FPS benchmark
    python3 profile_model.py --config configs/synapse/unet_resnet34.yaml --fps

    # FPS with custom settings
    python3 profile_model.py --config configs/synapse/unet_resnet34.yaml --fps --warmup 50 --runs 200 --batch_size 4

    # Profile all configs in a directory
    python3 profile_model.py --config_dir configs/synapse/

    # Batch profile with FPS
    python3 profile_model.py --config_dir configs/synapse/ --fps

    # Custom input size
    python3 profile_model.py --config configs/synapse/unet_resnet34.yaml --img_size 512

    # Show per-module breakdown
    python3 profile_model.py --config configs/synapse/unet_resnet34.yaml --detail
"""

import os
import sys
import argparse
import time
import yaml
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medseg.model_builder import build_model

# Trigger component registration
import medseg.models.encoders  # noqa: F401
import medseg.models.decoders  # noqa: F401
import medseg.models.skip_connections  # noqa: F401
import medseg.models.bottlenecks  # noqa: F401


# ============================================================
# FLOPs counting for common layer types
# ============================================================

def _conv_flops(module: nn.Conv2d, input_shape: torch.Size, output_shape: torch.Size) -> int:
    """FLOPs for Conv2d: kernel_h * kernel_w * in_ch * out_ch * out_h * out_w / groups."""
    batch, out_ch, out_h, out_w = output_shape
    in_ch = module.in_channels // module.groups
    kh, kw = module.kernel_size
    # Multiply-add = 2 ops per element
    flops = kh * kw * in_ch * out_ch * out_h * out_w // module.groups
    if module.bias is not None:
        flops += out_ch * out_h * out_w
    return flops


def _linear_flops(module: nn.Linear, input_shape: torch.Size, output_shape: torch.Size) -> int:
    """FLOPs for Linear: in_features * out_features * batch_elements."""
    # input_shape may be (B, *, in) → output_shape (B, *, out)
    batch_elements = 1
    for s in output_shape[:-1]:
        batch_elements *= s
    flops = module.in_features * module.out_features
    if module.bias is not None:
        flops += module.out_features
    # Multiply by spatial/batch elements excluding the last dim
    # But we report per-sample FLOPs, so divide by batch size
    per_sample = flops * (batch_elements // output_shape[0])
    return per_sample


def _bn_flops(module, input_shape: torch.Size, output_shape: torch.Size) -> int:
    """FLOPs for BatchNorm: 2 * num_elements (mean + variance normalization)."""
    # Per sample: 2 ops per element (subtract mean, divide by std)
    elements = 1
    for s in output_shape[1:]:  # skip batch
        elements *= s
    return 2 * elements


def _ln_flops(module: nn.LayerNorm, input_shape: torch.Size, output_shape: torch.Size) -> int:
    """FLOPs for LayerNorm."""
    elements = 1
    for s in output_shape[1:]:
        elements *= s
    return 2 * elements


def _gelu_flops(module, input_shape: torch.Size, output_shape: torch.Size) -> int:
    """FLOPs for GELU activation (approximate)."""
    elements = 1
    for s in output_shape[1:]:
        elements *= s
    return elements


def _softmax_flops(module, input_shape: torch.Size, output_shape: torch.Size) -> int:
    """FLOPs for Softmax."""
    elements = 1
    for s in output_shape[1:]:
        elements *= s
    return 3 * elements  # exp + sum + div


def _adaptive_pool_flops(module, input_shape: torch.Size, output_shape: torch.Size) -> int:
    """FLOPs for adaptive pooling."""
    # Each output element is an average of input elements
    in_elements = 1
    for s in input_shape[2:]:
        in_elements *= s
    out_elements = 1
    for s in output_shape[2:]:
        out_elements *= s
    return in_elements  # per sample, per channel


def _upsample_flops(module, input_shape: torch.Size, output_shape: torch.Size) -> int:
    """FLOPs for bilinear interpolation."""
    # Bilinear: 4 multiplications + 3 additions per output pixel
    out_elements = 1
    for s in output_shape[1:]:
        out_elements *= s
    return 7 * out_elements


# Map module types to FLOPs calculators
_FLOPS_HANDLERS = {
    nn.Conv2d: _conv_flops,
    nn.ConvTranspose2d: _conv_flops,  # same formula approximately
    nn.Linear: _linear_flops,
    nn.BatchNorm2d: _bn_flops,
    nn.BatchNorm1d: _bn_flops,
    nn.GroupNorm: _bn_flops,
    nn.InstanceNorm2d: _bn_flops,
    nn.LayerNorm: _ln_flops,
    nn.GELU: _gelu_flops,
    nn.Softmax: _softmax_flops,
    nn.AdaptiveAvgPool2d: _adaptive_pool_flops,
    nn.AdaptiveMaxPool2d: _adaptive_pool_flops,
    nn.Upsample: _upsample_flops,
    nn.UpsamplingBilinear2d: _upsample_flops,
}


# ============================================================
# Profiler core
# ============================================================

class ModelProfiler:
    """Profile a model's FLOPs and active parameter count via forward hooks.

    Only parameters belonging to modules that are actually called during
    the forward pass are counted. This correctly handles models with
    conditional branches, unused layers, etc.
    """

    def __init__(self, model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)):
        self.model = model
        self.input_size = input_size

        # Results
        self._module_flops: Dict[str, int] = OrderedDict()
        self._module_params: Dict[str, int] = OrderedDict()
        self._active_modules: set = set()
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks on all leaf modules."""
        for name, module in self.model.named_modules():
            # Only hook leaf modules (no children) or modules with known FLOPs
            hook = module.register_forward_hook(self._make_hook(name, module))
            self._hooks.append(hook)

    def _make_hook(self, name: str, module: nn.Module):
        def hook_fn(m, inp, out):
            self._active_modules.add(name)

            # Get input/output shapes
            if isinstance(inp, tuple) and len(inp) > 0:
                first_inp = inp[0]
            else:
                first_inp = inp

            if isinstance(first_inp, torch.Tensor):
                in_shape = first_inp.shape
            else:
                in_shape = None

            if isinstance(out, torch.Tensor):
                out_shape = out.shape
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                out_shape = out[0].shape
            else:
                out_shape = None

            # Count FLOPs for this module type
            flops = 0
            module_type = type(m)
            if module_type in _FLOPS_HANDLERS and in_shape is not None and out_shape is not None:
                try:
                    flops = _FLOPS_HANDLERS[module_type](m, in_shape, out_shape)
                except Exception:
                    flops = 0

            if name in self._module_flops:
                self._module_flops[name] += flops
            else:
                self._module_flops[name] = flops

            # Count parameters for this specific module (not children)
            own_params = sum(p.numel() for p in m.parameters(recurse=False))
            self._module_params[name] = own_params

        return hook_fn

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def profile(self) -> Dict[str, Any]:
        """Run forward pass and collect profiling data.

        Returns:
            Dict with keys:
                - total_flops: Total FLOPs (int)
                - total_params: Total model parameters (int) — includes unused
                - active_params: Parameters involved in forward pass (int)
                - flops_str: Human-readable FLOPs string
                - params_str: Human-readable active params string
                - module_details: Per-module breakdown list
        """
        self._module_flops.clear()
        self._module_params.clear()
        self._active_modules.clear()

        self.model.eval()
        device = next(self.model.parameters()).device if len(list(self.model.parameters())) > 0 else torch.device('cpu')
        x = torch.randn(*self.input_size, device=device)

        self._register_hooks()
        try:
            self.model(x)
        finally:
            self._remove_hooks()

        # Compute totals
        total_flops = sum(self._module_flops.values())
        total_params = sum(p.numel() for p in self.model.parameters())

        # Active params: only from modules that were called
        active_param_ids = set()
        for name, module in self.model.named_modules():
            if name in self._active_modules:
                for p in module.parameters(recurse=False):
                    active_param_ids.add(id(p))
        active_params = sum(
            p.numel() for p in self.model.parameters() if id(p) in active_param_ids
        )

        # Build per-module detail
        module_details = []
        for name in self._module_flops:
            if name not in self._active_modules:
                continue
            flops = self._module_flops[name]
            params = self._module_params.get(name, 0)
            if flops > 0 or params > 0:
                module_details.append({
                    'name': name,
                    'type': type(dict(self.model.named_modules())[name]).__name__,
                    'flops': flops,
                    'params': params,
                })

        return {
            'total_flops': total_flops,
            'total_params': total_params,
            'active_params': active_params,
            'flops_str': _format_number(total_flops, 'FLOPs'),
            'total_params_str': _format_number(total_params, 'params'),
            'active_params_str': _format_number(active_params, 'params'),
            'module_details': module_details,
        }


# ============================================================
# FPS Benchmark
# ============================================================

class FPSBenchmark:
    """Measure model inference speed (FPS).

    Performs warmup runs to stabilize, then times multiple inference runs
    to compute average FPS, latency, and throughput.
    Supports both CPU and GPU (with proper CUDA synchronization).
    """

    def __init__(self, model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224),
                 device: str = 'cpu', warmup: int = 30, runs: int = 100):
        self.model = model
        self.input_size = input_size
        self.device = device
        self.warmup = warmup
        self.runs = runs
        self.use_cuda = 'cuda' in device and torch.cuda.is_available()

    @torch.no_grad()
    def benchmark(self) -> Dict[str, Any]:
        """Run FPS benchmark.

        Returns:
            Dict with keys:
                - fps: Frames per second (float)
                - latency_ms: Average latency in milliseconds (float)
                - latency_std_ms: Latency standard deviation in ms (float)
                - throughput: Images per second considering batch size (float)
                - batch_size: Batch size used (int)
                - device: Device used (str)
                - warmup: Number of warmup iterations (int)
                - runs: Number of timed iterations (int)
        """
        self.model.eval()
        batch_size = self.input_size[0]
        x = torch.randn(*self.input_size, device=self.device)

        # Warmup
        for _ in range(self.warmup):
            self.model(x)
            if self.use_cuda:
                torch.cuda.synchronize()

        # Timed runs
        latencies = []
        for _ in range(self.runs):
            if self.use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.model(x)
            if self.use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

        latencies_tensor = torch.tensor(latencies)
        avg_latency = latencies_tensor.mean().item()
        std_latency = latencies_tensor.std().item()
        fps = 1000.0 / avg_latency  # frames per second (batch_size=1 equivalent)
        throughput = batch_size * 1000.0 / avg_latency  # images per second

        return {
            'fps': fps,
            'latency_ms': avg_latency,
            'latency_std_ms': std_latency,
            'throughput': throughput,
            'batch_size': batch_size,
            'device': self.device,
            'warmup': self.warmup,
            'runs': self.runs,
        }


def _format_number(n: int, suffix: str = '') -> str:
    """Format large number with appropriate unit."""
    if n >= 1e12:
        return f"{n / 1e12:.2f} T{suffix}"
    elif n >= 1e9:
        return f"{n / 1e9:.2f} G{suffix}"
    elif n >= 1e6:
        return f"{n / 1e6:.2f} M{suffix}"
    elif n >= 1e3:
        return f"{n / 1e3:.2f} K{suffix}"
    else:
        return f"{n} {suffix}"


# ============================================================
# CLI
# ============================================================

def profile_single_config(config_path: str, img_size: Optional[int] = None,
                          detail: bool = False, device: str = 'cpu',
                          fps: bool = False, warmup: int = 30,
                          runs: int = 100, batch_size: int = 1) -> Dict[str, Any]:
    """Profile a model from a YAML config file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get('model', cfg)
    if img_size is None:
        img_size = model_cfg.get('img_size', 224)
    in_channels = model_cfg.get('encoder', {}).get('in_channels', 3)

    model = build_model(cfg)
    model = model.to(device)

    profiler = ModelProfiler(model, input_size=(1, in_channels, img_size, img_size))
    result = profiler.profile()
    result['config'] = os.path.basename(config_path)

    # FPS benchmark
    if fps:
        bench = FPSBenchmark(
            model,
            input_size=(batch_size, in_channels, img_size, img_size),
            device=device,
            warmup=warmup,
            runs=runs,
        )
        result['fps'] = bench.benchmark()

    return result


def print_result(result: Dict[str, Any], detail: bool = False):
    """Print profiling results."""
    config_name = result.get('config', 'unknown')
    print(f"\n{'=' * 70}")
    print(f"  Model: {config_name}")
    print(f"{'=' * 70}")
    print(f"  FLOPs:            {result['flops_str']}")
    print(f"  Active Params:    {result['active_params_str']}")
    if result['total_params'] != result['active_params']:
        print(f"  Total Params:     {result['total_params_str']}  (includes {_format_number(result['total_params'] - result['active_params'], '')} unused)")

    # FPS results
    if 'fps' in result:
        fps_data = result['fps']
        print(f"  {'─' * 66}")
        print(f"  FPS:              {fps_data['fps']:.1f}")
        print(f"  Latency:          {fps_data['latency_ms']:.2f} ms ± {fps_data['latency_std_ms']:.2f} ms")
        if fps_data['batch_size'] > 1:
            print(f"  Throughput:       {fps_data['throughput']:.1f} img/s  (batch={fps_data['batch_size']})")
        print(f"  Device:           {fps_data['device']}  (warmup={fps_data['warmup']}, runs={fps_data['runs']})")

    print(f"{'=' * 70}")

    if detail and result['module_details']:
        print(f"\n  {'Module':<50} {'Type':<25} {'FLOPs':>14} {'Params':>12}")
        print(f"  {'-' * 50} {'-' * 25} {'-' * 14} {'-' * 12}")
        for m in result['module_details']:
            flops_s = _format_number(m['flops']) if m['flops'] > 0 else '-'
            params_s = _format_number(m['params']) if m['params'] > 0 else '-'
            name = m['name']
            if len(name) > 48:
                name = '...' + name[-45:]
            print(f"  {name:<50} {m['type']:<25} {flops_s:>14} {params_s:>12}")
        print()


def profile_directory(config_dir: str, img_size: Optional[int] = None,
                      device: str = 'cpu', fps: bool = False,
                      warmup: int = 30, runs: int = 100,
                      batch_size: int = 1) -> list:
    """Profile all YAML configs in a directory (recursively)."""
    results = []
    yaml_files = []
    for root, dirs, files in os.walk(config_dir):
        for f in sorted(files):
            if f.endswith('.yaml') and f != 'default.yaml':
                yaml_files.append(os.path.join(root, f))

    print(f"\nProfiling {len(yaml_files)} configs from {config_dir}...\n")
    header = f"{'Config':<40} {'FLOPs':>14} {'Active Params':>16} {'Total Params':>16}"
    if fps:
        header += f" {'FPS':>10} {'Latency':>14}"
    print(header)
    print(f"{'-' * 40} {'-' * 14} {'-' * 16} {'-' * 16}" + (' ' + '-' * 10 + ' ' + '-' * 14 if fps else ''))

    for yf in yaml_files:
        rel_path = os.path.relpath(yf, config_dir)
        try:
            result = profile_single_config(
                yf, img_size=img_size, device=device,
                fps=fps, warmup=warmup, runs=runs, batch_size=batch_size,
            )
            results.append(result)
            extra = ''
            if result['total_params'] != result['active_params']:
                extra = f" ({_format_number(result['total_params'])})"
            line = f"{rel_path:<40} {result['flops_str']:>14} {result['active_params_str']:>16}{extra:>16}"
            if fps and 'fps' in result:
                fps_data = result['fps']
                line += f" {fps_data['fps']:>9.1f} {fps_data['latency_ms']:>10.2f} ms"
            print(line)
        except Exception as e:
            print(f"{rel_path:<40} {'ERROR':>14} {str(e)[:40]}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Profile model FLOPs, parameter count, and FPS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 profile_model.py --config configs/synapse/unet_resnet34.yaml
  python3 profile_model.py --config configs/synapse/unet_resnet34.yaml --fps
  python3 profile_model.py --config configs/synapse/unet_resnet34.yaml --fps --warmup 50 --runs 200
  python3 profile_model.py --config configs/synapse/unet_resnet34.yaml --detail
  python3 profile_model.py --config_dir configs/synapse/
  python3 profile_model.py --config_dir configs/synapse/ --fps
  python3 profile_model.py --config_dir configs/ --img_size 512
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help='Path to a single YAML config file')
    group.add_argument('--config_dir', type=str, help='Directory of YAML configs to profile')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Override input image size (default: from config)')
    parser.add_argument('--detail', action='store_true',
                        help='Show per-module breakdown (single config only)')
    parser.add_argument('--fps', action='store_true',
                        help='Run FPS benchmark')
    parser.add_argument('--warmup', type=int, default=30,
                        help='Number of warmup iterations for FPS (default: 30)')
    parser.add_argument('--runs', type=int, default=100,
                        help='Number of timed iterations for FPS (default: 100)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for FPS benchmark (default: 1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for profiling (default: cpu)')
    args = parser.parse_args()

    if args.config:
        result = profile_single_config(
            args.config, img_size=args.img_size, device=args.device,
            fps=args.fps, warmup=args.warmup, runs=args.runs,
            batch_size=args.batch_size,
        )
        print_result(result, detail=args.detail)
    else:
        profile_directory(
            args.config_dir, img_size=args.img_size, device=args.device,
            fps=args.fps, warmup=args.warmup, runs=args.runs,
            batch_size=args.batch_size,
        )


if __name__ == '__main__':
    main()
