"""Unified weight downloader for medseg.

Each method in the project that needs a non-trivial pretrained checkpoint
(MedSAM, GroundingDINO, GloVe, etc.) registers a :class:`WeightSource`
here. At runtime callers invoke :func:`ensure_weight` with the registry
key; the downloader tries every URL in order (HuggingFace Hub first,
then direct HTTP mirrors), caches the result under
``$MEDSEG_WEIGHT_CACHE`` (or ``~/.cache/medseg/weights``), and returns the
local path.

When **every** automated source fails, the function raises
:class:`WeightDownloadError` carrying:
    * the canonical manual download URL,
    * the exact target path the user should place the file at,
    * any extra license / token instructions.

So a yaml that names a method requiring weights either Just Works (auto
download) or fails with an actionable message telling the user where to
get the file and where to put it. Per project policy there is no silent
substitution of a different checkpoint or random initialisation.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class WeightSource:
    """A single registered checkpoint, with auto-download + manual fallback URL.

    Attributes
    ----------
    name
        Registry key (e.g. ``"medsam_vit_b"``).
    filename
        File name written into the cache directory.
    sources
        Ordered list of auto-downloadable URLs / HF specs. Each entry is
        a callable ``(target_path: Path) -> None`` that fetches the file
        when invoked. Helpers :func:`_hf_file`, :func:`_http` produce
        these.
    sha256
        Optional sha256 hex digest of the expected file.
    manual_url
        Human-facing URL to show the user when every automatic source
        fails (e.g. the official GitHub release page or Google Drive
        folder).
    manual_instructions
        Extra multi-line text shown to the user on failure (license /
        HF token / decompression instructions).
    size_mb
        Rough file size for the log message; ``None`` if unknown.
    """

    name: str
    filename: str
    sources: List[Callable[[Path], None]] = field(default_factory=list)
    sha256: Optional[str] = None
    manual_url: str = ""
    manual_instructions: str = ""
    size_mb: Optional[int] = None


class WeightDownloadError(RuntimeError):
    """Raised when a required checkpoint cannot be auto-downloaded.

    The message always contains the canonical manual download URL and the
    exact target path so the user can drop the file in by hand.
    """


# ----------------------------------------------------------------------
# Cache root
# ----------------------------------------------------------------------


def default_cache_root() -> Path:
    """Return ``$MEDSEG_WEIGHT_CACHE`` or ``~/.cache/medseg/weights``."""
    env = os.environ.get("MEDSEG_WEIGHT_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "medseg" / "weights"


# ----------------------------------------------------------------------
# Source helpers — each returns a callable that performs one fetch
# ----------------------------------------------------------------------


def _hf_file(repo_id: str, filename: str, repo_type: str = "model") -> Callable[[Path], None]:
    """Build a fetcher that pulls ``filename`` from a HuggingFace repo."""

    def _fetch(target: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for HF-based downloads. "
                "Install it with `pip install huggingface_hub`."
            ) from e
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            cache_dir=str(target.parent.parent),
            local_dir=str(target.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may write to a different filename if HF caches by
        # blob hash; make sure the final path matches what the caller wants.
        if Path(local) != target:
            try:
                Path(local).replace(target)
            except OSError:
                # Cross-device rename: fall back to copy + unlink.
                import shutil
                shutil.copyfile(local, target)
                Path(local).unlink(missing_ok=True)

    _fetch.__name__ = f"hf_file:{repo_id}/{filename}"
    return _fetch


def _http(url: str) -> Callable[[Path], None]:
    """Build a fetcher that downloads ``url`` to the target path."""

    def _fetch(target: Path) -> None:
        import urllib.request

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                read = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    read += len(chunk)
                    if total and read % (16 * 1024 * 1024) < 1024 * 1024:
                        pct = 100.0 * read / total
                        logger.info(f"  {url}: {read/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)")
            tmp.replace(target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    _fetch.__name__ = f"http:{url}"
    return _fetch


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

WEIGHT_REGISTRY: dict[str, WeightSource] = {}


def register(src: WeightSource) -> WeightSource:
    """Register a weight source (idempotent on repeated import)."""
    WEIGHT_REGISTRY[src.name] = src
    return src


# --- MedSAM (Ma et al., Nature Communications 2024) ----------------------
register(WeightSource(
    name="medsam_vit_b",
    filename="medsam_vit_b.pth",
    sources=[
        _hf_file("SansuiHan/medical_models", "medsam_vit_b.pth"),
    ],
    manual_url=(
        "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN "
        "or https://zenodo.org/records/10689643"
    ),
    manual_instructions=(
        "Download medsam_vit_b.pth from the official MedSAM release and "
        "place it at the printed cache path. The HF SamModel format at "
        "flaviagiammarino/medsam-vit-base uses different key names and is "
        "not bit-equivalent — do not substitute it."
    ),
    size_mb=375,
))


# --- GroundingDINO (Liu et al., ECCV 2024) -------------------------------
register(WeightSource(
    name="groundingdino_swint_ogc",
    filename="groundingdino_swint_ogc.pth",
    sources=[
        _hf_file("ShilongLiu/GroundingDINO", "groundingdino_swint_ogc.pth"),
    ],
    manual_url=(
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha/groundingdino_swint_ogc.pth"
    ),
    manual_instructions=(
        "Swin-T variant of GroundingDINO (open-set object detector). "
        "Used as the prompt source for MedSAM/SAM2 in detector→segmenter "
        "pipelines (configs/text_guided/synapse_grounding_dino_*.yaml)."
    ),
    size_mb=694,
))

register(WeightSource(
    name="groundingdino_swinb_cogcoor",
    filename="groundingdino_swinb_cogcoor.pth",
    sources=[
        _hf_file("ShilongLiu/GroundingDINO", "groundingdino_swinb_cogcoor.pth"),
    ],
    manual_url=(
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    ),
    manual_instructions="Swin-B variant of GroundingDINO.",
    size_mb=938,
))


# --- SAM ViT-B (Kirillov et al., ICCV 2023) ------------------------------
register(WeightSource(
    name="sam_vit_b",
    filename="sam_vit_b_01ec64.pth",
    sources=[
        _http("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
    ],
    manual_url="https://github.com/facebookresearch/segment-anything#model-checkpoints",
    manual_instructions="Vanilla SAM ViT-B; required by SaLIP when sam_variant='vit_b'.",
    size_mb=375,
))

register(WeightSource(
    name="sam_vit_l",
    filename="sam_vit_l_0b3195.pth",
    sources=[
        _http("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
    ],
    manual_url="https://github.com/facebookresearch/segment-anything#model-checkpoints",
    manual_instructions="Vanilla SAM ViT-L.",
    size_mb=1250,
))

register(WeightSource(
    name="sam_vit_h",
    filename="sam_vit_h_4b8939.pth",
    sources=[
        _http("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
    ],
    manual_url="https://github.com/facebookresearch/segment-anything#model-checkpoints",
    manual_instructions="Vanilla SAM ViT-H (default backbone for many med-seg papers).",
    size_mb=2560,
))


# --- SAM-Med2D ViT-B (Cheng et al., 2024) --------------------------------
# Official OpenGVLab fine-tune on 4.6M medical image-mask pairs. The released
# checkpoint adds per-block bottleneck adapters and a custom prompt encoder
# tuned for medical clicks/boxes; it is NOT bit-equivalent to vanilla SAM.
register(WeightSource(
    name="sam_med2d_vit_b",
    filename="sam-med2d_b.pth",
    sources=[
        _hf_file("OpenGVLab/SAM-Med2D", "sam-med2d_b.pth"),
    ],
    manual_url=(
        "https://huggingface.co/OpenGVLab/SAM-Med2D/resolve/main/sam-med2d_b.pth "
        "or https://github.com/OpenGVLab/SAM-Med2D#-weights"
    ),
    manual_instructions=(
        "Cheng et al., 'SAM-Med2D' (arXiv 2308.16184, 2024). The official "
        "ViT-B checkpoint is fine-tuned on 4.6M medical image-mask pairs and "
        "supports point/box/mask prompts. Drop the file at the printed cache "
        "path. The vanilla SAM ViT-B weight cannot be substituted."
    ),
    size_mb=2467,
))


# --- GloVe 6B 300d (Pennington et al., EMNLP 2014) -----------------------
register(WeightSource(
    name="glove_6b_300d",
    filename="glove.6B.300d.txt",
    sources=[
        _hf_file("stanfordnlp/glove", "glove.6B.300d.txt", repo_type="dataset"),
    ],
    manual_url="https://nlp.stanford.edu/data/glove.6B.zip",
    manual_instructions=(
        "Stanford GloVe 6B 300-dim word embeddings, used by TGANet's "
        "EmbeddingFeatureFusion. Download the zip, extract "
        "glove.6B.300d.txt, and drop it at the printed path. The zip is "
        "822 MB; just the 300-d file is ~990 MB uncompressed."
    ),
    size_mb=1000,
))


# --- MediSee composite (LLaVA-Med + CLIP + MediSee fine-tune) ------------
# The 4 components are handled separately by medseg/inference/mllm/medisee/
# weights_loader.py (snapshot_download for HF dirs). Registered here only
# so `python -m medseg.utils.weight_downloader --list` enumerates them.
register(WeightSource(
    name="medisee_llava_med",
    filename="(snapshot)",
    sources=[],
    manual_url="https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b",
    manual_instructions=(
        "LLaVA-Med backbone for MediSee. Auto-downloaded via "
        "huggingface_hub.snapshot_download by the MediSee wrapper. "
        "~14 GB; requires HF account."
    ),
    size_mb=14000,
))

register(WeightSource(
    name="medisee_finetune",
    filename="(snapshot)",
    sources=[],
    manual_url="https://huggingface.co/Carryyy/MediSee",
    manual_instructions=(
        "MediSee fine-tune weights (ACM MM 2025). "
        "Auto-downloaded by the MediSee wrapper."
    ),
    size_mb=3000,
))


# ----------------------------------------------------------------------
# Core API
# ----------------------------------------------------------------------


def ensure_weight(
    name: str,
    cache_dir: Optional[str | Path] = None,
    verify: bool = True,
) -> Path:
    """Return a local path to the registered weight, downloading if needed.

    Parameters
    ----------
    name
        Registry key (e.g. ``"medsam_vit_b"``).
    cache_dir
        Override for the cache root. If ``None``, uses
        ``$MEDSEG_WEIGHT_CACHE`` or ``~/.cache/medseg/weights``.
    verify
        When ``True`` and the registered :attr:`WeightSource.sha256` is
        set, the downloaded file is verified after the fetch.

    Raises
    ------
    KeyError
        If ``name`` is not in :data:`WEIGHT_REGISTRY`.
    WeightDownloadError
        If every registered source fails.
    """
    if name not in WEIGHT_REGISTRY:
        raise KeyError(
            f"Unknown weight name '{name}'. "
            f"Known: {sorted(WEIGHT_REGISTRY.keys())}"
        )

    src = WEIGHT_REGISTRY[name]
    root = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_root()
    target = root / src.filename
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size > 0:
        if verify and src.sha256 and _sha256(target) != src.sha256:
            logger.warning(
                f"sha256 mismatch for {target}; re-downloading"
            )
            target.unlink()
        else:
            return target

    if not src.sources:
        raise WeightDownloadError(_manual_message(src, target))

    errors = []
    for fetch in src.sources:
        try:
            logger.info(f"Downloading {name} via {fetch.__name__} -> {target}")
            fetch(target)
            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError(
                    f"fetcher {fetch.__name__} returned without writing the target"
                )
            if verify and src.sha256 and _sha256(target) != src.sha256:
                target.unlink(missing_ok=True)
                raise RuntimeError(
                    f"sha256 mismatch after download via {fetch.__name__}"
                )
            logger.info(f"OK: {name} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
            return target
        except Exception as e:
            errors.append(f"  - {fetch.__name__}: {type(e).__name__}: {e}")
            target.unlink(missing_ok=True)

    raise WeightDownloadError(_manual_message(src, target, errors))


def _manual_message(
    src: WeightSource,
    target: Path,
    errors: Optional[List[str]] = None,
) -> str:
    """Compose the actionable failure message handed to the user."""
    parts = [
        "",
        f"Failed to auto-download weight '{src.name}'.",
        f"  expected file: {target}",
    ]
    if src.size_mb:
        parts.append(f"  approximate size: {src.size_mb} MB")
    if errors:
        parts.append("  attempted sources:")
        parts.extend(errors)
    if src.manual_url:
        parts.append(f"  manual download: {src.manual_url}")
    if src.manual_instructions:
        parts.append("  instructions:")
        parts.extend(f"    {line}" for line in src.manual_instructions.splitlines())
    parts.append(f"  → place the file at: {target}")
    parts.append("")
    return "\n".join(parts)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_path(name: str, cache_dir: Optional[str | Path] = None) -> Path:
    """Return the path a weight would live at, without triggering a download."""
    if name not in WEIGHT_REGISTRY:
        raise KeyError(f"Unknown weight name '{name}'")
    root = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_root()
    return root / WEIGHT_REGISTRY[name].filename


# ----------------------------------------------------------------------
# HF AutoModel/AutoTokenizer wrappers — clearer error on download failure
# ----------------------------------------------------------------------


def hf_from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    *args,
    **kwargs,
):
    """Wrap ``cls.from_pretrained`` so HF download failures surface with
    an actionable manual URL.

    Drop-in replacement for e.g. ``AutoModel.from_pretrained(...)``::

        from medseg.utils.weight_downloader import hf_from_pretrained
        model = hf_from_pretrained(AutoModel, "microsoft/BiomedVLP-CXR-BERT-specialized")
    """
    try:
        return cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    except Exception as e:
        raise WeightDownloadError(
            f"\nFailed to load HF model '{pretrained_model_name_or_path}' "
            f"via {cls.__name__}.from_pretrained.\n"
            f"  underlying error: {type(e).__name__}: {e}\n"
            f"  manual download: https://huggingface.co/{pretrained_model_name_or_path}\n"
            f"  instructions:\n"
            f"    1. If the repo is gated (e.g. Llama / CogVLM), "
            f"`huggingface-cli login` with an account that has access.\n"
            f"    2. If you are behind a proxy, set HF_ENDPOINT=https://hf-mirror.com.\n"
            f"    3. To use a local copy, pass the local directory path "
            f"instead of the repo id.\n"
        ) from e


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="medseg weight downloader / inspector"
    )
    parser.add_argument("--cache", type=str, default=None,
                        help="override cache root (defaults to $MEDSEG_WEIGHT_CACHE "
                             "or ~/.cache/medseg/weights)")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list registered weights")

    p_dl = sub.add_parser("download", help="download a registered weight")
    p_dl.add_argument("name", help="registry key (use `list` to see them)")

    sub.add_parser("check", help="check which weights are present in the cache")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "list" or args.cmd is None:
        root = Path(args.cache).expanduser() if args.cache else default_cache_root()
        print(f"cache root: {root}")
        for name in sorted(WEIGHT_REGISTRY.keys()):
            src = WEIGHT_REGISTRY[name]
            size = f"~{src.size_mb} MB" if src.size_mb else ""
            here = (root / src.filename).exists()
            print(f"  [{'OK' if here else '  '}] {name:35s} {size:10s} -> {src.filename}")
        return 0

    if args.cmd == "download":
        try:
            path = ensure_weight(args.name, cache_dir=args.cache)
            print(f"OK: {path}")
            return 0
        except WeightDownloadError as e:
            print(str(e))
            return 1

    if args.cmd == "check":
        root = Path(args.cache).expanduser() if args.cache else default_cache_root()
        missing = [n for n, s in WEIGHT_REGISTRY.items()
                   if s.sources and not (root / s.filename).exists()]
        if missing:
            print("missing weights (run `download <name>` to fetch):")
            for n in missing:
                print(f"  - {n}")
            return 1
        print("all auto-downloadable weights are present")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_cli())
