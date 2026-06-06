"""Text + image segmentation datasets (QaTa-COV19 / MosMedData+).

Both datasets follow the layout used by the LViT paper
(https://github.com/HUANGLIZI/LViT):

    <data_root>/
        Train Folder/
            Img/        case_001.png ...
            GT/         case_001.png ...
            Train_text.xlsx     (or .csv) -- columns: image, text
        Val Folder/
            Img/, GT/, Val_text.{xlsx|csv}
        Test Folder/
            Img/, GT/, Test_text.{xlsx|csv}

A more permissive layout is also supported (caption in a single CSV next
to images/ + masks/):

    <root>/
        images/
        masks/
        captions.csv        columns: image, text

The dataset returns a sample dict::

    {
        "image":     FloatTensor(3, H, W) in [0,1],
        "label":     LongTensor(H, W),
        "case_name": str,
        "text":      str,                              # raw caption
        "text_ids":  LongTensor(L) | None,             # if tokenizer set
        "text_mask": LongTensor(L) | None,
    }

When a HF tokenizer is available the caption is also tokenised so that
LanGuideMedSeg / LViT-style models can pick the right tensor.  The
training loop should pass ``text=batch["text"]`` (raw) or ``text={
'input_ids': batch['text_ids'], 'attention_mask': batch['text_mask']}``
depending on the model -- see model wrappers for the bridge.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def _read_caption_table(path: str) -> Dict[str, str]:
    """Parse a caption table (.csv / .xlsx / .tsv) into {image_name: text}.

    The file should expose two columns; the first must contain the image
    filename (with or without extension) and the second the caption.  Any
    extra columns are ignored.  Both pandas (preferred) and a minimal
    csv-only fallback are supported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    try:
        import pandas as pd  # type: ignore
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)
        cols = list(df.columns)
        assert len(cols) >= 2, f"caption table needs >=2 cols, got {cols}"
        img_col, txt_col = cols[0], cols[1]
        return {str(r[img_col]): str(r[txt_col]) for _, r in df.iterrows()}
    except ImportError:  # pragma: no cover
        # csv-only minimal fallback
        if ext in (".xlsx", ".xls"):
            raise RuntimeError("xlsx caption files require pandas+openpyxl")
        import csv
        out = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t" if ext == ".tsv" else ",")
            rows = list(reader)
        if not rows:
            return {}
        # skip header if first row looks like ('image', 'text')
        start = 1 if rows[0][0].lower() in ("image", "filename", "name") else 0
        for r in rows[start:]:
            if len(r) >= 2:
                out[str(r[0])] = str(r[1])
        return out


def _try_tokenizer(name: Optional[str], max_length: int):
    if not name:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        warnings.warn("[TextImageDataset] transformers not installed; caption stays raw")
        return None
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        tok.model_max_length = max_length
        return tok
    except Exception as exc:
        warnings.warn(f"[TextImageDataset] tokenizer '{name}' load failed: {exc}; raw only")
        return None


class TextImageDataset(Dataset):
    """Generic image + mask + caption dataset.

    Args:
        image_dir / mask_dir: explicit folders.  If only ``root_dir`` is
            given, ``<root>/images`` and ``<root>/masks`` are used.
        caption_file: path to a CSV / XLSX / TSV holding (filename, text).
            Filenames may include or omit extensions.
        img_size: target H=W resize.
        tokenizer_name: HF model id used to tokenise captions; None -> raw.
        text_max_length: tokenizer truncation length.
        transform: dict-based transform applied to {image, label}.
        img_suffix / mask_suffix: extension filters.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        caption_file: Optional[str] = None,
        img_size: int = 224,
        tokenizer_name: Optional[str] = None,
        text_max_length: int = 24,
        transform=None,
        img_suffix: str = ".png",
        mask_suffix: str = ".png",
    ):
        super().__init__()
        if image_dir is None or mask_dir is None:
            assert root_dir is not None, "must provide root_dir or image_dir+mask_dir"
            image_dir = image_dir or os.path.join(root_dir, "images")
            mask_dir = mask_dir or os.path.join(root_dir, "masks")
            if caption_file is None:
                # try common names
                for cand in ("captions.csv", "caption.csv", "text.csv", "text.xlsx"):
                    if os.path.exists(os.path.join(root_dir, cand)):
                        caption_file = os.path.join(root_dir, cand)
                        break
        self._image_dir = image_dir
        self._mask_dir = mask_dir
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.transform = transform
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
            warnings.warn(f"[TextImageDataset] image/mask dir missing: {image_dir} | {mask_dir}")
            self._captions: Dict[str, str] = {}
            self.samples: List[Tuple[str, str, str]] = []
        else:
            captions = _read_caption_table(caption_file) if caption_file else {}
            # normalise caption keys: strip extension to match basename
            norm = {}
            for k, v in captions.items():
                base = os.path.splitext(k)[0]
                norm[base] = v
                norm[k] = v  # keep original too
            self._captions = norm

            img_bases = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(image_dir)
                if f.endswith(img_suffix)
            )
            mask_bases = set(
                os.path.splitext(f)[0]
                for f in os.listdir(mask_dir)
                if f.endswith(mask_suffix)
            )
            self.samples = [
                (b + img_suffix, b + mask_suffix, self._captions.get(b, ""))
                for b in img_bases
                if b in mask_bases
            ]

        # tokenizer
        self.text_max_length = text_max_length
        self.tokenizer = _try_tokenizer(tokenizer_name, text_max_length)

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize(self.img_size, Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _load_mask(self, path: str) -> np.ndarray:
        m = Image.open(path).resize(self.img_size, Image.NEAREST)
        arr = np.array(m, dtype=np.int64)
        # binarise if mask is 0/255
        if arr.max() > 1:
            arr = (arr > 0).astype(np.int64)
        return arr

    def _tokenize(self, text: str):
        if self.tokenizer is None or not text:
            return None, None
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        return enc["input_ids"][0].long(), enc["attention_mask"][0].long()

    def __getitem__(self, idx):
        img_file, mask_file, caption = self.samples[idx]
        image = self._load_image(os.path.join(self._image_dir, img_file))
        mask = self._load_mask(os.path.join(self._mask_dir, mask_file))

        if self.transform is not None:
            sample = self.transform({"image": image, "label": mask})
            image, mask = sample["image"], sample["label"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()

        text_ids, text_mask = self._tokenize(caption)

        out = {
            "image": image,
            "label": mask,
            "case_name": img_file,
            "text": caption,
        }
        if text_ids is not None:
            out["text_ids"] = text_ids
            out["text_mask"] = text_mask
        return out


# ---------------------------------------------------------------------------
# concrete datasets matching LViT's directory convention
# ---------------------------------------------------------------------------


class _LViTSplitDataset(TextImageDataset):
    """Helper for LViT-style ``Train Folder / Val Folder / Test Folder`` layout."""

    SPLIT_TO_FOLDER = {
        "train": "Train Folder",
        "val": "Val Folder",
        "test": "Test Folder",
    }
    SPLIT_TO_TXT = {
        "train": "Train_text",
        "val": "Val_text",
        "test": "Test_text",
    }

    def __init__(self, data_root: str, split: str = "train", img_size: int = 224,
                 tokenizer_name: Optional[str] = None, text_max_length: int = 24,
                 transform=None):
        if split not in self.SPLIT_TO_FOLDER:
            raise ValueError(f"split must be one of {list(self.SPLIT_TO_FOLDER)}")
        folder = os.path.join(data_root, self.SPLIT_TO_FOLDER[split])
        image_dir = os.path.join(folder, "Img")
        mask_dir = os.path.join(folder, "GT")
        caption_file = None
        for ext in (".xlsx", ".csv", ".tsv"):
            cand = os.path.join(folder, self.SPLIT_TO_TXT[split] + ext)
            if os.path.exists(cand):
                caption_file = cand
                break
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            caption_file=caption_file,
            img_size=img_size,
            tokenizer_name=tokenizer_name,
            text_max_length=text_max_length,
            transform=transform,
        )
        self.data_root = data_root
        self.split = split


class QaTaCOV19Dataset(_LViTSplitDataset):
    """QaTa-COV19 (LViT enriched version) -- chest X-ray COVID-19 segmentation.

    Per-image radiology captions live in ``Train_text.xlsx`` etc.; refer to
    https://github.com/HUANGLIZI/LViT#datasets for download.
    """


class MosMedPlusDataset(_LViTSplitDataset):
    """MosMedData+ (LViT enriched version) -- COVID-19 CT slice segmentation.

    Same directory contract as :class:`QaTaCOV19Dataset`.
    """
