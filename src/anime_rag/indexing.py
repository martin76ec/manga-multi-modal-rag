"""Build and persist the multimodal panel index: one image embedding and one
text embedding per panel, living in the same CLIP vector space.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedder import ClipJapaneseEmbedder
from .panels import PanelDoc

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _has_meaningful_text(text: str, min_chars: int = 3) -> bool:
    """CLIP text embeddings of very short or punctuation-only strings (e.g.
    "……", a lone kana particle) tend to collapse near a generic point in
    embedding space and score spuriously high against unrelated queries.
    Panels whose dialogue doesn't clear this bar fall back to image-only
    scoring instead of polluting the text-score ranking."""
    return sum(len(w) for w in _WORD_RE.findall(text)) >= min_chars

META_FILENAME = "meta.jsonl"
IMAGE_EMB_FILENAME = "image_emb.npy"
TEXT_EMB_FILENAME = "text_emb.npy"
HAS_TEXT_FILENAME = "has_text.npy"


@dataclass
class Index:
    meta: list[dict]
    image_emb: np.ndarray
    text_emb: np.ndarray
    has_text: np.ndarray

    def __len__(self) -> int:
        return len(self.meta)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / META_FILENAME, "w", encoding="utf-8") as f:
            for row in self.meta:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        np.save(index_dir / IMAGE_EMB_FILENAME, self.image_emb)
        np.save(index_dir / TEXT_EMB_FILENAME, self.text_emb)
        np.save(index_dir / HAS_TEXT_FILENAME, self.has_text)

    @classmethod
    def load(cls, index_dir: Path) -> "Index":
        meta = []
        with open(index_dir / META_FILENAME, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    meta.append(json.loads(line))
        image_emb = np.load(index_dir / IMAGE_EMB_FILENAME)
        text_emb = np.load(index_dir / TEXT_EMB_FILENAME)
        has_text = np.load(index_dir / HAS_TEXT_FILENAME)
        return cls(meta=meta, image_emb=image_emb, text_emb=text_emb, has_text=has_text)

    @classmethod
    def merge(cls, indices: list["Index"]) -> "Index":
        meta = [row for idx in indices for row in idx.meta]
        image_emb = np.concatenate([idx.image_emb for idx in indices], axis=0)
        text_emb = np.concatenate([idx.text_emb for idx in indices], axis=0)
        has_text = np.concatenate([idx.has_text for idx in indices], axis=0)
        return cls(meta=meta, image_emb=image_emb, text_emb=text_emb, has_text=has_text)

    @classmethod
    def load_all(cls, index_dir: Path) -> "Index":
        """Load and merge every per-book sub-index under index_dir."""
        book_dirs = sorted(
            d for d in index_dir.iterdir() if d.is_dir() and (d / META_FILENAME).exists()
        )
        if not book_dirs:
            raise FileNotFoundError(f"No book indices found under {index_dir}")
        return cls.merge([cls.load(d) for d in book_dirs])


def build_index(
    panels: list[PanelDoc],
    embedder: ClipJapaneseEmbedder,
    image_batch_size: int = 32,
    text_batch_size: int = 64,
) -> Index:
    meta = [
        {
            "id": p.id,
            "book": p.book,
            "page_index": p.page_index,
            "bbox": list(p.bbox),
            "text": p.text,
            "characters": list(p.characters),
            "image_path": str(p.image_path),
        }
        for p in panels
    ]
    image_emb = embedder.embed_images([p.image_path for p in panels], batch_size=image_batch_size)
    text_emb = embedder.embed_texts([p.text for p in panels], batch_size=text_batch_size)
    has_text = np.array([_has_meaningful_text(p.text) for p in panels], dtype=bool)
    return Index(meta=meta, image_emb=image_emb, text_emb=text_emb, has_text=has_text)
