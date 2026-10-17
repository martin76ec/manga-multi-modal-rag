"""Multimodal retrieval over a built panel index.

Each panel carries two embeddings (image crop, dialogue text) in the same
CLIP space. A query -- text or image -- is embedded once and scored against
both, so a match can come from what a panel looks like, what was said in it,
or both at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .embedder import ClipJapaneseEmbedder
from .indexing import Index
from .translate import Translator, contains_japanese


@dataclass
class SearchResult:
    score: float
    image_score: float
    text_score: float | None
    meta: dict
    query_ja: str | None = None  # the Japanese text actually embedded, if translated


class Searcher:
    def __init__(
        self,
        index: Index,
        embedder: ClipJapaneseEmbedder | None = None,
        translator: Translator | None = None,
    ):
        self.index = index
        self.embedder = embedder or ClipJapaneseEmbedder()
        self.translator = translator or Translator()

    def _rank(self, query_emb: np.ndarray, top_k: int, w_img: float, w_text: float) -> list[SearchResult]:
        img_scores = self.index.image_emb @ query_emb
        text_scores = self.index.text_emb @ query_emb
        has_text = self.index.has_text

        scores = np.where(has_text, w_img * img_scores + w_text * text_scores, img_scores)

        order = np.argsort(-scores)[:top_k]
        results = []
        for i in order:
            results.append(
                SearchResult(
                    score=float(scores[i]),
                    image_score=float(img_scores[i]),
                    text_score=float(text_scores[i]) if has_text[i] else None,
                    meta=self.index.meta[i],
                )
            )
        return results

    def search_text(
        self, query: str, top_k: int = 10, w_img: float = 0.6, w_text: float = 0.4, translate: bool = True
    ) -> list[SearchResult]:
        query_ja = query
        if translate and not contains_japanese(query):
            query_ja = self.translator.en_to_ja([query])[0]

        query_emb = self.embedder.embed_texts([query_ja])[0]
        results = self._rank(query_emb, top_k, w_img, w_text)
        for r in results:
            r.query_ja = query_ja if query_ja != query else None
        return results

    def search_image(self, image_path: Path, top_k: int = 10, w_img: float = 0.6, w_text: float = 0.4) -> list[SearchResult]:
        image = Image.open(image_path).convert("RGB")
        query_emb = self.embedder.embed_image(image)
        return self._rank(query_emb, top_k, w_img, w_text)

    def search_by_text_embedding(
        self, japanese_text: str, top_k: int = 10, w_img: float = 0.6, w_text: float = 0.4
    ) -> list[SearchResult]:
        """Ranks by an already-prepared piece of Japanese text rather than
        translating `query` first -- used for HyDE, where the text to embed
        is an LLM-generated hypothetical passage, not the user's query
        itself (see scripts/eval.py)."""
        query_emb = self.embedder.embed_texts([japanese_text])[0]
        results = self._rank(query_emb, top_k, w_img, w_text)
        for r in results:
            r.query_ja = japanese_text
        return results

    def translate_result_texts(self, results: list[SearchResult]) -> list[str | None]:
        """English translations of each result's Japanese dialogue, aligned
        with `results` (None where a result has no dialogue text)."""
        idx_with_text = [i for i, r in enumerate(results) if r.meta.get("text", "").strip()]
        if not idx_with_text:
            return [None] * len(results)
        translated = self.translator.ja_to_en([results[i].meta["text"] for i in idx_with_text])
        out: list[str | None] = [None] * len(results)
        for i, t in zip(idx_with_text, translated):
            out[i] = t
        return out
