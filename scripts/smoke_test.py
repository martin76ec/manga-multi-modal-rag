"""Smoke test for the indexed RAG pipeline: checks that an index exists,
that image self-similarity is near-perfect (catches embedding/normalization
regressions), and that an English query gets translated and returns results.

Run via `make test` (indexes a couple of demo books first if needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anime_rag.embedder import ClipJapaneseEmbedder
from anime_rag.indexing import Index
from anime_rag.search import Searcher
from anime_rag.translate import contains_japanese

INDEX_DIR = Path("data/index")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def main() -> None:
    if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
        print(f"No index found under {INDEX_DIR}. Run `make index` first.")
        sys.exit(1)

    index = Index.load_all(INDEX_DIR)
    check("index loads with panels", len(index) > 0, f"{len(index)} panels")

    embedder = ClipJapaneseEmbedder()
    searcher = Searcher(index, embedder)

    # Image self-match: searching with a panel's own crop should return that
    # same panel as the #1 hit with near-1.0 similarity.
    probe_id = index.meta[0]["id"]
    probe_path = Path(index.meta[0]["image_path"])
    results = searcher.search_image(probe_path, top_k=1)
    check(
        "image self-match returns itself",
        bool(results) and results[0].meta["id"] == probe_id,
        f"got {results[0].meta['id'] if results else None!r}, expected {probe_id!r}",
    )
    check(
        "image self-match score is near 1.0",
        bool(results) and results[0].image_score > 0.99,
        f"score={results[0].image_score if results else None}",
    )

    # English query: should get translated to Japanese before embedding,
    # and return non-empty, sanely-scored results.
    results = searcher.search_text("police officer", top_k=5)
    check("english query returns results", len(results) > 0)
    check(
        "english query was translated to Japanese",
        bool(results) and bool(results[0].query_ja) and contains_japanese(results[0].query_ja),
        f"query_ja={results[0].query_ja if results else None!r}",
    )
    check(
        "top result score is in a sane range",
        bool(results) and 0.0 < results[0].score <= 1.0,
        f"score={results[0].score if results else None}",
    )

    # Known-answer check, only when the specific demo book is indexed.
    books_indexed = {m["book"] for m in index.meta}
    if "AisazuNihaIrarenai" in books_indexed:
        expected_id = "AisazuNihaIrarenai_057_000015bd"
        results = searcher.search_text("police officer", top_k=1)
        check(
            'known query "police officer" finds the expected panel',
            bool(results) and results[0].meta["id"] == expected_id,
            f"got {results[0].meta['id'] if results else None!r}, expected {expected_id!r}",
        )

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
