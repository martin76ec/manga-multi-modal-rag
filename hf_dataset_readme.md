---
license: other
---

# Manga109-s Panel Embeddings

Multimodal (image + text) embeddings and metadata for panels from the
[Manga109-s](https://huggingface.co/datasets/hal-utokyo/Manga109-s) dataset,
computed with [`line-corporation/clip-japanese-base`](https://huggingface.co/line-corporation/clip-japanese-base).

**Contains:** `index.parquet` — one row per panel: `id`, `book`, `page_index`,
`bbox`, `text` (dialogue), `characters`, `image_emb`, `text_emb` (512-d each).
**Does not contain:** any manga images. Manga109-s forbids redistributing the
dataset itself; this is a derived numeric artifact only, produced with a
tool at [martin76ec/manga-multi-modal-rag](https://github.com/martin76ec/manga-multi-modal-rag).

## Credit

Derived from **Manga109-s**, courtesy of the original manga authors and
[hal-utokyo](https://huggingface.co/datasets/hal-utokyo/Manga109-s). Please
cite:

```bibtex
@inproceedings{baek2026mangav26,
  title     = {{Manga109-v2026: Revisiting Manga109 Annotations for Modern Manga Understanding}},
  author    = {Baek, Jeonghun and Miyai, Atsuyuki and Onohara, Shota and Ikuta, Hikaru and Aizawa, Kiyoharu},
  booktitle = {Culture x AI Workshop at ICML 2026},
  year      = {2026},
}
```

Full license, usage terms, and additional citations:
https://huggingface.co/datasets/hal-utokyo/Manga109-s
