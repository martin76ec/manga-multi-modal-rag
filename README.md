# anime-rag

Multimodal retrieval over the [Manga109-s](https://huggingface.co/datasets/hal-utokyo/Manga109-s) manga
dataset. Each manga panel is embedded twice — once as an image crop, once as its
transcribed dialogue text — into the same vector space using a Japanese CLIP model
([`line-corporation/clip-japanese-base`](https://huggingface.co/line-corporation/clip-japanese-base)).
A query (text or image) is scored against both embeddings, so retrieval can match on
what a panel looks like, what was said in it, or a blend of both.

## Setup

```bash
uv sync
hf auth login                 # needed once, for the gated dataset
```

Manga109-s is a gated dataset — request access on the
[dataset page](https://huggingface.co/datasets/hal-utokyo/Manga109-s) first.

## Usage

```bash
# list the 87 available books
uv run anime-rag books

# extract + embed a few books (fetches only those books from the dataset zip,
# via HTTP range reads — no need to download the full ~3GB archive)
uv run anime-rag index AisazuNihaIrarenai ARMS

# search in English — the query is auto-translated to Japanese before
# embedding (the CLIP model and the dialogue are both Japanese-native), and
# each result's dialogue is translated back to English for display
uv run anime-rag search "police officer"

# search in Japanese directly (skips translation, since it's detected)
uv run anime-rag search "警察官"

# search by image — finds visually similar panels
uv run anime-rag search --image path/to/some_panel.jpg

# tune the image/text score blend, save result crops for inspection
uv run anime-rag search "missing son" --w-img 0.4 --w-text 0.6 --save-crops out/

# skip translation entirely (faster startup if you're querying in Japanese)
uv run anime-rag search "警察官" --no-translate

# interactive REPL: loads models once, then take as many questions as you like.
# plain text/image queries do direct retrieval; prefix with 'agent:' to route
# that one question through the MRKL+IRCoT agent instead (see below)
uv run anime-rag ask

# agentic Q&A: a MRKL+IRCoT agent that issues its own retrieval queries over
# several rounds and synthesizes a cited answer (needs HF Inference access)
uv run anime-rag agent "What happens when a police officer shows up at someone's house?"
```

## How it works

- `source.py` opens the dataset's zip archive through `HfFileSystem` and extracts
  only the requested books' images/annotations (partial HTTP range reads).
- `annotations.py` parses each book's Manga109 XML into pages of frames
  (panels), text boxes, and character face/body boxes.
- `panels.py` crops one image per frame and associates it with the dialogue
  text and character names whose boxes fall inside that frame.
- `embedder.py` wraps the Japanese CLIP model to embed panel crops and text
  into a shared space.
- `translate.py` translates English queries to Japanese before embedding,
  and Japanese result dialogue back to English for display, via
  `facebook/nllb-200-distilled-600M` (local, no API key needed).
- `indexing.py` builds and persists a per-book index (embeddings + metadata)
  under `data/index/`.
- `search.py` ranks panels for a query by a weighted blend of
  image-embedding and text-embedding cosine similarity. Panels whose
  dialogue is empty or too short to embed meaningfully (e.g. "……") fall back
  to image-only scoring.
- `agent.py` implements a **MRKL + IRCoT** retrieval agent on top of
  `search.py`: a MRKL-style explicit `Thought -> Action -> Observation ->
  Final Answer` transcript (`llm.py`, via the HF Inference API — the only
  part of this project that's not local) drives a single `search_manga`
  tool, but IRCoT-style, each retrieval query is the agent's own latest
  reasoning step rather than a fixed query, so it can chase multi-hop
  questions across several retrieval rounds and still ground its answer in
  cited book/page evidence (or admit it doesn't know, instead of guessing).
  The LLM (`google/gemma-4-26B-A4B-it`) is vision-capable, so each
  Observation includes the actual panel images, not just translated
  dialogue — it can judge what's visually happening in a panel even when
  the dialogue doesn't describe it.

Downloaded data and built indices live under `data/` (gitignored) — the
dataset license forbids redistribution.

`anime-rag agent` calls the Hugging Face Inference API (model:
`google/gemma-4-26B-A4B-it` by default) and needs network access plus an HF
token with inference access (`hf auth login`); everything else in this
project (embedding, translation, indexing) runs fully locally.

## Indexing the full dataset (e.g. on a bigger GPU)

The dataset has 87 books (~73k panels total once all are indexed). Embedding
that locally on a small/shared GPU takes a while, so on a bigger box (this
was written with an H200 in mind):

```bash
# on the bigger machine: clone this repo, then
uv sync
hf auth login                      # needs Manga109-s access

# index every book in one go; bump --batch-size way up on a big GPU
uv run anime-rag index --all --batch-size 256

# or, as a single script (also pushes the result to HF when given a repo id):
./scripts/h200_index_all.sh yourname/manga109-rag-index
```

`--all` opens the dataset zip once and indexes everything it finds via
`anime-rag books`, skipping any book already indexed (safe to re-run /
resume after an interruption).

### Moving the index to another machine

`push-index`/`pull-index` upload/download **only the built index**
(embeddings + metadata) via a HF dataset repo — never the actual panel crop
images, which live in a separate directory (`data/raw/crops/`) the license
is stricter about. This is derived numeric output, analogous to publishing
a pretrained model, not a copy of the dataset itself — see the license note
below. Push it **private**.

The index is consolidated into a single `index.parquet` file before
uploading — Parquet is the standard single-file format for embeddings +
metadata (what HF's own `datasets` library uses), and far easier to move
around than one directory per book (87 books = 348 small files otherwise).
`pull-index` downloads that one file and transparently re-splits it back
into the usual `data/index/<book>/` layout, so nothing else changes.

```bash
# on the box that built the index
uv run anime-rag push-index yourname/manga109-rag-index

# if you previously pushed with an older version of this tool that uploaded
# the raw per-book folders instead of a single file, add --clean once to
# remove that old structure from the repo after the parquet file is up
uv run anime-rag push-index yourname/manga109-rag-index --clean

# on another machine
uv run anime-rag pull-index yourname/manga109-rag-index

# the pulled index has no crop images yet -- regenerate them locally
# (re-extracts raw pages from the dataset zip, crops them; no GPU/model
# inference involved, so it's cheap even without the bigger box)
uv run anime-rag crops
```

Search/citations work immediately after `pull-index` (it's pure embedding
math); you only need `anime-rag crops` once you want to actually view a
result's panel image or use the vision-capable agent.

## License note

Manga109-s is distributed under its own license (non-commercial-friendly but
not open), which caps how much of any one book's pages may be published and
forbids redistributing the dataset itself. See the dataset page for details.
