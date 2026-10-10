#!/usr/bin/env bash
# Index the full Manga109-s dataset (all 87 books) on a beefy GPU box, then
# optionally push the resulting index (embeddings + metadata only, no
# images) to a private HF dataset repo so it can be pulled back onto a
# laptop without recomputing embeddings there.
#
# Usage:
#   ./scripts/h200_index_all.sh                          # index only
#   ./scripts/h200_index_all.sh yourname/manga109-rag-index   # index + push
#
# Expects: this repo checked out, `uv` installed, and `hf auth login` already
# done with an account that has Manga109-s access.

set -euo pipefail
cd "$(dirname "$0")/.."

PUSH_REPO="${1:-}"
BATCH_SIZE="${BATCH_SIZE:-256}"

echo "== uv sync =="
uv sync

echo "== indexing all books (batch size: $BATCH_SIZE) =="
uv run anime-rag index --all --batch-size "$BATCH_SIZE"

if [ -n "$PUSH_REPO" ]; then
    echo "== pushing index to https://huggingface.co/datasets/$PUSH_REPO (private) =="
    uv run anime-rag push-index "$PUSH_REPO"
else
    echo "== done. To push the index to HF later, run: =="
    echo "   uv run anime-rag push-index <yourname>/manga109-rag-index"
fi
