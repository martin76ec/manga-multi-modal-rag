.PHONY: install demo-index index-all test ask

BATCH_SIZE ?= 32

install:
	uv sync

# Small 2-book fixture used by `test` and `ask` -- NOT the full dataset.
# Already-indexed books are skipped, so re-running this is cheap/instant.
demo-index:
	uv run anime-rag index AisazuNihaIrarenai ARMS

# The real full-dataset indexing job (all 87 books). See scripts/h200_index_all.sh
# for the bigger-GPU version with pushing to HF built in.
index-all:
	uv run anime-rag index --all --batch-size $(BATCH_SIZE)

test: demo-index
	uv run python scripts/smoke_test.py

ask: demo-index
	uv run anime-rag ask
