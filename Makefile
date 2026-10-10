.PHONY: install demo-index index-all test ask

BATCH_SIZE ?= 32

install:
	uv sync

# Fixture used by `test` and `ask` to guarantee *some* books are indexed.
# Skips entirely if data/index already has anything (e.g. the full
# dataset) -- otherwise falls back to indexing 2 small books, enough for
# scripts/smoke_test.py's known-answer check.
demo-index:
	@if [ -d data/index ] && [ -n "$$(ls -A data/index 2>/dev/null)" ]; then \
		echo "demo-index: data/index already has books, skipping"; \
	else \
		uv run anime-rag index AisazuNihaIrarenai ARMS; \
	fi

# The real full-dataset indexing job (all 87 books). See scripts/h200_index_all.sh
# for the bigger-GPU version with pushing to HF built in.
index-all:
	uv run anime-rag index --all --batch-size $(BATCH_SIZE)

test: demo-index
	uv run python scripts/smoke_test.py

ask: demo-index
	uv run anime-rag ask
