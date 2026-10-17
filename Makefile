.PHONY: install demo-index index-all test ask eval

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

# RAGAS-style hit-rate/MRR + LLM-judge faithfulness/relevancy, swept over
# retrieval blend, agent max_steps/top_k, and the HyDE/IRCoT ablations.
# Runs against whatever is already indexed -- no re-indexing, no ragas
# package (see scripts/eval.py). The eval cases now span ~15 different
# books (not just a couple), so this needs the full `index-all` dataset,
# not the demo-index 2-book fixture -- no auto-index dependency here since
# that's a much bigger prerequisite than `test`/`ask` need.
eval:
	uv run python scripts/eval.py
