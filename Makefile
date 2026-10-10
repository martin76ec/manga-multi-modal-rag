.PHONY: install index test ask

install:
	uv sync

index:
	uv run anime-rag index AisazuNihaIrarenai ARMS

test: index
	uv run python scripts/smoke_test.py

ask: index
	uv run anime-rag ask
