"""Chat completion via the Hugging Face Inference API (hosted, not local).

Unlike the embedder and translator, which run locally, this calls out to a
model hosted by an HF Inference Provider -- no local download, but it needs
network access and an HF token with inference access.
"""

from __future__ import annotations

from collections import Counter

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"

# "auto" provider selection picks whichever provider is first in the user's
# HF settings, which hit a 429 (rate limit) on the free tier during testing.
# Falling back through a few known-live providers for this model makes
# single requests more resilient without needing a paid HF plan.
PROVIDER_FALLBACKS = ["featherless-ai", "novita", "scaleway"]


def is_degenerate(text: str, min_words: int = 30, max_top_word_frac: float = 0.4) -> bool:
    """Detects pathological repetition-loop output (observed from one
    provider occasionally), not a quality judgment on normal text. Two
    distinct patterns seen in practice:
    - the exact same word repeated dozens of times (e.g. "thought" x70)
    - a malformed tool-call template repeated with a *varying* argument
      each time (e.g. 'call:search_manga{query: "..."}thought' x10) --
      word-frequency alone misses this since punctuation glued to the
      varying text makes each "word" technically unique, so this also
      checks raw occurrences of "thought", which our prompt format expects
      at most once per well-formed reply.
    """
    if text.lower().count("thought") > 2:
        return True

    words = text.split()
    if len(words) < min_words:
        return False
    _, top_count = Counter(words).most_common(1)[0]
    return top_count / len(words) > max_top_word_frac


class ChatLLM:
    def __init__(self, model: str = DEFAULT_MODEL, providers: list[str] | None = None):
        self.model = model
        self.providers = providers or PROVIDER_FALLBACKS

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 300,
        stop: list[str] | None = None,
        frequency_penalty: float = 0.4,
    ) -> str:
        last_error: Exception | None = None
        # Each provider gets 2 tries: observed one provider occasionally
        # return finish_reason="tool_calls" with an empty `content` (and a
        # garbled `reasoning` field), or a degenerate repetition loop, for
        # no apparent reason -- a transient hosted-inference glitch, not
        # something retrying the same prompt against the same provider
        # failed to recover from in testing. frequency_penalty also
        # directly discourages the repetition loops at the source.
        for provider in self.providers:
            for _attempt in range(2):
                try:
                    client = InferenceClient(provider=provider)
                    resp = client.chat_completion(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        stop=stop,
                        frequency_penalty=frequency_penalty,
                    )
                    content = resp.choices[0].message.content
                    if not content:
                        last_error = RuntimeError(
                            f"{provider} returned empty content (finish_reason={resp.choices[0].finish_reason!r})"
                        )
                        continue
                    if is_degenerate(content):
                        last_error = RuntimeError(f"{provider} returned degenerate/repetitive content")
                        continue
                    return content
                except HfHubHTTPError as e:
                    last_error = e
                    continue
        raise RuntimeError(f"all providers failed for {self.model}: {last_error}") from last_error
