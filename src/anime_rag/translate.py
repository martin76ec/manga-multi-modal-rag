"""English <-> Japanese translation.

The CLIP model and the manga dialogue are both Japanese-native: an English
query fed straight to the Japanese tokenizer doesn't embed meaningfully, and
the retrieved dialogue is Japanese. This module bridges both directions so
the tool is usable without knowing Japanese -- queries get translated to
Japanese before embedding, and result text gets translated to English for
display.

Uses NLLB-200 (distilled 600M): tried the smaller Helsinki-NLP/opus-mt-en-jap
and staka/fugumt-en-ja Marian models first, but both produced incoherent or
repetition-looping output even on simple single-word queries. NLLB is larger
but reliably correct (e.g. "police officer" -> "警察官").
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase

MODEL_ID = "facebook/nllb-200-distilled-600M"
EN_CODE = "eng_Latn"
JA_CODE = "jpn_Jpan"

_JAPANESE_RE = re.compile(r"[぀-ヿ一-鿿]")


def contains_japanese(text: str) -> bool:
    return bool(_JAPANESE_RE.search(text))


class Translator:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: AutoModelForSeq2SeqLM | None = None
        self._tok_en: PreTrainedTokenizerBase | None = None
        self._tok_ja: PreTrainedTokenizerBase | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(self.device)
        self._model.eval()
        self._tok_en = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=EN_CODE)
        self._tok_ja = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=JA_CODE)

    @torch.no_grad()
    def _translate(self, texts: list[str], src_tok: PreTrainedTokenizerBase, target_code: str) -> list[str]:
        if not texts:
            return []
        self._ensure_loaded()
        safe_texts = [t if t.strip() else "." for t in texts]
        batch = src_tok(safe_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self._model.generate(
            **batch,
            forced_bos_token_id=src_tok.convert_tokens_to_ids(target_code),
            max_new_tokens=64,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )
        return src_tok.batch_decode(out, skip_special_tokens=True)

    def en_to_ja(self, texts: list[str]) -> list[str]:
        self._ensure_loaded()
        return self._translate(texts, self._tok_en, JA_CODE)

    def ja_to_en(self, texts: list[str]) -> list[str]:
        self._ensure_loaded()
        return self._translate(texts, self._tok_ja, EN_CODE)
