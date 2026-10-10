"""Shared image/text embedding space using a Japanese CLIP model.

Manga109-s dialogue is Japanese, and manga line-art is far from the natural
photos most CLIP models are trained on, so we use a Japanese-native CLIP
checkpoint (line-corporation/clip-japanese-base) rather than a generic
English CLIP. Images and text are embedded into the same vector space, which
is what makes cross-modal (text query -> image results) search possible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

# Silences two informational-only warnings this model's remote code always
# triggers (T5Tokenizer "legacy behaviour", and "using a slow image
# processor") -- neither is actionable for us, just noise on every load.
hf_logging.set_verbosity_error()

MODEL_ID = "line-corporation/clip-japanese-base"


class ClipJapaneseEmbedder:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(self.device)
        self.model.eval()

    @staticmethod
    def _normalize(x: torch.Tensor) -> np.ndarray:
        x = x / x.norm(dim=-1, keepdim=True)
        return x.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[i : i + batch_size]]
            tokens = self.tokenizer(batch).to(self.device)
            feats = self.model.get_text_features(**tokens)
            out.append(self._normalize(feats))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)

    @torch.no_grad()
    def embed_images(self, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            out.append(self._normalize(feats))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self.embed_images_pil([image])[0]

    @torch.no_grad()
    def embed_images_pil(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return self._normalize(feats)
