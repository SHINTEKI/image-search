"""CLIP model wrapper.

Loads an open_clip CLIP model once (lazy singleton) and exposes two methods that
project images and text into the *same* L2-normalized embedding space, so cosine
similarity between a text query and an image tells us how well they match.
"""
from __future__ import annotations

import functools
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from . import config


class ClipModel:
    """Thin wrapper around an open_clip model + preprocessing pipeline."""

    def __init__(self) -> None:
        import open_clip  # imported lazily so the CLI/help is fast

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.MODEL_NAME, pretrained=config.PRETRAINED
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)

    @property
    def dim(self) -> int:
        return self.model.text_projection.shape[1]

    @torch.no_grad()
    def encode_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        """Embed PIL images -> (N, D) float32 array, L2-normalized rows."""
        batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        feats = self.model.encode_image(batch)
        return _normalize(feats)

    @torch.no_grad()
    def encode_text(self, queries: list[str]) -> np.ndarray:
        """Embed text queries -> (N, D) float32 array, L2-normalized rows."""
        tokens = self.tokenizer(queries).to(self.device)
        feats = self.model.encode_text(tokens)
        return _normalize(feats)


def _normalize(feats: torch.Tensor) -> np.ndarray:
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


@functools.lru_cache(maxsize=1)
def get_model() -> ClipModel:
    """Return the process-wide singleton CLIP model, loading it on first use."""
    return ClipModel()
