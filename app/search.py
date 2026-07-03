"""In-memory search over the cached CLIP embeddings.

The index (embeddings matrix + metadata) is loaded once. Because all rows are
L2-normalized, cosine similarity is a single matrix multiply: `query @ matrix.T`.
At folder scale this is exact and sub-millisecond — no vector DB needed. See the
README for the scaling path (FAISS/HNSW) past ~100k images.
"""
from __future__ import annotations

import json

import numpy as np

from . import config


class SearchIndex:
    def __init__(self) -> None:
        if not config.EMBEDDINGS_PATH.exists() or not config.META_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Build it first: `uv run python -m app.index`"
            )
        self.matrix: np.ndarray = np.load(config.EMBEDDINGS_PATH)
        meta = json.loads(config.META_PATH.read_text())
        self.images: list[dict] = meta["images"]
        self.model_name: str = meta.get("model", config.MODEL_NAME)

    @property
    def count(self) -> int:
        return len(self.images)

    def _rank(self, query_vec: np.ndarray, top_k: int, exclude: int | None = None) -> list[dict]:
        # query_vec: (D,) normalized. Cosine sim vs every image = one matmul.
        scores = self.matrix @ query_vec
        order = np.argsort(-scores)
        results = []
        for idx in order:
            if exclude is not None and int(idx) == exclude:
                continue
            item = dict(self.images[int(idx)])
            item["score"] = round(float(scores[int(idx)]), 4)
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    def search_text(self, query: str, top_k: int = 24) -> list[dict]:
        from .model import get_model

        vec = get_model().encode_text([query])[0]
        return self._rank(vec, top_k)

    def search_by_id(self, image_id: int, top_k: int = 24) -> list[dict]:
        """Find images similar to an already-indexed image (click-to-similar)."""
        if not 0 <= image_id < self.count:
            raise IndexError(f"image_id {image_id} out of range [0, {self.count})")
        vec = self.matrix[image_id]
        return self._rank(vec, top_k, exclude=image_id)

    def search_by_vector(self, vec: np.ndarray, top_k: int = 24) -> list[dict]:
        """Find images similar to an arbitrary embedding (uploaded image)."""
        return self._rank(vec, top_k)

    def search_keyword(self, query: str, top_k: int = 24) -> list[dict]:
        """Plain filename substring match — a non-semantic baseline for contrast."""
        q = query.lower().strip()
        results = []
        for item in self.images:
            if q and q in item["filename"].lower():
                out = dict(item)
                out["score"] = 1.0
                results.append(out)
        return results[:top_k]
