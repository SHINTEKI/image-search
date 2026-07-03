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

# CLIP was trained on caption-style text ("a photo of a dog"), not bare words
# ("dog"), so wrapping the query in prompt templates and averaging the resulting
# embeddings (prompt ensembling, from the CLIP paper) makes queries land better
# in the model's distribution. The plain "{}" template keeps full-sentence
# queries from being over-wrapped.
PROMPT_TEMPLATES = (
    "a photo of {}.",
    "a photo of a {}.",
    "a close-up photo of {}.",
    "a photo containing {}.",
    "{}",
)


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

        # Top-k via argpartition (O(N)) instead of a full O(N log N) sort: we only
        # need the best `k` candidates, then sort just those. Grab one extra so an
        # excluded id can't shrink the result below top_k.
        k = min(top_k + (1 if exclude is not None else 0), scores.shape[0])
        candidates = np.argpartition(-scores, k - 1)[:k]
        order = candidates[np.argsort(-scores[candidates])]

        results = []
        for idx in order:
            i = int(idx)
            if exclude is not None and i == exclude:
                continue
            item = dict(self.images[i])
            item["score"] = round(float(scores[i]), 4)
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a text query with prompt ensembling -> one L2-normalized vector."""
        from .model import get_model

        prompts = [t.format(query) for t in PROMPT_TEMPLATES]
        vecs = get_model().encode_text(prompts)  # (T, D), each row normalized
        vec = vecs.mean(axis=0)                  # ensemble = mean of prompt embeddings
        return vec / np.linalg.norm(vec)         # renormalize so dot == cosine

    def search_text(self, query: str, top_k: int = 24) -> list[dict]:
        return self._rank(self._embed_query(query), top_k)

    def search_by_id(self, image_id: int, top_k: int = 24) -> list[dict]:
        """Find images similar to an already-indexed image (click-to-similar)."""
        if not 0 <= image_id < self.count:
            raise IndexError(f"image_id {image_id} out of range [0, {self.count})")
        vec = self.matrix[image_id]
        return self._rank(vec, top_k, exclude=image_id)

    def search_by_vector(self, vec: np.ndarray, top_k: int = 24) -> list[dict]:
        """Find images similar to an arbitrary embedding (uploaded image)."""
        return self._rank(vec, top_k)
