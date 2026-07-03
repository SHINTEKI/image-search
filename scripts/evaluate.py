"""Lightweight, reproducible evaluation for the semantic search index.

The sample images (Picsum) have no human relevance labels, so instead of inventing
subjective query→image ground truth we run two objective checks that need no labels:

  1. Retrieval sanity — every image must be its own nearest neighbour (self-cosine
     ≈ 1.0). Catches broken normalization / index corruption.

  2. Label round-trip (Recall@K) — we build a small vocabulary, use zero-shot CLIP
     to assign each image its best-matching label, then issue that label as a text
     query and check whether the image comes back in the top-K. This measures whether
     text→image retrieval actually recovers the concept CLIP itself sees in the image.
     It is a *self-consistency* metric, not a benchmark against external ground truth.

Plus a human-readable spot check: a few natural-language queries and their top hits.

Usage:  uv run python scripts/evaluate.py
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from app.model import get_model  # noqa: E402
from app.search import SearchIndex  # noqa: E402

# A generic vocabulary spanning common photo subjects. Zero-shot CLIP picks the
# best label per image; we then retrieve by that label.
VOCAB = [
    "a building", "a city street", "nature", "a forest", "a mountain",
    "the ocean", "a beach", "the sky", "a person", "an animal",
    "a flower", "food", "a car", "technology", "a black and white photo",
    "water", "a bridge", "a road", "trees", "a landscape",
]

SPOT_CHECK_QUERIES = [
    "green nature landscape",
    "black and white photo",
    "a building",
    "water and sky",
]


def evaluate() -> int:
    idx = SearchIndex()
    model = get_model()
    n = idx.count
    print(f"Evaluating index: {n} images ({idx.model_name})\n")

    # --- Check 1: retrieval sanity (self-cosine ≈ 1.0) ---
    self_sims = np.array([float(idx.matrix[i] @ idx.matrix[i]) for i in range(n)])
    print("1) Retrieval sanity")
    print(f"   self-cosine  min={self_sims.min():.4f}  mean={self_sims.mean():.4f} "
          f"(expected ≈ 1.0)")
    assert self_sims.min() > 0.99, "embeddings are not unit-normalized!"
    print("   ✓ all embeddings unit-normalized\n")

    # --- Check 2: label round-trip Recall@K ---
    label_vecs = model.encode_text(VOCAB)             # (V, D)
    sims = idx.matrix @ label_vecs.T                  # (N, V)
    best_label = sims.argmax(axis=1)                  # each image's top label

    for k in (1, 5, 10):
        hits = 0
        for img_id in range(n):
            label = VOCAB[best_label[img_id]]
            results = idx.search_text(label, top_k=k)
            if any(r["id"] == img_id for r in results):
                hits += 1
        print(f"2) Label round-trip Recall@{k:<2} = {hits}/{n} = {hits / n:.1%}")
    print()

    # --- Spot check: human-readable top hits ---
    print("3) Spot-check queries (top-3):")
    for q in SPOT_CHECK_QUERIES:
        top = idx.search_text(q, top_k=3)
        hits = ", ".join(f"{r['filename']}({r['score']})" for r in top)
        print(f"   {q!r:28} → {hits}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate())
