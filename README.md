# 🔍 Image Semantic Search

Natural-language semantic search over a folder of images, powered by **CLIP**.
Type what you're looking for — *"a dog on the beach"*, *"city at night"*, *"black and white photo"* —
and get the most relevant images back in milliseconds, through a clean, beautiful UI.

![Made with CLIP](https://img.shields.io/badge/model-CLIP%20ViT--B--32-6366f1)
![Backend](https://img.shields.io/badge/backend-FastAPI-009688)
![No API key](https://img.shields.io/badge/cost-%240%20%2F%20offline-22c55e)

> ## 🎥 [**▶ Watch the demo video**](https://drive.google.com/file/d/1F6mjKEvNoCLblKqF9TNx_sXfT7dBPFJw/view?usp=sharing)
> A short walkthrough of text search, image search, and find-similar.

---

## Why this design

The brief asks for **natural-language search** where **image serving is low-latency and cheap**, with a
**simple, beautiful UI**. The design maps directly onto those goals:

| Requirement | How it's met |
| --- | --- |
| Natural-language search | **CLIP** embeds images and text into the *same* vector space; cosine similarity ranks images by how well they match the query. |
| Low latency | Embeddings are **precomputed offline**; the online path is a single in-memory matrix multiply (**exact cosine, sub-millisecond** at folder scale). The model is warmed up at startup so even the first query is fast. |
| Cheap | **Local model, no API key, runs offline** — $0 per search. Images are served as **pre-generated thumbnails** (tiny, cache-friendly transfers); originals load only on demand. |
| Simple & beautiful UI | A single **dark-theme** page (Tailwind, no build step): tabbed search, responsive grid, similarity badges, click-to-find-similar, upload-to-search, lightbox. |
| Multiple search methods | **Two complementary modes**: text→image and image→image (upload or click-Similar). |

---

## Flow diagram

```
                    ┌───────────────────────── BUILD (offline, run once) ─────────────────────────┐
                    │                                                                             │
   images/  ──────► │   for each image:  CLIP image encoder ──► 512-d vector (L2-normalized)      │
   (a folder)       │                    generate thumbnail  ──► .cache/thumbs/                    │
                    │                                                                             │
                    │        stack vectors ──► .cache/embeddings.npy   +   meta.json               │
                    └─────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼   (loaded + warmed up at server startup)
   ┌──────────────────────────────────── SEARCH (online, milliseconds) ─────────────────────────────┐
   │                                                                                                │
   │  Browser UI ──query──► FastAPI ──► CLIP text encoder (prompt ensemble) ──► query vector         │
   │                                                              │                                 │
   │                                     cosine = query · matrixᵀ  (one matmul, all vectors in RAM)  │
   │                                                              │                                 │
   │                                    top-K via argpartition ───┴──► thumbnails + scores ──► grid  │
   │                                                                                                │
   │  Serving:  /thumb/{name}  small cached JPEG (fast, cheap)   ·   /image/{id}  original on demand │
   └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Text → image** runs the query through the CLIP text encoder. **Image → image** (upload, or click
*Similar* on a result) uses an image vector directly — cached ones are instant; an uploaded image is
embedded on the fly.

---

## Search methods (in the UI)

The UI has two tabs, plus a follow-up action available on every result:

1. **Search by text** — type natural language; ranked by CLIP cosine similarity.
2. **Search by image** — upload a photo; it's embedded with CLIP and matched against the index.
3. **Similar** (on any result) — click *Similar* on a card to search using **that image's own vector** —
   a fast way to keep exploring related pictures (no re-embedding needed; it reuses the cached vector).

---

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.13.

```bash
# 1. Install dependencies (CPU PyTorch by default — no GPU needed)
uv sync

# 2. (optional) grab 100 sample images, or drop your own into ./images
uv run python scripts/fetch_samples.py

# 3. Build the search index (embeds images + makes thumbnails; one-off, ~25s for 100 imgs on CPU)
uv run python -m app.index            # or: uv run python -m app.index /path/to/your/images

# 4. Run the app
uv run uvicorn app.server:app --port 8077
#    open http://127.0.0.1:8077
```

**To search your own folder:** point step 3 at it — `uv run python -m app.index ~/Pictures` — then
copy/symlink that folder to `./images` (or edit `IMAGES_DIR` in `app/config.py`) and restart the server.

---

## Project structure

```
app/
  config.py    # paths, model name, thumbnail size
  model.py     # CLIP wrapper — lazy singleton, L2-normalized encode_image / encode_text
  index.py     # CLI: scan folder → embed → thumbnails → cache (.npy + meta.json)
  search.py    # load cache once; prompt-ensembled text search / similar / by-vector
  server.py    # FastAPI: search API + thumbnail/image serving + serves the UI
web/
  index.html   # single-page dark-theme UI (Tailwind CDN, vanilla JS, no build)
images/        # your images (100 public samples included)
scripts/
  fetch_samples.py   # downloads the sample images
  evaluate.py        # label-free sanity + self-consistency evaluation
.cache/        # embeddings.npy + meta.json + thumbs/  (generated; gitignored)
```

---

## Core questions (answered)

**Q: How does the natural-language search actually work?**
CLIP was trained so that an image and its describing text land close together in a shared embedding space.
We embed every image once, embed the query text at search time, and rank images by cosine similarity to
the query. No labels, tags, or captions required.

**Q: What is "prompt ensembling", and why use it?**
CLIP is trained on caption-style text (*"a photo of a dog"*), not bare words (*"dog"*), so bare or short
queries land slightly out of distribution. We wrap the query in several templates
(*"a photo of {}"*, *"a close-up photo of {}"*, …), encode each, and **average the vectors** (a technique
from the CLIP paper). This nudges the query toward the model's distribution and makes results more robust —
e.g. `"building"` and `"a building"` return the same images. A plain `"{}"` template is kept so full
sentences aren't over-wrapped. Cost is negligible (a handful of text encodes, still sub-second).

**Q: Why 512 dimensions — could it be more precise?**
512 isn't a tuning knob we picked; it's the **output dimension of CLIP ViT-B-32** (fixed by the
architecture). "More precise" doesn't mean *more dimensions* — it means a **stronger model**, whose
dimension comes along for the ride: e.g. `ViT-L-14` outputs 768-d and is more accurate, at the cost of a
larger download and slower CPU inference. Swap `MODEL_NAME`/`PRETRAINED` in `app/config.py`; nothing else
changes. For a folder-scale, zero-setup demo, 512-d ViT-B-32 is the sweet spot.

**Q: Why is image serving low-latency and cheap?**
Two moves. (1) **All the expensive work is offline** — image embeddings and thumbnails are computed once at
index time, so the online request never touches the image encoder. Search itself is one matrix multiply
over vectors already in RAM (sub-ms), and top-K uses `argpartition` (O(N)) instead of a full sort. (2)
**Thumbnails, not originals** — the grid loads small cache-friendly JPEGs (`Cache-Control` set), so
transfers are tiny and bandwidth cost is minimal; originals are fetched only when a user opens one.

**Q: Why no vector database, and when would I switch — concretely?**
At folder scale an exact brute-force cosine (`query @ matrixᵀ`) is faster and simpler than a DB, whose
network/serialization overhead dominates at this size. Concrete thresholds (512-d float32 vectors):

| Images | Matrix size | Brute-force search | Recommendation |
| --- | --- | --- | --- |
| 100 – 10k | ≤ 20 MB | < 1 ms | **In-memory matmul** (this repo) |
| 10k – ~1M | 20 MB – 2 GB | ~1 – 100 ms | Still fine in RAM; add an **ANN index (FAISS / HNSWlib)** if latency matters |
| > ~1M (won't fit in RAM) | > 2 GB | too slow / too big | **Vector DB (Qdrant / pgvector)** for ANN + persistence + horizontal scale |

The retrieval layer is intentionally isolated in `app/search.py`, so swapping the backend leaves the
**embedding model and the HTTP API unchanged**.

**Q: Do you rerank results?**
No — this is single-stage retrieval (embed → cosine → sort), which is the right call at this scale where
recall is trivially complete. A two-stage **retrieve-then-rerank** pipeline (recall top-N cheaply, then
re-score with a heavier cross-modal model such as BLIP-2) is the natural next step for large corpora or
harder queries; it's noted here as a future direction rather than implemented, to avoid over-engineering.
Our prompt ensembling already improves ranking on the query side at near-zero cost.

**Q: What are the limitations?**
CLIP reflects its training distribution: strong on common scenes/objects, weaker on fine-grained detail,
text-in-images (OCR), or niche domains. Similarity scores are relative rankings, not calibrated
probabilities.

---

## Evaluation

The sample images have **no human relevance labels**, so rather than invent subjective query→image ground
truth, `scripts/evaluate.py` runs two **label-free, reproducible** checks (`uv run python scripts/evaluate.py`):

1. **Retrieval sanity** — every image must be its own nearest neighbour (self-cosine ≈ 1.0). This catches
   broken normalization or index corruption. *(Result: min = 1.000 ✓.)*
2. **Label round-trip Recall@K** — zero-shot CLIP assigns each image its best label from a small vocabulary,
   then we query by that label and check whether the image returns in the top-K. It measures whether
   text→image retrieval recovers the concept CLIP itself sees in the image — a **self-consistency** metric.
   *(Result on 100 samples: Recall@1 14%, @5 48%, @10 65%.)*

   > **Reading the numbers honestly:** Recall@1 looks low because the vocabulary overlaps heavily
   > (*forest / trees / nature / landscape* all describe similar green images), so same-concept images
   > compete for the top slot — it under-counts, it doesn't mean retrieval is broken. Recall@10 rising to
   > 65% shows the concept is reliably recovered within the first page. Plus a human-readable spot check of
   > natural-language queries and their top hits.

**How I'd evaluate this properly in production:** collect real query→click data and measure
**Precision@K / mAP / MRR** against human-judged relevance; run **A/B tests** on ranking changes (e.g.
prompt templates, a reranker); and track **CTR / zero-result rate** as live quality signals. A labelled
eval set (a few hundred query→relevant-image pairs) would replace the self-consistency proxy above.

---

## Tech stack

- **Model:** [open_clip](https://github.com/mlfoundations/open_clip) — CLIP `ViT-B-32` (`laion2b_s34b_b79k`)
- **Backend:** FastAPI + Uvicorn
- **Vectors:** NumPy (in-memory exact cosine, `argpartition` top-K)
- **Frontend:** single HTML page, Tailwind (CDN), vanilla JS — no build step
- **Deps:** managed by `uv`
