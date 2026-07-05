# Image Semantic Search

Search a folder of images by describing what you want — *"a dog on the beach"*,
*"city at night"*, *"black and white photo"* — and get the closest matches back
in milliseconds. Runs a local CLIP model, no API key, no network calls at search time.

![model](https://img.shields.io/badge/model-CLIP%20ViT--B--32-6366f1)
![backend](https://img.shields.io/badge/backend-FastAPI-009688)
![cost](https://img.shields.io/badge/cost-%240%20%2F%20offline-22c55e)

> ## 🎥 [**▶ Watch the demo video**](https://drive.google.com/file/d/1F6mjKEvNoCLblKqF9TNx_sXfT7dBPFJw/view?usp=sharing)
> A short walkthrough of text search, image search, and find-similar.

## How it works

CLIP embeds images and text into a shared vector space, so a text query and the
images it describes end up close together. Image embeddings are computed once,
offline, and cached. At search time only the query is embedded; ranking is a
single cosine-similarity matrix multiply over the cached vectors held in memory.
The model is loaded at startup rather than on first use, so the first query is
fast instead of paying a one-time load stall.

```
                    ┌───────────────── BUILD (offline, run once) ─────────────────┐
                    │                                                             │
   images/  ──────► │   each image:  CLIP image encoder ──► 512-d vector           │
   (a folder)       │                make thumbnail      ──► .cache/thumbs/        │
                    │                                                             │
                    │        stack vectors ──► .cache/embeddings.npy + meta.json   │
                    └─────────────────────────────────────────────────────────────┘
                                              │
                                              ▼   (loaded + warmed up at startup)
   ┌──────────────────────── SEARCH (online, milliseconds) ───────────────────────┐
   │                                                                              │
   │  UI ──query──► FastAPI ──► CLIP text encoder ──► query vector                 │
   │                                     │                                        │
   │                cosine = query · matrixᵀ   (one matmul, vectors in RAM)        │
   │                                     │                                        │
   │                top-K ───────────────┴──► thumbnails + scores ──► grid         │
   │                                                                              │
   │  /thumb/{name}  small cached JPEG    ·    /image/{id}  original on demand     │
   └──────────────────────────────────────────────────────────────────────────────┘
```

Image serving stays cheap because the grid loads pre-generated thumbnails (small,
cache-friendly JPEGs); originals are fetched only when you open one. Both media
routes check that the resolved path stays inside their directory, so a crafted
name (e.g. `../../etc/passwd`) can't escape it.

## Search modes

- **Search by text** — type natural language; ranked by CLIP cosine similarity.
- **Search by image** — upload a photo; it's embedded and matched against the index.
- **Similar** — click *Similar* on any result to search by that image's own cached
  vector (no re-embedding).

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.13.

```bash
uv sync                                   # install deps (CPU PyTorch, no GPU needed)
uv run python scripts/fetch_samples.py    # optional: grab 100 sample images
uv run python -m app.index                # build the index (~30s for 100 imgs on CPU)
uv run uvicorn app.server:app --port 8077 # then open http://127.0.0.1:8077
```

To index your own folder: `uv run python -m app.index /path/to/images`, then point
`IMAGES_DIR` in `app/config.py` at it (or symlink it to `./images`) and restart.

## Project layout

```
app/
  config.py    paths, model name, thumbnail size
  model.py     CLIP wrapper — lazy singleton, L2-normalized encode_image / encode_text
  index.py     CLI: scan folder → embed → thumbnails → cache (.npy + meta.json)
  search.py    load cache once; text search / similar / by-vector
  server.py    FastAPI: search API + thumbnail/image serving + serves the UI
web/
  index.html   single-page UI (Tailwind CDN, vanilla JS, no build step)
scripts/
  fetch_samples.py   downloads the sample images
.cache/        embeddings.npy + meta.json + thumbs/  (generated, gitignored)
```

## Notes

**Prompt ensembling.** CLIP is trained on caption-style text (*"a photo of a dog"*),
so bare queries land slightly out of distribution. Each query is wrapped in a few
templates, encoded, and averaged before ranking, which makes short queries more
robust (e.g. `building` and `a building` return the same images).

**No vector database.** For a folder of images an exact in-memory matmul is faster
and simpler than a DB. The retrieval layer is isolated in `app/search.py`, so it can
be swapped for an ANN index (FAISS / HNSWlib) or a vector DB (Qdrant / pgvector) past
~1M vectors without touching the model or the HTTP API:

| Images | Matrix (512-d f32) | Search | Backend |
| --- | --- | --- | --- |
| 100 – 10k | ≤ 20 MB | < 1 ms | in-memory matmul (this repo) |
| 10k – ~1M | 20 MB – 2 GB | ~1 – 100 ms | in-memory, add ANN index if needed |
| > ~1M | > 2 GB | — | vector DB for ANN + persistence |

**Model.** `ViT-B-32` (`laion2b_s34b_b79k`) via [open_clip](https://github.com/mlfoundations/open_clip),
512-d output. A stronger model (e.g. `ViT-L-14`, 768-d) is more accurate at the cost
of a larger download and slower CPU inference — swap `MODEL_NAME`/`PRETRAINED` in
`app/config.py`, nothing else changes.
         
**Limitations.** CLIP is strong on common scenes and objects, weaker on fine-grained
detail, text-in-images (OCR), and niche domains. Similarity scores are relative
rankings, not calibrated probabilities.

## Design Q&A

**Why CLIP for this task?**
The requirement is *natural-language* search with no pre-existing labels, tags, or
captions on the images. CLIP is trained with a contrastive objective that puts an
image and its describing text in the same space, so any free-text query can be
compared to any image with a dot product — exactly what's needed, and nothing has
to be labelled first. The trade-off is that CLIP is a general model: it's strong on
common subjects but weaker on fine-grained detail, OCR, and niche domains.   
     
**How is serving cheap and low-latency?**
By moving all the expensive work offline and keeping the online path trivial. Image
embeddings and thumbnails are computed once at index time, so a search request never
touches the image encoder — it only embeds the short query text and does one matrix
multiply over vectors already in RAM (sub-millisecond at folder scale; top-K via
`argpartition`, not a full sort). Images are served as small pre-generated thumbnails
with a `Cache-Control` header, so transfers are tiny and bandwidth is minimal;
originals load only when a user opens one. The cost of this design is an upfront
indexing pass (~30s for 100 images on CPU) and stale results if the folder changes
without re-indexing — an acceptable trade for a read-heavy search workload.

**Why no vector database?**
At folder scale an exact brute-force cosine (`query @ matrixᵀ`) is both faster and
simpler than a DB, whose network and serialization overhead would dominate at this
size — and it's exact, so there's no approximate-recall loss. A vector DB buys ANN
search, persistence, and horizontal scale, which only start to matter past ~1M
vectors (see the scaling table above). Adopting one earlier would be premature
complexity. Because retrieval is isolated in `app/search.py`, the switch can be made
later without touching the model or the HTTP API.

**Why full re-index instead of incremental?**
`app.index` rebuilds the whole index each run — it re-embeds every image and
overwrites the cache. The gain is simplicity and determinism: the index is
stateless and fully reproducible from one command, with none of the consistency
bugs incremental indexing invites (stale entries, orphaned thumbnails, drifting
ids). The cost is redundant work — ~30s for 100 images on CPU, and it grows with
the folder. Incremental indexing (hash each file, embed only new or changed ones,
append to the cache) only pays off once the folder is large enough that a full
rebuild is too slow or near-real-time updates are needed; at this scale it would be
premature. The trade-off it accepts today is that adding an image means re-running
the build.

**How is quality evaluated?**
The sample images have no human relevance labels, so `scripts/evaluate.py` runs two
*label-free* checks instead of inventing subjective ground truth. (1) A retrieval
sanity check: every image must be its own nearest neighbour (self-cosine ≈ 1.0),
which catches broken normalization or a corrupt index. (2) A label round-trip:
zero-shot CLIP assigns each image its best-matching label from a small vocabulary,
then that label is issued as a query to check whether the image returns in the
top-K — a self-consistency signal that text→image retrieval recovers the concept
the model sees. On the 100-image sample this gives Recall@1 ≈ 14%, @5 ≈ 48%,
@10 ≈ 65%; the low @1 is expected because the vocabulary contains overlapping
concepts (*nature* / *forest* / *trees* / *landscape* compete for the same images),
so @5–@10 are the meaningful figures. The trade-off is honesty over flattery: this
measures internal consistency, not agreement with human judgement, which would need
a hand-labelled query set. A spot check prints top hits for a few real queries.

**Why no reranking?**
This is single-stage retrieval (embed → cosine → sort), which is sufficient at this
scale where recall over a few hundred images is essentially complete. A two-stage
retrieve-then-rerank pipeline — recall top-N cheaply, then re-score with a heavier
cross-modal model (e.g. BLIP-2) — improves precision on large corpora or harder
queries, but adds latency and a second model to run and maintain. It's the natural
next step for scale, deliberately left out here to avoid over-engineering; query-side
prompt ensembling already improves ranking at near-zero cost.

## How this was built (AI coding agent)

Built with an AI coding agent (Claude Code). Because the scope was small, the work
ran in a **single main terminal, serially** — one focused thread of building and
iterating — rather than fanning out parallel agents, which would have added
coordination overhead without a real speedup at this size. A **second, separate
session** was used purely to **evaluate and debug the existing code** — an
independent pair of eyes that reviewed for bugs and consistency and drove the
label-free evaluation — kept apart from the main thread so review stayed unbiased by
the context that produced the code.

## Tech stack

open_clip · FastAPI + Uvicorn · NumPy (in-memory cosine) · single-page Tailwind/vanilla-JS UI · `uv`
