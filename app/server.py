"""FastAPI backend for image semantic search.

Startup loads the search index and warms up the CLIP model once. Endpoints:
  GET  /api/search?q=...                            -> natural-language (text -> image) search
  GET  /api/similar?id=N                            -> images similar to indexed image N
  POST /api/upload  (multipart image)               -> images similar to an uploaded image
  GET  /thumb/{name}                                -> cached thumbnail (small, fast, cheap)
  GET  /image/{id}                                  -> original full-resolution image
  GET  /                                            -> the single-page UI

Thumbnails are cache-friendly and tiny, so image serving stays low-latency and cheap;
originals are only fetched on demand (e.g. when a user opens one).
"""
from __future__ import annotations

import contextlib
import io

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import config
from .search import SearchIndex

_index: SearchIndex | None = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the index and warm up the CLIP model at startup, so the very first
    # query is fast (no lazy-load stall on the first request).
    from .model import get_model

    global _index
    _index = SearchIndex()
    get_model()  # preload CLIP weights now instead of on first search
    print(f"Loaded index: {_index.count} images ({_index.model_name}); model ready.")
    yield


app = FastAPI(title="Image Semantic Search", lifespan=lifespan)


def index() -> SearchIndex:
    if _index is None:  # pragma: no cover - startup guarantees this
        raise HTTPException(503, "Index not loaded")
    return _index


@app.get("/api/stats")
def stats() -> dict:
    idx = index()
    return {"count": idx.count, "model": idx.model_name}


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(24, ge=1, le=100),
) -> JSONResponse:
    """Natural-language (text -> image) semantic search."""
    results = index().search_text(q, top_k)
    return JSONResponse({"query": q, "results": results})


@app.get("/api/similar")
def api_similar(id: int = Query(..., ge=0), top_k: int = Query(24, ge=1, le=100)) -> JSONResponse:
    idx = index()
    if id >= idx.count:
        raise HTTPException(404, "Unknown image id")
    return JSONResponse({"method": "similar", "id": id, "results": idx.search_by_id(id, top_k)})


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), top_k: int = Query(24, ge=1, le=100)) -> JSONResponse:
    from .model import get_model

    data = await file.read()
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image")
    vec = get_model().encode_images([im])[0]
    return JSONResponse({"method": "upload", "results": index().search_by_vector(vec, top_k)})


@app.get("/thumb/{name}")
def thumb(name: str) -> FileResponse:
    path = (config.THUMBS_DIR / name).resolve()
    # Prevent path traversal: the resolved path must stay inside the thumbs dir.
    if not path.is_relative_to(config.THUMBS_DIR.resolve()) or not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/image/{image_id}")
def image(image_id: int) -> FileResponse:
    idx = index()
    if image_id < 0 or image_id >= idx.count:
        raise HTTPException(404, "Unknown image id")
    path = (config.IMAGES_DIR / idx.images[image_id]["path"]).resolve()
    if not path.is_relative_to(config.IMAGES_DIR.resolve()) or not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path)


# Serve the single-page UI at "/". Mounted last so /api and media routes win.
app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
