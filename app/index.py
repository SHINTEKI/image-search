"""Build the search index for a folder of images.

For every image we:
  1. compute a CLIP embedding (image -> vector), and
  2. generate a small thumbnail (cheap, low-latency serving).

Embeddings are stacked into one (N, D) matrix saved as embeddings.npy; per-image
metadata (filename, size) goes to meta.json. This is a one-off offline cost — the
online search path never touches the image encoder.

Usage:
    uv run python -m app.index [IMAGES_DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from . import config


def find_images(images_dir: pathlib.Path) -> list[pathlib.Path]:
    files = [
        p
        for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTS
    ]
    return files


def make_thumbnail(src: pathlib.Path, dest: pathlib.Path) -> None:
    """Write a small RGB thumbnail (longest edge = THUMB_MAX_EDGE)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((config.THUMB_MAX_EDGE, config.THUMB_MAX_EDGE))
        im.save(dest, format="JPEG", quality=82)


def build(images_dir: pathlib.Path, batch_size: int = 16) -> None:
    from .model import get_model

    files = find_images(images_dir)
    if not files:
        print(f"No images found in {images_dir}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Indexing {len(files)} images from {images_dir} ...")
    t0 = time.time()
    model = get_model()
    print(f"  loaded CLIP {config.MODEL_NAME} in {time.time() - t0:.1f}s")

    config.THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    all_vecs: list[np.ndarray] = []
    meta: list[dict] = []

    for start in range(0, len(files), batch_size):
        batch_files = files[start : start + batch_size]
        images = []
        for f in batch_files:
            with Image.open(f) as im:
                images.append(ImageOps.exif_transpose(im).convert("RGB"))
        vecs = model.encode_images(images)
        all_vecs.append(vecs)

        for f in batch_files:
            rel = f.relative_to(images_dir).as_posix()
            thumb_name = rel.replace("/", "__") + ".jpg"
            make_thumbnail(f, config.THUMBS_DIR / thumb_name)
            meta.append(
                {
                    "id": len(meta),
                    "path": rel,
                    "filename": f.name,
                    "thumb": thumb_name,
                }
            )
        print(f"  embedded {min(start + batch_size, len(files))}/{len(files)}")

    embeddings = np.concatenate(all_vecs, axis=0)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_PATH, embeddings)
    config.META_PATH.write_text(
        json.dumps(
            {
                "model": config.MODEL_NAME,
                "pretrained": config.PRETRAINED,
                "dim": int(embeddings.shape[1]),
                "count": len(meta),
                "images": meta,
            },
            indent=2,
        )
    )
    print(
        f"Done: {len(meta)} images, dim={embeddings.shape[1]}, "
        f"{time.time() - t0:.1f}s total -> {config.CACHE_DIR}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CLIP image search index.")
    parser.add_argument(
        "images_dir",
        nargs="?",
        default=str(config.IMAGES_DIR),
        help="Folder of images to index (default: ./images)",
    )
    args = parser.parse_args()
    build(pathlib.Path(args.images_dir).resolve())


if __name__ == "__main__":
    main()
