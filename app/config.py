"""Shared paths and constants for the image semantic search app."""
from __future__ import annotations

import pathlib

# Project root = parent of the app/ package.
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Default folder of images to index, and where the search index cache lives.
IMAGES_DIR = ROOT / "images"
CACHE_DIR = ROOT / ".cache"
THUMBS_DIR = CACHE_DIR / "thumbs"
EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
META_PATH = CACHE_DIR / "meta.json"

WEB_DIR = ROOT / "web"

# CLIP model. ViT-B-32 is small (~350MB) and fast on CPU while accurate enough
# for natural-language image search. `laion2b_s34b_b79k` is a strong open weight.
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# Thumbnail longest edge in pixels — small transfers keep serving cheap & fast.
THUMB_MAX_EDGE = 384

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
