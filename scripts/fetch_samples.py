"""Download a small, varied set of public-domain sample images into images/.

Uses Picsum (Lorem Picsum), which serves free-to-use photos. We walk Picsum photo
IDs and keep the ones that resolve, until we have TARGET images — giving a varied,
reproducible sample set that works out of the box.
"""
from __future__ import annotations

import pathlib
import sys

import requests

TARGET = 100
# Picsum IDs are sparse (some are missing); scan a wide range and take what exists.
ID_RANGE = range(1, 400)
URL = "https://picsum.photos/id/{id}/800/600"


def main() -> int:
    out = pathlib.Path(__file__).resolve().parent.parent / "images"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    existing = sorted(out.glob("picsum_*.jpg"))
    if len(existing) >= TARGET:
        print(f"Already have {len(existing)} images in {out}")
        return 0

    saved = len(existing)
    for pid in ID_RANGE:
        if saved >= TARGET:
            break
        dest = out / f"picsum_{pid:03d}.jpg"
        if dest.exists():
            continue
        try:
            r = session.get(URL.format(id=pid), timeout=30)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            dest.write_bytes(r.content)
            saved += 1
            print(f"  [{saved}/{TARGET}] saved {dest.name} ({len(r.content) // 1024} KB)")
        except Exception as e:  # noqa: BLE001 - best-effort downloader
            print(f"  skip id={pid}: {e}", file=sys.stderr)

    print(f"Done: {saved} images in {out}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
