"""Download a small, varied set of public-domain sample images into images/.

Uses Picsum (Lorem Picsum) which serves free-to-use photos. We request a fixed
list of photo IDs so the sample set is deterministic and reproducible.
"""
from __future__ import annotations

import pathlib
import sys

import requests

# A hand-picked spread of Picsum photo IDs covering people, nature, animals,
# architecture, food, tech and transport so text queries have variety to hit.
PHOTO_IDS = [
    10, 20, 24, 27, 28, 29, 33, 37, 40, 42,
    48, 58, 65, 76, 82, 96, 100, 106, 111, 119,
    129, 133, 142, 152, 164, 175, 180, 188, 199, 219,
]

BASE = "https://picsum.photos/id/{id}/800/600"


def main() -> int:
    out = pathlib.Path(__file__).resolve().parent.parent / "images"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    ok = 0
    for pid in PHOTO_IDS:
        dest = out / f"picsum_{pid:03d}.jpg"
        if dest.exists():
            ok += 1
            continue
        url = BASE.format(id=pid)
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            ok += 1
            print(f"  saved {dest.name} ({len(r.content) // 1024} KB)")
        except Exception as e:  # noqa: BLE001 - best-effort downloader
            print(f"  skip id={pid}: {e}", file=sys.stderr)
    print(f"Done: {ok}/{len(PHOTO_IDS)} images in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
