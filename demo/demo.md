# Demo video

> Record a short (~60–90s) screen capture and drop it here (or paste a link below).
> On macOS: `Cmd+Shift+5`. On Linux: OBS / `wf-recorder` / GNOME screen recorder.
> Then commit the file as `demo/demo.mp4` (or a `.gif`), or paste a YouTube/Loom link.

**Video:** _<paste link here, or add `demo/demo.mp4` to the repo>_

## Suggested 60-second script

1. **Open** `http://127.0.0.1:8077` — the dark-theme grid loads with a default query.
2. **Semantic search** — type `black and white photo`, hit Enter. Point out the similarity
   badges and the sub-second latency shown in the status line.
3. **Try a few queries** — click the example chips (`green nature landscape`, `a building`,
   `water and sky`) to show natural language across different concepts.
4. **Find similar** — click *Similar* on a result to search by that image's own vector.
5. **Search by image** — click *Search by image*, upload a photo, show the nearest matches.
6. **Keyword baseline** — switch to *Keyword*, search a filename fragment (e.g. `058`) to
   contrast the non-semantic method.
7. **Open an image** — click a thumbnail to open the full-resolution lightbox.

Close by noting: local CLIP model, no API key, embeddings precomputed offline, exact cosine
search in RAM → cheap + low latency.
