"""
Run every action category as a natural-language query against the demo video.

Two purposes:

1. **End-to-end test.** Exercises segmentation -> VideoMAE features -> caption
   model -> semantic search on a real 49 s video built from clips the model
   never saw.
2. **Pick the app's default query.** A visitor's first click has to succeed, so
   `app.py` pre-fills the search box with a query measured to return a correct
   top-1 hit here. Guessing which one works would be irresponsible.

Run:
    ./.venv/bin/python -u scripts/demo_query_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Codebase"))

DEMO = PROJECT_ROOT / "Test Videos" / "demo_concat.mp4"
TRUTH = PROJECT_ROOT / "Test Videos" / "demo_concat_truth.json"
OUT = PROJECT_ROOT / "scripts" / "demo_query_report.json"

# One natural-language query per category — phrased as a person would type it,
# not as the caption vocabulary words, so this tests semantic matching too.
QUERIES = {
    "fall": "someone falls to the ground",
    "grab": "a person grabbing something",
    "gun": "a person with a gun",
    "hit": "someone hitting another person",
    "kick": "a person kicking someone",
    "lying_down": "somebody lying on the floor",
    "run": "somebody running",
    "sit": "someone sitting down",
    "sneak": "a person sneaking around",
    "stand": "people standing still",
    "struggle": "people fighting",
    "throw": "someone throwing something",
    "walk": "a person walking",
}

THRESHOLD = 0.30
TOP_K = 5


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def main() -> int:
    from lstm_captioning import caption_video
    from segmentation import probe_video
    from timestamping import search_segments

    if not DEMO.exists():
        sys.exit(f"demo video missing: {DEMO}")

    info = probe_video(DEMO)
    print(f"demo: {info.duration:.1f}s, {info.n_frames} frames, {info.fps:.0f} fps")

    t0 = time.time()
    segments = caption_video(DEMO)
    caption_seconds = time.time() - t0
    print(f"captioned {len(segments)} windows in {caption_seconds:.1f}s "
          f"({caption_seconds / len(segments):.2f}s per window)\n")

    print("--- generated captions ---")
    for s in segments:
        print(f"  {s}")

    truth = json.loads(TRUTH.read_text())
    by_cat = {s["category"]: s for s in truth["segments"]}

    print("\n--- query sweep ---")
    rows, hits1, hits3 = [], 0, 0
    for cat, query in QUERIES.items():
        gt = by_cat.get(cat)
        if gt is None:
            continue
        res = search_segments(segments, query, similarity_threshold=THRESHOLD, top_k=TOP_K)
        top1 = bool(res) and overlaps(res[0].start, res[0].end, gt["start"], gt["end"])
        top3 = any(overlaps(h.start, h.end, gt["start"], gt["end"]) for h in res[:3])
        hits1 += top1
        hits3 += top3
        mark = "HIT " if top1 else ("top3" if top3 else "miss")
        got = (f"{res[0].start:5.1f}-{res[0].end:5.1f}s ({res[0].score:.2f})"
               if res else "no results")
        print(f"  {mark}  {cat:<11} {query:<32} want {gt['start']:5.1f}-{gt['end']:5.1f}s  got {got}")
        rows.append({
            "category": cat, "query": query,
            "expected": [gt["start"], gt["end"]],
            "top1_hit": top1, "top3_hit": top3,
            "results": [{"start": h.start, "end": h.end,
                         "score": round(h.score, 4), "text": h.text} for h in res[:3]],
        })

    n = len(rows)
    working = [r["query"] for r in rows if r["top1_hit"]]
    print(f"\n  top-1: {hits1}/{n} ({hits1 / n * 100:.0f}%)   "
          f"top-3: {hits3}/{n} ({hits3 / n * 100:.0f}%)")
    print(f"\n  queries safe to use as the app default ({len(working)}):")
    for q in working:
        print(f"    - {q}")

    OUT.write_text(json.dumps({
        "demo_video": DEMO.name,
        "duration": info.duration,
        "n_segments": len(segments),
        "caption_seconds": round(caption_seconds, 1),
        "threshold": THRESHOLD,
        "top1": hits1, "top3": hits3, "n_queries": n,
        "top1_rate": round(hits1 / n, 4), "top3_rate": round(hits3 / n, 4),
        "working_queries": working,
        "segments": [{"text": s.text, "start": s.start, "end": s.end} for s in segments],
        "rows": rows,
    }, indent=2))
    print(f"\nreport -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
