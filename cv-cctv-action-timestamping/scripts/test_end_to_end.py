"""
Stage 7 -- end-to-end test of the rewired pipeline.

Three things get checked:

1. **Feature parity.** `Codebase/feature_extraction.py` must produce exactly the
   features `scripts/extract_features.py` produced at training time. If they
   drift, the model sees a distribution it was never fitted on and every caption
   silently degrades.

2. **Interface.** `lstm_captioning` returns `list[CaptionSegment]` with real
   float seconds, and `timestamping` consumes that type and returns scored hits.
   This is the mismatch that made the original impossible to run.

3. **Search accuracy against known ground truth.** The demo video is built from
   13 clips the model never saw, with a JSON recording which action occupies
   which seconds. For each query we check whether the top hit's time range
   overlaps the correct ground-truth segment.

Run:
    ./.venv/bin/python -u scripts/test_end_to_end.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Codebase"))

DEMO = PROJECT_ROOT / "Test Videos" / "demo_concat.mp4"
TRUTH = PROJECT_ROOT / "Test Videos" / "demo_concat_truth.json"

# Natural-language queries an operator might type, and the category each should find.
QUERIES = [
    ("someone falls to the ground", "fall"),
    ("a person with a gun", "gun"),
    ("people fighting", "struggle"),
    ("somebody running", "run"),
    ("a person sneaking around", "sneak"),
    ("someone sitting down", "sit"),
]


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def main() -> int:
    failures: list[str] = []

    # ---------------------------------------------------------------- 1. parity
    print("=" * 66)
    print("1. FEATURE PARITY  (Codebase/ must match what training produced)")
    print("=" * 66)
    import feature_extraction as fe

    index = json.loads((PROJECT_ROOT / "features_index.json").read_text())
    stored = np.load(PROJECT_ROOT / "features.npy", mmap_mode="r")
    dataset = Path("/Users/rithishselvam/Downloads/archive/Videos/Videos")

    rng = np.random.default_rng(0)
    rows = rng.choice(len(index["clips"]), size=5, replace=False)
    worst = 0.0
    for r in rows:
        clip = index["clips"][int(r)]
        fresh = fe.video_clip_features(dataset / clip)
        ref = np.asarray(stored[int(r)])
        err = float(np.max(np.abs(fresh - ref)))
        worst = max(worst, err)
        ok = err == 0.0
        print(f"  {'OK  ' if ok else 'FAIL'} row {int(r):>3}  max_abs_err {err:.3e}  {clip.split('/')[-1][:40]}")
        if not ok:
            failures.append(f"feature parity broken for {clip} (err {err})")
    print(f"  -> worst error across 5 clips: {worst:.3e} "
          f"({'IDENTICAL' if worst == 0 else 'DRIFT DETECTED'})\n")

    # ------------------------------------------------------------- 2. captioning
    print("=" * 66)
    print("2. CAPTION THE DEMO VIDEO  (one model run)")
    print("=" * 66)
    from lstm_captioning import caption_video
    from segmentation import CaptionSegment, probe_video

    if not DEMO.exists():
        print(f"  demo video missing: {DEMO}")
        return 1
    info = probe_video(DEMO)
    print(f"  {DEMO.name}: {info.duration:.1f}s, {info.n_frames} frames, {info.fps:.0f} fps")

    t0 = time.time()
    segments = caption_video(DEMO)
    elapsed = time.time() - t0

    print(f"  -> {len(segments)} segments in {elapsed:.1f}s ({elapsed / len(segments):.2f}s per window)")
    print(f"  type check: all CaptionSegment = "
          f"{all(isinstance(s, CaptionSegment) for s in segments)}")
    print(f"  start/end are floats = "
          f"{all(isinstance(s.start, float) and isinstance(s.end, float) for s in segments)}")
    if not all(isinstance(s, CaptionSegment) for s in segments):
        failures.append("caption_video did not return CaptionSegment objects")

    print("\n  first 8 segments:")
    for s in segments[:8]:
        print(f"    {s}")

    # ----------------------------------------------------------------- 3. search
    print("\n" + "=" * 66)
    print("3. SEARCH vs GROUND TRUTH")
    print("=" * 66)
    from timestamping import SearchHit, search_segments

    truth = json.loads(TRUTH.read_text())
    by_cat = {s["category"]: s for s in truth["segments"]}

    hits_correct = 0
    tested = 0
    rows_out = []
    for query, want_cat in QUERIES:
        gt = by_cat.get(want_cat)
        if gt is None:
            continue
        tested += 1
        results = search_segments(segments, query, similarity_threshold=0.30, top_k=3)
        if not results:
            print(f"\n  '{query}'  -> NO RESULTS   (expected {want_cat} at "
                  f"{gt['start']:.1f}-{gt['end']:.1f}s)")
            rows_out.append({"query": query, "expected_category": want_cat,
                             "expected_range": [gt["start"], gt["end"]],
                             "top_hit": None, "correct": False})
            continue

        top = results[0]
        good = overlaps(top.start, top.end, gt["start"], gt["end"])
        hits_correct += good
        # was the right window anywhere in the top 3?
        in_top3 = any(overlaps(h.start, h.end, gt["start"], gt["end"]) for h in results)
        print(f"\n  '{query}'   expected {want_cat} at {gt['start']:.1f}-{gt['end']:.1f}s")
        for rank, h in enumerate(results, 1):
            mark = "<-- correct" if overlaps(h.start, h.end, gt["start"], gt["end"]) else ""
            print(f"    {rank}. [{h.start:5.1f}s-{h.end:5.1f}s] score {h.score:.3f}  {h.text}  {mark}")
        print(f"    top-1 {'HIT' if good else 'MISS'} | top-3 {'HIT' if in_top3 else 'MISS'}")
        rows_out.append({"query": query, "expected_category": want_cat,
                         "expected_range": [gt["start"], gt["end"]],
                         "top_hit": {"start": top.start, "end": top.end,
                                     "score": round(top.score, 4), "text": top.text},
                         "correct": bool(good), "in_top3": bool(in_top3)})

    top1 = hits_correct / tested if tested else 0.0
    top3 = sum(r.get("in_top3", False) for r in rows_out) / tested if tested else 0.0

    print("\n" + "=" * 66)
    print(f"  top-1 localisation: {hits_correct}/{tested} ({top1 * 100:.0f}%)")
    print(f"  top-3 localisation: {sum(r.get('in_top3', False) for r in rows_out)}/{tested} ({top3 * 100:.0f}%)")
    print("=" * 66)

    (PROJECT_ROOT / "scripts" / "e2e_report.json").write_text(json.dumps({
        "feature_parity_max_error": worst,
        "demo_video": DEMO.name,
        "demo_duration": info.duration,
        "n_segments": len(segments),
        "caption_seconds": round(elapsed, 1),
        "segments": [{"text": s.text, "start": s.start, "end": s.end} for s in segments],
        "queries": rows_out,
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "failures": failures,
    }, indent=2))

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nEnd-to-end pipeline runs. See scripts/e2e_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
