"""
Stage 6 -- build a demo video with known ground truth.

Concatenates several dataset clips from different action categories into one
30-60 s video, and writes a sidecar JSON recording exactly which action occupies
which time range. That gives search something to be *checked* against rather
than merely demonstrated on: if you search "someone falls" you know the answer
should be 12.0-16.0 s, not just that some timestamp came back.

Clips are drawn from the portion of the dataset the model never saw (indices
beyond the 60-per-category training sample), so the demo is an honest test.

Run:
    ./.venv/bin/python -u scripts/make_demo_video.py
    ./.venv/bin/python -u scripts/make_demo_video.py --categories fall gun run sit --seconds 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path("/Users/rithishselvam/Downloads/archive/Videos/Videos")
CAPTIONS_PATH = PROJECT_ROOT / "captions.json"
OUT_VIDEO = PROJECT_ROOT / "Test Videos" / "demo_concat.mp4"
OUT_TRUTH = PROJECT_ROOT / "Test Videos" / "demo_concat_truth.json"

OUT_FPS = 30.0
OUT_W, OUT_H = 640, 360
DEFAULT_CATEGORIES = ["walk", "fall", "gun", "run", "sit", "kick", "struggle", "throw"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    ap.add_argument("--seconds", type=float, default=45.0, help="target duration")
    ap.add_argument("--out", type=Path, default=OUT_VIDEO)
    args = ap.parse_args()

    used = set()
    if CAPTIONS_PATH.exists():
        used = set(json.loads(CAPTIONS_PATH.read_text())["clips"].keys())

    # Pick one unseen clip per requested category.
    chosen: list[tuple[str, Path]] = []
    for cat in args.categories:
        cdir = DATASET_DIR / cat
        if not cdir.is_dir():
            print(f"  skip unknown category {cat!r}")
            continue
        pool = [p for p in sorted(cdir.glob("*.mp4")) if f"{cat}/{p.name}" not in used]
        if not pool:
            pool = sorted(cdir.glob("*.mp4"))  # fall back to a seen clip
            print(f"  note: no unseen clip for {cat!r}, reusing a training clip")
        chosen.append((cat, pool[len(pool) // 2]))

    if not chosen:
        sys.exit("no usable categories")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                             OUT_FPS, (OUT_W, OUT_H))
    if not writer.isOpened():
        sys.exit(f"cannot open VideoWriter for {args.out}")

    segments = []
    frames_written = 0
    max_frames = int(args.seconds * OUT_FPS)

    for cat, path in chosen:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  skip unreadable {path.name}")
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or OUT_FPS
        start_sec = frames_written / OUT_FPS
        n_here = 0

        # Resample to OUT_FPS by index mapping, so timing stays truthful.
        src_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            src_frames.append(frame)
        cap.release()
        if not src_frames:
            continue

        out_n = max(1, int(round(len(src_frames) / src_fps * OUT_FPS)))
        for k in range(out_n):
            if frames_written >= max_frames:
                break
            src_idx = min(len(src_frames) - 1, int(round(k * len(src_frames) / out_n)))
            writer.write(cv2.resize(src_frames[src_idx], (OUT_W, OUT_H),
                                    interpolation=cv2.INTER_AREA))
            frames_written += 1
            n_here += 1

        if n_here:
            segments.append({
                "category": cat,
                "source_clip": f"{cat}/{path.name}",
                "start": round(start_sec, 3),
                "end": round((frames_written) / OUT_FPS, 3),
                "was_in_training_set": f"{cat}/{path.name}" in used,
            })
            print(f"  {cat:<10} {segments[-1]['start']:>6.2f}s - {segments[-1]['end']:>6.2f}s  {path.name}")
        if frames_written >= max_frames:
            break

    writer.release()
    duration = frames_written / OUT_FPS

    OUT_TRUTH.write_text(json.dumps({
        "video": args.out.name,
        "fps": OUT_FPS,
        "width": OUT_W, "height": OUT_H,
        "duration": round(duration, 3),
        "n_segments": len(segments),
        "segments": segments,
    }, indent=2))

    print(f"\nwrote {args.out}  ({duration:.1f}s, {frames_written} frames, "
          f"{args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {OUT_TRUTH}  ({len(segments)} ground-truth segments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
