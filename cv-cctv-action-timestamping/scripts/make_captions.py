"""
Stage 3 -- generate captions.json with a vision-language model.

For each selected clip we sample 3 frames, send them to Claude together with the
action label parsed from the clip's folder (ground truth), and store the returned
one-sentence caption.

Design notes
------------
* **Balanced selection.** 20 clips from each of the 13 categories = 260 clips.
  Selection is the first 20 by sorted filename, so the set is reproducible.
* **Round-robin ordering.** Clips are emitted one-per-category in rotation, so
  any prefix (e.g. ``--limit 10``) is spread across categories instead of being
  20 clips of "fall".
* **Resumable.** captions.json is rewritten atomically after every completed
  clip. Re-running skips clips already present, so an interrupted run continues
  where it stopped. ``--redo`` forces re-captioning.
* **Ground truth in the prompt.** The label comes from the directory name, not
  from the filename fields -- categories such as ``lying_down`` contain an
  underscore, and one file (UCFCRIME_Arrest007_lying_down.mp4) has no trailing
  instance number, so positional parsing is unreliable.

Run:
    . <scratchpad>/anthropic.env
    ./.venv/bin/python -u scripts/make_captions.py --limit 10
    ./.venv/bin/python -u scripts/make_captions.py            # all 260
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.environ.get("CCTV_DATASET", "/Users/rithishselvam/Downloads/archive/Videos/Videos"))
CAPTIONS_PATH = PROJECT_ROOT / "captions.json"

CLIPS_PER_CATEGORY = 60
FRAMES_PER_CLIP = 3
FRAME_POSITIONS = (0.2, 0.5, 0.8)  # avoid black/transitional first and last frames
MAX_EDGE = 512  # long-edge resize; ~262 image tokens per frame
JPEG_QUALITY = 85

MODEL = "claude-opus-5"
EFFORT = "low"  # captioning is not reasoning-heavy; low effort halves output tokens

# Human-readable gloss for each label, so the prompt does not lean on the bare
# folder name (e.g. "lying_down" -> "a person lying on the ground").
LABEL_GLOSS = {
    "fall": "a person falling to the ground",
    "grab": "one person grabbing another person or an object",
    "gun": "a person holding or brandishing a gun",
    "hit": "a person striking another person",
    "kick": "a person kicking another person or an object",
    "lying_down": "a person lying on the ground",
    "run": "a person running",
    "sit": "a person sitting down",
    "sneak": "a person moving furtively or sneaking",
    "stand": "a person standing",
    "struggle": "two or more people struggling or grappling",
    "throw": "a person throwing an object",
    "walk": "a person walking",
}

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": "One sentence, 8-12 words, present tense: subject + action + one setting + lighting.",
        }
    },
    "required": ["caption"],
    "additionalProperties": False,
}

# Closed vocabularies. The whole point is to bound the token inventory: with only
# a few hundred training clips, every word that appears once is a word the
# captioning model cannot learn. Restricting subjects, settings and lighting to
# fixed lists keeps the vocabulary in the low hundreds instead of ~800.
SUBJECTS = ["a man", "a woman", "a person", "two men", "two people", "several people", "a group"]
SETTINGS = [
    "street", "road", "sidewalk", "alley", "parking lot", "field", "yard",
    "shop", "store", "restaurant", "bar", "office", "lobby", "corridor",
    "hallway", "room", "stairwell", "station", "platform", "garage", "elevator",
]
# "in a dim room" was tried first and had to go: the model treated it as a second
# setting, producing "in a garage in a dim room". "in dim light" cannot be read
# that way.
LIGHTING = ["at night", "in daylight", "indoors", "in dim light"]

SYSTEM_PROMPT = f"""You write short, uniform captions for CCTV surveillance clips. They are training data for a small video-captioning model, so a tightly controlled vocabulary matters more than vividness.

You are shown 3 frames sampled from one short clip, in chronological order, plus the verified action label for that clip.

Write ONE present-tense sentence of 8 to 12 words, in exactly this shape:
    <subject> <action> in/on a <setting> <lighting>

Use ONLY these subjects: {', '.join(SUBJECTS)}
Use ONLY these settings: {', '.join(SETTINGS)}
Use ONLY these lighting phrases: {', '.join(LIGHTING)}

Rules:
- The action label is ground truth. Your caption must agree with it, even when the frames are dark, blurry, low-resolution or ambiguous.
- Pick the single closest setting from the list. If none fits well, choose the nearest general one ("street" outdoors, "room" indoors). Never invent a setting word.
- Name NO other objects. No cars, boxes, counters, shelves, escalators, tables, gates, stools, display cases. The setting word carries all the context needed.
- No adjectives beyond the lighting phrase. No colours, no clothing, no materials.
- No speculation about intent, identity, emotion, occupation or consequences.
- Never mention frames, images, cameras, footage, video, or the label itself.
- STOP after the lighting phrase. No trailing or subordinate clause -- nothing
  starting with "as", "while", "who", "and", "with", "near", "past", or a comma.
  Do not describe bystanders or any second action.
- Use exactly one setting word and exactly one lighting phrase. Never combine
  two settings ("a corridor in a dim room") or two lighting phrases.

Good: "A man kicks another person in a bar at night."
Good: "Several people struggle on a street at night."
Good: "A person lies on a road at night."
Bad:  "A man in dark clothing points a gun across glass display cases inside a jewelry shop." (too long, invented objects, clothing detail)
Bad:  "A man runs through a station indoors as several people walk nearby." (trailing clause)
Bad:  "A person sneaks along a corridor in a dim room at night." (two settings, two lighting phrases)"""

_write_lock = threading.Lock()


def discover_clips() -> list[dict]:
    """Balanced, deterministic, round-robin-ordered clip list."""
    cats = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    if not cats:
        sys.exit(f"No category folders under {DATASET_DIR}")

    per_cat: dict[str, list[Path]] = {}
    for c in cats:
        files = sorted((DATASET_DIR / c).glob("*.mp4"))[:CLIPS_PER_CATEGORY]
        if len(files) < CLIPS_PER_CATEGORY:
            sys.exit(f"Category {c!r} has only {len(files)} clips, need {CLIPS_PER_CATEGORY}")
        per_cat[c] = files

    # Round-robin so any prefix covers many categories.
    clips = []
    for i in range(CLIPS_PER_CATEGORY):
        for c in cats:
            p = per_cat[c][i]
            clips.append({"clip_id": f"{c}/{p.name}", "category": c, "path": p,
                          "source": p.name.split("_")[0]})
    return clips


def sample_frames(path: Path) -> tuple[list[bytes], dict]:
    """Return JPEG bytes for FRAMES_PER_CLIP frames plus clip metadata."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if n_frames <= 0:
        cap.release()
        raise RuntimeError(f"{path} reports {n_frames} frames")

    indices = [min(n_frames - 1, max(0, int(round(pos * (n_frames - 1))))) for pos in FRAME_POSITIONS]

    jpegs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            # Seeking can fail on some codecs; fall back to a sequential read.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(idx + 1):
                ok, frame = cap.read()
                if not ok:
                    break
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"{path}: failed to read frame {idx}")

        h, w = frame.shape[:2]
        scale = MAX_EDGE / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            cap.release()
            raise RuntimeError(f"{path}: JPEG encode failed for frame {idx}")
        jpegs.append(buf.tobytes())

    cap.release()
    meta = {"n_frames": n_frames, "fps": round(float(fps), 2), "frames_used": indices,
            "resolution": f"{w}x{h}"}
    return jpegs, meta


def caption_one(client, clip: dict) -> dict:
    """Caption a single clip. Raises on unrecoverable failure."""
    import anthropic

    jpegs, meta = sample_frames(clip["path"])
    gloss = LABEL_GLOSS.get(clip["category"], clip["category"].replace("_", " "))

    content: list[dict] = []
    for i, jpg in enumerate(jpegs, 1):
        content.append({"type": "text", "text": f"Frame {i} of {len(jpegs)}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.standard_b64encode(jpg).decode("ascii")},
        })
    content.append({
        "type": "text",
        "text": (f"Verified action label for this clip: {clip['category']} "
                 f"({gloss}).\n\nWrite the caption."),
    })

    last_err = None
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=300,
                # The system prompt is ~700 tokens and byte-identical on all 780
                # calls, so cache it: reads bill at ~0.1x. Only the per-clip
                # frames and label vary, and those sit after it in render order.
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": EFFORT,
                               "format": {"type": "json_schema", "schema": CAPTION_SCHEMA}},
                messages=[{"role": "user", "content": content}],
            )
            if resp.stop_reason == "refusal":
                cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
                raise RuntimeError(f"model refused (category={cat})")

            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            caption = json.loads(text)["caption"].strip()
            if not caption:
                raise RuntimeError("empty caption returned")

            return {
                "category": clip["category"],
                "source": clip["source"],
                "file": clip["clip_id"],
                "caption": caption,
                "model": resp.model,
                **meta,
                "usage": {
                    "in": resp.usage.input_tokens,
                    "out": resp.usage.output_tokens,
                    "cache_write": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                    "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
                },
            }
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            last_err = e
            time.sleep(2 ** attempt * 2)
        except (json.JSONDecodeError, KeyError) as e:
            last_err = RuntimeError(f"unparseable response: {e}")
            time.sleep(1)

    raise RuntimeError(f"{clip['clip_id']}: giving up after retries ({last_err!r})")


def load_captions() -> dict:
    if CAPTIONS_PATH.exists():
        return json.loads(CAPTIONS_PATH.read_text())
    return {"meta": {}, "clips": {}}


def save_captions(doc: dict) -> None:
    """Atomic rewrite so an interrupted run never leaves truncated JSON."""
    tmp = CAPTIONS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    tmp.replace(CAPTIONS_PATH)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="caption only the first N clips (0 = all 260)")
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls")
    ap.add_argument("--redo", action="store_true", help="re-caption clips already in captions.json")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return int(bool(sys.stderr.write("ANTHROPIC_API_KEY is not set.\n")))

    import anthropic

    clips = discover_clips()
    if args.limit:
        clips = clips[: args.limit]

    doc = load_captions()
    todo = clips if args.redo else [c for c in clips if c["clip_id"] not in doc["clips"]]

    print(f"selected {len(clips)} clips | already captioned {len(clips) - len(todo)} | to do {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return 0

    client = anthropic.Anthropic(max_retries=3)
    t_start = time.time()
    done = failed = 0
    tok_in = tok_out = tok_cw = tok_cr = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(caption_one, client, c): c for c in todo}
        for fut in as_completed(futures):
            clip = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 - one bad clip must not kill the run
                failed += 1
                print(f"  FAIL {clip['clip_id']}: {e}", flush=True)
                continue

            done += 1
            tok_in += rec["usage"]["in"]
            tok_out += rec["usage"]["out"]
            tok_cw += rec["usage"]["cache_write"]
            tok_cr += rec["usage"]["cache_read"]
            with _write_lock:
                doc["clips"][rec["file"]] = rec
                doc["meta"] = {
                    "model": MODEL, "effort": EFFORT,
                    "clips_per_category": CLIPS_PER_CATEGORY,
                    "frames_per_clip": FRAMES_PER_CLIP,
                    "frame_positions": list(FRAME_POSITIONS),
                    "max_edge_px": MAX_EDGE,
                    "count": len(doc["clips"]),
                }
                save_captions(doc)
            print(f"  [{done + failed}/{len(todo)}] {rec['category']:<11} {rec['caption']}", flush=True)

    elapsed = time.time() - t_start
    # claude-opus-5: $5/M in, $25/M out; cache writes 1.25x in, cache reads 0.1x in.
    cost = (tok_in * 5.0 + tok_cw * 6.25 + tok_cr * 0.5 + tok_out * 25.0) / 1e6
    print(f"\ncaptioned {done}, failed {failed}, in {elapsed:.0f}s")
    print(f"tokens: {tok_in} in / {tok_out} out / {tok_cw} cache-write / {tok_cr} cache-read")
    print(f"estimated cost: ${cost:.2f}")
    print(f"captions.json now holds {len(doc['clips'])} clips -> {CAPTIONS_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
