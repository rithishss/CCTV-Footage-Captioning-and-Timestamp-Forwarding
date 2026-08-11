"""
Motion-aware feature extraction with VideoMAE.

Why VideoMAE over I3D
---------------------
The linear probe showed VGG16+FFT features cap out near 31% validation accuracy.
That is a *representation* limit: VGG16 is an ImageNet **image** model applied to
frames independently, so nothing in the 25,088 spatial dims encodes motion, and
the 32 FFT bins are a crude hand-designed proxy for it.

`MCG-NJU/videomae-base` replaces that with a genuine spatiotemporal encoder:

* **Pretrained on Kinetics-400**, an action-recognition corpus — the pretraining
  objective is about motion, which is exactly the axis our 13 classes differ on.
* **Tube masking.** VideoMAE masks the same spatial patch across *all* frames,
  so the model cannot reconstruct a masked region by copying a neighbouring
  frame. It is forced to model temporal dynamics rather than exploit
  frame-to-frame redundancy. That is precisely the signal VGG16 cannot provide.
* **Native 16-frame input.** Our clips are already sampled at 16 frames, so the
  volume drops in without resampling.
* **Available and maintained in HuggingFace `transformers`** as a first-class
  model, whereas I3D has no official `transformers` implementation and would
  mean a third-party checkpoint of uncertain provenance.

I3D (or torchvision's `r2plus1d_18`) was the alternative. It is cheaper per clip
on CPU, but it is a smaller *supervised* Kinetics model, and we have MPS
available, so the capacity is worth more to us than the speed. If this proves too
slow, `r2plus1d_18` is the fallback.

Feature shape
-------------
VideoMAE emits `last_hidden_state` of (B, 1568, 768), where 1568 = 8 temporal
positions x 196 spatial patches (it halves the 16 input frames temporally).
Reshaping to (8, 196, 768) and mean-pooling over the spatial axis yields
**(8, 768)** — a temporal sequence, so it drops straight into the existing
BiLSTM encoder with T=8 instead of 16 and D=768 instead of 25120.

That is a 33x reduction in feature width, which is itself likely to help: 780
clips could never support 25,120 raw dims.

Run:
    ./.venv/bin/python -u scripts/extract_features_videomae.py --limit 10
    ./.venv/bin/python -u scripts/extract_features_videomae.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path("/Users/rithishselvam/Downloads/archive/Videos/Videos")
CAPTIONS_PATH = PROJECT_ROOT / "captions.json"
OUT_FEATURES = PROJECT_ROOT / "features_videomae.npy"
OUT_INDEX = PROJECT_ROOT / "features_videomae_index.json"

MODEL_ID = "MCG-NJU/videomae-base"
FRAMES_PER_CLIP = 16  # VideoMAE's native input length
IMG_SIZE = 224
TEMPORAL_OUT = 8  # VideoMAE halves 16 frames temporally
SPATIAL_PATCHES = 196  # 14 x 14
HIDDEN = 768
POOLED = HIDDEN * 2  # mean + std concatenated


def sample_indices(n_frames: int, k: int = FRAMES_PER_CLIP) -> list[int]:
    if n_frames <= 0:
        raise ValueError(f"clip reports {n_frames} frames")
    return [int(round(i * (n_frames - 1) / max(1, k - 1))) for i in range(k)]


def read_clip_rgb(path: Path) -> list[np.ndarray]:
    """16 RGB frames spanning the clip.

    Deliberately no CLAHE here: VideoMAE was pretrained on ordinary video and
    expects its own normalisation. Injecting a contrast transform it never saw
    would push inputs off-distribution.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        cap.release()
        raise RuntimeError(f"{path} reports {n_frames} frames")

    frames = []
    for idx in sample_indices(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(idx + 1):
                ok, frame = cap.read()
                if not ok:
                    break
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"{path}: cannot read frame {idx}")
        small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from transformers import VideoMAEImageProcessor, VideoMAEModel

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEModel.from_pretrained(MODEL_ID).to(device).eval()
    mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)
    print(f"loaded {MODEL_ID} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")

    clips = sorted(json.loads(CAPTIONS_PATH.read_text())["clips"].keys())
    if args.limit:
        clips = clips[: args.limit]

    feats = np.zeros((len(clips), TEMPORAL_OUT, POOLED), dtype=np.float32)
    index: list[str] = []
    failed: list[dict] = []
    t0 = time.time()

    batch_frames: list[list[np.ndarray]] = []
    batch_ids: list[str] = []

    def flush():
        if not batch_frames:
            return
        # (B, T, H, W, C) uint8 -> (B, T, C, H, W) normalised float
        arr = torch.from_numpy(np.stack(batch_frames)).float().div_(255.0)
        arr = arr.permute(0, 1, 4, 2, 3)
        arr = (arr - mean) / std
        with torch.no_grad():
            out = model(pixel_values=arr.to(device)).last_hidden_state  # (B, 1568, 768)
        b = out.shape[0]
        out = out.reshape(b, TEMPORAL_OUT, SPATIAL_PATCHES, HIDDEN)
        # mean+std over the spatial patches. The linear probe scored 47.4% with
        # mean+std vs 42.9% with mean alone -- the spread across patches carries
        # signal that averaging discards.
        out = torch.cat([out.mean(dim=2), out.std(dim=2)], dim=-1)  # (B, 8, 1536)
        out = out.float().cpu().numpy()
        for k, cid in enumerate(batch_ids):
            feats[len(index)] = out[k]
            index.append(cid)
        batch_frames.clear()
        batch_ids.clear()

    for i, clip_id in enumerate(clips):
        try:
            batch_frames.append(np.stack(read_clip_rgb(DATASET_DIR / clip_id)))
            batch_ids.append(clip_id)
        except Exception as e:  # noqa: BLE001
            failed.append({"clip": clip_id, "error": repr(e)})
            print(f"  FAIL {clip_id}: {e}", flush=True)
        if len(batch_frames) >= args.batch:
            flush()
        if (i + 1) % 50 == 0 or i + 1 == len(clips):
            el = time.time() - t0
            print(f"  {i + 1}/{len(clips)}  {el:.0f}s  ({(i + 1) / max(el, 1e-9):.2f} clips/s)", flush=True)
    flush()

    feats = feats[: len(index)]
    elapsed = time.time() - t0
    np.save(OUT_FEATURES, feats)
    OUT_INDEX.write_text(json.dumps({
        "features_file": OUT_FEATURES.name,
        "backbone": MODEL_ID,
        "shape": list(feats.shape),
        "dtype": str(feats.dtype),
        "frames_per_clip": FRAMES_PER_CLIP,
        "temporal_out": TEMPORAL_OUT,
        "hidden": HIDDEN,
        "pooled_dim": POOLED,
        "pooling": "mean+std over the 196 spatial patches, temporal axis kept",
        "extraction_seconds": round(elapsed, 1),
        "failed": failed,
        "clips": index,
    }, indent=2))

    print(f"\nshape {feats.shape} dtype {feats.dtype} ({feats.nbytes / 1e6:.0f} MB)")
    print(f"extracted {len(index)}/{len(clips)} in {elapsed:.0f}s, {len(failed)} failed")
    print(f"-> {OUT_FEATURES}\n-> {OUT_INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
