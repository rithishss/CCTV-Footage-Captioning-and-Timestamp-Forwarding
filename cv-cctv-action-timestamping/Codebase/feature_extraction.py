"""
Feature extraction — VideoMAE spatiotemporal backbone.

History of this module
----------------------
It began as VGG16 per-frame features plus a hand-designed FFT motion descriptor.
That worked, but a linear probe showed those features capped out near **30.8%**
validation action accuracy: VGG16 is an ImageNet *image* model applied to frames
independently, so nothing in its 25,088 dims encodes motion.

Replacing it with `MCG-NJU/videomae-base` — pretrained on Kinetics-400 with a
tube-masking objective that forces temporal modelling — raised the probe ceiling
to **47.4%** and the end-to-end model from 35.9% to **42.3%**.

What the old module got wrong, and which of those still matter:

* **RGB channel order.** VGG16's `preprocess_input` (caffe mode) expects RGB and
  flips internally; the original fed it BGR, so every feature it ever produced
  was channel-swapped. Still relevant: we convert BGR→RGB here too.
* **Variable-width temporal descriptor.** The original's was `n_frames - 1` wide,
  so an SVD fitted at one clip length could not transform another. Moot now —
  VideoMAE always emits a fixed (8, 1536).
* **Unbounded memory.** The original FFT'd every frame at native resolution
  (complex128, tens of GB on real footage). Moot now — no FFT at all.
* **Import-time side effects.** Importing the original pulled in
  `image_enhancement`, which globbed `./lol_dataset/...` and built `tf.data`
  pipelines at import. Torch/transformers are imported lazily here.

**No CLAHE.** The VGG16 path applied CLAHE contrast enhancement. VideoMAE was
pretrained on ordinary video with its own normalisation; injecting a contrast
transform it never saw during pretraining pushes inputs off-distribution. Frames
go in as-is apart from resize and the model's own mean/std.

This module is the single source of truth for feature computation. Training
(`scripts/extract_features_videomae.py`) and inference (`lstm_captioning.py`)
must agree exactly, or the model sees a distribution it was never fitted on.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

BACKBONE = "MCG-NJU/videomae-base"
FRAMES_PER_CLIP = 16  # VideoMAE's native input length
IMG_SIZE = 224
TEMPORAL_OUT = 8  # VideoMAE halves 16 input frames temporally
SPATIAL_PATCHES = 196  # 14 x 14
HIDDEN = 768
FEATURE_DIM = HIDDEN * 2  # mean + std over the spatial patches = 1536

_MODEL = None
_NORM = None


def get_backbone():
    """Load VideoMAE once per process (lazy: keeps this module cheap to import)."""
    global _MODEL, _NORM
    if _MODEL is None:
        import torch
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        processor = VideoMAEImageProcessor.from_pretrained(BACKBONE)
        model = VideoMAEModel.from_pretrained(BACKBONE).to(device).eval()
        _NORM = (
            torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1),
            torch.tensor(processor.image_std).view(1, 1, 3, 1, 1),
            device,
        )
        _MODEL = model
    return _MODEL, _NORM


# Kept for backwards compatibility with callers that expect the old name.
get_vgg16 = get_backbone


def sample_indices(n_frames: int, k: int = FRAMES_PER_CLIP) -> list[int]:
    """k frame indices spread evenly across `n_frames`."""
    if n_frames <= 0:
        raise ValueError(f"clip reports {n_frames} frames")
    return [int(round(i * (n_frames - 1) / max(1, k - 1))) for i in range(k)]


def prepare_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """One raw BGR frame -> 224x224 RGB, ready for VideoMAE normalisation."""
    small = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def features_from_frames(rgb_frames: list[np.ndarray]) -> np.ndarray:
    """(TEMPORAL_OUT, FEATURE_DIM) float32 from exactly FRAMES_PER_CLIP frames.

    VideoMAE emits `last_hidden_state` of (1, 1568, 768) = 8 temporal x 196
    spatial. We keep the temporal axis (so the BiLSTM encoder still sees a
    sequence) and pool the spatial axis with mean **and** std — the probe scored
    47.4% with mean+std vs 42.9% with mean alone, so the spread across patches
    carries signal that averaging throws away.
    """
    import torch

    if len(rgb_frames) != FRAMES_PER_CLIP:
        raise ValueError(f"expected {FRAMES_PER_CLIP} frames, got {len(rgb_frames)}")

    model, (mean, std, device) = get_backbone()
    arr = torch.from_numpy(np.stack(rgb_frames)[None, ...]).float().div_(255.0)
    arr = arr.permute(0, 1, 4, 2, 3)  # (1, T, C, H, W)
    arr = (arr - mean) / std
    with torch.no_grad():
        out = model(pixel_values=arr.to(device)).last_hidden_state
    out = out.reshape(1, TEMPORAL_OUT, SPATIAL_PATCHES, HIDDEN)
    out = torch.cat([out.mean(dim=2), out.std(dim=2)], dim=-1)
    return out.float().cpu().numpy()[0].astype(np.float32)


def video_clip_features(path: str | Path) -> np.ndarray:
    """(TEMPORAL_OUT, FEATURE_DIM) for a whole short clip."""
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
        frames.append(prepare_frame(frame))
    cap.release()
    return features_from_frames(frames)
