"""
Stage 4 -- extract hybrid spatial + temporal features for every captioned clip.

Rewrite of Codebase/feature_extraction.py. Fixes, in order of severity:

1. **RGB channel order (was silently wrong).** OpenCV yields BGR. Keras'
   ``vgg16.preprocess_input`` runs in caffe mode, which expects **RGB** and does
   its own RGB->BGR flip internally. The original fed it BGR, so every feature
   vector ever produced was computed on channel-swapped input. We convert
   BGR->RGB before preprocessing.

2. **Fixed-size temporal descriptor.** The original's descriptor was
   ``n_frames - 1`` long, so its width changed with clip duration and an SVD
   fitted on one clip length could not transform another. Ours is always
   ``TEMPORAL_BINS`` wide per timestep, whatever the clip length.

3. **Bounded memory.** The original ran ``np.fft.fftn`` over every frame of the
   video at native resolution (complex128 -- tens of GB for real footage). We
   FFT one 64x64 grayscale difference image at a time.

4. **Fixed timestep count.** Exactly ``FRAMES_PER_CLIP`` frames are sampled per
   clip regardless of length, so every row of features.npy has the same shape.

5. **Batched inference.** One ``model(batch)`` call per clip instead of
   ``model.predict()`` per frame.

Feature layout per timestep: [25088 VGG16 block5_pool] + [32 FFT radial bins]
                             = 25120 dims.

Run:
    ./.venv/bin/python -u scripts/extract_features.py --limit 10
    ./.venv/bin/python -u scripts/extract_features.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.environ.get("CCTV_DATASET", "/Users/rithishselvam/Downloads/archive/Videos/Videos"))
CAPTIONS_PATH = PROJECT_ROOT / "captions.json"
FEATURES_PATH = PROJECT_ROOT / "features.npy"
INDEX_PATH = PROJECT_ROOT / "features_index.json"

FRAMES_PER_CLIP = 16
IMG_SIZE = 224
SPATIAL_DIM = 25088  # 7 * 7 * 512
TEMPORAL_BINS = 32
MOTION_SIZE = 64  # grayscale difference images are computed at 64x64
FEATURE_DIM = SPATIAL_DIM + TEMPORAL_BINS


def sample_indices(n_frames: int, k: int = FRAMES_PER_CLIP) -> list[int]:
    """k frame indices spread evenly across the clip, clamped and de-duplicated
    by position (short clips repeat frames rather than failing)."""
    if n_frames <= 0:
        raise ValueError(f"clip reports {n_frames} frames")
    return [int(round(i * (n_frames - 1) / max(1, k - 1))) for i in range(k)]


def enhance(frame_bgr: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on the L channel.

    Note: the original called this `apply_zerodce`, but the body is CLAHE and
    always was. ZeroDCE is still unused -- the class in image_enhancement.py has
    no trained weights and two verified maths bugs, so switching to it would
    make results worse, not better. Named honestly here.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def radial_fft_profile(diff_gray: np.ndarray, bins: int = TEMPORAL_BINS) -> np.ndarray:
    """Fixed-width motion descriptor for one frame-difference image.

    2D FFT -> shifted magnitude -> mean magnitude in `bins` concentric radial
    rings. Ring k captures motion energy at a spatial frequency band, so the
    descriptor is the same width for any input and any clip length.
    """
    mag = np.abs(np.fft.fftshift(np.fft.fft2(diff_gray.astype(np.float32))))
    h, w = mag.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    idx = np.minimum((r_norm * bins).astype(np.int32), bins - 1)
    sums = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=bins)
    counts = np.bincount(idx.ravel(), minlength=bins).astype(np.float32)
    prof = sums / np.maximum(counts, 1.0)
    # log1p keeps the dynamic range sane before SVD sees it
    return np.log1p(prof).astype(np.float32)


def clip_features(path: Path, vgg, preprocess_input) -> np.ndarray:
    """(FRAMES_PER_CLIP, FEATURE_DIM) float32 for one clip."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        cap.release()
        raise RuntimeError(f"{path.name} reports {n_frames} frames")

    rgb_frames, motion_grays = [], []
    for idx in sample_indices(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:  # seek can fail on some codecs
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(idx + 1):
                ok, frame = cap.read()
                if not ok:
                    break
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"{path.name}: cannot read frame {idx}")

        small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        enhanced = enhance(small)
        # THE FIX: BGR -> RGB before preprocess_input (caffe mode expects RGB).
        rgb_frames.append(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        motion_grays.append(
            cv2.resize(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY),
                       (MOTION_SIZE, MOTION_SIZE), interpolation=cv2.INTER_AREA)
        )
    cap.release()

    batch = preprocess_input(np.asarray(rgb_frames, dtype=np.float32))
    import keras

    spatial = keras.ops.convert_to_numpy(vgg(batch, training=False))
    spatial = spatial.reshape(len(rgb_frames), -1).astype(np.float32)
    if spatial.shape[1] != SPATIAL_DIM:
        raise RuntimeError(f"expected {SPATIAL_DIM} spatial dims, got {spatial.shape[1]}")

    temporal = np.zeros((FRAMES_PER_CLIP, TEMPORAL_BINS), dtype=np.float32)
    for t in range(1, FRAMES_PER_CLIP):
        diff = cv2.absdiff(motion_grays[t], motion_grays[t - 1])
        temporal[t] = radial_fft_profile(diff)
    temporal[0] = temporal[1]  # t=0 has no predecessor; mirror t=1

    return np.concatenate([spatial, temporal], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (0 = all)")
    ap.add_argument("--out", type=Path, default=FEATURES_PATH)
    ap.add_argument("--index", type=Path, default=INDEX_PATH)
    args = ap.parse_args()

    if not CAPTIONS_PATH.exists():
        sys.exit("captions.json not found -- run scripts/make_captions.py first")
    clips = sorted(json.loads(CAPTIONS_PATH.read_text())["clips"].keys())
    if args.limit:
        clips = clips[: args.limit]

    import keras
    import torch
    from keras.applications.vgg16 import VGG16, preprocess_input

    print(f"keras {keras.__version__} / backend {keras.backend.backend()} / "
          f"mps {torch.backends.mps.is_available()}")
    vgg = VGG16(weights="imagenet", include_top=False)

    n = len(clips)
    feats = np.zeros((n, FRAMES_PER_CLIP, FEATURE_DIM), dtype=np.float32)
    index, failed = [], []
    t0 = time.time()

    for i, clip_id in enumerate(clips):
        path = DATASET_DIR / clip_id
        try:
            feats[len(index)] = clip_features(path, vgg, preprocess_input)
            index.append(clip_id)
        except Exception as e:  # noqa: BLE001 - one bad clip must not kill the run
            failed.append({"clip": clip_id, "error": repr(e)})
            print(f"  FAIL {clip_id}: {e}", flush=True)
        if (i + 1) % 50 == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"  {i + 1}/{n} clips  {el:.0f}s  ({(i + 1) / el:.1f} clips/s)", flush=True)

    feats = feats[: len(index)]  # drop trailing rows for failed clips
    elapsed = time.time() - t0

    np.save(args.out, feats)
    args.index.write_text(json.dumps({
        "features_file": args.out.name,
        "shape": list(feats.shape),
        "dtype": str(feats.dtype),
        "frames_per_clip": FRAMES_PER_CLIP,
        "spatial_dim": SPATIAL_DIM,
        "temporal_bins": TEMPORAL_BINS,
        "feature_dim": FEATURE_DIM,
        "img_size": IMG_SIZE,
        "motion_size": MOTION_SIZE,
        "extraction_seconds": round(elapsed, 1),
        "failed": failed,
        # row i of features.npy corresponds to clips[i]
        "clips": index,
    }, indent=2))

    print(f"\nshape {feats.shape} dtype {feats.dtype} "
          f"({feats.nbytes / 1e9:.2f} GB)")
    print(f"extracted {len(index)}/{n} clips in {elapsed:.0f}s, {len(failed)} failed")
    print(f"-> {args.out}\n-> {args.index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
