"""
Hybrid spatial + temporal feature extraction.

Rewrite of the original module. What changed and why:

* **RGB channel order.** OpenCV yields BGR; `vgg16.preprocess_input` runs in
  caffe mode, which expects **RGB** and performs its own RGB->BGR flip. The
  original passed BGR straight through, so every feature vector it produced was
  computed on channel-swapped input. Measured effect: cosine similarity 0.803
  between the correct and the buggy path on the same frame.

* **Fixed-width temporal descriptor.** The original's was `n_frames - 1` wide,
  so it changed with clip duration and an SVD fitted at one length could not
  transform another. Ours is always `TEMPORAL_BINS` per timestep.

* **Bounded memory.** The original ran `np.fft.fftn` over every frame of the
  video at native resolution (complex128 — tens of GB for real footage). This
  FFTs one 64x64 difference image at a time.

* **`apply_zerodce` renamed to `enhance`.** The function body is, and always
  was, CLAHE. ZeroDCE is not used: the class in `image_enhancement.py` has no
  trained weights and two verified maths bugs (its spatial-consistency kernels
  are shaped (1,3,1,3) where conv2d needs (3,3,1,1), and its TV loss subtracts
  mismatched slices), so using it would make results worse, not better.

* **No import-time side effects.** Importing the original pulled in
  `image_enhancement`, which globbed `./lol_dataset/...` and built `tf.data`
  pipelines at import. Keras is imported lazily here, inside `get_vgg16()`.

This module is the single source of truth for feature computation. Training
(`scripts/extract_features.py`) and inference (`Codebase/lstm_captioning.py`)
must produce byte-identical features or the model sees a distribution it was
never fitted on — `scripts/verify_feature_parity.py` checks exactly that.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FRAMES_PER_CLIP = 16
IMG_SIZE = 224
SPATIAL_DIM = 25088  # 7 * 7 * 512 from VGG16 block5_pool
TEMPORAL_BINS = 32
MOTION_SIZE = 64
FEATURE_DIM = SPATIAL_DIM + TEMPORAL_BINS  # 25120

_VGG = None


def get_vgg16():
    """Load VGG16 once per process. Keras is imported lazily so that modules
    which only need the CV helpers (e.g. segmentation) stay Keras-free."""
    global _VGG
    if _VGG is None:
        from keras.applications.vgg16 import VGG16

        _VGG = VGG16(weights="imagenet", include_top=False)
    return _VGG


def sample_indices(n_frames: int, k: int = FRAMES_PER_CLIP) -> list[int]:
    """k frame indices spread evenly across `n_frames`."""
    if n_frames <= 0:
        raise ValueError(f"clip reports {n_frames} frames")
    return [int(round(i * (n_frames - 1) / max(1, k - 1))) for i in range(k)]


def enhance(frame_bgr: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement on the L channel of LAB.

    (The original called this `apply_zerodce`; the body is CLAHE.)
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def prepare_frame(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One raw BGR frame -> (rgb_224 for VGG16, gray_64 for the motion FFT)."""
    small = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    enhanced = enhance(small)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)  # the fix
    gray = cv2.resize(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY),
                      (MOTION_SIZE, MOTION_SIZE), interpolation=cv2.INTER_AREA)
    return rgb, gray


def radial_fft_profile(diff_gray: np.ndarray, bins: int = TEMPORAL_BINS) -> np.ndarray:
    """Fixed-width motion descriptor for one frame-difference image.

    2-D FFT -> shifted magnitude -> mean magnitude within `bins` concentric
    radial rings. Ring k is motion energy in a spatial-frequency band, so the
    descriptor has the same width for any input and any clip length.
    """
    mag = np.abs(np.fft.fftshift(np.fft.fft2(diff_gray.astype(np.float32))))
    h, w = mag.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    idx = np.minimum(((r / r.max()) * bins).astype(np.int32), bins - 1)
    sums = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=bins)
    counts = np.bincount(idx.ravel(), minlength=bins).astype(np.float32)
    return np.log1p(sums / np.maximum(counts, 1.0)).astype(np.float32)


def features_from_frames(rgb_frames: list[np.ndarray],
                         gray_frames: list[np.ndarray],
                         vgg=None) -> np.ndarray:
    """(T, FEATURE_DIM) float32 from prepared frames."""
    import keras
    from keras.applications.vgg16 import preprocess_input

    vgg = vgg or get_vgg16()
    batch = preprocess_input(np.asarray(rgb_frames, dtype=np.float32))
    spatial = keras.ops.convert_to_numpy(vgg(batch, training=False))
    spatial = spatial.reshape(len(rgb_frames), -1).astype(np.float32)
    if spatial.shape[1] != SPATIAL_DIM:
        raise RuntimeError(f"expected {SPATIAL_DIM} spatial dims, got {spatial.shape[1]}")

    t_len = len(gray_frames)
    temporal = np.zeros((t_len, TEMPORAL_BINS), dtype=np.float32)
    for t in range(1, t_len):
        temporal[t] = radial_fft_profile(cv2.absdiff(gray_frames[t], gray_frames[t - 1]))
    if t_len > 1:
        temporal[0] = temporal[1]  # t=0 has no predecessor; mirror t=1

    return np.concatenate([spatial, temporal], axis=1)


def video_clip_features(path: str | Path, vgg=None) -> np.ndarray:
    """(FRAMES_PER_CLIP, FEATURE_DIM) for a whole short clip.

    Used by training. Inference goes through `segmentation` + this module's
    `features_from_frames` instead, one window at a time.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        cap.release()
        raise RuntimeError(f"{path} reports {n_frames} frames")

    rgb_frames, gray_frames = [], []
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
        rgb, gray = prepare_frame(frame)
        rgb_frames.append(rgb)
        gray_frames.append(gray)
    cap.release()

    return features_from_frames(rgb_frames, gray_frames, vgg)
