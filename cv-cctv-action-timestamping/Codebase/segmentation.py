"""
Temporal segmentation -- the layer the original project never had.

Why it has to exist
-------------------
The captioning model was trained on short clips and emits one caption per clip.
The original `lstm_captioning.lstm_captioning()` therefore shaped an entire
uploaded video as a single sample, produced exactly one caption keyed
``video_0``, and handed it to `timestamping`, which tried to read a `.start`
attribute off it. There was no mechanism by which a timestamp could ever exist.

This module cuts a video into overlapping fixed-length windows, each carrying a
real start and end time in seconds. Every downstream consumer works in terms of
`CaptionSegment`, so timestamps are structural rather than reconstructed.

Window size: 4.0 s
    The model has only ever seen clips of roughly this length -- the 780
    training clips have median duration 4.00 s and mean 3.97 s (p25 3.0,
    p75 5.0). Feeding it a window of a very different duration would be
    inference outside the training distribution.

Overlap: 50% (stride 2.0 s)
    An action that straddles a boundary would otherwise be split across two
    windows and captioned as two halves of nothing. With 50% overlap any event
    up to the window length is fully contained in at least one window.

    Tradeoff, stated plainly: overlap doubles the number of windows and
    therefore doubles inference cost. A 60 s video yields 29 windows at 50%
    overlap versus 15 with none. That is the price of not missing events at
    boundaries, and for a search tool a missed event is far worse than a slow
    query.

This module deliberately depends on nothing but OpenCV and numpy -- no Keras --
so segmentation logic can be tested without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# `prepare_frame` (resize -> CLAHE -> BGR->RGB, plus the 64x64 motion gray) is
# imported rather than duplicated: inference frames must be prepared exactly as
# training frames were, and two copies would eventually drift apart.
# feature_extraction imports Keras lazily, so this stays a cheap import.
from feature_extraction import prepare_frame

WINDOW_SECONDS = 4.0
OVERLAP = 0.5
STRIDE_SECONDS = WINDOW_SECONDS * (1.0 - OVERLAP)  # 2.0 s
FRAMES_PER_WINDOW = 16  # must match training (features are (16, 25120))
IMG_SIZE = 224
MOTION_SIZE = 64

# Safety valve: a multi-hour upload would otherwise allocate unbounded memory.
MAX_WINDOWS = 4000


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened or contains no usable frames."""


@dataclass(frozen=True)
class CaptionSegment:
    """One captioned slice of a video.

    The single currency of the caption -> search pipeline. `start` and `end` are
    seconds from the beginning of the video, as floats.
    """

    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __str__(self) -> str:  # e.g. "[12.0s-16.0s] a man runs ..."
        return f"[{self.start:.1f}s-{self.end:.1f}s] {self.text}"


@dataclass(frozen=True)
class VideoWindow:
    """A planned time window, before it has been captioned."""

    index: int
    start: float
    end: float
    frame_indices: tuple[int, ...]

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VideoInfo:
    path: str
    n_frames: int
    fps: float
    duration: float
    width: int
    height: int


def probe_video(path: str | Path) -> VideoInfo:
    """Read basic properties, raising VideoReadError on anything unusable."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoReadError(f"cannot open video: {path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Some containers report a bogus frame count; confirm at least one frame reads.
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise VideoReadError(f"video opened but no frame could be decoded: {path}")
    if n_frames <= 0:
        raise VideoReadError(f"video reports {n_frames} frames: {path}")
    if fps <= 0:
        fps = 30.0  # last-resort assumption; better than dividing by zero

    return VideoInfo(path, n_frames, fps, n_frames / fps, width, height)


def plan_windows(
    info: VideoInfo,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS,
    frames_per_window: int = FRAMES_PER_WINDOW,
) -> list[VideoWindow]:
    """Lay overlapping windows over the video's timeline.

    Edge cases handled:

    * **Video shorter than one window** -- a single window spanning the whole
      video. Its frames are still sampled to `frames_per_window`, repeating
      frames if necessary, so the model always receives its expected shape.
    * **Duration not a multiple of the stride** -- the trailing remainder would
      otherwise be dropped. A final window is anchored to the *end* of the video
      (`start = duration - window`) so the tail is always covered. It overlaps
      its predecessor by more than 50%, which is harmless.
    * **Absurdly long video** -- capped at MAX_WINDOWS with the stride widened
      to still span the whole timeline, so memory stays bounded.
    """
    duration = info.duration

    if duration <= window_seconds:
        starts = [0.0]
        ends = [duration]
    else:
        n_est = int(np.floor((duration - window_seconds) / stride_seconds)) + 1
        if n_est > MAX_WINDOWS:
            stride_seconds = (duration - window_seconds) / (MAX_WINDOWS - 1)
            n_est = MAX_WINDOWS
        starts = [i * stride_seconds for i in range(n_est)]
        ends = [s + window_seconds for s in starts]
        # Cover the tail if the last window stops short of the end.
        if ends[-1] < duration - 1e-3:
            starts.append(duration - window_seconds)
            ends.append(duration)

    windows: list[VideoWindow] = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        f0 = int(round(s * info.fps))
        f1 = int(round(e * info.fps)) - 1
        f0 = max(0, min(f0, info.n_frames - 1))
        f1 = max(f0, min(f1, info.n_frames - 1))
        if f1 == f0:
            idxs = tuple([f0] * frames_per_window)
        else:
            idxs = tuple(
                int(round(f0 + k * (f1 - f0) / (frames_per_window - 1)))
                for k in range(frames_per_window)
            )
        windows.append(VideoWindow(index=i, start=float(s), end=float(min(e, duration)),
                                   frame_indices=idxs))
    return windows


def read_window_frames(
    info: VideoInfo, windows: list[VideoWindow]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Decode every frame any window needs, in ONE sequential pass.

    Returns {frame_index: (rgb_224, gray_64)}. Sequential decoding is far
    cheaper than seeking per window, and with 50% overlap most frames are shared
    by two windows, so the cache is read twice on average.
    """
    needed = sorted({i for w in windows for i in w.frame_indices})
    if not needed:
        return {}

    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        raise VideoReadError(f"cannot open video: {info.path}")

    want = set(needed)
    last_wanted = needed[-1]
    pos = 0
    try:
        while pos <= last_wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if pos in want:
                cache[pos] = prepare_frame(frame)
            pos += 1
    finally:
        cap.release()

    if not cache:
        raise VideoReadError(f"decoded no usable frames from {info.path}")

    # A truncated file can leave late indices missing; substitute the last frame
    # we did decode rather than failing the whole video.
    if len(cache) < len(needed):
        fallback = cache[max(cache)]
        for i in needed:
            cache.setdefault(i, fallback)
    return cache


def segment_video(
    path: str | Path,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS,
) -> tuple[VideoInfo, list[VideoWindow]]:
    """Convenience wrapper: probe then plan."""
    info = probe_video(path)
    return info, plan_windows(info, window_seconds, stride_seconds)
