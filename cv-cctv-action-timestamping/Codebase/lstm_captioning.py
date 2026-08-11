"""
Caption a video as a list of time-stamped segments.

Rewrite of the original module. The original could not run at all:

* `lstm_captioning()` called `generate_caption_greedy(model, ..., tokenizer, ...)`
  but `model` and `tokenizer` were locals of `loading_and_preprocessing()`,
  which returned only `X_reduced` -- a guaranteed `NameError`.
* `tf.keras.preprocessing.text.tokenizer_from_json` was removed in Keras 3, so
  the module failed at import on any modern install.
* `features_concatenator` shaped the whole video as ONE sample and returned a
  dict `{"video_0": caption}`, so a timestamp could never exist.
* `features_concatenator` also built a one-element list and then took min/max
  over it -- its padding and dimension-alignment logic was a no-op.

Now: the video is cut into overlapping windows by `segmentation`, each window is
captioned, and the result is a `list[CaptionSegment]` carrying real start and
end times in seconds. Artifacts are loaded once per process and cached.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from feature_extraction import FEATURE_DIM, features_from_frames, get_vgg16  # noqa: E402
from segmentation import (  # noqa: E402
    CaptionSegment,
    VideoReadError,
    plan_windows,
    probe_video,
    read_window_frames,
)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ARTIFACT_DIR / "lstm_model.keras"
TOKENIZER_PATH = ARTIFACT_DIR / "tokenizer.json"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
SVD_PATH = ARTIFACT_DIR / "svd.pkl"

_ARTIFACTS = None


def load_artifacts():
    """Load model, tokenizer, SVD and scaler exactly once per process.

    The original constructed VGG16 and reloaded every artifact on each call.
    """
    global _ARTIFACTS
    if _ARTIFACTS is None:
        import keras

        missing = [p.name for p in (MODEL_PATH, TOKENIZER_PATH, SCALER_PATH, SVD_PATH)
                   if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing model artifacts: {', '.join(missing)}. "
                "Run scripts/train.py to regenerate them."
            )
        tokenizer = json.loads(TOKENIZER_PATH.read_text())
        _ARTIFACTS = {
            "model": keras.models.load_model(MODEL_PATH, compile=False),
            "tokenizer": tokenizer,
            "svd": joblib.load(SVD_PATH),
            "scaler": joblib.load(SCALER_PATH),
            "vgg": get_vgg16(),
            "max_length": int(tokenizer.get("max_length", 11)),
        }
    return _ARTIFACTS


def generate_caption_greedy(model, video_feature: np.ndarray, tokenizer: dict,
                            max_length: int | None = None) -> str:
    """Greedy autoregressive decode for a single window.

    `tokenizer` is our own dict, but `word_index` is looked up by the literal
    lowercase keys 'start' and 'end' -- the same contract the original assumed.
    """
    wi = tokenizer["word_index"]
    iw = tokenizer["index_word"]
    max_length = max_length or int(tokenizer.get("max_length", 11))

    start_token = wi.get("start")
    end_token = wi.get("end")
    if start_token is None or end_token is None:
        raise ValueError("tokenizer.json lacks the required 'start'/'end' tokens")

    seq = [start_token]
    words: list[str] = []
    for _ in range(max_length - 1):
        padded = np.array([seq + [0] * (max_length - 1 - len(seq))], dtype="int32")
        preds = model.predict([video_feature[None, ...], padded], verbose=0)
        nxt = int(np.argmax(preds[0, len(seq) - 1]))
        if nxt == end_token or nxt == 0:
            break
        words.append(iw.get(str(nxt), "<unk>"))
        seq.append(nxt)
    return " ".join(words)


def caption_video(video_path: str | Path, progress=None) -> list[CaptionSegment]:
    """Caption an entire video. Returns one CaptionSegment per window.

    `progress` is an optional callable(done, total) for UI feedback.
    """
    art = load_artifacts()
    info = probe_video(video_path)  # raises VideoReadError on bad input
    windows = plan_windows(info)
    frame_cache = read_window_frames(info, windows)

    segments: list[CaptionSegment] = []
    for n, w in enumerate(windows, 1):
        rgb = [frame_cache[i][0] for i in w.frame_indices]
        gray = [frame_cache[i][1] for i in w.frame_indices]

        feats = features_from_frames(rgb, gray, art["vgg"])          # (T, 25120)
        reduced = art["svd"].transform(feats)                         # (T, 1500)
        scaled = art["scaler"].transform(reduced).astype("float32")

        text = generate_caption_greedy(art["model"], scaled, art["tokenizer"],
                                       art["max_length"])
        segments.append(CaptionSegment(text=text, start=w.start, end=w.end))
        if progress:
            progress(n, len(windows))

    return segments


def lstm_captioning(video) -> list[CaptionSegment]:
    """Backwards-compatible entry point.

    Accepts a path, or a Streamlit `UploadedFile` / any file-like object with
    `.read()` (OpenCV needs a real path, so file-likes are spooled to a temp
    file first).
    """
    if hasattr(video, "read"):
        import tempfile

        suffix = Path(getattr(video, "name", "upload.mp4")).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            if hasattr(video, "seek"):
                video.seek(0)
            tmp.write(video.read())
            tmp_path = tmp.name
        try:
            return caption_video(tmp_path)
        finally:
            os.unlink(tmp_path)
    return caption_video(video)


__all__ = [
    "CaptionSegment",
    "VideoReadError",
    "caption_video",
    "generate_caption_greedy",
    "load_artifacts",
    "lstm_captioning",
]
