"""
CCTV Footage Search — Streamlit front end.

Complete rewrite. The original could not run: it rendered an undefined variable
(`video_html` where `full_html` was defined), inlined the entire uploaded video
into the DOM as base64, ran the whole inference pipeline twice per click, and
relied on `<script>` handlers that `st.markdown` strips before rendering.

Fixes for the seven catalogued bugs are marked A1..A7 inline.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DEMO_VIDEO = PROJECT_ROOT / "Test Videos" / "demo_concat.mp4"

REQUIRED_ARTIFACTS = {
    "lstm_model.keras": "the trained captioning model",
    "tokenizer.json": "the caption vocabulary",
    "svd.pkl": "the feature projection",
    "scaler.pkl": "the feature scaler",
}

MAX_UPLOAD_MB = 100
MAX_DURATION_SECONDS = 5 * 60

# Measured, not guessed: scripts/demo_query_sweep.py runs all 13 action
# categories against the sample video and checks each top hit against ground
# truth. "somebody running" is the query that lands a correct **top-1** hit, so
# a visitor's first click succeeds. On top-3 the sweep scores 7/13, which is why
# the results list shows five ranked candidates rather than a single answer.
DEFAULT_QUERY = "somebody running"

# Suggestions offered as one-click chips. Ordered by how well they did in the
# sweep: correct top-1 first, then correct within top-3.
SUGGESTED_QUERIES = [
    "somebody running",           # top-1 hit
    "someone throwing something", # top-3
    "someone sitting down",       # top-3
    "someone falls to the ground",# top-3
    "a person grabbing something",# top-3
]

ACTION_CATEGORIES = [
    "fall", "grab", "gun", "hit", "kick", "lying_down", "run",
    "sit", "sneak", "stand", "struggle", "throw", "walk",
]

st.set_page_config(page_title="CCTV Footage Search — Rithish S",
                   page_icon="🎥", layout="wide")


# --------------------------------------------------------------------------- #
# Artifact / model loading
# --------------------------------------------------------------------------- #
def missing_artifacts() -> list[str]:
    return [f"`{n}` ({d})" for n, d in REQUIRED_ARTIFACTS.items()
            if not (PROJECT_ROOT / n).exists()]


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """A6: cache the heavy objects so they survive every widget rerun.

    The original rebuilt VGG16 and reloaded all four artifacts on each call.
    """
    from lstm_captioning import load_artifacts
    from timestamping import get_embedder

    art = load_artifacts()
    get_embedder()  # warm the sentence embedder too
    return art


@st.cache_data(show_spinner=False)
def caption_cached(video_path: str, file_hash: str, _progress_key: str):
    """A3: run the model ONCE per distinct video.

    Keyed on a content hash, so re-searching, changing the threshold, or any
    other rerun reuses the captions instead of re-inferring. The original ran
    the entire pipeline twice for a single click.
    """
    from lstm_captioning import caption_video

    return caption_video(video_path)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def run_pipeline(video_path: str, file_hash: str):
    """Caption a video with staged progress. Returns (segments, error)."""
    from segmentation import VideoReadError, plan_windows, probe_video

    status = st.status("Analysing footage…", expanded=True)
    try:
        status.write("**Reading video…**")
        info = probe_video(video_path)
        if info.duration > MAX_DURATION_SECONDS:
            status.update(label="Video too long", state="error")
            return None, (f"That video is {info.duration / 60:.1f} minutes long. "
                          f"The limit is {MAX_DURATION_SECONDS // 60} minutes — "
                          "please trim it and try again.")
        windows = plan_windows(info)
        status.write(f"Found **{info.duration:.1f}s** of footage → "
                     f"**{len(windows)} overlapping 4-second windows**.")

        status.write("**Loading models…**")
        load_pipeline()

        status.write(f"**Extracting features and generating captions…** "
                     f"({len(windows)} windows, roughly "
                     f"{max(1, round(len(windows) * 0.45))}s)")
        t0 = time.time()
        segments = caption_cached(video_path, file_hash, "v1")
        elapsed = time.time() - t0

        if not segments:
            status.update(label="No segments produced", state="error")
            return None, ("No segments could be produced from this video. It may "
                          "be too short or the frames may be unreadable.")

        status.write(f"Generated **{len(segments)} captions** in {elapsed:.1f}s.")
        status.update(label=f"Done — {len(segments)} segments captioned",
                      state="complete", expanded=False)
        return segments, None

    except VideoReadError as e:
        status.update(label="Could not read the video", state="error")
        return None, (f"That file could not be read as video ({e}). "
                      "Try re-encoding it as a standard MP4.")
    except FileNotFoundError as e:
        status.update(label="Missing model files", state="error")
        return None, str(e)
    except Exception as e:  # noqa: BLE001 — never show a visitor a stack trace
        status.update(label="Analysis failed", state="error")
        return None, f"Something went wrong while analysing this video: {e}"


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🎥 CCTV Footage Search")
st.markdown(
    "Search surveillance footage by **describing what you're looking for**. "
    "The video is split into overlapping four-second windows, each one is "
    "captioned by a trained model, and your query is matched against those "
    "captions semantically — so *“someone falls over”* also finds *“a person "
    "lies on the ground”*."
)

gone = missing_artifacts()
if gone:
    st.error(
        "**The model files are missing, so search is unavailable.**\n\n"
        + "\n".join(f"- Missing: {m}" for m in gone)
        + "\n\nRegenerate them with `python scripts/train.py` "
          "(see README.md), then reload this page."
    )
    st.stop()

with st.expander("ℹ️  How this works & what it can't do", expanded=False):
    st.markdown(
        f"""
**Three stages.**

1. **Segmentation** — the video is cut into 4-second windows overlapping by 50%,
   so an action crossing a boundary is still fully inside at least one window.
2. **Captioning** — each window goes through a VGG16 visual encoder plus a
   motion descriptor, then a BiLSTM-with-attention decoder writes one short
   sentence describing it.
3. **Semantic search** — your query and every caption are embedded, and results
   are ranked by cosine similarity. It matches *meaning*, not keywords.

**How accurate is it, honestly?**

On held-out clips the model names the correct action **about 36% of the time**,
against a 7.7% baseline for guessing among 13 categories. That is far better
than chance and genuinely useful for narrowing down where to look — but it is
**not** a reliable detector, and roughly two out of three captions will name the
wrong action. Treat results as *candidate moments worth reviewing*, not answers.

The scene descriptions (indoors/outdoors, street/shop/corridor, day/night) are
noticeably more reliable than the action verbs.

On the 49-second sample video specifically, searching all 13 actions puts the
correct moment **first** 1 time in 13, and **somewhere in the top three** 7
times in 13. That is why results are shown as a ranked shortlist: scanning
three or four candidates is realistic, trusting the first one is not.

**The 13 actions it was trained on**

`{"` · `".join(ACTION_CATEGORIES)}`

Anything outside that list will be described using the closest of these words.

**The biggest limitation.** The model was trained on isolated 3–4 second clips,
each containing exactly one action, centred and already trimmed. Continuous
footage is a harder problem: actions start and stop mid-window, overlap, and are
surrounded by long stretches of nothing happening. Expect noticeably weaker
results on real continuous CCTV than the accuracy figure above suggests.
"""
    )

st.divider()

# ---- input ----------------------------------------------------------------
if "video_path" not in st.session_state:
    st.session_state.video_path = None
    st.session_state.video_hash = None
    st.session_state.video_label = None
    st.session_state.seek_to = 0

left, right = st.columns([3, 2])

with left:
    uploaded = st.file_uploader(
        f"Upload CCTV footage (MP4, max {MAX_UPLOAD_MB} MB / "
        f"{MAX_DURATION_SECONDS // 60} minutes)",
        type=["mp4", "mov", "avi"],
    )

with right:
    st.write("")
    st.write("")
    if DEMO_VIDEO.exists():
        if st.button("▶️  Try the sample video", type="primary", use_container_width=True):
            st.session_state.video_path = str(DEMO_VIDEO)
            st.session_state.video_hash = "demo-concat-v1"
            st.session_state.video_label = "Sample: 13 actions, 49 seconds"
            st.session_state.seek_to = 0
        st.caption("49 s of real CCTV clips covering all 13 actions.")
    else:
        st.caption("Sample video not found in this deployment.")

if uploaded is not None:
    data = uploaded.getvalue()
    size_mb = len(data) / 1e6
    if size_mb > MAX_UPLOAD_MB:
        st.error(f"That file is **{size_mb:.0f} MB**, over the {MAX_UPLOAD_MB} MB "
                 "limit. Please trim or compress it and try again.")
    else:
        digest = file_digest(data)
        if st.session_state.video_hash != digest:
            suffix = Path(uploaded.name).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(data)
            tmp.close()
            st.session_state.video_path = tmp.name
            st.session_state.video_hash = digest
            st.session_state.video_label = f"{uploaded.name} ({size_mb:.1f} MB)"
            st.session_state.seek_to = 0

if not st.session_state.video_path:
    st.info("Upload a clip or press **Try the sample video** to get started.")
    st.caption("Built by Rithish S · [github.com/rithishss](https://github.com/rithishss)")
    st.stop()

# ---- analyse --------------------------------------------------------------
segments, error = run_pipeline(st.session_state.video_path, st.session_state.video_hash)
if error:
    st.error(error)
    st.caption("Built by Rithish S · [github.com/rithishss](https://github.com/rithishss)")
    st.stop()

# ---- player ---------------------------------------------------------------
# A4: hand st.video the file directly. The original base64-inlined the whole
# video into the DOM — an 87 MB string for a 65 MB clip.
# A2/A5: `start_time` is Streamlit's supported seek mechanism, so no <script>
# handlers are needed and nothing gets stripped.
st.subheader(st.session_state.video_label or "Footage")
st.video(st.session_state.video_path, start_time=int(st.session_state.seek_to))
if st.session_state.seek_to:
    st.caption(f"⏱️ Jumped to **{fmt_ts(st.session_state.seek_to)}**. "
               "Press play if the video doesn't resume automatically.")

st.divider()

# ---- search ---------------------------------------------------------------
st.subheader("Search this footage")

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

st.caption("Try one of these, or type your own:")
chip_cols = st.columns(len(SUGGESTED_QUERIES))
for col, suggestion in zip(chip_cols, SUGGESTED_QUERIES):
    if col.button(suggestion, key=f"chip_{suggestion}", use_container_width=True):
        st.session_state.query_text = suggestion
        st.rerun()

sc1, sc2 = st.columns([4, 1])
with sc1:
    query = st.text_input("Describe what you're looking for",
                          value=st.session_state.query_text,
                          placeholder=f"e.g. {DEFAULT_QUERY}",
                          label_visibility="collapsed")
with sc2:
    threshold = st.slider("Sensitivity", 0.20, 0.60, 0.30, 0.05,
                          help="Lower finds more, with more false positives.")

if query and query.strip():
    from timestamping import search_segments

    with st.spinner("Searching captions…"):
        hits = search_segments(segments, query, similarity_threshold=threshold, top_k=5)

    if not hits:
        st.warning(
            f"Nothing matched **“{query}”** above the current sensitivity. "
            "Try lowering the slider, or rephrase using one of the 13 supported "
            "actions listed under *How this works*."
        )
    else:
        st.success(f"**{len(hits)}** matching moment"
                   f"{'s' if len(hits) != 1 else ''} — best first.")
        for rank, hit in enumerate(hits, 1):
            c1, c2, c3 = st.columns([1.4, 5.6, 1])
            with c1:
                st.markdown(f"**{fmt_ts(hit.start)} – {fmt_ts(hit.end)}**")
                st.progress(min(1.0, max(0.0, hit.score)),
                            text=f"{hit.score:.2f}")
            with c2:
                st.markdown(f"*{hit.text}*")
            with c3:
                # A7: real float seconds drive a real seek.
                if st.button("Jump", key=f"jump_{rank}", use_container_width=True):
                    st.session_state.seek_to = hit.start
                    st.rerun()

st.divider()

# ---- all captions ---------------------------------------------------------
# The visitor should be able to see what the model actually produced, rather
# than seek buttons appearing from nowhere.
with st.expander(f"📝  All {len(segments)} generated captions", expanded=False):
    st.caption("Every four-second window, in order. This is the raw model output "
               "that search runs over.")
    for i, seg in enumerate(segments):
        cc1, cc2, cc3 = st.columns([1.4, 5.6, 1])
        cc1.markdown(f"`{fmt_ts(seg.start)} – {fmt_ts(seg.end)}`")
        cc2.write(seg.text)
        if cc3.button("Jump", key=f"seg_jump_{i}", use_container_width=True):
            st.session_state.seek_to = seg.start
            st.rerun()

st.caption("Built by Rithish S · [github.com/rithishss](https://github.com/rithishss)")
