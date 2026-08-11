"""
Semantic search over captioned video segments.

Rewrite of the original module, which could not work:

* It received `dict[str, str]` from `lstm_captioning()` but iterated it as a list
  of objects with `.text` and `.start` (a `youtube-transcript-api` shape) --
  `AttributeError` on the first chunk.
* It computed `total_seconds` and then **discarded it**, returning caption *text*
  while `app.py` interpolated the return value into `seekTo({ts})` and `{ts}s`.
  The function's contract and its only caller disagreed completely.
* It rebuilt the SentenceTransformer on every call.
* `convert_timestamp_to_seconds` parsed `'HH:MM:SS.fff'`, a format nothing in
  this codebase ever produced. **It is deleted** -- a dead parser for a format
  that does not exist only invites confusion later.

Now: search consumes `list[CaptionSegment]` and returns ranked `SearchHit`s,
each carrying the segment (with real float seconds) and a similarity score.
Callers get the top candidates, not a single take-it-or-leave-it answer -- an
operator scanning footage wants to see the shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from segmentation import CaptionSegment

EMBED_MODEL = "minishlab/potion-base-8M"
DEFAULT_THRESHOLD = 0.35
DEFAULT_TOP_K = 5

_EMBEDDER = None


def get_embedder():
    """Load the sentence embedder once per process.

    `model2vec` static embeddings: numpy-only inference, no torch/transformers
    stack, ~30 MB. Replaces `sentence-transformers`, which would have pulled a
    second copy of torch into the deployment image.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from model2vec import StaticModel

        _EMBEDDER = StaticModel.from_pretrained(EMBED_MODEL)
    return _EMBEDDER


@dataclass(frozen=True)
class SearchHit:
    """One matching segment and how well it matched."""

    segment: CaptionSegment
    score: float

    # Convenience passthroughs so callers never have to reach inside.
    @property
    def text(self) -> str:
        return self.segment.text

    @property
    def start(self) -> float:
        return self.segment.start

    @property
    def end(self) -> float:
        return self.segment.end

    def __str__(self) -> str:
        return f"[{self.start:.1f}s-{self.end:.1f}s] {self.score:.3f}  {self.text}"


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


def merge_adjacent(hits: list[SearchHit], gap: float = 0.01) -> list[SearchHit]:
    """Collapse overlapping/adjacent hits into one span.

    Windows overlap by 50%, so a single real event usually matches two or three
    consecutive windows. Without merging, the operator sees the same incident
    three times. The merged span keeps the best-scoring caption.
    """
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h.segment.start)
    merged: list[SearchHit] = []
    cur = ordered[0]
    for nxt in ordered[1:]:
        if nxt.segment.start <= cur.segment.end + gap:
            best = cur if cur.score >= nxt.score else nxt
            cur = SearchHit(
                segment=CaptionSegment(
                    text=best.segment.text,
                    start=cur.segment.start,
                    end=max(cur.segment.end, nxt.segment.end),
                ),
                score=max(cur.score, nxt.score),
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return sorted(merged, key=lambda h: h.score, reverse=True)


def search_segments(
    segments: list[CaptionSegment],
    query: str,
    similarity_threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    merge: bool = True,
) -> list[SearchHit]:
    """Rank segments by semantic similarity to `query`.

    Returns up to `top_k` hits above `similarity_threshold`, highest score
    first. Empty list if nothing clears the bar.
    """
    if not segments or not query or not query.strip():
        return []

    embedder = get_embedder()
    corpus = _normalise(np.asarray(embedder.encode([s.text for s in segments])))
    q = _normalise(np.asarray(embedder.encode([query]))[0])
    scores = corpus @ q

    hits = [SearchHit(segment=s, score=float(sc))
            for s, sc in zip(segments, scores) if sc > similarity_threshold]
    if merge:
        hits = merge_adjacent(hits)
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def streamlit_timestamping(
    video,
    search_word: str,
    similarity_threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchHit]:
    """Caption a video and search it in one call.

    The model runs **once** here. The original `app.py` called the whole
    pipeline twice per click -- two complete VGG16 passes over the footage for
    one search.

    For interactive use, prefer calling `lstm_captioning.caption_video()` once
    and then `search_segments()` per query, so re-searching the same video does
    not re-run the model.
    """
    from lstm_captioning import lstm_captioning

    segments = lstm_captioning(video)
    return search_segments(segments, search_word, similarity_threshold, top_k)


__all__ = [
    "SearchHit",
    "get_embedder",
    "merge_adjacent",
    "search_segments",
    "streamlit_timestamping",
]
