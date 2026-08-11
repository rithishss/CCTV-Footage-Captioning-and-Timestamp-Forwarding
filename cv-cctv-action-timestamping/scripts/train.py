"""
Stage 5 -- fit the transforms and train the captioning model.

Produces all four missing artifacts:
    svd.pkl           TruncatedSVD, 25120 -> 1500 components
    scaler.pkl        StandardScaler fitted on the SVD output
    tokenizer.json    word_index containing literal lowercase 'start' / 'end'
    lstm_model.keras  BiLSTM encoder + attention decoder

Two deliberate choices worth knowing about:

* **Masked accuracy.** Padding tokens are excluded from both loss and accuracy.
  Counting them is the standard way to manufacture a 95%+ "accuracy" that means
  nothing -- most timesteps in a padded batch are padding. Every number this
  script reports is over real tokens only.

* **MAX_LENGTH = 11**, as specified. Captions run up to 14 words, so
  start + words + end exceeds 11 for a substantial share of the set; the
  truncation rate is measured and printed rather than hidden.

Run:
    ./.venv/bin/python -u scripts/train.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_PATH = PROJECT_ROOT / "captions.json"
FEATURES_PATH = PROJECT_ROOT / "features.npy"
INDEX_PATH = PROJECT_ROOT / "features_index.json"

SVD_PATH = PROJECT_ROOT / "svd.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer.json"
MODEL_PATH = PROJECT_ROOT / "lstm_model.keras"
REPORT_PATH = PROJECT_ROOT / "scripts" / "train_report.json"

N_COMPONENTS = 1500
MAX_LENGTH = 11  # total tokens per caption, including 'start' and 'end'
VAL_FRACTION = 0.2
SEED = 42

# Small model on purpose: ~624 training clips cannot support a large one.
ENC_UNITS = 128  # Bidirectional -> 256 wide
DEC_UNITS = 256  # must equal 2 * ENC_UNITS for keras.layers.Attention
EMBED_DIM = 128
DROPOUT = 0.3
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 12

PAD, OOV, START, END = "<pad>", "<unk>", "start", "end"


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def build_tokenizer(captions: list[str]) -> dict:
    """word_index with 0 reserved for padding and the literal tokens
    'start'/'end' present, because lstm_captioning.py looks them up by name."""
    vocab = sorted({w for c in captions for w in tokenize(c)})
    # START/END must not collide with a real caption word.
    for reserved in (START, END):
        if reserved in vocab:
            vocab.remove(reserved)
    words = [OOV, START, END] + vocab
    word_index = {w: i + 1 for i, w in enumerate(words)}  # 0 = padding
    return {
        "word_index": word_index,
        "index_word": {str(i): w for w, i in word_index.items()},
        "pad_token": PAD,
        "pad_index": 0,
        "oov_token": OOV,
        "start_token": START,
        "end_token": END,
        "max_length": MAX_LENGTH,
        "vocab_size": len(word_index) + 1,  # +1 for padding index 0
    }


def encode(caption: str, tok: dict) -> tuple[list[int], bool]:
    wi = tok["word_index"]
    ids = [wi[START]] + [wi.get(w, wi[OOV]) for w in tokenize(caption)] + [wi[END]]
    truncated = len(ids) > MAX_LENGTH
    ids = ids[:MAX_LENGTH] + [0] * max(0, MAX_LENGTH - len(ids))
    return ids, truncated


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(vocab_size: int, timesteps: int, feat_dim: int):
    import keras
    from keras import layers

    video = layers.Input(shape=(timesteps, feat_dim), name="video")
    tokens = layers.Input(shape=(MAX_LENGTH - 1,), dtype="int32", name="caption")

    enc = layers.Bidirectional(
        layers.LSTM(ENC_UNITS, return_sequences=True, dropout=DROPOUT), name="encoder"
    )(video)  # -> (T, 2*ENC_UNITS)

    emb = layers.Embedding(vocab_size, EMBED_DIM, mask_zero=False, name="embedding")(tokens)
    dec = layers.LSTM(DEC_UNITS, return_sequences=True, dropout=DROPOUT, name="decoder")(emb)

    # keras.layers.Attention requires query and value to share a last dim,
    # which is why DEC_UNITS == 2 * ENC_UNITS.
    ctx = layers.Attention(name="attention")([dec, enc])
    merged = layers.Concatenate(name="concat")([dec, ctx])
    merged = layers.Dropout(DROPOUT)(merged)
    out = layers.Dense(vocab_size, activation="softmax", name="output")(merged)

    return keras.Model([video, tokens], out, name="cctv_captioner")


def masked_loss_and_metric():
    import keras
    from keras import ops

    def loss_fn(y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        mask = ops.cast(ops.not_equal(y_true, 0), "float32")
        ce = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return ops.sum(ce * mask) / ops.maximum(ops.sum(mask), 1.0)

    def acc_fn(y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        mask = ops.cast(ops.not_equal(y_true, 0), "float32")
        pred = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
        hit = ops.cast(ops.equal(pred, y_true), "float32")
        return ops.sum(hit * mask) / ops.maximum(ops.sum(mask), 1.0)

    acc_fn.__name__ = "masked_acc"
    return loss_fn, acc_fn


def greedy_decode(model, video_feat: np.ndarray, tok: dict) -> str:
    """Autoregressive greedy decode for one clip."""
    wi, iw = tok["word_index"], tok["index_word"]
    start_id, end_id = wi[START], wi[END]
    seq = [start_id]
    words: list[str] = []
    for _ in range(MAX_LENGTH - 1):
        padded = np.array([seq + [0] * (MAX_LENGTH - 1 - len(seq))], dtype="int32")
        preds = model.predict([video_feat[None, ...], padded], verbose=0)
        nxt = int(np.argmax(preds[0, len(seq) - 1]))
        if nxt in (end_id, 0):
            break
        words.append(iw[str(nxt)])
        seq.append(nxt)
    return " ".join(words)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()

    for p in (CAPTIONS_PATH, FEATURES_PATH, INDEX_PATH):
        if not p.exists():
            sys.exit(f"missing {p}")

    import keras

    rng = np.random.default_rng(SEED)
    keras.utils.set_random_seed(SEED)

    # ---------------- load + align ----------------
    caps_doc = json.loads(CAPTIONS_PATH.read_text())["clips"]
    index = json.loads(INDEX_PATH.read_text())
    clip_ids: list[str] = index["clips"]
    X_raw = np.load(FEATURES_PATH)
    assert X_raw.shape[0] == len(clip_ids), "features/index length mismatch"

    captions = [caps_doc[c]["caption"] for c in clip_ids]
    categories = [caps_doc[c]["category"] for c in clip_ids]
    n_clips, timesteps, feat_dim = X_raw.shape
    print(f"features {X_raw.shape}  captions {len(captions)}  categories {len(set(categories))}")

    report: dict = {"n_clips": n_clips, "timesteps": timesteps, "raw_feature_dim": feat_dim}

    # ---------------- SVD ----------------
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    flat = X_raw.reshape(-1, feat_dim)
    print(f"\nfitting TruncatedSVD({N_COMPONENTS}) on {flat.shape} ...", flush=True)
    t0 = time.time()
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED, algorithm="randomized", n_iter=7)
    reduced = svd.fit_transform(flat)
    ev = float(svd.explained_variance_ratio_.sum())
    print(f"  done in {time.time() - t0:.0f}s | explained variance {ev * 100:.4f}%")
    report["svd"] = {"components": N_COMPONENTS, "explained_variance_ratio": ev,
                     "fit_seconds": round(time.time() - t0, 1)}

    scaler = StandardScaler().fit(reduced)
    X = scaler.transform(reduced).reshape(n_clips, timesteps, N_COMPONENTS).astype("float32")
    joblib.dump(svd, SVD_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  -> {SVD_PATH.name} ({SVD_PATH.stat().st_size / 1e6:.0f} MB), {SCALER_PATH.name}")
    del flat, reduced, X_raw

    # ---------------- tokenizer ----------------
    tok = build_tokenizer(captions)
    seqs, truncs = [], 0
    for c in captions:
        ids, tr = encode(c, tok)
        seqs.append(ids)
        truncs += tr
    seqs = np.asarray(seqs, dtype="int32")
    TOKENIZER_PATH.write_text(json.dumps(tok, indent=2))

    vocab_size = tok["vocab_size"]
    print(f"\ntokenizer: vocab {vocab_size} (incl. pad/unk/start/end)")
    print(f"  'start' -> {tok['word_index'][START]}   'end' -> {tok['word_index'][END]}")
    print(f"  captions truncated at MAX_LENGTH={MAX_LENGTH}: {truncs}/{len(captions)} "
          f"({100 * truncs / len(captions):.1f}%)")
    report["tokenizer"] = {"vocab_size": vocab_size, "max_length": MAX_LENGTH,
                           "truncated": truncs, "truncated_pct": round(100 * truncs / len(captions), 1),
                           "start_id": tok["word_index"][START], "end_id": tok["word_index"][END]}

    # ---------------- stratified split ----------------
    from sklearn.model_selection import train_test_split

    idx = np.arange(n_clips)
    tr_i, va_i = train_test_split(idx, test_size=VAL_FRACTION, random_state=SEED, stratify=categories)
    print(f"\nsplit: train {len(tr_i)} / val {len(va_i)} (stratified by category)")
    import collections
    vc = collections.Counter(categories[i] for i in va_i)
    print(f"  every category present in val: {len(vc) == len(set(categories))} ({dict(sorted(vc.items()))})")
    report["split"] = {"train": len(tr_i), "val": len(va_i), "val_per_category": dict(sorted(vc.items()))}

    dec_in, target = seqs[:, :-1], seqs[:, 1:]

    # ---------------- train ----------------
    loss_fn, acc_fn = masked_loss_and_metric()
    model = build_model(vocab_size, timesteps, N_COMPONENTS)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=[acc_fn])
    print(f"\nmodel params: {model.count_params():,}")

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    ]
    t0 = time.time()
    hist = model.fit(
        [X[tr_i], dec_in[tr_i]], target[tr_i],
        validation_data=([X[va_i], dec_in[va_i]], target[va_i]),
        epochs=args.epochs, batch_size=BATCH_SIZE, callbacks=cbs, verbose=2,
    )
    train_seconds = time.time() - t0

    best = int(np.argmin(hist.history["val_loss"]))
    res = {
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": best + 1,
        "train_loss": float(hist.history["loss"][best]),
        "val_loss": float(hist.history["val_loss"][best]),
        "train_acc": float(hist.history["masked_acc"][best]),
        "val_acc": float(hist.history["val_masked_acc"][best]),
        "train_seconds": round(train_seconds, 1),
        "params": int(model.count_params()),
    }
    report["training"] = res
    print(f"\nbest epoch {res['best_epoch']}/{res['epochs_run']}  "
          f"train_acc {res['train_acc']:.4f}  val_acc {res['val_acc']:.4f}  "
          f"(masked; padding excluded)")

    model.save(MODEL_PATH)
    print(f"  -> {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    # ---------------- qualitative check on held-out clips ----------------
    print("\n--- generated vs actual (5 held-out clips) ---")
    samples = rng.choice(va_i, size=min(5, len(va_i)), replace=False)
    pairs = []
    for i in samples:
        gen = greedy_decode(model, X[i], tok)
        pairs.append({"clip": clip_ids[i], "category": categories[i],
                      "actual": captions[i], "generated": gen})
        print(f"  [{categories[i]}] {clip_ids[i].split('/')[-1]}")
        print(f"     actual:    {captions[i]}")
        print(f"     generated: {gen}")
    report["samples"] = pairs

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
