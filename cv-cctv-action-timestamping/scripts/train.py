"""
Stage 5 -- fit the transforms and train the captioning model.

Produces all four missing artifacts:
    svd.pkl           TruncatedSVD, 25120 -> 256 components
    scaler.pkl        StandardScaler fitted on the SVD output
    tokenizer.json    word_index containing literal lowercase 'start' / 'end'
    lstm_model.keras  BiLSTM encoder + attention decoder

Two deliberate choices worth knowing about:

* **Masked accuracy.** Padding tokens are excluded from both loss and accuracy.
  Counting them is the standard way to manufacture a 95%+ "accuracy" that means
  nothing -- most timesteps in a padded batch are padding. Every number this
  script reports is over real tokens only.

* **MAX_LENGTH = 14.** Captions run up to 14 words, so start + words + end
  needs 14 slots to avoid truncating the tail of a caption. (It was 11
  originally; that silently truncated 42.2% of the set.)

* **N_COMPONENTS = 256.** 1500 components against 780 clips overfitted badly:
  validation action accuracy 11.5% vs 34.0% at 256, despite *higher* explained
  variance (93.47% vs 70.32%). The discarded variance was per-clip appearance
  detail the model was memorising.

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

N_COMPONENTS = 256
MAX_LENGTH = 14  # total tokens per caption, including 'start' and 'end'
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


def _identity_projection():
    """Stands in for TruncatedSVD when --no-svd is used.

    Keeps the four-artifact contract intact: inference always calls
    svd.transform(...) then scaler.transform(...), so a no-op projection means
    lstm_captioning.py needs no special case.

    This is sklearn's own FunctionTransformer rather than a custom class on
    purpose. A hand-rolled class gets pickled as `__main__.<Name>`, which then
    fails to unpickle from any *other* entry point -- including the Streamlit
    app. FunctionTransformer lives in sklearn, so it loads anywhere sklearn is
    installed.
    """
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(feature_names_out="one-to-one")


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

    # return_state so the decoder can be *initialised* from the video, not just
    # attend to it afterwards. Previously the decoder LSTM started from a zero
    # state and video only reached it through Attention applied AFTER the
    # recurrence, so video never entered the decoder's dynamics at all -- the
    # most likely reason it could not exploit better features.
    enc_seq, f_h, f_c, b_h, b_c = layers.Bidirectional(
        layers.LSTM(ENC_UNITS, return_sequences=True, return_state=True, dropout=DROPOUT),
        name="encoder",
    )(video)  # enc_seq -> (T, 2*ENC_UNITS)

    # Concatenating the forward and backward states gives exactly DEC_UNITS
    # (= 2 * ENC_UNITS), so it drops straight into the decoder's initial state.
    state_h = layers.Concatenate(name="state_h")([f_h, b_h])
    state_c = layers.Concatenate(name="state_c")([f_c, b_c])

    emb = layers.Embedding(vocab_size, EMBED_DIM, mask_zero=False, name="embedding")(tokens)
    dec = layers.LSTM(DEC_UNITS, return_sequences=True, dropout=DROPOUT, name="decoder")(
        emb, initial_state=[state_h, state_c]
    )

    # keras.layers.Attention requires query and value to share a last dim,
    # which is why DEC_UNITS == 2 * ENC_UNITS.
    ctx = layers.Attention(name="attention")([dec, enc_seq])
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
    global MAX_LENGTH, N_COMPONENTS, SVD_PATH, SCALER_PATH, TOKENIZER_PATH, MODEL_PATH, REPORT_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--max-length", type=int, default=None,
                    help="override MAX_LENGTH; pair with --outdir for comparison runs")
    ap.add_argument("--components", type=int, default=None,
                    help="override the SVD component count")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="write artifacts here instead of the project root (for comparisons)")
    ap.add_argument("--features", type=Path, default=None,
                    help="alternative features .npy (its index is <stem>_index.json)")
    ap.add_argument("--no-svd", action="store_true",
                    help="skip TruncatedSVD entirely; scale the raw features instead")
    args = ap.parse_args()

    if args.max_length:
        MAX_LENGTH = args.max_length
    if args.components:
        N_COMPONENTS = args.components

    global FEATURES_PATH, INDEX_PATH
    if args.features:
        FEATURES_PATH = args.features
        stem = args.features.stem
        cand = args.features.with_name(f"{stem}_index.json")
        INDEX_PATH = cand if cand.exists() else args.features.with_name("features_index.json")
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        SVD_PATH = args.outdir / "svd.pkl"
        SCALER_PATH = args.outdir / "scaler.pkl"
        TOKENIZER_PATH = args.outdir / "tokenizer.json"
        MODEL_PATH = args.outdir / "lstm_model.keras"
        REPORT_PATH = args.outdir / "train_report.json"

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
    if args.no_svd:
        # Identity "SVD" so the artifact contract and the inference path stay
        # unchanged: lstm_captioning always does svd.transform then
        # scaler.transform. With a compact backbone the projection may simply
        # not be needed, which is what --no-svd tests.
        print(f"\nSKIPPING SVD (--no-svd): scaling raw {feat_dim}-d features", flush=True)
        svd = _identity_projection().fit(flat[:1])
        reduced = flat
        n_out = feat_dim
        report["svd"] = {"components": None, "skipped": True, "raw_dim": feat_dim}
    else:
        print(f"\nfitting TruncatedSVD({N_COMPONENTS}) on {flat.shape} ...", flush=True)
        t0 = time.time()
        svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED,
                           algorithm="randomized", n_iter=7)
        reduced = svd.fit_transform(flat)
        ev = float(svd.explained_variance_ratio_.sum())
        n_out = N_COMPONENTS
        print(f"  done in {time.time() - t0:.0f}s | explained variance {ev * 100:.4f}%")
        report["svd"] = {"components": N_COMPONENTS, "explained_variance_ratio": ev,
                         "fit_seconds": round(time.time() - t0, 1)}

    scaler = StandardScaler().fit(reduced)
    X = scaler.transform(reduced).reshape(n_clips, timesteps, n_out).astype("float32")
    joblib.dump(svd, SVD_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  -> {SVD_PATH.name} ({SVD_PATH.stat().st_size / 1e6:.1f} MB), {SCALER_PATH.name}")
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
    model = build_model(vocab_size, timesteps, n_out)
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
