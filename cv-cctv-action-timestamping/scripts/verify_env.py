"""
Stage 2 environment smoke test.

Verifies that every dependency the rebuilt pipeline needs is importable and
actually functional -- not just installed. Also probes the dataset and the test
videos so Stage 4 can be designed against real numbers instead of guesses.

Backend note: TensorFlow's native extension crashes on this machine
(macOS 26.4.1 / arm64) with "mutex lock failed: Invalid argument" on import,
on both 2.20.0 and 2.21.0. We therefore run Keras 3 on its PyTorch backend,
which keeps the .keras artifact format and adds MPS acceleration.

Run:
    KERAS_BACKEND=torch ./.venv/bin/python -u scripts/verify_env.py
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
# Keras' LSTM uses an orthogonal initializer, which needs linalg_qr -- an op MPS
# does not implement. Without this fallback, building any LSTM raises
# NotImplementedError. It only affects unimplemented ops; the rest still runs on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path("/Users/rithishselvam/Downloads/archive/Videos/Videos")
TEST_VIDEO_DIR = PROJECT_ROOT / "Test Videos"

CLIPS_PER_CATEGORY = 20  # balanced sample -> 13 x 20 = 260 clips
FRAMES_PER_CLIP = 16  # sampled timesteps per clip (Stage 4)

results = {}
failures = []


def check(name):
    """Run a check, record pass/fail, and keep going on failure."""

    def wrap(fn):
        print(f"\n{'=' * 62}\n{name}\n{'=' * 62}", flush=True)
        try:
            out = fn()
            results[name] = out
            print("  PASS", flush=True)
        except Exception as exc:  # noqa: BLE001 - we want every failure, not the first
            failures.append((name, repr(exc)))
            results[name] = f"FAILED: {exc!r}"
            print(f"  FAIL: {exc!r}", flush=True)
        return fn

    return wrap


@check("1. Interpreter and platform")
def _interpreter():
    print(f"  python    {sys.version.split()[0]}")
    print(f"  platform  {platform.platform()}")
    print(f"  machine   {platform.machine()}")
    assert sys.version_info[:2] == (3, 11), "expected Python 3.11"
    return {"python": sys.version.split()[0], "platform": platform.platform()}


@check("2. Core imports")
def _imports():
    import cv2
    import joblib
    import keras
    import numpy as np
    import sklearn
    import streamlit
    import torch

    versions = {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "keras": keras.__version__,
        "keras_backend": keras.backend.backend(),
        "torch": torch.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "streamlit": streamlit.__version__,
    }
    for k, v in versions.items():
        print(f"  {k:<15} {v}")
    assert keras.backend.backend() == "torch", "KERAS_BACKEND must be torch"
    return versions


@check("3. TensorFlow is absent (it crashes on this OS)")
def _no_tf():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("  tensorflow not installed -- correct; Keras runs on torch instead")
        return "absent"
    raise AssertionError("tensorflow is installed and will crash on import")


@check("4. Compute device")
def _device():
    import torch

    mps = torch.backends.mps.is_available()
    print(f"  MPS (Apple GPU) available: {mps}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    device = "mps" if mps else "cpu"
    print(f"  -> Stage 4 will extract features on: {device}")
    return {"mps": mps, "device": device}


@check("5. VGG16 loads and yields the expected feature width")
def _vgg16():
    import numpy as np
    from keras.applications.vgg16 import VGG16

    t0 = time.time()
    base = VGG16(weights="imagenet", include_top=False)
    load_s = time.time() - t0

    import keras

    dummy = np.zeros((2, 224, 224, 3), dtype="float32")
    # On the torch backend the result is an MPS tensor; np.asarray() on it raises.
    # keras.ops.convert_to_numpy handles the device copy for us.
    out = keras.ops.convert_to_numpy(base(dummy, training=False))
    flat = int(np.prod(out.shape[1:]))

    print(f"  block5_pool output  {out.shape}  -> flattened {flat}")
    print(f"  weight load         {load_s:.1f}s")
    assert flat == 25088, f"expected 25088, got {flat}"
    return {"flat_dim": flat, "load_s": round(load_s, 1)}


@check("6. VGG16 throughput (sizes the Stage 4 extraction run)")
def _throughput():
    import numpy as np
    import torch
    from keras.applications.vgg16 import VGG16

    base = VGG16(weights="imagenet", include_top=False)
    batch = np.random.rand(16, 224, 224, 3).astype("float32")

    base(batch, training=False)  # warm up
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    t0 = time.time()
    for _ in range(3):
        base(batch, training=False)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    per_frame = (time.time() - t0) / 48

    fps = 1.0 / per_frame
    total_frames = CLIPS_PER_CATEGORY * 13 * FRAMES_PER_CLIP
    est_min = total_frames / fps / 60
    print(f"  {fps:.1f} frames/sec at batch=16")
    print(f"  => {total_frames} frames (260 clips x {FRAMES_PER_CLIP}) ~= {est_min:.1f} min")
    return {"frames_per_sec": round(fps, 1), "est_extract_min": round(est_min, 1)}


@check("7. A BiLSTM + attention model builds and saves as .keras")
def _keras_model():
    import numpy as np
    from keras import layers, models

    # Miniature stand-in for the Stage 5 architecture, to prove the format works.
    vid = layers.Input(shape=(FRAMES_PER_CLIP, 64), name="video")
    cap = layers.Input(shape=(None,), dtype="int32", name="caption")
    # Bidirectional doubles the width (32 -> 64), and keras.layers.Attention
    # requires query and value to share a last dimension, so the decoder LSTM
    # must be 64 units too. Stage 5's real model has to honour the same rule.
    enc = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(vid)
    emb = layers.Embedding(50, 32)(cap)
    dec = layers.LSTM(64, return_sequences=True)(emb)
    att = layers.Attention()([dec, enc])
    out = layers.Dense(50, activation="softmax")(layers.Concatenate()([dec, att]))
    model = models.Model([vid, cap], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    tmp = PROJECT_ROOT / "scripts" / "_smoke.keras"
    model.save(tmp)
    size_kb = tmp.stat().st_size / 1024
    reloaded = models.load_model(tmp)
    tmp.unlink()

    x = [np.zeros((1, FRAMES_PER_CLIP, 64), "float32"), np.zeros((1, 5), "int32")]
    pred = np.asarray(reloaded.predict(x, verbose=0))
    print(f"  built, saved ({size_kb:.0f} KB), reloaded; predict -> {pred.shape}")
    assert pred.shape == (1, 5, 50)
    return {"params": int(model.count_params()), "saved_kb": round(size_kb)}


@check("8. sklearn SVD + scaler survive a joblib round-trip")
def _sklearn_roundtrip():
    import joblib
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.random((200, 600)).astype("float32")

    svd = TruncatedSVD(n_components=50, random_state=0).fit(X)
    Xr = svd.transform(X)
    scaler = StandardScaler().fit(Xr)

    tmp = PROJECT_ROOT / "scripts" / "_smoke.pkl"
    joblib.dump({"svd": svd, "scaler": scaler}, tmp)
    back = joblib.load(tmp)
    tmp.unlink()

    assert np.allclose(back["svd"].transform(X), Xr)
    print(f"  round-trip exact; explained variance on random data {svd.explained_variance_ratio_.sum():.4f}")
    return "ok"


@check("9. Semantic search works without sentence-transformers")
def _model2vec():
    import numpy as np
    from model2vec import StaticModel

    t0 = time.time()
    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    print(f"  loaded potion-base-8M in {time.time() - t0:.1f}s")

    texts = [
        "two people fighting in a corridor",
        "a person sitting calmly on a bench",
        "someone falls to the ground",
    ]
    emb = model.encode(texts)
    q = model.encode(["a brawl breaks out"])[0]

    emb_n = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sims = emb_n @ (q / np.linalg.norm(q))
    for t, s in zip(texts, sims):
        print(f"  {s:+.3f}  {t}")
    print(f"  embedding dim {emb.shape[1]}")

    assert sims[0] > sims[1], "'brawl' must rank fighting above sitting calmly"
    return {"dim": int(emb.shape[1]), "fight_vs_sit": [float(sims[0]), float(sims[1])]}


@check("10. Dataset is readable; balanced 20/category sample is viable")
def _dataset():
    import cv2
    import numpy as np

    assert DATASET_DIR.is_dir(), f"dataset not found at {DATASET_DIR}"
    cats = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    counts = {c: len(list((DATASET_DIR / c).glob("*.mp4"))) for c in cats}
    print(f"  {len(cats)} categories, {sum(counts.values())} clips total")
    for c in cats:
        flag = "" if counts[c] >= CLIPS_PER_CATEGORY else "  <-- TOO FEW"
        print(f"    {c:<12} {counts[c]:>4}{flag}")

    short = [c for c in cats if counts[c] < CLIPS_PER_CATEGORY]
    assert not short, f"categories with fewer than {CLIPS_PER_CATEGORY} clips: {short}"
    print(f"  balanced sample of {CLIPS_PER_CATEGORY}/category = {CLIPS_PER_CATEGORY * len(cats)} clips: OK")

    # Probe the clips we would actually select (deterministic: first N by name).
    frames, fps_list, sizes, unreadable = [], [], [], []
    for c in cats:
        for p in sorted((DATASET_DIR / c).glob("*.mp4"))[:3]:
            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            f = cap.get(cv2.CAP_PROP_FPS)
            w, h = int(cap.get(3)), int(cap.get(4))
            ok, _ = cap.read()
            cap.release()
            if not ok or n <= 0:
                unreadable.append(p.name)
                continue
            frames.append(n)
            fps_list.append(f)
            sizes.append((w, h))

    frames = np.array(frames)
    print(f"\n  probed {len(frames)} clips, {len(unreadable)} unreadable")
    if unreadable:
        print(f"    unreadable: {unreadable[:5]}")
    print(f"  frame count  min={frames.min()} med={int(np.median(frames))} max={frames.max()}")
    print(f"  fps          {sorted({round(f, 1) for f in fps_list})}")
    print(f"  resolutions  {sorted({f'{w}x{h}' for w, h in sizes})}")
    print(f"  duration     med={np.median(frames) / np.median(fps_list):.1f}s")
    print(f"\n  FRAMES_PER_CLIP={FRAMES_PER_CLIP} vs shortest clip {frames.min()} frames"
          f" -> {'OK' if frames.min() >= FRAMES_PER_CLIP else 'must pad short clips'}")

    return {
        "categories": counts,
        "total_clips": sum(counts.values()),
        "frames_min": int(frames.min()),
        "frames_median": int(np.median(frames)),
        "frames_max": int(frames.max()),
        "unreadable_probed": len(unreadable),
        "resolutions": sorted({f"{w}x{h}" for w, h in sizes}),
    }


@check("11. Filename -> label parsing (Stage 3 depends on this)")
def _filenames():
    cats = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())
    sources, mismatches = {}, []
    for c in cats:
        for p in (DATASET_DIR / c).glob("*.mp4"):
            src = p.name.split("_")[0]
            sources[src] = sources.get(src, 0) + 1
            # Category can itself contain "_" (lying_down), so match on the
            # directory name rather than splitting into fixed fields.
            if f"_{c}_" not in p.name:
                mismatches.append(p.name)
    print(f"  sources: {sources}")
    print(f"  filenames whose folder label is not embedded in the name: {len(mismatches)}")
    if mismatches:
        print(f"    e.g. {mismatches[:5]}")
    return {"sources": sources, "label_mismatches": len(mismatches)}


@check("12. Bundled test videos are readable")
def _test_videos():
    import cv2

    out = {}
    for p in sorted(TEST_VIDEO_DIR.glob("*.mp4")):
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f = cap.get(cv2.CAP_PROP_FPS) or 0
        w, h = int(cap.get(3)), int(cap.get(4))
        cap.release()
        dur = n / f if f else 0
        print(f"  {p.name:<28} {n:>6} frames  {f:>5.1f} fps  {w}x{h}  {dur:>6.1f}s")
        out[p.name] = {"frames": n, "fps": round(f, 2), "size": f"{w}x{h}", "seconds": round(dur, 1)}
    assert out, "no test videos found"
    return out


print(f"\n\n{'=' * 62}\nSUMMARY\n{'=' * 62}")
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for name, err in failures:
        print(f"  - {name}: {err}")
else:
    print("All checks passed.")

report = PROJECT_ROOT / "scripts" / "env_report.json"
report.write_text(json.dumps(results, indent=2, default=str))
print(f"\nMachine-readable report -> {report}")
sys.exit(1 if failures else 0)
