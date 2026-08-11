"""
Stage 4 guard -- prove features.npy row i really is the clip features_index.json
says it is.

A silent misalignment between features and captions would train the model on
mismatched pairs. It would not raise, would not look wrong in any shape check,
and would only show up as inexplicably bad captions much later. So we verify it
directly: pick random rows, re-extract those clips from the source video, and
require the arrays to match.

Run:
    ./.venv/bin/python -u scripts/verify_alignment.py [--n 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import DATASET_DIR, clip_features  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feats = np.load(PROJECT_ROOT / "features.npy")
    index = json.loads((PROJECT_ROOT / "features_index.json").read_text())
    caps = json.loads((PROJECT_ROOT / "captions.json").read_text())["clips"]
    clip_ids = index["clips"]

    print(f"features {feats.shape} | index {len(clip_ids)} clips | captions {len(caps)}")
    if feats.shape[0] != len(clip_ids):
        print(f"FAIL: features has {feats.shape[0]} rows but index lists {len(clip_ids)}")
        return 1

    missing = [c for c in clip_ids if c not in caps]
    if missing:
        print(f"FAIL: {len(missing)} indexed clips have no caption, e.g. {missing[:3]}")
        return 1
    print("every indexed clip has a caption: OK")

    from keras.applications.vgg16 import VGG16, preprocess_input

    vgg = VGG16(weights="imagenet", include_top=False)

    rng = np.random.default_rng(args.seed)
    rows = rng.choice(len(clip_ids), size=min(args.n, len(clip_ids)), replace=False)

    ok = bad = 0
    worst = 0.0
    print(f"\nre-extracting {len(rows)} random rows and comparing:")
    for r in rows:
        clip_id = clip_ids[int(r)]
        fresh = clip_features(DATASET_DIR / clip_id, vgg, preprocess_input)
        stored = feats[int(r)]
        # Exact equality is expected (deterministic pipeline); allow a hair of
        # float slack in case the GPU scheduler reorders a reduction.
        max_abs = float(np.max(np.abs(fresh - stored)))
        rel = max_abs / max(1e-6, float(np.max(np.abs(stored))))
        match = rel < 1e-4
        worst = max(worst, rel)
        ok, bad = (ok + 1, bad) if match else (ok, bad + 1)
        status = "OK  " if match else "FAIL"
        print(f"  {status} row {int(r):>4}  rel_err {rel:.2e}  {caps[clip_id]['category']:<11} "
              f"{clip_id.split('/')[-1][:44]}")
        if not match:
            print(f"        stored mean {stored.mean():.4f} vs fresh {fresh.mean():.4f}")

    print(f"\n{ok}/{len(rows)} rows verified, worst relative error {worst:.2e}")
    if bad:
        print("ALIGNMENT IS BROKEN -- do not train on this array.")
        return 1

    # Cross-check: category implied by the clip path must equal the stored
    # caption's category, for every row (cheap, catches index scrambling).
    wrong = [c for c in clip_ids if c.split("/")[0] != caps[c]["category"]]
    if wrong:
        print(f"FAIL: {len(wrong)} clips whose folder disagrees with caption category")
        return 1
    print("folder/category cross-check over all rows: OK")
    print("\nALIGNMENT VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
