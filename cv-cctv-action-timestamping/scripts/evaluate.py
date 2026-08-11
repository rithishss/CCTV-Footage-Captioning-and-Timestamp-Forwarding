"""
Stage 5 evaluation -- measure what actually matters for timestamp search.

Masked token accuracy is a weak proxy: most tokens in these captions are
function words and setting/lighting boilerplate that a decoder can predict from
the language model alone. A caption can score well on tokens while naming the
wrong action, which is precisely the failure that breaks search.

So this script decodes every validation clip and asks a sharper question:
**does the generated caption contain the correct action verb for its category?**

Run:
    ./.venv/bin/python -u scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Surface forms the model may emit for each category's action.
ACTION_WORDS = {
    "fall": {"falls", "fall", "lies", "lie"},
    "grab": {"grabs", "grab", "holds", "hold"},
    "gun": {"gun", "points", "brandishes", "brandish"},
    "hit": {"hits", "hit", "strikes", "strike"},
    "kick": {"kicks", "kick"},
    "lying_down": {"lies", "lie", "lying"},
    "run": {"runs", "run"},
    "sit": {"sits", "sit"},
    "sneak": {"sneaks", "sneak", "crouches"},
    "stand": {"stands", "stand"},
    "struggle": {"struggle", "struggles", "grapple"},
    "throw": {"throws", "throw"},
    "walk": {"walks", "walk"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "scripts" / "eval_report.json")
    args = ap.parse_args()

    import keras
    import train as T  # reuse greedy_decode / tokenize

    tok = json.loads((args.artifacts / "tokenizer.json").read_text())
    T.MAX_LENGTH = tok["max_length"]
    model = keras.models.load_model(args.artifacts / "lstm_model.keras", compile=False)
    svd = joblib.load(args.artifacts / "svd.pkl")
    scaler = joblib.load(args.artifacts / "scaler.pkl")

    caps = json.loads((PROJECT_ROOT / "captions.json").read_text())["clips"]
    clip_ids = json.loads((PROJECT_ROOT / "features_index.json").read_text())["clips"]
    X_raw = np.load(PROJECT_ROOT / "features.npy")
    categories = [caps[c]["category"] for c in clip_ids]

    # Reproduce the exact split used in training.
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(clip_ids))
    tr_i, va_i = train_test_split(idx, test_size=T.VAL_FRACTION,
                                  random_state=T.SEED, stratify=categories)

    n, timesteps, feat_dim = X_raw.shape

    def project(rows):
        flat = X_raw[rows].reshape(-1, feat_dim)
        z = scaler.transform(svd.transform(flat))
        return z.reshape(len(rows), timesteps, -1).astype("float32")

    results = {}
    for split_name, rows in (("val", va_i), ("train", tr_i[: len(va_i)])):
        Z = project(rows)
        hits = exact = 0
        per_cat = collections.Counter()
        per_cat_total = collections.Counter()
        records = []
        for k, r in enumerate(rows):
            gen = T.greedy_decode(model, Z[k], tok)
            cat = categories[r]
            actual = caps[clip_ids[r]]["caption"]
            words = set(gen.lower().split())
            hit = bool(words & ACTION_WORDS.get(cat, set()))
            hits += hit
            per_cat[cat] += hit
            per_cat_total[cat] += 1
            exact += gen.strip().lower() == actual.strip().lower().rstrip(".")
            records.append({"clip": clip_ids[r], "category": cat,
                            "actual": actual, "generated": gen, "action_hit": hit})
        acc = hits / len(rows)
        results[split_name] = {
            "n": len(rows),
            "action_accuracy": round(acc, 4),
            "exact_match": round(exact / len(rows), 4),
            "per_category": {c: round(per_cat[c] / per_cat_total[c], 3) for c in sorted(per_cat_total)},
            "records": records,
        }
        print(f"\n=== {split_name} (n={len(rows)}) ===")
        print(f"  action-word accuracy : {acc * 100:.1f}%   (chance ~= {100 / 13:.1f}%)")
        print(f"  exact caption match  : {exact / len(rows) * 100:.1f}%")
        print("  per category:")
        for c in sorted(per_cat_total):
            bar = "#" * int(20 * per_cat[c] / per_cat_total[c])
            print(f"    {c:<11} {per_cat[c]:>2}/{per_cat_total[c]:<3} {bar}")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
