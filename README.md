# CCTV Footage Search

Search surveillance video by **describing what you're looking for**.

Type *"someone falls to the ground"* and get back the moments in the footage
where that seems to happen, ranked, each with a timestamp you can jump to. It
matches meaning rather than keywords, so *"someone falls over"* also surfaces a
clip captioned *"a person lies on the ground"*.

**Who it's for:** anyone with hours of surveillance footage and no practical way
to find the ninety seconds that matter. Reviewing CCTV is mostly scrubbing, and
this narrows where to look.

---

## How it works

Three stages, in order:

**1 · Segmentation** — the video is cut into **4-second windows overlapping by
50%**. Four seconds because that is the median length of the clips the model was
trained on; the overlap means an action crossing a boundary is still fully
contained in at least one window.

**2 · Captioning** — each window goes through
[VideoMAE](https://huggingface.co/MCG-NJU/videomae-base), a spatiotemporal
encoder pretrained on Kinetics-400, producing an 8×1536 descriptor per window.
A BiLSTM encoder with an attention decoder then writes one short sentence
describing it. The decoder's initial state comes from the encoder, so the video
shapes the sentence from the first word rather than only being consulted at the
end.

**3 · Semantic search** — the query and every generated caption are embedded
with [model2vec](https://github.com/MinishLab/model2vec) static embeddings and
ranked by cosine similarity. Overlapping matches are merged so one real event
doesn't appear three times.

---

## Results, honestly

Measured on 156 held-out clips the model never saw during training.

| Metric | Value | Baseline |
|---|---|---|
| **Names the correct action** | **42.3%** | 7.7% (chance across 13 categories) |
| Masked token accuracy | 76.1% | — |
| Exact caption match | 10.3% | — |

**42.3% means roughly three in five captions name the wrong action.** That is
5.5× better than guessing and genuinely useful for narrowing a search, but it is
not a detector. Treat results as candidate moments worth reviewing.

Scene descriptions — indoors/outdoors, street/shop/corridor, day/night — are
noticeably more reliable than the action verbs.

### On the bundled demo video

A 49-second clip assembled from 13 source videos **the model never saw at any
stage** (not trained on, not even captioned), with ground-truth time ranges
recorded in `Test Videos/demo_concat_truth.json`. Searching all 13 actions:

| | Rate |
|---|---|
| Correct moment ranked **first** | 1 / 13 |
| Correct moment **in the top three** | 7 / 13 |

That gap is why the app shows a ranked shortlist of five rather than one answer:
scanning three or four candidates is realistic; trusting the first is not.
Continuous footage is also harder than the held-out clips — see Limitations.

### The finding that mattered most

A linear probe (plain logistic regression predicting the action category
directly from the features) turned out to be the most useful diagnostic in the
project, because it separates *"the features don't contain the signal"* from
*"the model isn't using the signal"*:

| Features | Probe ceiling | Caption model |
|---|---|---|
| VGG16 frames + FFT motion bins | 30.8% | 35.9% |
| **VideoMAE** | **47.4%** | **42.3%** |

The original pipeline used VGG16 — an ImageNet **image** model applied to frames
independently. Nothing in its 25,088 dimensions encodes motion, and the
hand-designed FFT bins were a crude substitute. Swapping in VideoMAE, whose
tube-masking pretraining forces genuine temporal modelling, raised the ceiling by
**16.6 points**.

But the first VideoMAE attempt scored only 36.5% — it had moved the ceiling
without reaching it. The fix was architectural: the decoder LSTM had been
starting from a zero state, with video reaching it only through attention applied
*after* the recurrence. Initialising the decoder from the encoder took it to
42.3%. Both halves were needed; neither alone was enough.

---

## Limitations

- **Thirteen actions only:** `fall` `grab` `gun` `hit` `kick` `lying_down` `run`
  `sit` `sneak` `stand` `struggle` `throw` `walk`. Anything else gets described
  with the closest of these words.
- **Trained on isolated clips.** Every training clip was 3–4 seconds, trimmed,
  containing exactly one action. Continuous footage is harder: actions start
  mid-window, overlap, and sit inside long stretches of nothing. Expect
  noticeably weaker results than 42.3% suggests.
- **Small training set** — 780 clips, 60 per category. More data is the most
  obvious remaining lever.
- **`lying_down` is the weakest category** (1/12 on validation).
- **Not real time.** Roughly 0.6 s of compute per 4-second window, so a
  one-minute video takes about 20 seconds.
- **Uploads are capped** at 100 MB and 5 minutes.

---

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/rithishss/CCTV-Footage-Captioning-and-Timestamp-Forwarding.git
cd CCTV-Footage-Captioning-and-Timestamp-Forwarding/cv-cctv-action-timestamping

python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

./.venv/bin/python -m streamlit run Codebase/app.py
```

Then open http://localhost:8501 and press **Try the sample video**. The four
model artifacts are committed, so nothing needs training to run the app.

> **Note on TensorFlow.** This project runs **Keras 3 on its PyTorch backend**.
> TensorFlow's native extension aborts on import on macOS 26 (`mutex lock
> failed: Invalid argument`, on both 2.20.0 and 2.21.0). Keras 3 is
> backend-agnostic, so the `.keras` model format is unaffected. `KERAS_BACKEND`
> is set in code — no shell export needed.

### Reproducing the model

Only needed if you want to retrain. Requires the
[CCTV Action Recognition dataset](https://www.kaggle.com/datasets/jonathannield/cctv-action-recognition-dataset)
and an Anthropic API key for the captioning step.

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt

export ANTHROPIC_API_KEY=sk-ant-...
./.venv/bin/python -u scripts/make_captions.py              # 780 captions, ~7 min, ~$3
./.venv/bin/python -u scripts/extract_features_videomae.py  # ~7.5 min
./.venv/bin/python -u scripts/train.py --features features_videomae.npy --no-svd
./.venv/bin/python -u scripts/evaluate.py                   # action accuracy
./.venv/bin/python -u scripts/demo_query_sweep.py           # end-to-end search test
```

---

## Repository layout

```
Codebase/
  app.py                  Streamlit front end
  segmentation.py         windowing + CaptionSegment (the timestamps live here)
  feature_extraction.py   VideoMAE backbone
  lstm_captioning.py      caption a video -> list[CaptionSegment]
  timestamping.py         semantic search -> ranked SearchHits
  image_enhancement.py    unused ZeroDCE (see below)
scripts/                  data prep, training, evaluation, tests
Test Videos/              sample footage + the demo clip and its ground truth
lstm_model.keras  tokenizer.json  scaler.pkl  svd.pkl   the trained artifacts
```

---

Built by Rithish S · [github.com/rithishss](https://github.com/rithishss)
