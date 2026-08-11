# Deploying to Streamlit Community Cloud

Click-by-click. Everything that can be prepared in advance already has been —
what's left needs your browser and your GitHub account.

---

## Before you start

`main` is already pushed. The repo carries ~100 MB of video and model files;
every individual file is under GitHub's 100 MB limit (largest is
`Test Videos/Random_test_video1.mp4` at 62 MB), so no Git LFS is needed. You may
see GitHub's "large file" *warning* for anything over 50 MB — a warning, not a
rejection.

`requirements.txt` lives at the **repository root**. Streamlit Community Cloud
will not find it anywhere else.

There is deliberately **no `packages.txt`**. It isn't needed —
`opencv-python-headless` links `libgthread` (from glib, already present in the
base image) rather than `libGL`. Adding one also carries a trap: Streamlit
passes every line of `packages.txt` straight to `apt`, so `#` comment lines are
treated as package names and the build fails with `Unable to locate package the`.
If you ever do need one, it must be bare package names only, one per line.

---

## Deploy

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Authorise Streamlit to read your repositories if prompted.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in exactly:

   | Field | Value |
   |---|---|
   | **Repository** | `rithishss/CCTV-Footage-Captioning-and-Timestamp-Forwarding` |
   | **Branch** | `main` |
   | **Main file path** | `cv-cctv-action-timestamping/Codebase/app.py` |
   | **App URL** | your choice — e.g. `cctv-footage-search` |

   The main file path is the one people get wrong. It is **not** `app.py` and
   **not** `Codebase/app.py` — the project sits in a nested directory.

5. Click **Advanced settings** and set **Python version** to **3.11**.
   The pins in `requirements.txt` are verified against 3.11; 3.12+ may resolve
   different wheels.
6. Click **Deploy**.

---

## What to expect

**First build takes 10–20 minutes.** It is installing PyTorch, Keras and
Transformers. This is normal — watch the log pane rather than assuming it hung.

**First app load takes another 1–2 minutes**, because it downloads the VideoMAE
weights (~350 MB) from HuggingFace and caches them. Subsequent loads are fast.

**First search on a video takes ~20 seconds** for the 49-second sample: 24
windows at roughly 0.6 s each. The progress panel names each stage. Results are
cached per video, so searching again is instant.

To sanity-check the deployment: press **Try the sample video**, wait for
*"Done — 24 segments captioned"*, then click the **somebody running** chip. It
should return one result around **0:08–0:14**. That is the query verified to
land a correct top-1 hit.

---

## If the build succeeds but the app crashes on `import cv2`

**This already happened once. The cause was not OpenCV.**

`requirements.txt` was at `cv-cctv-action-timestamping/requirements.txt`, but
Streamlit Community Cloud looks for it at the **repository root**. Finding no
dependency file, it installed nothing, the build reported success, and the app
died on the first third-party import — which happens to be `cv2` in
`segmentation.py`.

The error type is the giveaway:

| Error | Means |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | the package was never installed — check where `requirements.txt` lives |
| `ImportError: libGL.so.1: cannot open shared object file` | package installed, **system** library missing — would need a `packages.txt` |

`requirements.txt` now sits at the repo root.
`cv-cctv-action-timestamping/requirements.txt` remains as a one-line
`-r ../requirements.txt` pointer so local installs from inside the project
directory still work.

## If the build fails

Work down this list.

**`ERROR: Could not find a version that satisfies the requirement torch==2.9.1+cpu`**
The `--extra-index-url` line at the top of `requirements.txt` was dropped or
reordered. It must remain the first non-comment line. If Streamlit's installer
ignores it, replace the two torch lines with a plain `torch==2.9.1` — the build
gets much larger but will resolve.

**`ImportError: libGL.so.1: cannot open shared object file`**
Something pulled in `opencv-python` instead of `opencv-python-headless` — check
nothing added it transitively, the headless build is deliberate. Only if that is
genuinely the case would you add a `packages.txt` at the repo root containing
exactly two lines, `libgl1` and `libglib2.0-0`, and nothing else: Streamlit feeds
every line to `apt`, so a comment line becomes a package name and fails the
build.

**Build succeeds, app crashes with `FileNotFoundError` naming an artifact**
One of `lstm_model.keras` / `tokenizer.json` / `scaler.pkl` / `svd.pkl` didn't
reach GitHub. Verify with:

```bash
git ls-files cv-cctv-action-timestamping/ | grep -E "keras|pkl|tokenizer"
```

All four must be listed. The app shows a clear error naming the missing file
rather than a stack trace, so the message tells you which one.

**App boots but "Sample video not found in this deployment"**
`Test Videos/demo_concat.mp4` wasn't pushed. Confirm with
`git ls-files "cv-cctv-action-timestamping/Test Videos"`.

**"Oh no. Error running app" with no useful log**
Almost always memory. Streamlit Community Cloud gives about 1 GB, and VideoMAE
(~350 MB) plus Keras plus the LSTM is close to it. Options, cheapest first:
reboot the app from the ⋮ menu; remove `Random_test_video1.mp4` and
`MalAct_test_video.mp4` from the repo to shrink the image; or, if it recurs,
switch the backbone to `torchvision`'s `r2plus1d_18`, which is roughly a third
the size (this costs accuracy and means retraining).

**Timeout during model download on first load**
Reboot the app. The HuggingFace cache usually survives and the second attempt
completes.

---

## After it's live

Add the URL to the repo's About section and to the top of `README.md`.

If you want the app to sleep less often, Streamlit Community Cloud wakes apps on
visit — the first visitor after idle waits through the model load again. Nothing
to configure; just know that a cold first click is slow.
