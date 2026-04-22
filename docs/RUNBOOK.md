# Runbook (copy-paste commands)

All commands assume repository root: `CIS5190_FinalProj/` (adjust paths if your layout differs).

## Course training defaults (Img2GPS §2.3)

Document your exact settings in [RESULTS.md](RESULTS.md):

- Image size **224×224**, normalized inputs
- **Lat/lon** normalized in training with dataset mean/std; **inverse transform** for metrics and submission
- Baseline reference: **ResNet-18**, 2-D output, **MSE**, **Adam** `lr=1e-3`, LR scheduler, **10–15 epochs** (your runs may differ if justified)

**Evaluation region** for the course leaderboard is **Penn campus** (see [DATA.md](DATA.md)); internal val uses **[`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img)** (see [DATA.md](DATA.md) Phase B).

### Augmentation (Phase B — training only)

**Policy:** RandAugment + RandomErasing + conservative RandomResizedCrop + optional Gaussian blur — see [DATA.md](DATA.md) § *Augmentation (training only — Group 5 policy)*.

```python
from training.augmentation import build_train_transforms, build_eval_transforms

train_tfms = build_train_transforms()   # random — use only on train split
eval_tfms = build_eval_transforms()     # val, released_img, preprocess.py alignment
```

### Build local train export (subset + dedup)

Requires `datasets` + `huggingface_hub` (see [requirements.txt](../requirements.txt)). From repo root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_group5_train_dataset.py --n 1500 --seed 51905 --out data/group5_train
```

**Where files go:** `data/group5_train/images/*.jpg`, `data/group5_train/metadata.csv`, `data/group5_train/build_manifest.json`. Each image **appends** a row to `metadata.csv` as it is written (with flush), so a **partial** run still has **labels** for every file on disk — you can build a Hub push from that folder without re-streaming the parent dataset.

**If the folder looks empty in Cursor/VS Code:** everything under `data/*` is **gitignored**, so some editors **hide** those paths in the file tree. The data is still on disk — confirm in Terminal: `ls -la data/group5_train/images | head`. You can toggle “exclude Git Ignore” / show ignored files in the explorer if your editor supports it.

**Disk space (`OSError: [Errno 28] No space left on device`):** By default the builder uses **`streaming=True`** for the train pool so it does **not** materialize a multi‑GB Arrow cache. If you still see **Errno 28**, free space on the volume that holds `~/.cache/huggingface/` (or set `HF_HOME` / `HF_DATASETS_CACHE` to a larger disk), and remove any **partial** dataset build under that cache. Only use **`--materialize`** if you have plenty of free disk and want the non-streaming code path.

**First run still downloads image bytes** over the network when streaming; let the command **finish**. If you interrupt it, delete the broken output dir and run again.

**Process ends with `Killed` (no Python traceback):** Usually **out of memory**. The builder streams train rows and should **write each JPEG as soon as a row is accepted** (so 2000 images are not all held in RAM). Still use a modest **`--shuffle-buffer-size`** (e.g. **512**), add **swap** on small instances, or upgrade RAM. `ps` showing **~15 GiB** for one Python process often meant an **older** script version that buffered all PIL images — **git pull** and re-run. Confirm OOM with `dmesg -T | tail -20`.

Dedup rule: **SHA-256 of lossless RGB PNG bytes** per image; drop any train row whose hash appears in **`gydou/released_img`**. Document counts in the script’s `build_manifest.json`.

### Push train dataset to Hugging Face Hub

1. **Create the dataset repo on Hugging Face** (one-time): [https://huggingface.co/new-dataset](https://huggingface.co/new-dataset) — choose an owner (your user or org) and a name, e.g. `cis5190-group5-train`. You can start **private** and switch to public later if the license allows.

2. **Authenticate** (pick one):
   - **CLI:** `hf auth login` (new) — or legacy `huggingface-cli login` if your `huggingface_hub` version still supports it — paste a **write** token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). If you see a deprecation notice, use **`hf`**.
   - **Env:** `export HF_TOKEN=hf_...` in the same shell before running the script.

3. **Build + push in one go** (after `metadata.csv` and `images/` exist locally, or run the same command to rebuild and push):

```bash
source .venv/bin/activate
python scripts/build_group5_train_dataset.py \
  --n 1500 \
  --seed 51905 \
  --out data/group5_train \
  --push-to-hub YOUR_USERNAME/cis5190-group5-train
```

Use **`--private`** if the Hub repo should stay private: append `--private` to the command above.

4. **Record the URL** in [SUBMISSION_LOG.md](SUBMISSION_LOG.md). **Group 5 (published):** [`https://huggingface.co/datasets/prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train). The dataset card should cite the parent [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) and state that **`gydou/released_img` is not in train** (see [DATA.md](DATA.md)).

### Load Hub datasets (Phase B)

```python
from datasets import load_dataset

val_ds = load_dataset("gydou/released_img", split="train")  # ~100; use as val only
full_train_pool = load_dataset("heidiywseo/5190-image-dataset", split="train")
# Subset + dedup vs val per docs/DATA.md; then train with aug on train only
```

## Environment (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## AWS EC2 (training)

See [AWS_SETUP.md](AWS_SETUP.md) for launch and SSH.

- Git commit hash used on the instance: **TBD** (run `git rev-parse --short HEAD` on EC2 in the repo you trained from)
- Training command: [Phase C — ViT partial-freeze training](#phase-c--vit-partial-freeze-training) below
- Artifacts on the instance (example repo folder `~/CIS5190-FinalProject-Spring2026`): `artifacts/model.pt`, `artifacts/norm_stats.json`, `artifacts/train_log.txt`, `artifacts/best_model.pt`

## Phase C — ViT partial-freeze training

Run from repo root on EC2. The training script prepends the repo root to `sys.path` so `python training/run_train_vit.py` can import `training.augmentation` (plain `python training/...` otherwise only adds the `training/` folder to the path and breaks that import).

```bash
source .venv/bin/activate
python training/run_train_vit.py \
  --hf-train prith27/cis5190-group5-train \
  --hf-val gydou/released_img \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-3 \
  --backbone-lr 1e-5 \
  --unfreeze-blocks 3 \
  --amp \
  --output-dir artifacts \
  --num-workers 4 2>&1 | tee artifacts/train_log.txt
```

Resume after interruption:

```bash
source .venv/bin/activate
python training/run_train_vit.py \
  --hf-train prith27/cis5190-group5-train \
  --hf-val gydou/released_img \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-3 \
  --backbone-lr 1e-5 \
  --unfreeze-blocks 3 \
  --amp \
  --output-dir artifacts \
  --num-workers 4 \
  --resume artifacts/latest_checkpoint.pt
```

Training script startup prints train label normalization stats to copy into `model.py`:

```text
LAT_MEAN=...
LAT_STD=...
LON_MEAN=...
LON_STD=...
```

Try `--unfreeze-blocks 0`, `3`, and `6` for quick ablations (record in RESULTS).

## After training — copy artifacts to your laptop

On EC2, outputs are under **`<REPO>/artifacts/`** (not on your Mac until you copy them). Replace `REPO` with your clone path (e.g. `CIS5190_FinalProject-Spring2026` or `CIS5190_FinalProj`).

**Git:** `artifacts/*` is gitignored (only `artifacts/.gitkeep` is tracked). Do not commit weights or logs; record `model.pt` SHA-256 in [SUBMISSION_LOG.md](SUBMISSION_LOG.md).

**On your laptop** (from the directory where you keep the project):

```bash
mkdir -p artifacts

scp -i ~/.ssh/cis5190_aws \
  ubuntu@<PUBLIC_IP_OR_DNS>:~/<REPO>/artifacts/model.pt \
  ubuntu@<PUBLIC_IP_OR_DNS>:~/<REPO>/artifacts/norm_stats.json \
  ubuntu@<PUBLIC_IP_OR_DNS>:~/<REPO>/artifacts/train_log.txt \
  ubuntu@<PUBLIC_IP_OR_DNS>:~/<REPO>/artifacts/best_model.pt \
  ./artifacts/
```

Then:

1. Open `artifacts/norm_stats.json` (or the `LAT_MEAN=...` lines at the top of `train_log.txt`) and paste the four numbers into **`model.py`** (`LAT_MEAN`, `LAT_STD`, `LON_MEAN`, `LON_STD`).
2. Put `model.pt` in the repo root if you want the eval command as documented: `cp artifacts/model.pt ./model.pt` (optional).
3. Record `shasum -a 256 model.pt` (or `artifacts/model.pt`) in [SUBMISSION_LOG.md](SUBMISSION_LOG.md).
4. Commit and push `model.py`, `preprocess.py`, docs, and (if your team tracks weights in Git — many use LFS or omit) `model.pt`.

## Export checkpoint (reference)

Canonical output paths from the training script:

- `artifacts/latest_checkpoint.pt` — every epoch, full training state (resume)
- `artifacts/best_model.pt` — best val Haversine
- `artifacts/model.pt` — final **state_dict** for submission (loaded from best weights at end of run)

Checksum after copy:

```bash
shasum -a 256 artifacts/model.pt
```

## Phase D — Local grader parity (`eval_project_a.py`)

**1. Export** [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) to `data/val_released_img/` (gitignored with other `data/*`):

```bash
source .venv/bin/activate
pip install -r requirements.txt
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$(pwd)/.hf_cache/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$(pwd)/.hf_cache/hub}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
python scripts/export_released_img_for_eval.py --out data/val_released_img
```

**2. Run** the course-style evaluator (weights path may be `artifacts/model.pt`):

```bash
source .venv/bin/activate
python project-resources/Img2GPS/eval_project_a.py \
  --model model.py \
  --preprocess preprocess.py \
  --weights artifacts/model.pt \
  --csv data/val_released_img/metadata.csv \
  --batch-size 32
```

Expected output includes: `avg_distance_m`, `mae (deg)`, `rmse (deg)`, `avg_infer_ms`. Log numbers in [RESULTS.md](RESULTS.md).

**Note:** On macOS, if `torch==2.9.1` is not available for your Python, install the closest supported `torch` / `torchvision` pair for local smoke tests; EC2/grader should follow [BACKEND_ENV.md](BACKEND_ENV.md).

## Phase E — Hugging Face leaderboard

The **exact** leaderboard URL, upload widget, and any **CLI** are defined by the course (Canvas, **Project_Submission.pdf**, staff announcement). Use those as the source of truth; the checklist below is generic.

### Files to have ready (Project A)

| File | Ready? | Notes |
|------|--------|--------|
| `model.py` | Yes | `get_model()`, `Model.predict` → raw degrees; norms in file match training. |
| `preprocess.py` | Yes | `prepare_data(csv_path)` → `(X, y)` with `y` in raw degrees. |
| `model.pt` | Yes | Same checkpoint you evaluated; path often `artifacts/model.pt` locally. |

**Before upload:** run Phase D [eval](#phase-d--local-grader-parity-eval_project_a) one more time on `data/val_released_img` so the three files on disk are the set you mean to submit.

### Upload (typical course flow)

1. Log in to [Hugging Face](https://huggingface.co/) (team account or designated uploader).
2. Open the **course leaderboard / submission** page from Canvas or the PDF.
3. Enter **Group ID `5`** and your chosen **Alias** (often **best score is kept per alias** — pick a stable final name, e.g. `group5-img2gps-v1`).
4. Upload **`model.py`**, **`preprocess.py`**, **`model.pt`** (names must match what the grader expects — usually exactly these).
5. Submit and wait for the run to finish; note **mean Haversine (m)** when it appears.

**Optional (CLI):** If staff publish a Hub repo or `huggingface-cli` / `hf` upload pattern, follow their snippet; otherwise use the web UI.

### After submission — log and verify

- Record **UTC time**, **alias**, **mean_haversine_m**, and **success/error** in [SUBMISSION_LOG.md](SUBMISSION_LOG.md) and the leaderboard table in [RESULTS.md](RESULTS.md).
- If the run **fails**, read the error carefully: common issues are wrong file names, `model.pt` not loading (key mismatch), timeout, or missing dependency (must match [BACKEND_ENV.md](BACKEND_ENV.md)).
- Re-download or re-export submitted `model.py` from Git if you fix bugs; **recompute `model.pt` SHA** if you change weights.

### What to check if something breaks

- [ ] `model.py` and `preprocess.py` are the same versions you tested with `eval_project_a.py`.
- [ ] `LAT_MEAN`, `LAT_STD`, `LON_MEAN`, `LON_STD` match the training run that produced **`model.pt`**.
- [ ] `model.pt` is the **full** ViT `state_dict` (not an empty or wrong checkpoint).
- [ ] File size / upload completed (no truncated upload).
- [ ] Only approved packages in submitted code paths ([BACKEND_ENV.md](BACKEND_ENV.md)).
