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

**Where files go:** `data/group5_train/images/*.jpg`, `data/group5_train/metadata.csv`, `data/group5_train/build_manifest.json`.

**If the folder looks empty in Cursor/VS Code:** everything under `data/*` is **gitignored**, so some editors **hide** those paths in the file tree. The data is still on disk — confirm in Terminal: `ls -la data/group5_train/images | head`. You can toggle “exclude Git Ignore” / show ignored files in the explorer if your editor supports it.

**Disk space (`OSError: [Errno 28] No space left on device`):** By default the builder uses **`streaming=True`** for the train pool so it does **not** materialize a multi‑GB Arrow cache. If you still see **Errno 28**, free space on the volume that holds `~/.cache/huggingface/` (or set `HF_HOME` / `HF_DATASETS_CACHE` to a larger disk), and remove any **partial** dataset build under that cache. Only use **`--materialize`** if you have plenty of free disk and want the non-streaming code path.

**First run still downloads image bytes** over the network when streaming; let the command **finish**. If you interrupt it, delete the broken output dir and run again.

**Process ends with `Killed` (no Python traceback):** Usually the **Linux OOM killer** — streaming shuffle + image decode can use a lot of RAM. Try a **smaller** shuffle buffer, e.g. `python scripts/build_group5_train_dataset.py ... --shuffle-buffer-size 512`, or add **swap** on the instance (`sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`), or use an instance with **more memory**. Confirm with `dmesg -T | tail -20` (look for “Out of memory”).

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

4. **Record the URL** in [SUBMISSION_LOG.md](SUBMISSION_LOG.md) (e.g. `https://huggingface.co/datasets/YOUR_USERNAME/cis5190-group5-train`) and add a dataset card on the Hub citing the parent [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) and stating **`gydou/released_img` is not in train** (see [DATA.md](DATA.md)).

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

See [AWS_SETUP.md](AWS_SETUP.md) for launch and SSH. After Phase C, fill in:

- Git commit hash used on the instance: TBD
- Exact training command or notebook cells: TBD
- How `model.pt` was copied off the instance: TBD

## Export checkpoint

TBD: path to canonical `model.pt` and how it was produced.

## Local evaluation (Project A)

```bash
python project-resources/Img2GPS/eval_project_a.py \
  --model model.py \
  --preprocess preprocess.py \
  --weights model.pt \
  --csv path/to/val/metadata.csv \
  --batch-size 32
```

Expected output includes: `avg_distance_m`, `mae (deg)`, `rmse (deg)`, `avg_infer_ms`.

## Hugging Face leaderboard

TBD: exact UI steps or CLI for uploading `model.py`, `preprocess.py`, `model.pt` once course provides the workflow.

Record **Group ID** (**5**) and **Alias** in [SUBMISSION_LOG.md](SUBMISSION_LOG.md).
