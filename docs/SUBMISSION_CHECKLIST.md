# Submission checklist (Project A — Img2GPS)

Use this before every Hugging Face upload. Sources: **Project_Submission.pdf**, **CIS 4190_5190 Final Project Descriptions.pdf** (Spring 2026), and local **eval_project_a.py** behavior.

## Files to upload

- [x] `model.py` — importable; exposes `get_model()` **or** class **`Model`** or **`IMG2GPS`**, instantiable **with no required arguments**
- [x] `preprocess.py` — exposes `prepare_data(csv_path: str) -> (X, y)`
- [x] `model.pt` — if the model needs weights; `state_dict` keys must match `model.state_dict()` after instantiation

## `prepare_data` contract

- [x] Returns `(X, y)` where `X` is suitable for `model.predict(batch)` or `model(batch)`
- [x] **`y` is raw latitude/longitude in decimal degrees** (not normalized), aligned with `X`

## Model output contract

- [x] Backend calls `model.predict(batch)` if present; else `model(batch)`
- [x] Each prediction is **`[lat, lon]` in decimal degrees** (raw), comparable to CSV ground truth

## Normalization (if used in training)

- [x] Any z-score or other normalization of coordinates is **undone inside inference** so **exported predictions are in degrees**
- [x] Per course text: if you use normalization, **hard-code the statistics in `model.py`** (not only in a notebook)

## Weights

- [x] `model.pt` loads with the grader’s `torch.load` + `load_state_dict` (fix key prefixes if you trained with `DataParallel`, etc.)

## Dependencies

- [ ] Only use packages in the approved backend list, or document **Ed** approval in [BACKEND_ENV.md](BACKEND_ENV.md)

## Integrity

- [ ] No attempt to bypass the I/O contract; TAs can read submitted source

## Course baseline alignment (Final Project Descriptions §2.3)

- [x] Inputs resized to **224×224** (and normalized) in **`preprocess.py`** to match training
- [ ] Model performance **≥ course ResNet-18 baseline** expectation (full fine-tune, MSE, Adam 1e-3, ~10–15 epochs in doc); document if you deviate — see internal val ~**62 m** in [RESULTS.md](RESULTS.md); confirm vs course baseline when available
- [x] If targets were z-scored in training, **`predict`** still returns **raw degrees**

## Training integrity

- [x] **[`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img)** is **not** in the training split (staff discourage training on it; Ed #348)
- [x] Train is a **deduplicated** subset of [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) — published: [`prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train) (**N=2000**, **seed=51905**; see [DATA.md](DATA.md) / [SUBMISSION_LOG.md](SUBMISSION_LOG.md))
- [ ] **Official representative test sample** from the course PDF (§2.2), if separate from above, is **not** in training

## Local sanity check

- [x] `eval_project_a.py` completes on exported val — see [RUNBOOK.md](RUNBOOK.md) Phase D (`data/val_released_img/metadata.csv`, `--weights artifacts/model.pt`)

## Submission metadata

- [x] **Group ID** (**5**) recorded in [SUBMISSION_LOG.md](SUBMISSION_LOG.md)
- [ ] **Alias** and leaderboard rows filled after Phase E upload
- [x] Leaderboard metric is **mean Haversine distance (meters)** — lower is better
