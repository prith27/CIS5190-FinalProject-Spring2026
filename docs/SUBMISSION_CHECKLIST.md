# Submission checklist (Project A — Img2GPS)

Use this before every Hugging Face upload. Sources: **Project_Submission.pdf**, **CIS 4190_5190 Final Project Descriptions.pdf** (Spring 2026), and local **eval_project_a.py** behavior.

## Files to upload

- [ ] `model.py` — importable; exposes `get_model()` **or** class **`Model`** or **`IMG2GPS`**, instantiable **with no required arguments**
- [ ] `preprocess.py` — exposes `prepare_data(csv_path: str) -> (X, y)`
- [ ] `model.pt` — if the model needs weights; `state_dict` keys must match `model.state_dict()` after instantiation

## `prepare_data` contract

- [ ] Returns `(X, y)` where `X` is suitable for `model.predict(batch)` or `model(batch)`
- [ ] **`y` is raw latitude/longitude in decimal degrees** (not normalized), aligned with `X`

## Model output contract

- [ ] Backend calls `model.predict(batch)` if present; else `model(batch)`
- [ ] Each prediction is **`[lat, lon]` in decimal degrees** (raw), comparable to CSV ground truth

## Normalization (if used in training)

- [ ] Any z-score or other normalization of coordinates is **undone inside inference** so **exported predictions are in degrees**
- [ ] Per course text: if you use normalization, **hard-code the statistics in `model.py`** (not only in a notebook)

## Weights

- [ ] `model.pt` loads with the grader’s `torch.load` + `load_state_dict` (fix key prefixes if you trained with `DataParallel`, etc.)

## Dependencies

- [ ] Only use packages in the approved backend list, or document **Ed** approval in [BACKEND_ENV.md](BACKEND_ENV.md)

## Integrity

- [ ] No attempt to bypass the I/O contract; TAs can read submitted source

## Course baseline alignment (Final Project Descriptions §2.3)

- [ ] Inputs resized to **224×224** (and normalized) in **`preprocess.py`** to match training
- [ ] Model performance **≥ course ResNet-18 baseline** expectation (full fine-tune, MSE, Adam 1e-3, ~10–15 epochs in doc); document if you deviate
- [ ] If targets were z-scored in training, **`predict`** still returns **raw degrees**

## Training integrity

- [ ] **[`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img)** is **not** in the training split (staff discourage training on it; Ed #348)
- [ ] Train is a **deduplicated** subset of [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) (document **seed** and **N**)
- [ ] **Official representative test sample** from the course PDF (§2.2), if separate from above, is **not** in training

## Local sanity check

- [ ] `python project-resources/Img2GPS/eval_project_a.py --model model.py --preprocess preprocess.py --weights model.pt --csv <val.csv>` completes without error

## Submission metadata

- [ ] **Group ID** (**5**) and **Alias** recorded in [SUBMISSION_LOG.md](SUBMISSION_LOG.md)
- [ ] Leaderboard metric is **mean Haversine distance (meters)** — lower is better
