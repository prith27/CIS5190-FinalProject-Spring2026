# Submission checklist (Project A — Img2GPS)

Use this before every Hugging Face upload. Source: **Project_Submission.pdf** and local **eval_project_a.py** behavior.

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

## Local sanity check

- [ ] `python project-resources/Img2GPS/eval_project_a.py --model model.py --preprocess preprocess.py --weights model.pt --csv <val.csv>` completes without error

## Submission metadata

- [ ] **Group ID** (**5**) and **Alias** recorded in [SUBMISSION_LOG.md](SUBMISSION_LOG.md)
- [ ] Leaderboard metric is **mean Haversine distance (meters)** — lower is better
