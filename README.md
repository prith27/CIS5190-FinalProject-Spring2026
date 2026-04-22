# CIS 5190 — Img2GPS (Project A)

Predict **latitude and longitude** (decimal degrees) from an image. This repository holds **course submission code** (`model.py`, `preprocess.py`, `model.pt`), **documentation** for reproducibility and the final report, and pointers to **Hugging Face** artifacts.

**Course scope (Spring 2026):** Evaluation targets **Penn campus** imagery in the **rectangle from 33rd & Walnut to 34th & Spruce** ([CIS 4190_5190 Final Project Descriptions.pdf](CIS%204190_5190%20Final%20Project%20Descriptions.pdf) §2.1). Train on a **curated** dataset aligned with that region (including **public** geo-tagged sources); see [docs/DATA.md](docs/DATA.md).

## Status

| Phase | Status |
|-------|--------|
| A — Repo & docs | Complete |
| B — Data & HF Dataset | **Complete** — team train: [`prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train) — [docs/DATA.md](docs/DATA.md), [docs/SUBMISSION_LOG.md](docs/SUBMISSION_LOG.md) |
| C — Train on AWS | **Implemented** — partial-freeze ViT-B/16 training pipeline at [training/run_train_vit.py](training/run_train_vit.py) |
| D — `model.py` / `preprocess.py` + local eval | **Verified** — `eval_project_a.py` on exported `gydou/released_img` (~62.2 m); see [docs/RUNBOOK.md](docs/RUNBOOK.md), [docs/RESULTS.md](docs/RESULTS.md) |
| E — HF leaderboard | TBD |
| F — Report & optional video | TBD |

## Quick links

| Resource | Location |
|----------|----------|
| Data protocol & HF Dataset URL | [docs/DATA.md](docs/DATA.md) |
| **Team train dataset (Hub)** | [`prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train) |
| AWS EC2 setup | [docs/AWS_SETUP.md](docs/AWS_SETUP.md) |
| Commands (train, eval, upload) | [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Experiment log | [docs/RESULTS.md](docs/RESULTS.md) |
| HF submission / Group ID / Alias | [docs/SUBMISSION_LOG.md](docs/SUBMISSION_LOG.md) |
| Grader I/O checklist | [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md) |
| Environment parity | [docs/BACKEND_ENV.md](docs/BACKEND_ENV.md) |
| Report draft (5-page body) | [docs/REPORT_OUTLINE.md](docs/REPORT_OUTLINE.md) |
| Phase doc audits | [docs/DOC_REVIEW.md](docs/DOC_REVIEW.md) |
| Phase C training script | [training/run_train_vit.py](training/run_train_vit.py) |
| Submission model entrypoint | [model.py](model.py) |
| Submission preprocess entrypoint | [preprocess.py](preprocess.py) |

## Course resources (local copies)

- Baseline notebook: `project-resources/Img2GPS/Release_baseline_model.ipynb` — **not in Git** (too large); keep your copy from the course zip or strip outputs and track a lean version if the team agrees.
- EXIF → CSV: [project-resources/Img2GPS/Release_post_process.ipynb](project-resources/Img2GPS/Release_post_process.ipynb)
- Local evaluator: [project-resources/Img2GPS/eval_project_a.py](project-resources/Img2GPS/eval_project_a.py)
- Reference CSV: [project-resources/Img2GPS/reference/metadata.csv](project-resources/Img2GPS/reference/metadata.csv) — reference **images** are **gitignored**; add JPGs locally next to this CSV for eval.
- Templates: [project-resources/model_template.py](project-resources/model_template.py), [project-resources/preprocess_template.py](project-resources/preprocess_template.py)

## Artifacts

Store exported checkpoints under **`artifacts/`** (directory is **gitignored** except `artifacts/.gitkeep`; do not push weights or logs). Record `model.pt` **SHA-256** in [docs/SUBMISSION_LOG.md](docs/SUBMISSION_LOG.md). Produce weights with `training/run_train_vit.py` on EC2 and `scp` locally as needed.

## Reproduce local eval (after Phase D)

From the repo root, with `model.py`, `preprocess.py`, and `model.pt` in place (e.g. `artifacts/model.pt`):

```bash
python project-resources/Img2GPS/eval_project_a.py \
  --model model.py \
  --preprocess preprocess.py \
  --weights model.pt \
  --csv path/to/val/metadata.csv
```

## Non-goals

- **Inference does not read EXIF GPS** from the query image; the model predicts location from pixels only (training labels may come from EXIF or external metadata).

## Team (Group ID **5**)

| Name | Role (update as you divide work) |
|------|----------------------------------|
| **Prithvi Seshadri** | TBD |
| **Vamsi Krishna Naghichetty Kishore Kumar** | TBD |
| **Ishita Munshi** | TBD |

Use **Group ID `5`** on Hugging Face leaderboard submissions and anywhere the course form asks for it. Details also in [docs/SUBMISSION_LOG.md](docs/SUBMISSION_LOG.md).

## Dataset deliverable (“do we curate?”)

**Yes.** The course requires **collecting, curating, cleaning, and splitting** data you use ([Final Project Descriptions](CIS%204190_5190%20Final%20Project%20Descriptions.pdf) §1.2) and submitting **the dataset you used** (§1.5). **Curation of public imagery** (with clear protocol, license, and HF Dataset link) satisfies this; you do not have to be the photographer. Details: [docs/DATA.md](docs/DATA.md).

## License / data

- See [docs/DATA.md](docs/DATA.md) for dataset license and attribution.
