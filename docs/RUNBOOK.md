# Runbook (copy-paste commands)

All commands assume repository root: `CIS5190_FinalProj/` (adjust paths if your layout differs).

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
