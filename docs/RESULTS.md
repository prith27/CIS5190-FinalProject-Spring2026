# Results and experiment log

**Rule:** One table for all runs. Do not duplicate conflicting numbers elsewhere; link from README/report to this file.

**Internal val:** [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) (export to CSV for `eval_project_a.py` if needed). **Train:** published as [`prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train) (**N=2000**, **seed=51905**; parent [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset)) — log **aug policy** (`group5_randa_m7`; see [DATA.md](DATA.md)) in **notes** for each training run.

## Experiment table

| run_id | date (UTC) | env | instance_type | commit | epochs | batch | lr | val_avg_dist_m | val_mae_deg | val_rmse_deg | notes |
|--------|------------|-----|-----------------|--------|--------|-------|-----|----------------|-------------|--------------|--------|
| vit-b16-pf3-v1 | 2026-04-22 | aws-ec2 | TBD | 29fa0e4 | 20 | 32 | head=1e-3, backbone=1e-5 | **62.176** | 0.000404 | 0.000541 | Best val Haversine at **epoch 15** (62.176 m on `gydou/released_img`); exported `artifacts/model.pt` from best checkpoint. Epoch 20 val: 62.675 m. ViT-B/16, `unfreeze_blocks=3`, `group5_randa_m7`, AMP, AdamW + CosineAnneal. |

**env:** `local` or `aws-ec2`  
**val_*:** from `eval_project_a.py` on your validation CSV unless noted

### Latest completed EC2 run (internal val: `gydou/released_img`)

- **Best val Haversine:** **62.176 m** at **epoch 15** (`train_log.txt` / training script best checkpoint).
- **Epoch 20 val Haversine:** 62.675 m (cosine LR ~0).
- **Label norm stats:** `artifacts/norm_stats.json` (same numbers in `model.py`); train Hub `prith27/cis5190-group5-train`.
- **Weights:** `artifacts/model.pt` (gitignored locally; SHA-256 in [SUBMISSION_LOG.md](SUBMISSION_LOG.md)).
- **Fill in:** EC2 `instance_type` if you want the table complete; optional `eval_project_a.py` on an exported val CSV for a second sanity metric.

## Leaderboard (Hugging Face)

| alias | date (UTC) | mean_haversine_m | notes |
|-------|------------|------------------|--------|
| TBD | TBD | TBD | |

## Figures

TBD: link or path to learning curves, error maps (for report appendix).
