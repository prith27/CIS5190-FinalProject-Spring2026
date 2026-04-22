# Results and experiment log

**Rule:** One table for all runs. Do not duplicate conflicting numbers elsewhere; link from README/report to this file.

**Internal val:** [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) (export to CSV for `eval_project_a.py` if needed). **Train:** published as [`prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train) (**N=2000**, **seed=51905**; parent [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset)) — log **aug policy** (`group5_randa_m7`; see [DATA.md](DATA.md)) in **notes** for each training run.

## Experiment table

| run_id | date (UTC) | env | instance_type | commit | epochs | batch | lr | val_avg_dist_m | val_mae_deg | val_rmse_deg | notes |
|--------|------------|-----|-----------------|--------|--------|-------|-----|----------------|-------------|--------------|--------|
| vit-b16-pf3-v1 | TBD | aws-ec2 | TBD | TBD | 20 | 32 | head=1e-3, backbone=1e-5 | TBD | TBD | TBD | vit_b_16 partial-freeze (unfreeze_blocks=3), group5_randa_m7, AMP, AdamW+CosineAnneal |

**env:** `local` or `aws-ec2`  
**val_*:** from `eval_project_a.py` on your validation CSV unless noted

## Leaderboard (Hugging Face)

| alias | date (UTC) | mean_haversine_m | notes |
|-------|------------|------------------|--------|
| TBD | TBD | TBD | |

## Figures

TBD: link or path to learning curves, error maps (for report appendix).
