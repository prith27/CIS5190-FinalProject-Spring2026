# Results and experiment log

**Rule:** One table for all runs. Do not duplicate conflicting numbers elsewhere; link from README/report to this file.

**Internal val:** [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) (export to CSV for `eval_project_a.py` if needed). **Train:** subset of [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) — log **N**, **seed**, and **aug policy** (`group5_randa_m7` = RandAugment `num_ops=2` `magnitude=7` + RandomErasing + crop/flip/blur; see [DATA.md](DATA.md)) in **notes**.

## Experiment table

| run_id | date (UTC) | env | instance_type | commit | epochs | batch | lr | val_avg_dist_m | val_mae_deg | val_rmse_deg | notes |
|--------|------------|-----|-----------------|--------|--------|-------|-----|----------------|-------------|--------------|--------|
| — | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |

**env:** `local` or `aws-ec2`  
**val_*:** from `eval_project_a.py` on your validation CSV unless noted

## Leaderboard (Hugging Face)

| alias | date (UTC) | mean_haversine_m | notes |
|-------|------------|------------------|--------|
| TBD | TBD | TBD | |

## Figures

TBD: link or path to learning curves, error maps (for report appendix).
