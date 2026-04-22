# Hugging Face submission log

## Team

- **Group ID:** **5** (use on all Hugging Face leaderboard submissions)
- **Course:** CIS 5190 — Spring 2026 (confirm term on Canvas if needed)

**Members**

| Name |
|------|
| Prithvi Seshadri |
| Vamsi Krishna Naghichetty Kishore Kumar |
| Ishita Munshi |

## Hugging Face Dataset (team data)

- **Team train subset (publish this):** [`prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train) — **2k** rows; **seed** `51905`; **dedup** SHA-256 PNG vs [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img); parent [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset). Built with [`scripts/build_group5_train_dataset.py`](../scripts/build_group5_train_dataset.py) on EC2. **Cite the parent** on the dataset card; **released_img not in train.**
- **Revision / commit:** (fill after push) dataset repo `main`; Git commit that matches export script: TBD

### Reference datasets (do not confuse with team repo)

| Dataset | URL | Use for Group 5 |
|---------|-----|-----------------|
| Released sample / **val** | [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) | **Validation only** (~100); **not** in training (Ed #348) |
| Train **pool** | [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) | Subset → **your** HF train dataset after dedup |

## Artifacts

| File | Purpose | SHA-256 | Notes |
|------|---------|---------|--------|
| `model.pt` | Weights | `3473d56ec0986c22bfc72fa97addf5428ed6f0a0a71c2d021e4a196f227c29a9` | vit-b16-pf3-v1; best val **62.176 m** (`gydou/released_img`). File is **gitignored**; keep a copy locally / upload to HF per course workflow. |
| `model.py` | Entry point | `29fa0e4` | ViT-B/16 partial-freeze; `LAT_*` / `LON_*` match `artifacts/norm_stats.json`. |
| `preprocess.py` | `prepare_data` | `29fa0e4` | Eval transforms aligned with `group5_randa_m7` eval path. |

## Leaderboard submissions

| datetime (UTC) | Alias | mean_haversine_m | outcome |
|----------------|-------|------------------|---------|
| TBD | TBD | TBD | success / error |

**Note:** Leaderboard may show **best per alias** — plan aliases (e.g. `teamname-v1-final`) accordingly.

## Git

- **Canonical commit for submission:** `29fa0e4` (update if you amend after further edits)
- **Phase C training script commit (`training/run_train_vit.py`):** same tree as `29fa0e4` unless changed
