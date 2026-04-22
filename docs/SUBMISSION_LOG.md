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
| `model.pt` | Weights | TBD | Canonical checkpoint for leaderboard |
| `model.py` | Entry point | TBD | Git commit hash preferred |
| `preprocess.py` | `prepare_data` | TBD | |

## Leaderboard submissions

| datetime (UTC) | Alias | mean_haversine_m | outcome |
|----------------|-------|------------------|---------|
| TBD | TBD | TBD | success / error |

**Note:** Leaderboard may show **best per alias** — plan aliases (e.g. `teamname-v1-final`) accordingly.

## Git

- **Canonical commit for submission:** TBD
