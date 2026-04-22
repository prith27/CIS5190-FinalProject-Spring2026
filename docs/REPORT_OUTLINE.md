# Report outline (CIS5190_FinalProj_Template.pdf)

**Report title line (template):** `[Title]` — **Project:** Img2GPS — **Team / Group ID:** 5

**Team members (for title page / §4):**

- Prithvi Seshadri  
- Vamsi Krishna Naghichetty Kishore Kumar  
- Ishita Munshi  

**Target:** Main body **≤ 5 pages** excluding references. Important findings must be in the main body; appendix optional for extra plots/proofs.

Paste draft paragraphs below each heading as phases complete.

---

## 1. Introduction

TBD (< 1 page): project highlights, model approach, key findings (core + exploratory).

---

## 2. Core Components

### 2.1 Data

Address **Final Project Descriptions** §2.1–2.2 and §2.4, plus **Ed #348**:

- **Validation:** [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) (~100) — **val / internal eval only**; **not** used in training.
- **Training pool / published train:** Subset of [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) — team Hub: [`prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train) (**N=2000**, **seed=51905**, **dedup** vs val; optional bbox filter not used).
- **Evaluation region:** Penn campus rectangle **33rd & Walnut** to **34th & Spruce** (Figure 1).
- **Augmentation:** train-time only — **RandAugment** (moderate magnitude) + **RandomErasing** + conservative **RandomResizedCrop** + flip + light **GaussianBlur**; val **Resize(256) → CenterCrop(224)** + ImageNet norm (see [DATA.md](DATA.md)).
- **Leakage / integrity:** document dedup; staff may use a **different** hidden test set for leaderboard.
- **License & attribution** for parent `5190-image-dataset`.

Draft paragraphs TBD below.

### 2.2 Model

Final model is a `torchvision` ViT-B/16 backbone with pretrained ImageNet initialization and the classification head replaced by a 2D regression layer (`Linear(768, 2)`) for normalized latitude/longitude prediction. We use partial fine-tuning to avoid overfitting on the 2k-image train split: freeze patch embedding + early/mid transformer blocks and only unfreeze the last 3 encoder blocks (`encoder_layer_9..11`), `encoder.ln`, and the regression head. This keeps strong pretrained visual features while adapting high-level representations to Penn campus localization cues.

Training objective is MSE on z-scored targets (`lat`, `lon`) using train-set normalization stats computed from `prith27/cis5190-group5-train`. We use AdamW with differential learning rates (head `1e-3`, unfrozen backbone `1e-5`), cosine annealing, and AMP for efficient EC2 GPU training. Checkpointing is done every epoch (`latest_checkpoint.pt`) with best-by-val-Haversine tracking (`best_model.pt`) and final state export to `model.pt`. At inference time, `model.py` denormalizes predictions back to raw decimal degrees as required by the grader contract.

### 2.3 Evaluation

Internal evaluation uses `gydou/released_img` as a fixed validation split (never used in training) with deterministic preprocessing: Resize(256) -> CenterCrop(224) -> ImageNet normalization. We report mean Haversine distance in meters as the primary metric, with MAE/RMSE in degrees as secondary diagnostics for debugging optimization behavior. Best checkpoint selection is based on validation mean Haversine, not train loss, to reduce overfitting risk.

For local reproducibility before leaderboard submission, we run `project-resources/Img2GPS/eval_project_a.py` against `model.py`, `preprocess.py`, and exported `model.pt`. The final candidate is selected from best validation Haversine among ablations (primarily `--unfreeze-blocks` and batch/learning-rate adjustments), then recorded in `docs/RESULTS.md` and `docs/SUBMISSION_LOG.md` with run metadata and artifact hash.

---

## 3. Exploratory Components

For **each** exploratory track, use the rubric type that applies (Code / Technique / Analysis / Teaching / Application / Datasets / Comparison). Remove unused types.

### 3.1 Datasets (recommended): augmentation / train expansion

TBD: default **group5_randa_m7** vs simpler ColorJitter baseline vs **TrivialAugmentWide** swap (see `training/augmentation.py`); optional added real images in sparse map cells; **val** always [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img); report Haversine **before/after**; cite staff exploratory note.

### 3.2 [Optional second exploratory]

TBD

---

## 4. Team Contributions

One sentence per member (data, model, AWS, HF, report, exploratory, etc.):

- **Prithvi Seshadri:** TBD  
- **Vamsi Krishna Naghichetty Kishore Kumar:** TBD  
- **Ishita Munshi:** TBD  

---

## 5. Extra Credit (optional video)

TBD: link to video (max 10 minutes; **all** team members must appear for full extra credit).

---

## 6. Acknowledgments (optional, not in page limit)

TBD

---

## 7. Suggestions for future iterations (optional)

TBD

---

## 8. Appendix (optional — plots, extra tables)

TBD: point to [RESULTS.md](RESULTS.md) and figures on disk.

---

## References

TBD: IM2GPS, PlaNet, Transloc4D, etc., as appropriate.
