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
- **Training pool:** [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset); **your** published subset of size **N** with fixed **seed**, **dedup** vs val, optional bbox filter; **HF link** to **team** dataset repo.
- **Evaluation region:** Penn campus rectangle **33rd & Walnut** to **34th & Spruce** (Figure 1).
- **Augmentation:** train-time only — **RandAugment** (moderate magnitude) + **RandomErasing** + conservative **RandomResizedCrop** + flip + light **GaussianBlur**; val **Resize(256) → CenterCrop(224)** + ImageNet norm (see [DATA.md](DATA.md)).
- **Leakage / integrity:** document dedup; staff may use a **different** hidden test set for leaderboard.
- **License & attribution** for parent `5190-image-dataset`.

Draft paragraphs TBD below.

### 2.2 Model

TBD: design considerations, iterative improvements, final architecture (backbone, head, losses, normalization).

### 2.3 Evaluation

TBD: internal evaluation protocol, **leaderboard** protocol, metrics (mean Haversine m, MAE/RMSE if used), how final submission was chosen.

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
