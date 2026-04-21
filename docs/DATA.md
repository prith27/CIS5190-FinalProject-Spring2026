# Data protocol and Hugging Face Dataset

**Course references:** [CIS 4190_5190 Final Project Descriptions.pdf](../CIS%204190_5190%20Final%20Project%20Descriptions.pdf) (Spring 2026), [Project_Submission.pdf](../Project_Submission.pdf).

---

## Phase B plan — Group 5 (train / val / augmentation)

This section is the **active Phase B** recipe. It aligns with Ed **Image2GPS data collection #348**: use **`gydou/released_img`** for **internal validation only** (staff **discourage** putting it in **training**); staff also confirmed **online** imagery is allowed (**Yue Wang**).

### Validation (fixed — do not train on this)

| Item | Value |
|------|--------|
| **Hub dataset** | [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) |
| **Role** | **Validation / sanity checks** only (~**100** rows; split labeled `train` on Hub — treat as **`val` locally**) |
| **Rule** | **Exclude** every example in this dataset from the **training** corpus (no leakage). |

Use this split for **local `eval_project_a.py`**, learning curves, and reporting **internal val Haversine**. Course **leaderboard** may use a **different** hidden set.

### Training (subset + your published dataset)

| Item | Value |
|------|--------|
| **Source pool** | [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) (~**3.35k** rows, `train` split) |
| **Team action** | Draw a **reproducible subset** (e.g. **N = 1,000–2,000**) with **`shuffle(seed=…)` + `select(range(N))`** (or stratified by grid if you implement it). |
| **Dedup vs val** | Before training, **remove** any row that **matches** `gydou/released_img` (e.g. same image bytes, or **(lat, lon)** rounded + perceptual hash — document the rule). |
| **License** | Before publishing a derivative on HF, read the **5190-image-dataset** card and **only** redistribute if the license allows; **cite the parent** on your dataset card. |

**Team HF Dataset (deliverable):** Publish **your** subset (post-dedup) under your org, e.g. `your-org/cis5190-group5-train`, with:

- Parent link: `heidiywseo/5190-image-dataset`
- **N**, **random seed**, **dedup** method, **bbox** filter (if any), **date** / revision
- Statement that **`gydou/released_img` is not included** in train

Record the final URL in [SUBMISSION_LOG.md](SUBMISSION_LOG.md) and the report.

### Augmentation (training only — Group 5 policy)

Apply **only on the training split** during optimization (not on `gydou/released_img` val). **Validation / `preprocess.py`:** deterministic resize + center crop + ImageNet normalization (see below).

**Rationale:** Augmentation **improves robustness per location** (lighting, scale, occlusion, mild blur); it does **not** add new GPS labels — still prioritize **spatial coverage** in the chosen **N** examples. For **geo-localization**, very aggressive geometry (huge rotations / extreme crops) can erase campus cues (façades, paths), so we use **strong but bounded** policies and document **magnitude** for reproducibility.

#### Validation / inference (no randomness)

Used for `gydou/released_img`, leaderboard inference, and the submitted **`preprocess.py`** path:

1. **Resize** shorter edge to **256** px (keep aspect ratio).
2. **Center crop** **224×224**.
3. **ToTensor** + **ImageNet** `normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`.

(`training/augmentation.py` → `build_eval_transforms()`.)

#### Training (advanced stack — default)

Order matters (PIL ops first, then tensor ops):

| Step | Transform | Role |
|------|-----------|------|
| 1 | **RandomResizedCrop(224)** | `scale=(0.7, 1.0)`, `ratio=(0.85, 1.15)` — keeps more scene context than ImageNet-default extremes; still 224×224 for ResNet-style backbones. |
| 2 | **RandomHorizontalFlip** | `p=0.5` — campus symmetry is imperfect but flip helps lighting/crop diversity. |
| 3 | **RandomApply(GaussianBlur)** | `kernel_size=3`, `sigma=(0.1, 1.0)`, `p=0.1` — mild defocus / sensor variability. |
| 4 | [**RandAugment**](https://arxiv.org/abs/1909.13719) | `num_ops=2`, `magnitude=7` (torchvision) — **policy search** over augmentations (color, contrast, posterize, etc.) without hand-tuning each knob. **Moderate magnitude** limits destructive warps for building-heavy scenes. |
| 5 | **ToTensor** | — |
| 6 | [**RandomErasing**](https://arxiv.org/abs/1708.04896) | `p=0.1`, `scale=(0.02, 0.12)`, `ratio=(0.3, 3.3)` — occlusion robustness on tensor. |
| 7 | **Normalize** | Same ImageNet mean/std as eval. |

Implementation: `training/augmentation.py` → `build_train_transforms()`.

#### Alternatives / ablations (report-friendly)

- **Simpler baseline:** drop RandAugment + RandomErasing; use **ColorJitter**(brightness/contrast/saturation **0.2**, hue **0.02**) after the crop + flip — easier to describe, often weaker.
- **Different policy search:** swap RandAugment for **TrivialAugmentWide** (single random op per step; strong and stable) — change one line in `build_train_transforms()`.
- **If val Haversine worsens:** lower RandAugment **magnitude** (e.g. **5**) or disable **GaussianBlur** / **RandomErasing** first; log outcomes in [RESULTS.md](RESULTS.md).

**Hyperparameters to log:** `aug_policy` name (e.g. `group5_randa_m7`), RandAugment `(num_ops, magnitude)`, crop `(scale, ratio)`, RandomErasing `(p, scale)`.

### Phase B execution checklist

1. `load_dataset("gydou/released_img")` → export or stream **val** manifest; **never** merge into train.
2. `heidiywseo/5190-image-dataset` train split → **shuffle(seed)** → take **N** rows → **dedup** against val → optional **bbox** filter. The repo script [`scripts/build_group5_train_dataset.py`](../scripts/build_group5_train_dataset.py) does this using **streaming** by default (saves disk); optional **`--materialize`** uses full cache + map-style shuffle if you have space.
3. Push **team train** dataset to HF; update **SUBMISSION_LOG** + report.
4. Implement **train** `transforms` with aug; **val** = deterministic preprocess only.
5. Log **N_train**, **N_val**, **seed**, aug policy in [RESULTS.md](RESULTS.md) first run.

---

## Do we need to “curate” a dataset?

**Yes.** The course **basic components** require you to **collect, curate, clean, and split** a dataset ([Final Project Descriptions §1.2](../CIS%204190_5190%20Final%20Project%20Descriptions.pdf)). The **May 6** deliverables include **“The dataset you collected and used”** (§1.5).

**Curation** here means: **subset + dedup + documented seed + HF publication** of **your** training split, with **clear** attribution to [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset), plus **validation** protocol using [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) **without** training on it.

**Licensing:** Anything you publish on **Hugging Face** must be **redistributable** under stated terms; verify the **5190** dataset license before mirroring.

---

## Dataset size (minimum N?)

The **Final Project Descriptions** and **Project_Submission.pdf** **do not** specify a minimum **N**. Team target: **~1,000–2,000** train images from the **5190** pool (after dedup), plus **train-time augmentation**. Adjust **N** if val error plateaus or overfits.

---

## Problem scope (official test region)

Per **Image2GPS §2.1** in the Final Project Descriptions:

- The **evaluation** focus is **University of Pennsylvania campus** imagery.
- The **testing region** is the **rectangle from 33rd & Walnut to 34th & Spruce** (see **Figure 1** in that PDF).
- **Metric:** mean **Haversine distance (meters)**; lower is better (same as leaderboard).

**Bounding box for optional filtering** (fill from Figure 1 if you subset by geo):

| | Value |
|--|--------|
| Min latitude | TBD |
| Max latitude | TBD |
| Min longitude | TBD |
| Max longitude | TBD |

---

## Official / staff-released sample (`gydou/released_img`)

Treat [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) as the **released sample** for **your** validation pipeline. Per Ed #348: **not for training**. Dedup train against it.

---

## Ideal in-person protocol (course §2.4.1) vs our sources

The course describes walkway, multi-view, phone-upright capture. **Our train pool** is **online / course-adjacent** imagery from [`5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset). In the **report**, state **domain gap** (e.g. vs walkway phone photos) and how **aug + coverage** mitigate it.

---

## Preprocessing alignment (course baseline §2.3)

- Resize inputs to **224×224**
- Input **normalization** (ImageNet-style or as in baseline notebook)
- **Target normalization:** mean/variance of lat/lon on **train**; **invert** for metrics and **`predict`** (raw **degrees** for the grader)

Document exact transforms in [RUNBOOK.md](RUNBOOK.md).

---

## File layout (exports for training scripts)

Typical layout after export (local or HF):

- `images/` — JPEG/PNG
- `metadata.csv` — `file_name`, `Latitude`, `Longitude`

Compatible with grader column aliases (see [Project_Submission.pdf](../Project_Submission.pdf)).

---

## Cleaning

- Failed decodes, corrupt files → drop row  
- **Dedup:** vs **val** set; optional near-duplicate GPS bins  
- Document counts removed

---

## Splits (Group 5)

| Split | Source | Rows (approx.) | Notes |
|-------|--------|----------------|--------|
| **Validation** | [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) | ~100 | **Not** in train |
| **Train** | Subset of [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) | **N** (e.g. 1k–2k) | Seed + dedup documented |

---

## Final dataset statistics

TBD after Phase B: **N_train**, **N_val**, map / histogram of train lat-lon, aug policy version.

---

## Hugging Face Dataset (team publish)

- **Train subset URL:** TBD — your team repo after push  
- **Val reference:** [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img) (cite; do not re-host as “your train”)  
- **Card:** parent **`5190-image-dataset`**, license, **seed**, **N**, **dedup**, aug summary

---

## Known limitations

Domain shift vs staff-held test; GPS noise; possible overlap risk if dedup is weak — document mitigation.
