# Documentation phase audits

Complete an entry **before closing each phase**. Second reviewer should scan **docs/** only when possible.

---

## Phase A — 2026-04-21

- **Reviewer(s):** TBD (team)
- [x] Files in documentation map created/updated for this phase
- [x] RUNBOOK: structure present (training TBD in Phase C)
- [x] RESULTS / SUBMISSION_LOG: templates ready
- [x] REPORT_OUTLINE: headings match course template
- [x] No contradictions between README, DATA, RESULTS, SUBMISSION_LOG (no filled conflicting numbers yet)

**Notes:** Phase A establishes repo layout and doc skeletons. AWS and HF URLs filled in later phases.

**Update:** Group ID **5** and roster (Prithvi Seshadri, Vamsi Krishna Naghichetty Kishore Kumar, Ishita Munshi) recorded in README, SUBMISSION_LOG, REPORT_OUTLINE, RUNBOOK, SUBMISSION_CHECKLIST.

**Update:** [DATA.md](DATA.md), [README.md](README.md), [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md), [REPORT_OUTLINE.md](REPORT_OUTLINE.md), [RUNBOOK.md](RUNBOOK.md) aligned with **CIS 4190_5190 Final Project Descriptions.pdf** (campus test box, baseline §2.3, test-sample exclusion §2.2, curation deliverable).

---

## Cleanup — 2026-04-21

- Removed experimental **`data/penn_wikimedia_curated/`** build (wrong / not vetted for licenses). No dataset scraper scripts retained in repo.
- **`data/*`** gitignored except [`data/.gitkeep`](../data/.gitkeep); team will add a **properly sourced** dataset and HF link when ready.

---

## Phase B plan locked — 2026-04-21

- [DATA.md](DATA.md) **Phase B**: val = [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img); train = reproducible subset of [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) + **dedup** + **train-only augmentation**; team HF: [`prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train) (**N=2000**, seed 51905).
- Updated [README.md](../README.md), [RUNBOOK.md](RUNBOOK.md), [SUBMISSION_LOG.md](SUBMISSION_LOG.md), [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md), [REPORT_OUTLINE.md](REPORT_OUTLINE.md).

### Phase B implementation — 2026-04-21

- **Aug policy** documented in [DATA.md](DATA.md): RandAugment + RandomErasing + geo-aware crop bounds; eval pipeline fixed; ablations (TrivialAugmentWide, simpler ColorJitter) noted.
- Code: [`training/augmentation.py`](../training/augmentation.py), [`scripts/build_group5_train_dataset.py`](../scripts/build_group5_train_dataset.py); [RUNBOOK.md](RUNBOOK.md) commands.
- [x] Team **train** Hub: [`prith27/cis5190-group5-train`](https://huggingface.co/datasets/prith27/cis5190-group5-train) — [SUBMISSION_LOG.md](SUBMISSION_LOG.md) updated.

### Phase B dataset published — 2026-04-21

- **URL:** [https://huggingface.co/datasets/prith27/cis5190-group5-train](https://huggingface.co/datasets/prith27/cis5190-group5-train) (2k rows, seed 51905). README, DATA, SUBMISSION_LOG, RUNBOOK, RESULTS, REPORT_OUTLINE, SUBMISSION_CHECKLIST cross-links updated for Git push.

---

## Phase C — 2026-04-22

- **Reviewer(s):** TBD
- [x] All docs touched by this phase updated
- [x] RUNBOOK: training + resume + export commands documented
- [x] RESULTS: experiment row added with run id, units, and notes
- [x] SUBMISSION_LOG: artifact rows clarified for model/model.pt/preprocess metadata
- [x] REPORT_OUTLINE: Phase C methodology/evaluation text added
- [x] Cross-check: README links updated for `model.py`, `preprocess.py`, and training script

**Notes:** Implemented partial-freeze ViT Phase C pipeline (`training/run_train_vit.py`) plus submission files (`model.py`, `preprocess.py`). **Update:** vit-b16-pf3-v1 EC2 run logged in RESULTS; `model.py` norms from `artifacts/norm_stats.json`; `model.pt` SHA in SUBMISSION_LOG; `/artifacts/*` gitignored (keep `.gitkeep` only).

---

## Phase D — 2026-04-22

- **Reviewer(s):** TBD
- [x] [RUNBOOK.md](RUNBOOK.md): Phase D export + `eval_project_a.py` commands
- [x] [RESULTS.md](RESULTS.md): local eval **avg_distance_m 62.196** (N=100 exported `gydou/released_img`)
- [x] [scripts/export_released_img_for_eval.py](../scripts/export_released_img_for_eval.py) added
- [x] [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md): core I/O items checked post-verify
- [ ] Phase E leaderboard rows / alias (pending)

---

## Template for future phases

```markdown
## Phase [X] — [YYYY-MM-DD]
- **Reviewer(s):**
- [ ] All docs touched by this phase updated
- [ ] RUNBOOK: new commands tested where applicable
- [ ] RESULTS or SUBMISSION_LOG: numbers with units + run id
- [ ] REPORT_OUTLINE: paste-ready paragraphs for §2/§3
- [ ] Cross-check: README links still valid
```
