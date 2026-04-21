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

- [DATA.md](DATA.md) **Phase B**: val = [`gydou/released_img`](https://huggingface.co/datasets/gydou/released_img); train = reproducible subset of [`heidiywseo/5190-image-dataset`](https://huggingface.co/datasets/heidiywseo/5190-image-dataset) + **dedup** + **train-only augmentation**; team HF publish TBD.
- Updated [README.md](../README.md), [RUNBOOK.md](RUNBOOK.md), [SUBMISSION_LOG.md](SUBMISSION_LOG.md), [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md), [REPORT_OUTLINE.md](REPORT_OUTLINE.md).

### Phase B implementation — 2026-04-21

- **Aug policy** documented in [DATA.md](DATA.md): RandAugment + RandomErasing + geo-aware crop bounds; eval pipeline fixed; ablations (TrivialAugmentWide, simpler ColorJitter) noted.
- Code: [`training/augmentation.py`](../training/augmentation.py), [`scripts/build_group5_train_dataset.py`](../scripts/build_group5_train_dataset.py); [RUNBOOK.md](RUNBOOK.md) commands.
- [ ] Team still: run builder → publish HF train repo → fill [SUBMISSION_LOG.md](SUBMISSION_LOG.md) URL.

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
